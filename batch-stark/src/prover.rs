// Modified by ReallyMe Technologies Limited, 2026.
// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Batch-STARK prover: commits traces, computes quotient polynomials, and
//! produces opening proofs for multiple AIR instances in a single FRI batch.

use alloc::vec;
use alloc::vec::Vec;

#[cfg(debug_assertions)]
use p3_air::DebugConstraintBuilder;
use p3_air::symbolic::{AirLayout, SymbolicExpressionExt, UndeclaredNextRowRead};
use p3_air::{Air, RowWindow};
use p3_challenger::GrindingChallenger;
#[cfg(feature = "secret-memory")]
use p3_commit::{
    OpeningRequest, SecretDetachedPcsCommitter, SecretGeneratedQuotientCommitError,
    SecretOpeningRequest, SecretPcsCommitter, SecretPcsMixedOpener, SecretPcsMixedOpeningRequest,
    SecretProverData, SecretQuotientCommitmentInput, SecretQuotientCommitmentPlan,
    SecretUnivariateStarkPcsCommitter, SecretUnivariateStarkPcsEvaluator,
};
use p3_commit::{Pcs, PolynomialSpace, UnivariateStarkPcs};
use p3_field::{
    Algebra, BasedVectorSpace, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
    PrimeField64,
};
use p3_lookup::logup::LogUpGadget;
use p3_lookup::{
    InteractionSymbolicBuilder, Lookup, LookupProtocol, LookupTerminal,
    check_multiplicity_height_bound,
};
#[cfg(feature = "secret-memory")]
use p3_lookup::{Kind as LookupKind, SecretLookupGenerationError};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
#[cfg(feature = "secret-memory")]
use p3_matrix::secret::SecretMatrixError;
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{
    AirConstraints, OpenedValues, PackedChallenge, PackedVal, PreprocessedOpenedValues,
    ProverConstraintFolder,
};
use p3_util::{DisjointMutPtr, log2_strict_usize};
use tracing::{debug_span, info_span, instrument};
#[cfg(feature = "secret-memory")]
use zeroize::Zeroize;

use crate::ProvingError;
use crate::common::ProverData;
use crate::config::{
    Challenge, Commitment, Domain, PcsProverError, StarkGenericConfig as SGC, Val,
};
use crate::folder::ProverConstraintFolderWithLookups;
use crate::proof::{BatchCommitments, BatchOpenedValues, BatchProof, OpenedValuesWithLookups};
#[cfg(feature = "secret-memory")]
use crate::symbolic::get_log_num_quotient_chunks_for_domain;
use crate::symbolic::{evaluate_constraints, get_constraint_layout, get_symbolic_constraints};
use crate::transcript::{BatchProverTranscript, BatchShape};

/// Per-instance quotient output: the chunk domains and their committed LDE matrices.
type InstanceQuotient<SC> = (Vec<Domain<SC>>, Vec<RowMajorMatrix<Val<SC>>>);

/// Stable failures from guarded quotient evaluation.
#[cfg(feature = "secret-memory")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SecretQuotientError {
    /// The AIR requests a boundary-I/O binding unsupported by batch-STARK.
    #[error("guarded quotient evaluation does not support boundary-I/O public values")]
    UnsupportedPublicBoundary,
    /// A domain or evaluation matrix has inconsistent geometry.
    #[error("guarded quotient evaluation shape is invalid")]
    InvalidShape,
    /// A derived row, buffer, or packing dimension cannot be represented.
    #[error("guarded quotient evaluation dimensions overflow")]
    DimensionOverflow,
    /// A secret-derived output or scratch buffer could not be reserved.
    #[error("guarded quotient evaluation allocation failed")]
    AllocationFailed,
}

/// Failure while moving a guarded extension-field quotient into its PCS
/// commitment owner.
#[cfg(feature = "secret-memory")]
#[derive(Debug, thiserror::Error)]
pub enum SecretQuotientCommitError<PcsError> {
    /// The batch adapter could not reserve its guarded PCS input collection.
    #[error("guarded quotient batch allocation failed")]
    AllocationFailed,
    /// Guarded extension-to-base conversion rejected the quotient matrix.
    #[error("guarded quotient representation conversion failed")]
    Matrix(#[from] SecretMatrixError),
    /// The guarded PCS rejected quotient splitting, extension, or commitment.
    #[error("guarded quotient PCS commitment failed")]
    Pcs(PcsError),
}

/// Failure while acquiring guarded quotient-domain matrices and evaluating
/// their AIR quotient.
#[cfg(feature = "secret-memory")]
#[derive(Debug, thiserror::Error)]
pub enum SecretPcsQuotientError<PcsError> {
    /// The guarded main-trace commitment could not be evaluated on the quotient domain.
    #[error("guarded main-trace domain evaluation failed")]
    MainEvaluation(PcsError),
    /// The guarded permutation commitment could not be evaluated on the quotient domain.
    #[error("guarded permutation domain evaluation failed")]
    PermutationEvaluation(PcsError),
    /// AIR quotient evaluation rejected the acquired matrix geometry or allocation.
    #[error("guarded AIR quotient evaluation failed")]
    Quotient(#[from] SecretQuotientError),
}

/// Failure while transferring guarded main traces into the PCS commitment
/// owner required by the batch transcript.
#[cfg(feature = "secret-memory")]
#[derive(Debug, thiserror::Error)]
pub enum SecretMainTraceCommitError<PcsError> {
    /// A batch commitment must contain at least one main trace.
    #[error("guarded main-trace batch is empty")]
    EmptyBatch,
    /// Trace and domain collections must describe the same batch.
    #[error("guarded main-trace batch shape is invalid")]
    InputCountMismatch,
    /// The commitment-input vector could not be reserved.
    #[error("guarded main-trace commitment allocation failed")]
    AllocationFailed,
    /// A source trace could not be copied into a distinct guarded owner.
    #[error("guarded main-trace copy failed")]
    Matrix(#[from] SecretMatrixError),
    /// The guarded PCS rejected the main-trace commitment.
    #[error("guarded main-trace PCS commitment failed")]
    Pcs(PcsError),
}

/// Failure while flattening and committing guarded lookup-permutation traces.
#[cfg(feature = "secret-memory")]
#[derive(Debug, thiserror::Error)]
pub enum SecretPermutationCommitError<PcsError> {
    /// A permutation commitment must contain at least one generated trace.
    #[error("guarded permutation-trace batch is empty")]
    EmptyBatch,
    /// The commitment-input vector could not be reserved.
    #[error("guarded permutation-trace commitment allocation failed")]
    AllocationFailed,
    /// A guarded extension-field trace could not be flattened safely.
    #[error("guarded permutation-trace representation conversion failed")]
    Matrix(#[from] SecretMatrixError),
    /// The guarded PCS rejected the permutation commitment.
    #[error("guarded permutation-trace PCS commitment failed")]
    Pcs(PcsError),
}

/// Stable failures from the end-to-end guarded batch-STARK prover.
///
/// The variants identify protocol phases without carrying witness bytes,
/// public inputs, arbitrary strings, or allocator-provided diagnostics.
#[cfg(feature = "secret-memory")]
#[derive(Debug, thiserror::Error)]
pub enum SecretBatchProvingError<CommitError, EvaluationError, OpenError>
where
    CommitError: core::fmt::Debug,
    EvaluationError: core::fmt::Debug,
    OpenError: core::fmt::Debug,
{
    /// At least one AIR instance is required by the batch proof format.
    #[error("guarded batch prover received an empty batch")]
    EmptyBatch,
    /// AIRs, traces, public values, or cached lookup metadata are misaligned.
    #[error("guarded batch prover input counts are inconsistent")]
    InputCountMismatch,
    /// Batch-STARK requires public values to be bound by AIR constraints.
    #[error("guarded batch prover does not support boundary-I/O public values")]
    UnsupportedPublicBoundary,
    /// A public-value vector does not match its AIR declaration.
    #[error("guarded batch prover public-value count is invalid")]
    PublicValueCountMismatch,
    /// A source trace is empty, non-power-of-two, or has invalid row geometry.
    #[error("guarded batch prover trace shape is invalid")]
    InvalidTraceShape,
    /// Cached preprocessing metadata is inconsistent with the AIR batch.
    #[error("guarded batch prover preprocessing metadata is invalid")]
    InvalidPreprocessing,
    /// Public preprocessing exists but its reusable prover data is absent.
    #[error("guarded batch prover preprocessing data is missing")]
    MissingPreprocessingData,
    /// Lookup transcript metadata would violate its canonical layout.
    #[error("guarded batch prover lookup layout is invalid")]
    InvalidLookupLayout,
    /// The LogUp multiplicity bound is not sound for these trace heights.
    #[error("guarded batch prover lookup multiplicity bound is exceeded")]
    LookupMultiplicityBound,
    /// A domain size, chunk count, offset, or allocation length overflowed.
    #[error("guarded batch prover dimensions overflow")]
    DimensionOverflow,
    /// The PCS could not construct a required disjoint quotient domain.
    #[error("guarded batch prover quotient domain is unsupported")]
    UnsupportedQuotientDomain,
    /// The PCS domain cannot express the AIR's next-row opening point.
    #[error("guarded batch prover next-row opening is unsupported")]
    UnsupportedNextRow,
    /// A protocol-owned vector allocation failed.
    #[error("guarded batch prover allocation failed")]
    AllocationFailed,
    /// Guarded LogUp generation rejected witness or protocol geometry.
    #[error("guarded batch prover lookup generation failed")]
    LookupGeneration(#[source] SecretLookupGenerationError),
    /// The PCS could not reconstruct a committed main trace for LogUp generation.
    #[error("guarded batch prover lookup trace reconstruction failed")]
    LookupTraceEvaluation(EvaluationError),
    /// The guarded main-trace commitment failed.
    #[error("guarded batch prover main-trace commitment failed")]
    MainCommitment(#[source] SecretMainTraceCommitError<CommitError>),
    /// The guarded lookup-permutation commitment failed.
    #[error("guarded batch prover permutation commitment failed")]
    PermutationCommitment(#[source] SecretPermutationCommitError<CommitError>),
    /// Guarded quotient-domain evaluation or AIR evaluation failed.
    #[error("guarded batch prover quotient evaluation failed")]
    QuotientEvaluation(#[source] SecretPcsQuotientError<EvaluationError>),
    /// The shared guarded quotient commitment failed.
    #[error("guarded batch prover quotient commitment failed")]
    QuotientCommitment(#[source] SecretQuotientCommitError<CommitError>),
    /// The guarded randomization-polynomial commitment failed.
    #[error("guarded batch prover randomization commitment failed")]
    RandomizationCommitment(CommitError),
    /// A hiding PCS failed to return its required randomization commitment.
    #[error("guarded batch prover randomization commitment is missing")]
    MissingRandomizationCommitment,
    /// The guarded mixed public/private PCS opening failed.
    #[error("guarded batch prover opening failed")]
    Opening(OpenError),
    /// The PCS returned opened values inconsistent with the requested layout.
    #[error("guarded batch prover returned an invalid opening shape")]
    InvalidOpeningShape,
}

/// A completed phase in the guarded batch-STARK proving lifecycle.
///
/// Observers receive only protocol structure. No witness, public-input,
/// commitment, challenge, or proof value crosses this boundary.
#[cfg(feature = "secret-memory")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SecretBatchProvingPhase {
    /// Input geometry and transcript metadata have been validated and derived.
    Preparation,
    /// The guarded main-trace commitment has been constructed.
    MainCommitment,
    /// Lookup permutation traces have been generated and source traces erased.
    LookupGeneration,
    /// The optional lookup-permutation commitment has been constructed.
    PermutationCommitment,
    /// One instance quotient has been evaluated on its quotient domain.
    QuotientEvaluation { instance_index: usize },
    /// The shared quotient commitment, including nested instance evaluation.
    QuotientCommitment,
    /// The optional hiding randomization commitment has been constructed.
    RandomizationCommitment,
    /// The mixed PCS opening proof has been constructed.
    Opening,
    /// Public proof assembly has completed.
    ProofAssembly,
}

/// Result of comparing one directly reconstructed quotient chunk with the
/// corresponding interleaved rows of the verified full-domain evaluation.
///
/// This contains no witness values. It is intended for audit and profiling
/// runs that need to validate a future bounded-memory quotient strategy
/// against the established prover path.
#[cfg(feature = "secret-memory")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SecretQuotientChunkAuditStatus {
    /// Both guarded matrices have identical shape and field elements.
    Equivalent,
    /// Reconstruction completed but produced a different guarded matrix.
    Mismatch,
    /// The direct reconstruction rejected its inputs with a typed error.
    EvaluationFailed,
}

/// Receives non-secret guarded-prover phase boundaries.
///
/// The hook is deliberately infallible: diagnostics must not change proof
/// success, transcript state, or secret-owner cleanup. Implementations should
/// avoid blocking in production; profiling tools may perform heavier sampling
/// when the caller explicitly opts into the observed proving entry point.
#[cfg(feature = "secret-memory")]
pub trait SecretBatchProvingObserver {
    /// Record that `phase` is about to begin.
    fn phase_started(&mut self, _phase: SecretBatchProvingPhase) {}

    /// Record that `phase` completed.
    fn phase_completed(&mut self, _phase: SecretBatchProvingPhase) {}

    /// Opt into the expensive quotient-chunk equivalence audit.
    ///
    /// The default remains disabled so ordinary proving does not duplicate
    /// secret polynomial evaluation or change its memory profile.
    fn quotient_chunk_audit_requested(&self) -> bool {
        false
    }

    /// Record the non-secret result for one instance and quotient chunk.
    fn quotient_chunk_audited(
        &mut self,
        _instance_index: usize,
        _chunk_index: usize,
        _status: SecretQuotientChunkAuditStatus,
    ) {
    }
}

#[cfg(feature = "secret-memory")]
struct NoopSecretBatchProvingObserver;

#[cfg(feature = "secret-memory")]
impl SecretBatchProvingObserver for NoopSecretBatchProvingObserver {}

/// Guarded PCS prover data used by the batch-STARK secret-memory capability.
#[cfg(feature = "secret-memory")]
pub type SecretPcsData<SC> = SecretProverData<
    <<SC as SGC>::Pcs as SecretPcsCommitter<
        <SC as SGC>::Challenge,
        <SC as SGC>::Challenger,
    >>::SecretData,
>;

/// Output of guarded PCS domain acquisition followed by AIR quotient evaluation.
#[cfg(feature = "secret-memory")]
pub type SecretPcsQuotientResult<SC> = Result<
    SecretProverData<RowMajorMatrix<<SC as SGC>::Challenge>>,
    SecretPcsQuotientError<
        <<SC as SGC>::Pcs as SecretUnivariateStarkPcsEvaluator<
            <SC as SGC>::Challenge,
            <SC as SGC>::Challenger,
        >>::SecretEvaluationError,
    >,
>;

/// Output of guarded PCS domain acquisition followed by AIR quotient
/// evaluation in the base-field layout consumed by the PCS.
#[cfg(feature = "secret-memory")]
type SecretPcsBaseQuotientResult<SC> = Result<
    SecretProverData<RowMajorMatrix<Val<SC>>>,
    SecretPcsQuotientError<
        <<SC as SGC>::Pcs as SecretUnivariateStarkPcsEvaluator<
            <SC as SGC>::Challenge,
            <SC as SGC>::Challenger,
        >>::SecretEvaluationError,
    >,
>;

/// Chunked base-field quotient output produced without a full intermediate.
#[cfg(feature = "secret-memory")]
type SecretPcsBaseQuotientChunksResult<SC> = Result<
    Vec<SecretProverData<RowMajorMatrix<Val<SC>>>>,
    SecretPcsQuotientError<
        <<SC as SGC>::Pcs as SecretUnivariateStarkPcsEvaluator<
            <SC as SGC>::Challenge,
            <SC as SGC>::Challenger,
        >>::SecretEvaluationError,
    >,
>;

/// Remaining guarded chunks for the instance currently being committed.
#[cfg(feature = "secret-memory")]
type PendingSecretQuotientChunks<SC> = (
    usize,
    alloc::vec::IntoIter<SecretProverData<RowMajorMatrix<Val<SC>>>>,
);

/// Output of the guarded batch main-trace commitment handoff.
#[cfg(feature = "secret-memory")]
pub type SecretMainTraceCommitResult<SC> =
    Result<
        (Commitment<SC>, SecretPcsData<SC>),
        SecretMainTraceCommitError<
            <<SC as SGC>::Pcs as SecretPcsCommitter<
                <SC as SGC>::Challenge,
                <SC as SGC>::Challenger,
            >>::SecretCommitError,
        >,
    >;

/// One guarded lookup-permutation trace paired with its PCS evaluation domain.
#[cfg(feature = "secret-memory")]
pub type SecretPermutationTrace<SC> = (
    Domain<SC>,
    SecretProverData<RowMajorMatrix<<SC as SGC>::Challenge>>,
);

/// Output of the guarded batch permutation-trace commitment handoff.
#[cfg(feature = "secret-memory")]
pub type SecretPermutationCommitResult<SC> =
    Result<
        (Commitment<SC>, SecretPcsData<SC>),
        SecretPermutationCommitError<
            <<SC as SGC>::Pcs as SecretPcsCommitter<
                <SC as SGC>::Challenge,
                <SC as SGC>::Challenger,
            >>::SecretCommitError,
        >,
    >;

/// Output of the guarded quotient commitment handoff.
#[cfg(feature = "secret-memory")]
pub type SecretQuotientCommitResult<SC> =
    Result<
        (
            Commitment<SC>,
            SecretProverData<
                <<SC as SGC>::Pcs as SecretPcsCommitter<
                    <SC as SGC>::Challenge,
                    <SC as SGC>::Challenger,
                >>::SecretData,
            >,
        ),
        SecretQuotientCommitError<
            <<SC as SGC>::Pcs as SecretPcsCommitter<
                <SC as SGC>::Challenge,
                <SC as SGC>::Challenger,
            >>::SecretCommitError,
        >,
    >;

/// Result of the complete guarded batch-STARK proving pipeline.
#[cfg(feature = "secret-memory")]
pub type SecretBatchProvingResult<SC> =
    Result<
        BatchProof<SC>,
        SecretBatchProvingError<
            <<SC as SGC>::Pcs as SecretPcsCommitter<
                <SC as SGC>::Challenge,
                <SC as SGC>::Challenger,
            >>::SecretCommitError,
            <<SC as SGC>::Pcs as SecretUnivariateStarkPcsEvaluator<
                <SC as SGC>::Challenge,
                <SC as SGC>::Challenger,
            >>::SecretEvaluationError,
            <<SC as SGC>::Pcs as p3_commit::SecretPcsOpener<
                <SC as SGC>::Challenge,
                <SC as SGC>::Challenger,
            >>::SecretOpenError,
        >,
    >;

/// One guarded instance quotient awaiting the shared batch commitment.
#[cfg(feature = "secret-memory")]
pub struct SecretBatchQuotient<SC>
where
    SC: SGC,
    SC::Challenge: Zeroize,
{
    domain: Domain<SC>,
    evaluations: SecretProverData<RowMajorMatrix<SC::Challenge>>,
    num_chunks: usize,
}

#[cfg(feature = "secret-memory")]
impl<SC> SecretBatchQuotient<SC>
where
    SC: SGC,
    SC::Challenge: Zeroize,
{
    /// Bind one extension-field quotient to its domain and chunk count.
    pub const fn new(
        domain: Domain<SC>,
        evaluations: SecretProverData<RowMajorMatrix<SC::Challenge>>,
        num_chunks: usize,
    ) -> Self {
        Self {
            domain,
            evaluations,
            num_chunks,
        }
    }

    fn into_parts(
        self,
    ) -> (
        Domain<SC>,
        SecretProverData<RowMajorMatrix<SC::Challenge>>,
        usize,
    ) {
        (self.domain, self.evaluations, self.num_chunks)
    }
}

#[cfg(feature = "secret-memory")]
struct SecretQuotientScratch<SC: SGC>
where
    Val<SC>: Zeroize,
    PackedVal<SC>: Zeroize,
    PackedChallenge<SC>: Zeroize,
{
    permutation: SecretProverData<Vec<PackedChallenge<SC>>>,
    main: SecretProverData<Vec<PackedVal<SC>>>,
    preprocessed: SecretProverData<Vec<PackedVal<SC>>>,
    scalar_pack: SecretProverData<Vec<Val<SC>>>,
    extension_coefficients: SecretProverData<Vec<PackedVal<SC>>>,
}

#[cfg(feature = "secret-memory")]
fn guarded_vec_with_capacity<T: Zeroize>(
    capacity: usize,
) -> Result<SecretProverData<Vec<T>>, SecretQuotientError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| SecretQuotientError::AllocationFailed)?;
    Ok(SecretProverData::new(values))
}

#[cfg(feature = "secret-memory")]
impl<SC> SecretQuotientScratch<SC>
where
    SC: SGC,
    Val<SC>: Zeroize,
    PackedVal<SC>: Zeroize,
    PackedChallenge<SC>: Zeroize,
{
    fn new(
        permutation_capacity: usize,
        main_capacity: usize,
        preprocessed_capacity: usize,
        pack_width: usize,
        extension_degree: usize,
    ) -> Result<Self, SecretQuotientError> {
        Ok(Self {
            permutation: guarded_vec_with_capacity(permutation_capacity)?,
            main: guarded_vec_with_capacity(main_capacity)?,
            preprocessed: guarded_vec_with_capacity(preprocessed_capacity)?,
            scalar_pack: guarded_vec_with_capacity(pack_width)?,
            extension_coefficients: guarded_vec_with_capacity(extension_degree)?,
        })
    }

    fn erase_for_reuse(&mut self) {
        self.permutation.zeroize();
        self.main.zeroize();
        self.preprocessed.zeroize();
        self.scalar_pack.zeroize();
        self.extension_coefficients.zeroize();
    }
}

/// A single AIR instance bundled with its execution trace, public inputs,
/// and lookup declarations.
///
/// One or more of these are passed to the batch prover.
#[derive(Debug)]
pub struct StarkInstance<'a, SC: SGC, A> {
    /// The AIR (constraint system) for this instance.
    pub air: &'a A,
    /// Execution trace as a row-major matrix over the base field.
    pub trace: &'a RowMajorMatrix<Val<SC>>,
    /// Public input values exposed by this instance.
    pub public_values: Vec<Val<SC>>,
}

impl<'a, SC: SGC, A> StarkInstance<'a, SC, A> {
    /// Build instances from parallel slices of AIRs, traces, and public values.
    ///
    /// # Panics
    /// Panics if `airs`, `traces`, and `public_values` don't all have the same length.
    pub fn new_multiple(
        airs: &'a [A],
        traces: &'a [&'a RowMajorMatrix<Val<SC>>],
        public_values: &[Vec<Val<SC>>],
    ) -> Vec<Self> {
        assert_eq!(airs.len(), traces.len(), "airs/traces length mismatch");
        assert_eq!(
            airs.len(),
            public_values.len(),
            "airs/public_values length mismatch"
        );
        airs.iter()
            .zip(traces.iter())
            .zip(public_values.iter())
            .map(|((air, trace), public_values)| Self {
                air,
                trace,
                public_values: public_values.clone(),
            })
            .collect()
    }
}

/// An AIR instance whose source trace remains under a zeroizing owner.
///
/// [`prove_batch_secret`] preserves guarded ownership through the complete
/// batch-STARK proving path. The narrower [`prove_batch_guarded_inputs`]
/// staging API erases only this source matrix and must not be treated as an
/// end-to-end erasure boundary.
#[cfg(feature = "secret-memory")]
pub struct GuardedStarkInstance<'a, SC: SGC, A>
where
    Val<SC>: Zeroize,
{
    /// The AIR (constraint system) for this instance.
    pub air: &'a A,
    /// Execution trace owned by an erasing allocation guard.
    pub trace: SecretProverData<RowMajorMatrix<Val<SC>>>,
    /// Public input values exposed by this instance.
    pub public_values: Vec<Val<SC>>,
}

#[cfg(feature = "secret-memory")]
impl<SC: SGC, A> GuardedStarkInstance<'_, SC, A>
where
    Val<SC>: Zeroize,
{
    /// Return the number of execution-trace rows without releasing the guarded allocation.
    pub fn trace_height(&self) -> usize {
        self.trace.height()
    }
}

/// Commit a batch of guarded main traces without exposing an ordinary matrix
/// copy.
///
/// This compatibility helper preserves the source traces for callers that
/// need to reuse them. Each PCS input is copied directly into a second
/// zeroizing owner, which the guarded PCS consumes and retains through its
/// commitment-data lifecycle. The complete guarded batch prover instead uses
/// an internal consuming handoff and reconstructs lookup inputs from committed
/// coefficients after Fiat-Shamir fixes the lookup challenges.
#[cfg(feature = "secret-memory")]
pub fn commit_main_traces_secret<SC>(
    pcs: &SC::Pcs,
    domains: &[Domain<SC>],
    traces: &[SecretProverData<RowMajorMatrix<Val<SC>>>],
) -> SecretMainTraceCommitResult<SC>
where
    SC: SGC,
    Val<SC>: PrimeField64 + Zeroize,
    Domain<SC>: Clone,
    SC::Pcs: SecretPcsCommitter<SC::Challenge, SC::Challenger>,
{
    if traces.is_empty() {
        return Err(SecretMainTraceCommitError::EmptyBatch);
    }
    if domains.len() != traces.len() {
        return Err(SecretMainTraceCommitError::InputCountMismatch);
    }

    let mut owned_traces = Vec::new();
    owned_traces
        .try_reserve_exact(traces.len())
        .map_err(|_| SecretMainTraceCommitError::AllocationFailed)?;
    for trace in traces {
        owned_traces.push(trace.try_clone_guarded()?);
    }

    let mut evaluations = Vec::new();
    evaluations
        .try_reserve_exact(owned_traces.len())
        .map_err(|_| SecretMainTraceCommitError::AllocationFailed)?;
    for (domain, trace) in domains.iter().cloned().zip(owned_traces) {
        evaluations.push((domain, trace));
    }
    pcs.commit_secret(evaluations)
        .map_err(SecretMainTraceCommitError::Pcs)
}

/// Consume guarded main traces into the PCS commitment owner.
///
/// This is the end-to-end batch prover path. Unlike the public compatibility
/// helper, it does not retain a duplicate source matrix after commitment.
#[cfg(feature = "secret-memory")]
fn commit_owned_main_traces_secret<SC>(
    pcs: &SC::Pcs,
    domains: &[Domain<SC>],
    traces: Vec<SecretProverData<RowMajorMatrix<Val<SC>>>>,
) -> SecretMainTraceCommitResult<SC>
where
    SC: SGC,
    Val<SC>: PrimeField64 + Zeroize,
    Domain<SC>: Clone,
    SC::Pcs: SecretDetachedPcsCommitter<SC::Challenge, SC::Challenger>,
{
    if traces.is_empty() {
        return Err(SecretMainTraceCommitError::EmptyBatch);
    }
    if domains.len() != traces.len() {
        return Err(SecretMainTraceCommitError::InputCountMismatch);
    }

    let mut evaluations = Vec::new();
    evaluations
        .try_reserve_exact(traces.len())
        .map_err(|_| SecretMainTraceCommitError::AllocationFailed)?;
    for (domain, trace) in domains.iter().cloned().zip(traces) {
        evaluations.push((domain, trace));
    }

    pcs.commit_detached_secret(evaluations)
        .map_err(SecretMainTraceCommitError::Pcs)
}

/// Flatten and commit generated lookup-permutation traces without releasing
/// their guarded extension-field allocations.
///
/// The representation change consumes and erases each extension-field owner,
/// then transfers the guarded base-field matrix directly into the PCS.
#[cfg(feature = "secret-memory")]
pub fn commit_permutation_traces_secret<SC>(
    pcs: &SC::Pcs,
    traces: Vec<SecretPermutationTrace<SC>>,
) -> SecretPermutationCommitResult<SC>
where
    SC: SGC,
    Val<SC>: Zeroize,
    SC::Challenge: Zeroize,
    SC::Pcs: SecretPcsCommitter<SC::Challenge, SC::Challenger>,
{
    if traces.is_empty() {
        return Err(SecretPermutationCommitError::EmptyBatch);
    }

    let mut evaluations = Vec::new();
    evaluations
        .try_reserve_exact(traces.len())
        .map_err(|_| SecretPermutationCommitError::AllocationFailed)?;
    for (domain, trace) in traces {
        let flattened = trace
            .try_flatten_extension_to_base::<Val<SC>>()
            .map_err(SecretPermutationCommitError::Matrix)?;
        evaluations.push((domain, flattened));
    }

    pcs.commit_secret(evaluations)
        .map_err(SecretPermutationCommitError::Pcs)
}

/// Consume guarded lookup-permutation traces into detached PCS state.
///
/// The public helper above preserves its retained-data behavior for callers
/// that use it independently. The end-to-end prover can detach immediately
/// because later quotient evaluation reconstructs from coefficients and later
/// openings authenticate against the detached Merkle state.
#[cfg(feature = "secret-memory")]
fn commit_owned_permutation_traces_secret<SC>(
    pcs: &SC::Pcs,
    traces: Vec<SecretPermutationTrace<SC>>,
) -> SecretPermutationCommitResult<SC>
where
    SC: SGC,
    Val<SC>: Zeroize,
    SC::Challenge: Zeroize,
    SC::Pcs: SecretDetachedPcsCommitter<SC::Challenge, SC::Challenger>,
{
    if traces.is_empty() {
        return Err(SecretPermutationCommitError::EmptyBatch);
    }

    let mut evaluations = Vec::new();
    evaluations
        .try_reserve_exact(traces.len())
        .map_err(|_| SecretPermutationCommitError::AllocationFailed)?;
    for (domain, trace) in traces {
        let flattened = trace
            .try_flatten_extension_to_base::<Val<SC>>()
            .map_err(SecretPermutationCommitError::Matrix)?;
        evaluations.push((domain, flattened));
    }

    pcs.commit_detached_secret(evaluations)
        .map_err(SecretPermutationCommitError::Pcs)
}

trait BatchTrace<T> {
    fn matrix(&self) -> &RowMajorMatrix<T>;

    fn release_source(&mut self) {}
}

struct BorrowedBatchTrace<'a, T> {
    matrix: &'a RowMajorMatrix<T>,
}

impl<T> BatchTrace<T> for BorrowedBatchTrace<'_, T> {
    fn matrix(&self) -> &RowMajorMatrix<T> {
        self.matrix
    }
}

#[cfg(feature = "secret-memory")]
struct GuardedBatchTrace<T>
where
    T: Zeroize,
{
    matrix: SecretProverData<RowMajorMatrix<T>>,
}

#[cfg(feature = "secret-memory")]
impl<T> BatchTrace<T> for GuardedBatchTrace<T>
where
    T: Clone + Send + Sync + Zeroize,
{
    fn matrix(&self) -> &RowMajorMatrix<T> {
        &self.matrix
    }

    fn release_source(&mut self) {
        self.matrix.zeroize();
        self.matrix.values = Vec::new();
        self.matrix.width = 0;
    }
}

#[cfg(all(test, feature = "secret-memory"))]
mod guarded_trace_tests {
    use alloc::vec;

    use p3_commit::SecretProverData;
    use p3_matrix::dense::RowMajorMatrix;

    use super::{BatchTrace, GuardedBatchTrace};

    #[test]
    fn release_source_erases_shape_and_releases_backing_allocation() {
        let matrix = RowMajorMatrix::new(vec![0xA5_u8; 16], 4);
        let mut trace = GuardedBatchTrace {
            matrix: SecretProverData::new(matrix),
        };

        trace.release_source();

        assert!(trace.matrix.values.is_empty());
        assert_eq!(trace.matrix.values.capacity(), 0);
        assert_eq!(trace.matrix.width, 0);
    }
}

/// Generate a batch STARK proof for all provided instances.
///
/// # Overview
///
/// Runs the full prover pipeline:
/// - Commit to execution traces (single batched commitment).
/// - Generate and commit permutation traces for lookup arguments.
/// - Compute quotient polynomials per instance.
/// - Commit to quotient chunks.
/// - Open all commitments at a random out-of-domain point.
///
/// # Arguments
///
/// - `config`: STARK configuration (PCS, challenger, ZK flag).
/// - `instances`: one entry per AIR instance with its trace and public values.
/// - `prover_data`: precomputed common data and preprocessed commitments.
///
/// # Returns
///
/// A self-contained batch proof that can be verified with `verify_batch`.
///
/// # Errors
///
/// No partial proof is ever emitted.
///
/// - An empty list of instances is rejected up front as an empty batch.
/// - Configuration or disclosure-budget failures return the PCS error with its proving phase.
#[instrument(skip_all)]
pub fn prove_batch<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>
        + Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
    #[cfg(not(debug_assertions))] A: for<'a> Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
>(
    config: &SC,
    instances: &[StarkInstance<'_, SC, A>],
    prover_data: &ProverData<SC>,
) -> Result<BatchProof<SC>, ProvingError<PcsProverError<SC>>>
where
    SC: SGC,
    Val<SC>: PrimeField64,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
    Domain<SC>: Send + Sync,
    SC::Pcs: Sync,
    PcsProverError<SC>: Send,
    SC::Challenger: GrindingChallenger<Witness = Val<SC>>,
    <SC::Pcs as p3_commit::Pcs<SC::Challenge, SC::Challenger>>::ProverData: Sync,
    <SC::Pcs as p3_commit::Pcs<SC::Challenge, SC::Challenger>>::Commitment: Sync,
{
    // A batch commitment needs at least one matrix.
    // Reject an empty batch here, before the commitment scheme fails on it.
    if instances.is_empty() {
        return Err(ProvingError::EmptyBatch);
    }

    let airs = instances
        .iter()
        .map(|instance| instance.air)
        .collect::<Vec<_>>();
    let public_values = instances
        .iter()
        .map(|instance| instance.public_values.as_slice())
        .collect::<Vec<_>>();
    let mut traces = instances
        .iter()
        .map(|instance| BorrowedBatchTrace {
            matrix: instance.trace,
        })
        .collect::<Vec<_>>();
    prove_batch_inner(config, &airs, &mut traces, &public_values, prover_data)
}

/// Generate a batch STARK proof and release guarded source traces at their last use.
///
/// This is an incremental source-memory hardening boundary. It avoids retaining the original trace
/// matrices throughout quotient construction and FRI opening, but it does not turn the ordinary PCS
/// prover data into guarded memory. Callers must not treat it as a complete secret-prover API.
#[cfg(feature = "secret-memory")]
#[instrument(skip_all)]
pub fn prove_batch_guarded_inputs<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>
        + Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
    #[cfg(not(debug_assertions))] A: for<'a> Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
>(
    config: &SC,
    instances: Vec<GuardedStarkInstance<'_, SC, A>>,
    prover_data: &ProverData<SC>,
) -> Result<BatchProof<SC>, ProvingError<PcsProverError<SC>>>
where
    SC: SGC,
    Val<SC>: PrimeField64 + Zeroize,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
    Domain<SC>: Send + Sync,
    SC::Pcs: Sync,
    PcsProverError<SC>: Send,
    SC::Challenger: GrindingChallenger<Witness = Val<SC>>,
    <SC::Pcs as p3_commit::Pcs<SC::Challenge, SC::Challenger>>::ProverData: Sync,
    <SC::Pcs as p3_commit::Pcs<SC::Challenge, SC::Challenger>>::Commitment: Sync,
{
    let mut airs = Vec::with_capacity(instances.len());
    let mut public_values = Vec::with_capacity(instances.len());
    let mut traces = Vec::with_capacity(instances.len());
    for instance in instances {
        airs.push(instance.air);
        public_values.push(instance.public_values);
        traces.push(GuardedBatchTrace {
            matrix: instance.trace,
        });
    }
    let public_value_refs = public_values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    prove_batch_inner(config, &airs, &mut traces, &public_value_refs, prover_data)
}

/// Generate a batch-STARK proof while retaining all witness-derived PCS data
/// under zeroizing ownership until its final protocol use.
///
/// Public preprocessing remains in the ordinary reusable prover-data owner: it
/// is fixed by the proving key and is intentionally not classified as witness
/// material. Main traces, lookup permutation traces, quotient evaluations,
/// quotient LDEs, optional hiding polynomials, and private opening rounds use
/// the guarded PCS capabilities throughout.
#[cfg(feature = "secret-memory")]
#[instrument(skip_all)]
pub fn prove_batch_secret<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>
        + Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
    #[cfg(not(debug_assertions))] A: for<'a> Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
>(
    config: &SC,
    instances: Vec<GuardedStarkInstance<'_, SC, A>>,
    prover_data: &ProverData<SC>,
) -> SecretBatchProvingResult<SC>
where
    SC: SGC,
    Val<SC>: PrimeField64 + Zeroize,
    SC::Challenge: Zeroize,
    PackedVal<SC>: Zeroize,
    PackedChallenge<SC>: Zeroize,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
    Domain<SC>: Clone,
    SC::Pcs: SecretDetachedPcsCommitter<SC::Challenge, SC::Challenger>
        + SecretPcsMixedOpener<SC::Challenge, SC::Challenger>
        + SecretUnivariateStarkPcsCommitter<SC::Challenge, SC::Challenger>
        + SecretUnivariateStarkPcsEvaluator<SC::Challenge, SC::Challenger>,
    SC::Challenger: GrindingChallenger<Witness = Val<SC>>,
{
    let mut observer = NoopSecretBatchProvingObserver;
    prove_batch_secret_with_observer(config, instances, prover_data, &mut observer)
}

/// Generate a guarded batch-STARK proof and report non-secret phase boundaries.
///
/// This is the diagnostic form of [`prove_batch_secret`]. The observer sees
/// only a typed phase identifier after each phase completes and cannot alter
/// transcript contents. Observer work is outside the cryptographic protocol;
/// callers should use the ordinary entry point unless profiling is explicitly
/// required.
#[cfg(feature = "secret-memory")]
#[instrument(skip_all)]
pub fn prove_batch_secret_with_observer<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>
        + Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
    #[cfg(not(debug_assertions))] A: for<'a> Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
>(
    config: &SC,
    instances: Vec<GuardedStarkInstance<'_, SC, A>>,
    prover_data: &ProverData<SC>,
    observer: &mut impl SecretBatchProvingObserver,
) -> SecretBatchProvingResult<SC>
where
    SC: SGC,
    Val<SC>: PrimeField64 + Zeroize,
    SC::Challenge: Zeroize,
    PackedVal<SC>: Zeroize,
    PackedChallenge<SC>: Zeroize,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
    Domain<SC>: Clone,
    SC::Pcs: SecretDetachedPcsCommitter<SC::Challenge, SC::Challenger>
        + SecretPcsMixedOpener<SC::Challenge, SC::Challenger>
        + SecretUnivariateStarkPcsCommitter<SC::Challenge, SC::Challenger>
        + SecretUnivariateStarkPcsEvaluator<SC::Challenge, SC::Challenger>,
    SC::Challenger: GrindingChallenger<Witness = Val<SC>>,
{
    type Error<Config> = SecretBatchProvingError<
        <<Config as SGC>::Pcs as SecretPcsCommitter<
            <Config as SGC>::Challenge,
            <Config as SGC>::Challenger,
        >>::SecretCommitError,
        <<Config as SGC>::Pcs as SecretUnivariateStarkPcsEvaluator<
            <Config as SGC>::Challenge,
            <Config as SGC>::Challenger,
        >>::SecretEvaluationError,
        <<Config as SGC>::Pcs as p3_commit::SecretPcsOpener<
            <Config as SGC>::Challenge,
            <Config as SGC>::Challenger,
        >>::SecretOpenError,
    >;

    observer.phase_started(SecretBatchProvingPhase::Preparation);
    let num_instances = instances.len();
    if num_instances == 0 {
        return Err(Error::<SC>::EmptyBatch);
    }

    let common = &prover_data.common;
    if common.lookups.len() != num_instances {
        return Err(Error::<SC>::InputCountMismatch);
    }
    if let Some(global) = &common.preprocessed {
        if global.instances.len() != num_instances
            || global.matrix_to_instance.len()
                != global
                    .instances
                    .iter()
                    .filter(|entry| entry.is_some())
                    .count()
            || prover_data.prover_only.preprocessed_prover_data.is_none()
        {
            return Err(Error::<SC>::InvalidPreprocessing);
        }
        for (matrix_index, &instance_index) in global.matrix_to_instance.iter().enumerate() {
            let Some(Some(metadata)) = global.instances.get(instance_index) else {
                return Err(Error::<SC>::InvalidPreprocessing);
            };
            if metadata.matrix_index != matrix_index || metadata.width == 0 {
                return Err(Error::<SC>::InvalidPreprocessing);
            }
        }
    } else if prover_data.prover_only.preprocessed_prover_data.is_some() {
        return Err(Error::<SC>::InvalidPreprocessing);
    }

    let lookup_gadget = LogUpGadget::new();
    if lookup_gadget.num_challenges() != 2 {
        return Err(Error::<SC>::InvalidLookupLayout);
    }

    let total_lookups = common.lookups.iter().try_fold(0_usize, |total, lookups| {
        total
            .checked_add(lookups.len())
            .ok_or(Error::<SC>::DimensionOverflow)
    })?;
    let mut global_lookup_widths = Vec::new();
    global_lookup_widths
        .try_reserve_exact(total_lookups)
        .map_err(|_| Error::<SC>::AllocationFailed)?;
    for lookups in &common.lookups {
        for (expected_column, lookup) in lookups.iter().enumerate() {
            if lookup.column != expected_column || lookup.elements.is_empty() {
                return Err(Error::<SC>::InvalidLookupLayout);
            }
            let tuple_width = lookup.elements[0].len();
            if lookup
                .elements
                .iter()
                .any(|elements| elements.len() != tuple_width)
            {
                return Err(Error::<SC>::InvalidLookupLayout);
            }
            if let LookupKind::Global(name) = &lookup.kind {
                if let Some((_, width)) = global_lookup_widths
                    .iter()
                    .find(|(existing, _)| *existing == name.as_str())
                {
                    if *width != tuple_width {
                        return Err(Error::<SC>::InvalidLookupLayout);
                    }
                } else {
                    global_lookup_widths.push((name.as_str(), tuple_width));
                }
            }
        }
    }

    let mut airs = Vec::new();
    let mut public_values = Vec::new();
    let mut traces = Vec::new();
    airs.try_reserve_exact(num_instances)
        .map_err(|_| Error::<SC>::AllocationFailed)?;
    public_values
        .try_reserve_exact(num_instances)
        .map_err(|_| Error::<SC>::AllocationFailed)?;
    traces
        .try_reserve_exact(num_instances)
        .map_err(|_| Error::<SC>::AllocationFailed)?;
    for instance in instances {
        if !instance.air.public_boundary_io().is_empty() {
            return Err(Error::<SC>::UnsupportedPublicBoundary);
        }
        if instance.public_values.len() != instance.air.num_public_values() {
            return Err(Error::<SC>::PublicValueCountMismatch);
        }
        let trace_height = instance.trace.height();
        if instance.trace.width == 0
            || instance.trace.width != instance.air.width()
            || trace_height == 0
            || !trace_height.is_power_of_two()
            || instance.trace.values.len()
                != trace_height
                    .checked_mul(instance.trace.width)
                    .ok_or(Error::<SC>::DimensionOverflow)?
        {
            return Err(Error::<SC>::InvalidTraceShape);
        }
        airs.push(instance.air);
        public_values.push(instance.public_values);
        traces.push(instance.trace);
    }
    let public_value_refs = public_values.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let degrees = traces
        .iter()
        .map(|trace| trace.height())
        .collect::<Vec<_>>();
    check_multiplicity_height_bound(&common.lookups, &degrees)
        .map_err(|_| Error::<SC>::LookupMultiplicityBound)?;

    let is_zk = config.is_zk();
    let extension_factor = is_zk.checked_add(1).ok_or(Error::<SC>::DimensionOverflow)?;
    let pcs = config.pcs();
    let mut trace_domains = Vec::new();
    let mut extended_trace_domains = Vec::new();
    let mut log_extended_degrees = Vec::new();
    trace_domains
        .try_reserve_exact(num_instances)
        .map_err(|_| Error::<SC>::AllocationFailed)?;
    extended_trace_domains
        .try_reserve_exact(num_instances)
        .map_err(|_| Error::<SC>::AllocationFailed)?;
    log_extended_degrees
        .try_reserve_exact(num_instances)
        .map_err(|_| Error::<SC>::AllocationFailed)?;
    for &degree in &degrees {
        let extended_degree = degree
            .checked_mul(extension_factor)
            .ok_or(Error::<SC>::DimensionOverflow)?;
        trace_domains.push(pcs.natural_domain_for_degree(degree));
        extended_trace_domains.push(pcs.natural_domain_for_degree(extended_degree));
        let log_degree =
            usize::try_from(degree.ilog2()).map_err(|_| Error::<SC>::DimensionOverflow)?;
        log_extended_degrees.push(
            log_degree
                .checked_add(is_zk)
                .ok_or(Error::<SC>::DimensionOverflow)?,
        );
    }

    let all_lookups = common
        .lookups
        .iter()
        .map(|lookups| &**lookups)
        .collect::<Vec<_>>();
    let mut preprocessed_widths = Vec::new();
    let mut log_num_quotient_chunks = Vec::new();
    let mut num_quotient_chunks = Vec::new();
    preprocessed_widths
        .try_reserve_exact(num_instances)
        .map_err(|_| Error::<SC>::AllocationFailed)?;
    log_num_quotient_chunks
        .try_reserve_exact(num_instances)
        .map_err(|_| Error::<SC>::AllocationFailed)?;
    num_quotient_chunks
        .try_reserve_exact(num_instances)
        .map_err(|_| Error::<SC>::AllocationFailed)?;
    for index in 0..num_instances {
        let preprocessed_width = common
            .preprocessed
            .as_ref()
            .and_then(|global| global.instances[index].as_ref())
            .map_or(0, |metadata| metadata.width);
        preprocessed_widths.push(preprocessed_width);
        let layout = AirLayout {
            preprocessed_width,
            main_width: airs[index].width(),
            num_public_values: airs[index].num_public_values(),
            num_periodic_columns: airs[index].num_periodic_columns(),
            ..Default::default()
        };
        let log_chunks = get_log_num_quotient_chunks_for_domain::<_, SC::Challenge, A, _>(
            airs[index],
            layout,
            trace_domains[index],
            all_lookups[index],
            is_zk,
            &lookup_gadget,
        );
        let chunk_log = log_chunks
            .checked_add(is_zk)
            .ok_or(Error::<SC>::DimensionOverflow)?;
        let shift = u32::try_from(chunk_log).map_err(|_| Error::<SC>::DimensionOverflow)?;
        let chunks = 1_usize
            .checked_shl(shift)
            .ok_or(Error::<SC>::DimensionOverflow)?;
        log_num_quotient_chunks.push(log_chunks);
        num_quotient_chunks.push(chunks);
    }

    let num_lookup_instances = all_lookups
        .iter()
        .filter(|lookups| !lookups.is_empty())
        .count();
    let shape = BatchShape {
        trace_widths: airs.iter().map(|air| air.width()).collect(),
        public_value_counts: public_value_refs
            .iter()
            .map(|values| values.len())
            .collect(),
        preprocessed_widths: preprocessed_widths.clone(),
        has_preprocessed_commitment: common.preprocessed.is_some(),
        num_lookup_instances,
        lookup_pow_bits: config.lookup_proof_of_work_bits(),
        has_randomization_commitment: SC::Pcs::ZK,
        ood_pow_bits: config.ood_proof_of_work_bits(),
    };

    let mut challenger = config.initialise_challenger();
    let mut transcript =
        BatchProverTranscript::<SC::Challenger, Val<SC>, Challenge<SC>, Commitment<SC>>::new(
            &mut challenger,
            shape,
        );
    transcript.instance_bindings(&log_extended_degrees);
    observer.phase_completed(SecretBatchProvingPhase::Preparation);

    observer.phase_started(SecretBatchProvingPhase::MainCommitment);
    let (main_commitment, mut main_data) =
        commit_owned_main_traces_secret::<SC>(pcs, &extended_trace_domains, traces)
            .inspect_err(|_| transcript.abort())
            .map_err(Error::<SC>::MainCommitment)?;
    transcript.main_phase(main_commitment.clone(), &public_value_refs);
    transcript.preprocessed_phase(
        common
            .preprocessed
            .as_ref()
            .map(|global| global.commitment.clone()),
    );
    observer.phase_completed(SecretBatchProvingPhase::MainCommitment);

    observer.phase_started(SecretBatchProvingPhase::LookupGeneration);
    let (lookup_challenges, lookup_pow_witness) =
        transcript.lookup_phase(&all_lookups, &lookup_gadget);
    let mut lookup_terminals = Vec::new();
    let mut permutation_traces = Vec::new();
    lookup_terminals
        .try_reserve_exact(num_instances)
        .map_err(|_| {
            transcript.abort();
            Error::<SC>::AllocationFailed
        })?;
    permutation_traces
        .try_reserve_exact(num_lookup_instances)
        .map_err(|_| {
            transcript.abort();
            Error::<SC>::AllocationFailed
        })?;
    for index in 0..num_instances {
        if all_lookups[index].is_empty() {
            lookup_terminals.push(None);
            continue;
        }
        let trace = pcs
            .get_source_evaluations_secret(
                &main_data,
                index,
                trace_domains[index],
                extended_trace_domains[index],
            )
            .map_err(|source| {
                transcript.abort();
                Error::<SC>::LookupTraceEvaluation(source)
            })?;
        if trace.width != airs[index].width()
            || trace.height() != degrees[index]
            || trace.values.len()
                != trace.height().checked_mul(trace.width).ok_or_else(|| {
                    transcript.abort();
                    Error::<SC>::DimensionOverflow
                })?
        {
            transcript.abort();
            return Err(Error::<SC>::InvalidTraceShape);
        }
        let preprocessed = airs[index].preprocessed_trace();
        let periodic_columns = airs[index].periodic_columns();
        let (permutation, terminal) = lookup_gadget
            .generate_permutation_secret::<Val<SC>, SC::Challenge>(
                &trace,
                &preprocessed,
                periodic_columns.as_ref(),
                public_value_refs[index],
                all_lookups[index],
                &lookup_challenges[index],
            )
            .map_err(|source| {
                transcript.abort();
                Error::<SC>::LookupGeneration(source)
            })?;
        let (Some(permutation), Some(terminal)) = (permutation, terminal) else {
            transcript.abort();
            return Err(Error::<SC>::InvalidLookupLayout);
        };
        lookup_terminals.push(Some(terminal));
        permutation_traces.push((extended_trace_domains[index], permutation));
    }
    observer.phase_completed(SecretBatchProvingPhase::LookupGeneration);

    observer.phase_started(SecretBatchProvingPhase::PermutationCommitment);
    let mut permutation_commit_and_data = if permutation_traces.is_empty() {
        None
    } else {
        Some(
            commit_owned_permutation_traces_secret::<SC>(pcs, permutation_traces)
                .inspect_err(|_| transcript.abort())
                .map_err(Error::<SC>::PermutationCommitment)?,
        )
    };
    observer.phase_completed(SecretBatchProvingPhase::PermutationCommitment);

    let terminal_values = lookup_terminals
        .iter()
        .flatten()
        .map(|terminal| terminal.0)
        .collect::<Vec<_>>();
    let alpha = transcript.permutation_phase(
        permutation_commit_and_data
            .as_ref()
            .map(|(commitment, _)| commitment.clone()),
        &terminal_values,
    );

    let mut permutation_indices = Vec::new();
    permutation_indices
        .try_reserve_exact(num_instances)
        .map_err(|_| {
            transcript.abort();
            Error::<SC>::AllocationFailed
        })?;
    let mut next_permutation_index = 0_usize;
    for lookups in &all_lookups {
        permutation_indices.push(next_permutation_index);
        if !lookups.is_empty() {
            next_permutation_index = next_permutation_index.checked_add(1).ok_or_else(|| {
                transcript.abort();
                Error::<SC>::DimensionOverflow
            })?;
        }
    }

    let mut quotient_plans = Vec::new();
    let mut quotient_chunk_ranges = Vec::new();
    quotient_plans
        .try_reserve_exact(num_instances)
        .map_err(|_| {
            transcript.abort();
            Error::<SC>::AllocationFailed
        })?;
    quotient_chunk_ranges
        .try_reserve_exact(num_instances)
        .map_err(|_| {
            transcript.abort();
            Error::<SC>::AllocationFailed
        })?;
    let mut quotient_chunk_end = 0_usize;
    for index in 0..num_instances {
        let quotient_log = log_extended_degrees[index]
            .checked_add(log_num_quotient_chunks[index])
            .ok_or_else(|| {
                transcript.abort();
                Error::<SC>::DimensionOverflow
            })?;
        let shift = u32::try_from(quotient_log).map_err(|_| {
            transcript.abort();
            Error::<SC>::DimensionOverflow
        })?;
        let quotient_size = 1_usize.checked_shl(shift).ok_or_else(|| {
            transcript.abort();
            Error::<SC>::DimensionOverflow
        })?;
        let quotient_domain = extended_trace_domains[index]
            .try_create_disjoint_domain(quotient_size)
            .ok_or_else(|| {
                transcript.abort();
                Error::<SC>::UnsupportedQuotientDomain
            })?;
        let range_start = quotient_chunk_end;
        quotient_chunk_end = quotient_chunk_end
            .checked_add(num_quotient_chunks[index])
            .ok_or_else(|| {
                transcript.abort();
                Error::<SC>::DimensionOverflow
            })?;
        quotient_chunk_ranges.push((range_start, quotient_chunk_end));
        quotient_plans.push(SecretQuotientCommitmentPlan::new(
            quotient_domain,
            SC::Challenge::DIMENSION,
            num_quotient_chunks[index],
        ));
    }

    observer.phase_started(SecretBatchProvingPhase::QuotientCommitment);
    // Production evaluation is bounded to one final quotient chunk. Audit
    // callers may additionally retain the established full-domain result long
    // enough to compare each emitted chunk without changing the transcript.
    let mut pending_audit_chunks: Option<PendingSecretQuotientChunks<SC>> = None;
    let quotient_result = <SC::Pcs as SecretUnivariateStarkPcsCommitter<
        SC::Challenge,
        SC::Challenger,
    >>::commit_generated_quotient_chunks_secret(
        pcs,
        quotient_plans,
        |index, plan, chunk_index, chunk_domain| {
            if chunk_index == 0 {
                if pending_audit_chunks.is_some() {
                    transcript.abort();
                    return Err(Error::<SC>::QuotientEvaluation(
                        SecretPcsQuotientError::Quotient(SecretQuotientError::InvalidShape),
                    ));
                }
                observer.phase_started(SecretBatchProvingPhase::QuotientEvaluation {
                    instance_index: index,
                });
            } else if observer.quotient_chunk_audit_requested() && pending_audit_chunks.is_none() {
                transcript.abort();
                return Err(Error::<SC>::QuotientEvaluation(
                    SecretPcsQuotientError::Quotient(SecretQuotientError::InvalidShape),
                ));
            }
            let layout = AirLayout {
                preprocessed_width: preprocessed_widths[index],
                main_width: airs[index].width(),
                num_public_values: airs[index].num_public_values(),
                num_periodic_columns: airs[index].num_periodic_columns(),
                ..Default::default()
            };
            let preprocessed_on_chunk_domain = match common
                .preprocessed
                .as_ref()
                .and_then(|global| global.instances[index].as_ref())
            {
                Some(metadata) => {
                    let public_data = prover_data
                        .prover_only
                        .preprocessed_prover_data
                        .as_ref()
                        .ok_or_else(|| {
                            transcript.abort();
                            Error::<SC>::MissingPreprocessingData
                        })?;
                    Some(
                        pcs.get_evaluations_on_domain_no_random(
                            public_data,
                            metadata.matrix_index,
                            chunk_domain,
                        )
                        .to_row_major_matrix(),
                    )
                }
                None => None,
            };
            let permutation_data = permutation_commit_and_data
                .as_ref()
                .filter(|_| !all_lookups[index].is_empty())
                .map(|(_, data)| (data, permutation_indices[index]));
            let permutation_values = lookup_terminals[index]
                .iter()
                .map(|terminal| terminal.0)
                .collect::<Vec<_>>();
            let quotient_chunk = quotient_value_chunk_base_from_secret_pcs::<SC, A, _>(
                pcs,
                airs[index],
                public_value_refs[index],
                layout,
                trace_domains[index],
                chunk_domain,
                &main_data,
                index,
                permutation_data,
                all_lookups[index],
                &permutation_values,
                &lookup_gadget,
                &lookup_challenges[index],
                preprocessed_on_chunk_domain.as_ref(),
                alpha,
            )
            .map_err(|source| {
                transcript.abort();
                Error::<SC>::QuotientEvaluation(source)
            })?;
            if observer.quotient_chunk_audit_requested() {
                if chunk_index == 0 {
                    let quotient_domain = *plan.domain();
                    let audit_preprocessed = match common
                        .preprocessed
                        .as_ref()
                        .and_then(|global| global.instances[index].as_ref())
                    {
                        Some(metadata) => {
                            let public_data = prover_data
                                .prover_only
                                .preprocessed_prover_data
                                .as_ref()
                                .ok_or_else(|| {
                                    transcript.abort();
                                    Error::<SC>::MissingPreprocessingData
                                })?;
                            Some(
                                pcs.get_evaluations_on_domain_no_random(
                                    public_data,
                                    metadata.matrix_index,
                                    quotient_domain,
                                )
                                .to_row_major_matrix(),
                            )
                        }
                        None => None,
                    };
                    let audit_chunks = quotient_value_chunks_base_from_secret_pcs::<SC, A, _>(
                        pcs,
                        airs[index],
                        public_value_refs[index],
                        layout,
                        trace_domains[index],
                        quotient_domain,
                        &main_data,
                        index,
                        permutation_data,
                        all_lookups[index],
                        &permutation_values,
                        &lookup_gadget,
                        &lookup_challenges[index],
                        audit_preprocessed.as_ref(),
                        alpha,
                        plan.num_chunks(),
                    )
                    .map_err(|source| {
                        transcript.abort();
                        Error::<SC>::QuotientEvaluation(source)
                    })?;
                    if audit_chunks.len() != plan.num_chunks() {
                        transcript.abort();
                        return Err(Error::<SC>::QuotientEvaluation(
                            SecretPcsQuotientError::Quotient(SecretQuotientError::InvalidShape),
                        ));
                    }
                    pending_audit_chunks = Some((index, audit_chunks.into_iter()));
                }

                let (audit_index, audit_chunks) =
                    pending_audit_chunks.as_mut().ok_or_else(|| {
                        transcript.abort();
                        Error::<SC>::QuotientEvaluation(SecretPcsQuotientError::Quotient(
                            SecretQuotientError::InvalidShape,
                        ))
                    })?;
                if *audit_index != index {
                    transcript.abort();
                    return Err(Error::<SC>::QuotientEvaluation(
                        SecretPcsQuotientError::Quotient(SecretQuotientError::InvalidShape),
                    ));
                }
                let status = match audit_chunks.next() {
                    Some(expected)
                        if expected.width == quotient_chunk.width
                            && expected.values == quotient_chunk.values =>
                    {
                        SecretQuotientChunkAuditStatus::Equivalent
                    }
                    Some(_) => SecretQuotientChunkAuditStatus::Mismatch,
                    None => SecretQuotientChunkAuditStatus::EvaluationFailed,
                };
                observer.quotient_chunk_audited(index, chunk_index, status);
                if audit_chunks.len() == 0 {
                    pending_audit_chunks = None;
                }
            }

            let next_chunk_index = chunk_index.checked_add(1).ok_or_else(|| {
                transcript.abort();
                Error::<SC>::DimensionOverflow
            })?;
            if next_chunk_index == plan.num_chunks() {
                observer.phase_completed(SecretBatchProvingPhase::QuotientEvaluation {
                    instance_index: index,
                });
            }
            Ok(quotient_chunk)
        },
    );
    let (quotient_commitment, mut quotient_data) = match quotient_result {
        Ok(output) => output,
        Err(SecretGeneratedQuotientCommitError::Generation(error)) => return Err(error),
        Err(SecretGeneratedQuotientCommitError::Commitment(source)) => {
            transcript.abort();
            return Err(Error::<SC>::QuotientCommitment(
                SecretQuotientCommitError::Pcs(source),
            ));
        }
    };
    observer.phase_completed(SecretBatchProvingPhase::QuotientCommitment);

    observer.phase_started(SecretBatchProvingPhase::RandomizationCommitment);
    let mut randomization = pcs
        .get_opt_randomization_poly_commitment_secret(extended_trace_domains.iter().copied())
        .inspect_err(|_| transcript.abort())
        .map_err(Error::<SC>::RandomizationCommitment)?;
    if SC::Pcs::ZK != randomization.is_some() {
        transcript.abort();
        return Err(Error::<SC>::MissingRandomizationCommitment);
    }
    observer.phase_completed(SecretBatchProvingPhase::RandomizationCommitment);
    let randomization_commitment = randomization
        .as_ref()
        .map(|(commitment, _)| commitment.clone());
    transcript.quotient_phase(
        quotient_commitment.clone(),
        randomization_commitment.clone(),
    );
    let (zeta, ood_pow_witness) = transcript.ood_phase();

    observer.phase_started(SecretBatchProvingPhase::Opening);
    let opening_layout = p3_uni_stark::StarkOpeningLayout::new(SC::Pcs::ZK);
    let opening_result = {
        let mut rounds = Vec::new();
        rounds.try_reserve_exact(5).map_err(|_| {
            transcript.abort();
            Error::<SC>::AllocationFailed
        })?;

        if let Some((_, randomization_data)) = randomization.as_mut() {
            let points = trace_domains.iter().map(|_| vec![zeta]).collect();
            rounds.push(SecretPcsMixedOpeningRequest::Private(
                SecretOpeningRequest {
                    prover_data: randomization_data,
                    points,
                },
            ));
        }

        let mut main_points = Vec::new();
        main_points.try_reserve_exact(num_instances).map_err(|_| {
            transcript.abort();
            Error::<SC>::AllocationFailed
        })?;
        for index in 0..num_instances {
            if airs[index].main_next_row_columns().is_empty() {
                main_points.push(vec![zeta]);
            } else {
                let next = trace_domains[index].next_point(zeta).ok_or_else(|| {
                    transcript.abort();
                    Error::<SC>::UnsupportedNextRow
                })?;
                main_points.push(vec![zeta, next]);
            }
        }
        rounds.push(SecretPcsMixedOpeningRequest::Private(
            SecretOpeningRequest {
                prover_data: &mut main_data,
                points: main_points,
            },
        ));

        let mut quotient_points = Vec::new();
        quotient_points
            .try_reserve_exact(quotient_chunk_end)
            .map_err(|_| {
                transcript.abort();
                Error::<SC>::AllocationFailed
            })?;
        quotient_points.resize_with(quotient_chunk_end, || vec![zeta]);
        rounds.push(SecretPcsMixedOpeningRequest::Private(
            SecretOpeningRequest {
                prover_data: &mut quotient_data,
                points: quotient_points,
            },
        ));

        if let Some(global) = &common.preprocessed {
            let public_data = prover_data
                .prover_only
                .preprocessed_prover_data
                .as_ref()
                .ok_or_else(|| {
                    transcript.abort();
                    Error::<SC>::MissingPreprocessingData
                })?;
            let mut points = Vec::new();
            points
                .try_reserve_exact(global.matrix_to_instance.len())
                .map_err(|_| {
                    transcript.abort();
                    Error::<SC>::AllocationFailed
                })?;
            for &instance_index in &global.matrix_to_instance {
                if airs[instance_index]
                    .preprocessed_next_row_columns()
                    .is_empty()
                {
                    points.push(vec![zeta]);
                } else {
                    let next = trace_domains[instance_index]
                        .next_point(zeta)
                        .ok_or_else(|| {
                            transcript.abort();
                            Error::<SC>::UnsupportedNextRow
                        })?;
                    points.push(vec![zeta, next]);
                }
            }
            rounds.push(SecretPcsMixedOpeningRequest::Public(OpeningRequest {
                prover_data: public_data,
                points,
            }));
        }

        if let Some((_, permutation_data)) = permutation_commit_and_data.as_mut() {
            let mut points = Vec::new();
            points
                .try_reserve_exact(num_lookup_instances)
                .map_err(|_| {
                    transcript.abort();
                    Error::<SC>::AllocationFailed
                })?;
            for (domain, lookups) in trace_domains.iter().zip(&all_lookups) {
                if !lookups.is_empty() {
                    let next = domain.next_point(zeta).ok_or_else(|| {
                        transcript.abort();
                        Error::<SC>::UnsupportedNextRow
                    })?;
                    points.push(vec![zeta, next]);
                }
            }
            rounds.push(SecretPcsMixedOpeningRequest::Private(
                SecretOpeningRequest {
                    prover_data: permutation_data,
                    points,
                },
            ));
        }

        transcript
            .delegate(|challenger| pcs.open_secret_with_public_preprocessing(rounds, challenger))
    };
    let (opened_values, opening_proof) = opening_result.map_err(|source| {
        transcript.abort();
        Error::<SC>::Opening(source)
    })?;
    transcript.finish();
    observer.phase_completed(SecretBatchProvingPhase::Opening);

    observer.phase_started(SecretBatchProvingPhase::ProofAssembly);
    let trace_openings = opened_values
        .get(opening_layout.trace)
        .ok_or(Error::<SC>::InvalidOpeningShape)?;
    let quotient_openings = opened_values
        .get(opening_layout.quotient)
        .ok_or(Error::<SC>::InvalidOpeningShape)?;
    if trace_openings.len() != num_instances || quotient_openings.len() != quotient_chunk_end {
        return Err(Error::<SC>::InvalidOpeningShape);
    }
    let preprocessed_openings = if common.preprocessed.is_some() {
        Some(
            opened_values
                .get(opening_layout.preprocessed)
                .ok_or(Error::<SC>::InvalidOpeningShape)?,
        )
    } else {
        None
    };
    let permutation_index = if common.preprocessed.is_some() {
        opening_layout
            .preprocessed
            .checked_add(1)
            .ok_or(Error::<SC>::DimensionOverflow)?
    } else {
        opening_layout.preprocessed
    };
    let permutation_openings = if permutation_commit_and_data.is_some() {
        Some(
            opened_values
                .get(permutation_index)
                .ok_or(Error::<SC>::InvalidOpeningShape)?,
        )
    } else {
        None
    };
    let random_openings = if randomization.is_some() {
        Some(
            opened_values
                .first()
                .ok_or(Error::<SC>::InvalidOpeningShape)?,
        )
    } else {
        None
    };

    let mut proof_instances = Vec::new();
    proof_instances
        .try_reserve_exact(num_instances)
        .map_err(|_| Error::<SC>::AllocationFailed)?;
    let mut permutation_opening_index = 0_usize;
    for index in 0..num_instances {
        let trace_values = trace_openings
            .get(index)
            .ok_or(Error::<SC>::InvalidOpeningShape)?;
        let expected_trace_points = if airs[index].main_next_row_columns().is_empty() {
            1
        } else {
            2
        };
        if trace_values.len() != expected_trace_points {
            return Err(Error::<SC>::InvalidOpeningShape);
        }
        let trace_local = trace_values[0].clone();
        let trace_next = trace_values.get(1).cloned();

        let (chunk_start, chunk_end) = quotient_chunk_ranges[index];
        let mut quotient_chunks = Vec::new();
        quotient_chunks
            .try_reserve_exact(chunk_end - chunk_start)
            .map_err(|_| Error::<SC>::AllocationFailed)?;
        for values in quotient_openings
            .get(chunk_start..chunk_end)
            .ok_or(Error::<SC>::InvalidOpeningShape)?
        {
            if values.len() != 1 {
                return Err(Error::<SC>::InvalidOpeningShape);
            }
            quotient_chunks.push(values[0].clone());
        }

        let preprocessed = match (
            common
                .preprocessed
                .as_ref()
                .and_then(|global| global.instances[index].as_ref()),
            preprocessed_openings,
        ) {
            (Some(metadata), Some(all_values)) => {
                let values = all_values
                    .get(metadata.matrix_index)
                    .ok_or(Error::<SC>::InvalidOpeningShape)?;
                let expected = if airs[index].preprocessed_next_row_columns().is_empty() {
                    1
                } else {
                    2
                };
                if values.len() != expected {
                    return Err(Error::<SC>::InvalidOpeningShape);
                }
                Some(PreprocessedOpenedValues {
                    local: values[0].clone(),
                    next: values.get(1).cloned(),
                })
            }
            (None, _) => None,
            _ => return Err(Error::<SC>::InvalidOpeningShape),
        };

        let (permutation_local, permutation_next) = if all_lookups[index].is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let values = permutation_openings
                .and_then(|all_values| all_values.get(permutation_opening_index))
                .ok_or(Error::<SC>::InvalidOpeningShape)?;
            if values.len() != 2 {
                return Err(Error::<SC>::InvalidOpeningShape);
            }
            permutation_opening_index = permutation_opening_index
                .checked_add(1)
                .ok_or(Error::<SC>::DimensionOverflow)?;
            (values[0].clone(), values[1].clone())
        };

        let random = match random_openings {
            Some(all_values) => {
                let values = all_values
                    .get(index)
                    .ok_or(Error::<SC>::InvalidOpeningShape)?;
                if values.len() != 1 {
                    return Err(Error::<SC>::InvalidOpeningShape);
                }
                Some(values[0].clone())
            }
            None => None,
        };

        proof_instances.push(OpenedValuesWithLookups {
            base_opened_values: OpenedValues {
                trace_local,
                trace_next,
                preprocessed,
                quotient_chunks,
                random,
            },
            permutation_local,
            permutation_next,
        });
    }
    if permutation_openings.is_some_and(|values| values.len() != permutation_opening_index) {
        return Err(Error::<SC>::InvalidOpeningShape);
    }

    let proof = BatchProof {
        commitments: BatchCommitments {
            main: main_commitment,
            quotient_chunks: quotient_commitment,
            random: randomization_commitment,
            permutation: permutation_commit_and_data
                .as_ref()
                .map(|(commitment, _)| commitment.clone()),
        },
        opened_values: BatchOpenedValues {
            instances: proof_instances,
        },
        opening_proof,
        lookup_terminals,
        degree_bits: log_extended_degrees,
        lookup_pow_witness,
        ood_pow_witness,
    };
    observer.phase_completed(SecretBatchProvingPhase::ProofAssembly);
    Ok(proof)
}

fn prove_batch_inner<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>
        + Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
    #[cfg(not(debug_assertions))] A: for<'a> Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
    T,
>(
    config: &SC,
    airs: &[&A],
    traces: &mut [T],
    pub_vals: &[&[Val<SC>]],
    prover_data: &ProverData<SC>,
) -> Result<BatchProof<SC>, ProvingError<PcsProverError<SC>>>
where
    SC: SGC,
    Val<SC>: PrimeField64,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
    Domain<SC>: Send + Sync,
    SC::Pcs: Sync,
    PcsProverError<SC>: Send,
    SC::Challenger: GrindingChallenger<Witness = Val<SC>>,
    <SC::Pcs as p3_commit::Pcs<SC::Challenge, SC::Challenger>>::ProverData: Sync,
    <SC::Pcs as p3_commit::Pcs<SC::Challenge, SC::Challenger>>::Commitment: Sync,
    T: BatchTrace<Val<SC>>,
{
    // Public inputs reach this proof only through AIR constraints.
    // A cell listed for backend binding would go completely unbound.
    assert!(
        airs.iter().all(|air| air.public_boundary_io().is_empty()),
        "batch-stark does not support boundary-IO public values; bind them with AIR constraints"
    );
    assert_eq!(airs.len(), traces.len(), "airs/traces length mismatch");
    assert_eq!(
        airs.len(),
        pub_vals.len(),
        "airs/public-values length mismatch"
    );

    let common = &prover_data.common;
    // TODO: Extend if additional lookup gadgets are added.
    let lookup_gadget = LogUpGadget::new();

    let pcs = config.pcs();

    // Collect per-instance degree information.
    let degrees: Vec<usize> = traces.iter().map(|trace| trace.matrix().height()).collect();
    let log_degrees: Vec<usize> = degrees.iter().copied().map(log2_strict_usize).collect();
    // Extended degree accounts for the ZK blinding factor (2x when ZK is enabled).
    let log_ext_degrees: Vec<usize> = log_degrees.iter().map(|&d| d + config.is_zk()).collect();

    // Fail fast: a wrapped multiplicity makes this proof unverifiable.
    // The verifier enforces the same bound.
    check_multiplicity_height_bound(&common.lookups, &degrees)
        .expect("LogUp multiplicity height-bound violated");

    // Read lookups from the keygen-cached CommonData (not from instances).
    let all_lookups: Vec<&[Lookup<Val<SC>>]> = common.lookups.iter().map(|l| &**l).collect();
    // Per-AIR lookup terminal: `Some(terminal)` once filled, `None` for AIRs with no lookups.
    let mut lookup_terminals: Vec<Option<LookupTerminal<SC::Challenge>>> =
        all_lookups.iter().map(|_| None).collect();

    // Base and extended domains for every instance.
    let (trace_domains, ext_trace_domains): (Vec<Domain<SC>>, Vec<Domain<SC>>) = degrees
        .iter()
        .map(|&deg| {
            (
                pcs.natural_domain_for_degree(deg),
                pcs.natural_domain_for_degree(deg * (config.is_zk() + 1)),
            )
        })
        .unzip();

    // Determine preprocessed widths and quotient chunk counts per instance.
    let mut preprocessed_widths = Vec::with_capacity(airs.len());
    let chunk_counts = airs
        .iter()
        .zip(pub_vals.iter())
        .enumerate()
        .map(|(i, (air, _pv))| {
            // Width of the preprocessed trace for this instance (0 if absent).
            let pre_w = common
                .preprocessed
                .as_ref()
                .and_then(|g| g.instances[i].as_ref().map(|m| m.width))
                .unwrap_or(0);
            preprocessed_widths.push(pre_w);

            let layout = AirLayout {
                preprocessed_width: pre_w,
                main_width: air.width(),
                num_public_values: air.num_public_values(),
                num_periodic_columns: air.num_periodic_columns(),
                ..Default::default()
            };

            // Infer the log of the quotient polynomial degree from symbolic analysis.
            //
            // The same evaluation checks the next-row declaration.
            // Why: with an empty declaration the next row is never opened, and the verifier folds zeros for it.
            // A proof of an AIR reading such a row would always be rejected.
            let constraints =
                info_span!("infer log of constraint degree", air_idx = i).in_scope(|| {
                    let evaluated = evaluate_constraints::<_, SC::Challenge, A, _>(
                        air,
                        layout,
                        all_lookups[i],
                        &lookup_gadget,
                    );
                    AirConstraints::new(*air, evaluated)
                });
            constraints.check_next_row_reads().map_err(
                |UndeclaredNextRowRead { trace, column }| ProvingError::UndeclaredNextRowRead {
                    air: Some(i),
                    trace,
                    column,
                },
            )?;
            let lq_chunks = constraints
                .log_num_quotient_chunks(pcs.natural_domain_for_degree(degrees[i]), config.is_zk());
            // Actual number of quotient chunks (doubled when ZK is enabled).
            let n_chunks = 1 << (lq_chunks + config.is_zk());
            Ok((lq_chunks, n_chunks))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (log_num_quotient_chunks, num_quotient_chunks): (Vec<usize>, Vec<usize>) =
        chunk_counts.into_iter().unzip();

    let n_instances = airs.len();
    let widths: Vec<usize> = airs.iter().map(|a| A::width(a)).collect();

    // Transcript: describe the run, then seed a driver from that description.
    //
    // Every number here comes from the AIRs, the common data, or the config.
    // None of it is read back from the proof this run is about to build.
    let shape = BatchShape {
        trace_widths: widths,
        public_value_counts: pub_vals.iter().map(|pv| pv.len()).collect(),
        preprocessed_widths: preprocessed_widths.clone(),
        has_preprocessed_commitment: common.preprocessed.is_some(),
        num_lookup_instances: all_lookups.iter().filter(|l| !l.is_empty()).count(),
        lookup_pow_bits: config.lookup_proof_of_work_bits(),
        has_randomization_commitment: SC::Pcs::ZK,
        ood_pow_bits: config.ood_proof_of_work_bits(),
    };

    let mut challenger = config.initialise_challenger();
    let mut transcript =
        BatchProverTranscript::<SC::Challenger, Val<SC>, Challenge<SC>, Commitment<SC>>::new(
            &mut challenger,
            shape,
        );

    // The size of each instance is the prover's to choose, so it is absorbed.
    transcript.instance_bindings(&log_ext_degrees);

    // Transcript: Main trace commitment

    // Build PCS inputs for every instance and commit in a single batch.
    let main_commit_inputs = traces
        .iter()
        .zip(ext_trace_domains.iter().cloned())
        .map(|(trace, dom)| (dom, trace.matrix().clone()))
        .collect::<Vec<_>>();
    let (main_commit, main_data) = pcs
        .commit(main_commit_inputs)
        .inspect_err(|_| transcript.abort())
        .map_err(|source| ProvingError::Pcs {
            phase: "trace commitment",
            source,
        })?;

    transcript.main_phase(main_commit.clone(), pub_vals);
    transcript.preprocessed_phase(common.preprocessed.as_ref().map(|g| g.commitment.clone()));

    // Transcript: Lookup challenges and permutation traces

    // Grind before the lookup challenges, then draw them.
    //
    // Why: the main-trace commitment and every public value are already bound.
    // The witness therefore commits the prover to the trace before the pair exists.
    // Searching for a favourable pair costs `2^lookup_proof_of_work_bits` per candidate.
    let (challenges_per_instance, lookup_pow_witness) =
        transcript.lookup_phase(&all_lookups, &lookup_gadget);

    // Generate permutation traces for instances that have lookups.
    let mut permutation_commit_inputs = Vec::with_capacity(n_instances);
    for (i, (trace, ext_domain)) in traces
        .iter()
        .zip(ext_trace_domains.iter().cloned())
        .enumerate()
    {
        if !all_lookups[i].is_empty() {
            // Compute the permutation argument trace and the AIR's single terminal.
            let periodic_columns = airs[i].periodic_columns();
            let (generated_perm, terminal) = lookup_gadget
                .generate_permutation::<Val<SC>, SC::Challenge>(
                    trace.matrix(),
                    &airs[i].preprocessed_trace(),
                    periodic_columns.as_ref(),
                    pub_vals[i],
                    all_lookups[i],
                    &challenges_per_instance[i],
                );

            // Record the AIR's terminal for transcript observation and proof emission.
            lookup_terminals[i] = terminal;

            #[cfg(debug_assertions)]
            {
                use crate::check_constraints::check_constraints;

                let preprocessed_trace = airs[i].preprocessed_trace();

                let perm_vals: Vec<SC::Challenge> = terminal.iter().map(|t| t.0).collect();
                let lookup_constraints_inputs = (all_lookups[i], &lookup_gadget);
                check_constraints(
                    airs[i],
                    trace.matrix(),
                    &preprocessed_trace,
                    &generated_perm,
                    &challenges_per_instance[i],
                    &perm_vals,
                    pub_vals[i],
                    lookup_constraints_inputs,
                );
            }

            // Consume the generated matrix directly; no extra clone before flattening.
            permutation_commit_inputs.push((ext_domain, generated_perm.flatten_to_base()));
        }
    }

    // Debug-only: verify that all lookup sums balance across instances.
    #[cfg(debug_assertions)]
    {
        use p3_lookup::debug_util::{LookupDebugInstance, check_lookups};

        let preprocessed_traces: Vec<_> = airs.iter().map(|air| air.preprocessed_trace()).collect();
        let periodic_columns: Vec<_> = airs
            .iter()
            .map(|air| air.periodic_columns().into_owned())
            .collect();
        let debug_instances: Vec<_> = traces
            .iter()
            .zip(preprocessed_traces.iter())
            .zip(periodic_columns.iter())
            .zip(all_lookups.iter())
            .zip(pub_vals.iter())
            .map(
                |((((trace, prep), periodic), lookups), public_values)| LookupDebugInstance {
                    main_trace: trace.matrix(),
                    preprocessed_trace: prep,
                    public_values,
                    periodic_columns: periodic,
                    lookups,
                    permutation_challenges: &[],
                },
            )
            .collect();
        check_lookups(&debug_instances);
    }

    // Commit all permutation traces (if any).
    let permutation_commit_and_data = if !permutation_commit_inputs.is_empty() {
        Some(
            pcs.commit(permutation_commit_inputs)
                .inspect_err(|_| transcript.abort())
                .map_err(|source| ProvingError::Pcs {
                    phase: "permutation commitment",
                    source,
                })?,
        )
    } else {
        None
    };

    // Main-trace and lookup-permutation construction are the final consumers of the source
    // matrices. Guarded callers erase and deallocate them here, before quotient and FRI work reach
    // their peak. Borrowed callers retain the historical ownership behavior.
    for trace in traces {
        trace.release_source();
    }

    // Transcript: bind the permutation commitment and every terminal, then draw alpha.
    let terminal_values: Vec<Challenge<SC>> =
        lookup_terminals.iter().flatten().map(|t| t.0).collect();
    let alpha: Challenge<SC> = transcript.permutation_phase(
        permutation_commit_and_data
            .as_ref()
            .map(|(commitment, _)| commitment.clone()),
        &terminal_values,
    );

    // Capture only the permutation prover data;
    //
    // The commitment isn't read in the parallel closure below.
    let permutation_data = permutation_commit_and_data.as_ref().map(|(_, data)| data);

    // Permutation-matrix index per instance: the prefix count of prior instances
    // that contribute a permutation trace. Precomputing this removes the only
    // cross-iteration dependency, so each instance's quotient is independent.
    let perm_indices: Vec<usize> = all_lookups
        .iter()
        .scan(0usize, |next, lookups| {
            let idx = *next;
            if !lookups.is_empty() {
                *next += 1;
            }
            Some(idx)
        })
        .collect();

    // Each instance's quotient chunks are independent, so compute them in
    // parallel. `quotient_values` already parallelises over rows; with many
    // instances this fills the cores that a single instance leaves idle.
    let per_instance: Result<Vec<InstanceQuotient<SC>>, PcsProverError<SC>> = (0..n_instances)
        .into_par_iter()
        .map(|i| {
            let _air_span = info_span!("compute quotient", air_idx = i).entered();

            let log_chunks = log_num_quotient_chunks[i];
            let n_chunks = num_quotient_chunks[i];
            // Build the quotient domain: disjoint from the trace domain,
            // with size = ext_degree * num_quotient_chunks.
            let quotient_domain =
                ext_trace_domains[i].create_disjoint_domain(1 << (log_ext_degrees[i] + log_chunks));

            let sym_layout = AirLayout {
                preprocessed_width: preprocessed_widths[i],
                main_width: airs[i].width(),
                num_public_values: airs[i].num_public_values(),
                num_periodic_columns: airs[i].num_periodic_columns(),
                ..Default::default()
            };

            // Debug-only: verify the static constraint-count hint matches symbolic analysis.
            debug_assert!(
                airs[i].num_constraints().is_none_or(|n| {
                    n == get_symbolic_constraints(
                        airs[i],
                        sym_layout,
                        all_lookups[i],
                        &lookup_gadget,
                    )
                    .0
                    .len()
                }),
                "num_constraints() = {} but symbolic evaluation found {} base constraints",
                airs[i].num_constraints().unwrap(),
                get_symbolic_constraints(airs[i], sym_layout, all_lookups[i], &lookup_gadget,)
                    .0
                    .len(),
            );

            // Evaluate the committed main trace on the quotient domain via LDE.
            let trace_on_quotient_domain =
                pcs.get_evaluations_on_domain(&main_data, i, quotient_domain);

            // Evaluate the permutation trace on the quotient domain (if lookups exist).
            let permutation_on_quotient_domain = permutation_data
                .filter(|_| !all_lookups[i].is_empty())
                .map(|perm_data| {
                    pcs.get_evaluations_on_domain(perm_data, perm_indices[i], quotient_domain)
                });

            // Evaluate preprocessed columns on the quotient domain (if present).
            let preprocessed_on_quotient_domain = common
                .preprocessed
                .as_ref()
                .and_then(|g| g.instances[i].as_ref())
                .map(|meta| {
                    let preprocessed_prover_data = prover_data
                        .prover_only
                        .preprocessed_prover_data
                        .as_ref()
                        .expect(
                            "preprocessed_prover_data must exist when preprocessed columns exist",
                        );
                    pcs.get_evaluations_on_domain_no_random(
                        preprocessed_prover_data,
                        meta.matrix_index,
                        quotient_domain,
                    )
                });

            // Compute quotient(x) = constraints(x) / Z_H(x) on the quotient domain.
            let perm_vals: Vec<_> = lookup_terminals[i].iter().map(|t| t.0).collect();
            let q_values = quotient_values(
                pcs,
                airs[i],
                pub_vals[i],
                sym_layout,
                trace_domains[i],
                quotient_domain,
                &trace_on_quotient_domain,
                permutation_on_quotient_domain.as_ref(),
                all_lookups[i],
                &perm_vals,
                &lookup_gadget,
                &challenges_per_instance[i],
                preprocessed_on_quotient_domain.as_ref(),
                alpha,
            );

            // Flatten extension values to base field and split into degree-bounded chunks.
            let q_flat = RowMajorMatrix::new_col(q_values).flatten_to_base();
            let chunk_mats = quotient_domain.split_evals(n_chunks, q_flat);
            let chunk_domains = quotient_domain.split_domains(n_chunks);

            // Compute low-degree extensions of each chunk for commitment.
            let evals = chunk_domains.iter().zip(chunk_mats).map(|(d, m)| (*d, m));
            let ldes = pcs.get_quotient_ldes(evals, n_chunks)?;

            Ok((chunk_domains, ldes))
        })
        .collect();

    // Concatenate in instance order so the commit layout stays deterministic.
    let mut quotient_chunk_domains = Vec::new();
    let mut quotient_chunk_mats = Vec::new();
    let mut quotient_chunk_ranges = Vec::with_capacity(n_instances);
    for (chunk_domains, ldes) in
        per_instance
            .inspect_err(|_| transcript.abort())
            .map_err(|source| ProvingError::Pcs {
                phase: "quotient evaluations",
                source,
            })?
    {
        let start = quotient_chunk_domains.len();
        quotient_chunk_domains.extend(chunk_domains);
        quotient_chunk_mats.extend(ldes);
        let end = quotient_chunk_domains.len();
        quotient_chunk_ranges.push((start, end));
    }

    // Commit all quotient chunks in a single batch.
    let (quotient_commit, quotient_data) = pcs
        .commit_ldes(quotient_chunk_mats)
        .inspect_err(|_| transcript.abort())
        .map_err(|source| ProvingError::Pcs {
            phase: "quotient commitment",
            source,
        })?;

    // Transcript: Optional ZK randomization polynomial
    //
    // When ZK is enabled, commit to a random extension-field polynomial of
    // degree 2n. The PCS later adds (R(X) - R(z)) / (X - z) to the batch,
    // hiding the trace values at the query points.
    //
    // TODO: This approach is only statistically ZK.
    // A perfectly-ZK version would use a true extension-field polynomial.
    let (opt_r_commit, opt_r_data) = if SC::Pcs::ZK {
        let (r_commit, r_data) = pcs
            .get_opt_randomization_poly_commitment(ext_trace_domains.iter().copied())
            .inspect_err(|_| transcript.abort())
            .map_err(|source| ProvingError::Pcs {
                phase: "randomization commitment",
                source,
            })?
            .expect("ZK is enabled, so we should have randomization commitments");
        (Some(r_commit), Some(r_data))
    } else {
        (None, None)
    };

    transcript.quotient_phase(quotient_commit.clone(), opt_r_commit.clone());

    // Transcript: OOD opening

    // Grind before the out-of-domain point, then sample it.
    //
    // Why: every commitment and lookup terminal above is already bound.
    // The witness commits the prover to the whole batch before it learns the point.
    let (zeta, ood_pow_witness): (Challenge<SC>, Val<SC>) = transcript.ood_phase();

    // Build the opening rounds and produce the FRI opening proof.
    let opening_layout = p3_uni_stark::StarkOpeningLayout::new(SC::Pcs::ZK);
    let opening_result = {
        let mut rounds = Vec::new();

        // Round 0 (optional): randomization polynomial opened at zeta per instance.
        let round0 = opt_r_data.as_ref().map(|r_data| {
            let round0_points = trace_domains.iter().map(|_| vec![zeta]).collect();
            (r_data, round0_points)
        });
        rounds.extend(round0);

        // Round 1: main trace. Open at zeta; also at the next domain point
        // if the AIR accesses the next row.
        let round1_points = trace_domains
            .iter()
            .enumerate()
            .map(|(i, dom)| {
                if !airs[i].main_next_row_columns().is_empty() {
                    vec![
                        zeta,
                        dom.next_point(zeta)
                            .expect("domain should support next_point operation"),
                    ]
                } else {
                    vec![zeta]
                }
            })
            .collect::<Vec<_>>();
        rounds.push((&main_data, round1_points));

        // Round 2: quotient chunks, each opened at zeta only.
        let round2_points = quotient_chunk_ranges
            .iter()
            .cloned()
            .flat_map(|(s, e)| (s..e).map(|_| vec![zeta]))
            .collect::<Vec<_>>();
        rounds.push((&quotient_data, round2_points));

        // Round 3 (optional): preprocessed columns. Open at zeta, and also
        // at the next-row point if the AIR reads preprocessed next-row columns.
        if let Some(global) = &common.preprocessed {
            let preprocessed_prover_data = prover_data
                .prover_only
                .preprocessed_prover_data
                .as_ref()
                .expect("preprocessed_prover_data must exist when preprocessed columns exist");
            let pre_points = global
                .matrix_to_instance
                .iter()
                .map(|&inst_idx| {
                    if !airs[inst_idx].preprocessed_next_row_columns().is_empty() {
                        let zeta_next_i = trace_domains[inst_idx]
                            .next_point(zeta)
                            .expect("domain should support next_point operation");
                        vec![zeta, zeta_next_i]
                    } else {
                        vec![zeta]
                    }
                })
                .collect();
            rounds.push((preprocessed_prover_data, pre_points));
        }

        // Round 4 (optional): permutation traces for instances with lookups.
        // Always opened at both zeta and the next-row point.
        let lookup_points: Vec<_> = trace_domains
            .iter()
            .zip(&all_lookups)
            .filter(|&(_, lookups)| !lookups.is_empty())
            .map(|(dom, _)| {
                vec![
                    zeta,
                    dom.next_point(zeta)
                        .expect("domain should support next_point operation"),
                ]
            })
            .collect();

        if let Some((_, perm_data)) = &permutation_commit_and_data {
            let lookup_round = (perm_data, lookup_points);
            rounds.push(lookup_round);
        }

        // The opening argument runs on the same sponge, under its own description.
        transcript.delegate(|challenger| {
            pcs.open_with_preprocessing(
                rounds.into_iter().map(Into::into).collect(),
                challenger,
                common
                    .preprocessed
                    .as_ref()
                    .map(|_| opening_layout.preprocessed),
            )
        })
    };

    let (opened_values, opening_proof) = opening_result
        .inspect_err(|_| transcript.abort())
        .map_err(|source| ProvingError::Pcs {
            phase: "opening",
            source,
        })?;
    transcript.finish();

    // Parse opened values into per-instance structures

    // Permutation round follows preprocessed (if present), else takes its slot.
    let permutation_idx = if common.preprocessed.is_some() {
        opening_layout.preprocessed + 1
    } else {
        opening_layout.preprocessed
    };

    // Main trace opened values: one entry per instance.
    let trace_values_for_mats = &opened_values[opening_layout.trace];
    assert_eq!(trace_values_for_mats.len(), n_instances);

    let mut per_instance = Vec::with_capacity(n_instances);

    // Preprocessed openings (if a global preprocessed commitment exists).
    let preprocessed_openings = common
        .preprocessed
        .as_ref()
        .map(|_| &opened_values[opening_layout.preprocessed]);

    // Iterator over permutation opened values (one per instance with lookups).
    let is_lookup = permutation_commit_and_data.is_some();
    let permutation_values_for_mats = if is_lookup {
        &opened_values[permutation_idx]
    } else {
        &vec![]
    };
    let mut permutation_values_for_mats = permutation_values_for_mats.iter();

    // Iterate over quotient chunk ranges to assemble per-instance opened values.
    let mut quotient_openings_iter = opened_values[opening_layout.quotient].iter();
    for (i, (s, e)) in quotient_chunk_ranges.iter().copied().enumerate() {
        // Optional randomization polynomial opening.
        let random = if opt_r_data.is_some() {
            Some(opened_values[0][i][0].clone())
        } else {
            None
        };

        // Main trace: local row always present; next row only if AIR uses it.
        let tv = &trace_values_for_mats[i];
        let trace_local = tv[0].clone();
        let trace_next = if !airs[i].main_next_row_columns().is_empty() {
            Some(tv[1].clone())
        } else {
            None
        };

        // Quotient chunks: collect the zeta-point opening of each chunk.
        let mut qcs = Vec::with_capacity(e - s);
        for _ in s..e {
            let mat_vals = quotient_openings_iter
                .next()
                .expect("chunk index in bounds");
            qcs.push(mat_vals[0].clone());
        }

        // Preprocessed openings: local and optionally next row.
        let preprocessed = match (&common.preprocessed, preprocessed_openings) {
            (Some(global), Some(pre_round)) => global.instances[i].as_ref().map(|meta| {
                let vals = &pre_round[meta.matrix_index];
                let next = if !airs[i].preprocessed_next_row_columns().is_empty() {
                    assert_eq!(
                        vals.len(),
                        2,
                        "expected two opening points (zeta, zeta_next) for preprocessed trace"
                    );
                    Some(vals[1].clone())
                } else {
                    assert_eq!(
                        vals.len(),
                        1,
                        "expected one opening point (zeta) for preprocessed trace"
                    );
                    None
                };
                PreprocessedOpenedValues {
                    local: vals[0].clone(),
                    next,
                }
            }),
            _ => None,
        };

        // Permutation openings: present only for instances with lookups.
        let (permutation_local, permutation_next) = if !all_lookups[i].is_empty() {
            let perm_v = permutation_values_for_mats
                .next()
                .expect("instance should have permutation openings");
            (perm_v[0].clone(), perm_v[1].clone())
        } else {
            (vec![], vec![])
        };

        // Assemble the complete opened values for this instance.
        let base_opened = OpenedValues {
            trace_local,
            trace_next,
            preprocessed,
            quotient_chunks: qcs,
            random,
        };

        per_instance.push(OpenedValuesWithLookups {
            base_opened_values: base_opened,
            permutation_local,
            permutation_next,
        });
    }

    // Extract the permutation commitment (if any) for inclusion in the proof.
    let permutation = permutation_commit_and_data
        .as_ref()
        .map(|(comm, _)| comm.clone());

    // Assemble the final proof structure.
    Ok(BatchProof {
        commitments: BatchCommitments {
            main: main_commit,
            quotient_chunks: quotient_commit,
            random: opt_r_commit,
            permutation,
        },
        opened_values: BatchOpenedValues {
            instances: per_instance,
        },
        opening_proof,
        lookup_terminals,
        degree_bits: log_ext_degrees,
        lookup_pow_witness,
        ood_pow_witness,
    })
}

/// Evaluate the quotient polynomial q(x) = C(x) / Z_H(x) over the quotient
/// domain using packed SIMD arithmetic.
///
/// # Overview
///
/// For each packed chunk of domain points, this function:
/// - Loads the main, preprocessed, and permutation traces at the current and
///   next rows.
/// - Evaluates all AIR constraints (including lookup constraints) into a
///   constraint folder.
/// - Combines constraints using random powers of alpha.
/// - Divides by the vanishing polynomial to get the quotient value.
///
/// # Performance
///
/// Constraint and permutation buffers are allocated once per rayon task and
/// reused across chunk iterations via `for_each_init`, reducing allocator
/// pressure in the hot loop.
///
/// The output is written directly into a pre-allocated buffer through
/// disjoint mutable pointer access, avoiding an intermediate `collect`.
#[instrument(name = "compute quotient polynomial", skip_all, level = "debug")]
#[allow(clippy::too_many_arguments)]
pub fn quotient_values<SC, A, Mat, LG>(
    pcs: &SC::Pcs,
    air: &A,
    public_values: &[Val<SC>],
    layout: AirLayout,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    trace_on_quotient_domain: &Mat,
    opt_permutation_on_quotient_domain: Option<&Mat>,
    lookups: &[Lookup<Val<SC>>],
    permutation_vals: &[SC::Challenge],
    lookup_gadget: &LG,
    permutation_challenges: &[SC::Challenge],
    preprocessed_on_quotient_domain: Option<&Mat>,
    alpha: SC::Challenge,
) -> Vec<SC::Challenge>
where
    SC: SGC,
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
    LG: LookupProtocol + Sync,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
{
    // Public inputs reach this proof only through AIR constraints.
    // A cell listed for backend binding would go completely unbound.
    assert!(
        air.public_boundary_io().is_empty(),
        "batch-stark does not support boundary-IO public values; bind them with AIR constraints"
    );

    let quotient_size = quotient_domain.size();
    let main_width = trace_on_quotient_domain.width();
    let (perm_width, perm_height) = opt_permutation_on_quotient_domain
        .as_ref()
        .map_or((0, 0), |mat| (mat.width(), mat.height()));

    // Extension field dimension (number of base-field coordinates per extension element).
    let ext_degree = SC::Challenge::DIMENSION;

    // Compute selector polynomials (is_first_row, is_last_row, etc.) over the quotient domain.
    let mut sels = debug_span!("Compute Selectors")
        .in_scope(|| trace_domain.selectors_on_coset(quotient_domain));

    // Distance between the "current" and "next" row in the quotient domain.
    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    // The verifier folds zeros for a next row the AIR does not declare.
    // The prover folds the same values, so both sides evaluate one constraint polynomial.
    let main_next = !air.main_next_row_columns().is_empty();
    let preprocessed_next = !air.preprocessed_next_row_columns().is_empty();

    // Pad selectors so every packed chunk can load `pack_width` scalars. When
    // `quotient_size` is not a multiple of `pack_width`, extend by repeating
    // coset values modulo `quotient_size` (matching wrapped trace indexing).
    let pack_width = PackedVal::<SC>::WIDTH;
    let padded_sel_len = quotient_size.next_multiple_of(pack_width);
    let pad_selectors = |v: &mut Vec<Val<SC>>| {
        debug_assert_eq!(v.len(), quotient_size);
        for idx in quotient_size..padded_sel_len {
            v.push(v[idx % quotient_size]);
        }
    };
    pad_selectors(&mut sels.is_first_row);
    pad_selectors(&mut sels.is_last_row);
    pad_selectors(&mut sels.is_transition);
    pad_selectors(&mut sels.inv_vanishing);

    // Decompose alpha into per-constraint powers, split by base vs extension constraints.
    let constraint_layout = get_constraint_layout(air, layout, lookups, lookup_gadget);
    let (base_alpha_powers, ext_alpha_powers) = constraint_layout.decompose_alpha(alpha);

    let periodic_cols = air.periodic_columns();
    let periodic_table =
        pcs.build_periodic_lde_table(&periodic_cols, trace_domain, quotient_domain);

    // The packed row groups of the periodic table repeat every `groups_in_period`
    // groups, so only those are materialized and group `g` reads `g % groups_in_period`.
    let ncols = periodic_table.width();
    let groups_in_period = periodic_table.packed_group_period(pack_width);
    let periodic_packed: Vec<PackedVal<SC>> = if periodic_table.is_empty() {
        Vec::new()
    } else {
        let periodic_table_ref = &periodic_table;
        (0..groups_in_period)
            .flat_map(move |group| {
                let i_start = group * pack_width;
                (0..ncols).map(move |col_idx| {
                    PackedVal::<SC>::from_fn(|offset| {
                        *periodic_table_ref.get(i_start + offset, col_idx)
                    })
                })
            })
            .collect()
    };

    // Broadcast scalar challenges to packed representations for SIMD evaluation.
    let packed_perm_challenges: Vec<PackedChallenge<SC>> = permutation_challenges
        .iter()
        .map(|&p_c| PackedChallenge::<SC>::from(p_c))
        .collect();
    let permutation_vals_packed: Vec<PackedChallenge<SC>> = permutation_vals
        .iter()
        .map(|&v| PackedChallenge::<SC>::from(v))
        .collect();

    // Capacities for the reusable per-task buffers.
    let n_base = constraint_layout.base_indices.len();
    let n_ext = constraint_layout.ext_indices.len();
    let constraint_count = constraint_layout.total_constraints();
    let perm_cols = if perm_width > 0 {
        perm_width / ext_degree
    } else {
        0
    };

    // Pre-allocate the output buffer and obtain a disjoint-write pointer.
    // SAFETY: Each chunk writes to `result[i_start..i_start + chunk_emit]` with
    // `chunk_emit = min(pack_width, quotient_size - i_start)`; ranges are disjoint.
    let mut result = SC::Challenge::zero_vec(quotient_size);
    let result_ptr = DisjointMutPtr::new(&mut result);

    (0..quotient_size)
        .into_par_iter()
        .step_by(pack_width)
        .for_each_init(
            // Per-task initialization: allocate constraint, permutation and
            // packed-row buffers once. These are cleared (without
            // deallocating) and reused on every iteration.
            || {
                (
                    Vec::with_capacity(n_base),
                    Vec::with_capacity(n_ext),
                    Vec::with_capacity(2 * perm_cols),
                    Vec::with_capacity(2 * main_width),
                    Vec::with_capacity(
                        2 * preprocessed_on_quotient_domain.map_or(0, |p| p.width()),
                    ),
                )
            },
            |(base_buf, ext_buf, perm_buf, main_buf, prep_buf), i_start| {
                let chunk_emit = pack_width.min(quotient_size - i_start);
                // Load SIMD-packed selector values for this chunk.
                let i_range = i_start..i_start + pack_width;
                let is_first_row =
                    *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
                let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
                let is_transition =
                    *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
                let inv_vanishing = *PackedVal::<SC>::from_slice(&sels.inv_vanishing[i_range]);

                // Pack the main trace rows (current + next) for this chunk.
                //
                // An undeclared next row holds zeros, the values the verifier folds for it.
                main_buf.clear();
                main_buf.extend(
                    trace_on_quotient_domain.vertically_packed_row::<PackedVal<SC>>(i_start),
                );
                if main_next {
                    main_buf.extend(
                        trace_on_quotient_domain
                            .vertically_packed_row::<PackedVal<SC>>(i_start + next_step),
                    );
                } else {
                    main_buf.resize(2 * main_width, PackedVal::<SC>::ZERO);
                }
                let main = RowMajorMatrixView::new(main_buf.as_slice(), main_width);

                // Pack preprocessed rows if the AIR has preprocessed columns.
                let preprocessed_view = preprocessed_on_quotient_domain.map_or_else(
                    || RowMajorMatrixView::new(&[], 0),
                    |preprocessed| {
                        prep_buf.clear();
                        prep_buf
                            .extend(preprocessed.vertically_packed_row::<PackedVal<SC>>(i_start));
                        if preprocessed_next {
                            prep_buf.extend(
                                preprocessed
                                    .vertically_packed_row::<PackedVal<SC>>(i_start + next_step),
                            );
                        } else {
                            prep_buf.resize(2 * preprocessed.width(), PackedVal::<SC>::ZERO);
                        }
                        RowMajorMatrixView::new(prep_buf.as_slice(), preprocessed.width())
                    },
                );

                // Build a packed permutation matrix from element-wise reads.
                // The buffer is cleared and refilled each iteration without reallocating.
                perm_buf.clear();
                if let Some(permutation_on_quotient_domain) =
                    opt_permutation_on_quotient_domain.as_ref()
                {
                    // Current-row permutation columns.
                    perm_buf.extend((0..perm_width).step_by(ext_degree).map(|col| {
                        PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
                            PackedVal::<SC>::from_fn(|offset| {
                                permutation_on_quotient_domain
                                    .get((i_start + offset) % perm_height, col + i)
                                    .unwrap()
                            })
                        })
                    }));
                    // Next-row permutation columns.
                    perm_buf.extend((0..perm_width).step_by(ext_degree).map(|col| {
                        PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
                            PackedVal::<SC>::from_fn(|offset| {
                                permutation_on_quotient_domain
                                    .get((i_start + next_step + offset) % perm_height, col + i)
                                    .unwrap()
                            })
                        })
                    }));
                }
                let permutation = RowMajorMatrixView::new(perm_buf.as_slice(), perm_cols);

                // Swap in the reusable constraint buffers (already cleared).
                base_buf.clear();
                ext_buf.clear();
                let periodic_values: &[PackedVal<SC>] = if periodic_packed.is_empty() {
                    &[]
                } else {
                    let group = (i_start / pack_width) % groups_in_period;
                    &periodic_packed[group * ncols..group * ncols + ncols]
                };
                let inner_folder = ProverConstraintFolder {
                    main,
                    preprocessed: preprocessed_view,
                    preprocessed_window: RowWindow::from_view(&preprocessed_view),
                    periodic_values,
                    public_values,
                    is_first_row,
                    is_last_row,
                    is_transition,
                    base_alpha_powers: &base_alpha_powers,
                    ext_alpha_powers: &ext_alpha_powers,
                    base_constraints: core::mem::take(base_buf),
                    ext_constraints: core::mem::take(ext_buf),
                    constraint_index: 0,
                    constraint_count,
                };

                // Wrap the inner folder with lookup-specific fields and evaluate.
                let mut folder = ProverConstraintFolderWithLookups {
                    inner: inner_folder,
                    permutation: permutation.as_view(),
                    permutation_challenges: &packed_perm_challenges,
                    permutation_values: &permutation_vals_packed,
                };
                lookup_gadget.eval_air_and_lookups(air, &mut folder, lookups);

                // Combine all constraints with alpha powers and divide by the vanishing polynomial.
                let quotient = folder.inner.finalize_constraints() * inv_vanishing;

                // Reclaim buffers for reuse in the next iteration.
                *base_buf = folder.inner.base_constraints;
                *ext_buf = folder.inner.ext_constraints;

                // Unpack the SIMD quotient into individual extension-field values
                // and write them directly into the pre-allocated output buffer.
                // SAFETY: Each i_start is unique and targets a disjoint slice.
                let out = unsafe { result_ptr.slice_mut(i_start, chunk_emit) };
                for (idx_in_packing, slot) in out.iter_mut().enumerate() {
                    *slot = quotient.extract(idx_in_packing);
                }
            },
        );

    result
}

#[cfg(feature = "secret-memory")]
#[allow(clippy::too_many_arguments)]
fn append_guarded_permutation_row<SC, Mat>(
    matrix: &Mat,
    row_start: usize,
    matrix_height: usize,
    matrix_width: usize,
    extension_degree: usize,
    pack_width: usize,
    scalar_pack: &mut SecretProverData<Vec<Val<SC>>>,
    extension_coefficients: &mut SecretProverData<Vec<PackedVal<SC>>>,
    destination: &mut SecretProverData<Vec<PackedChallenge<SC>>>,
) -> Result<(), SecretQuotientError>
where
    SC: SGC,
    Val<SC>: Zeroize,
    PackedVal<SC>: Zeroize,
    PackedChallenge<SC>: Zeroize,
    Mat: Matrix<Val<SC>>,
{
    for column_start in (0..matrix_width).step_by(extension_degree) {
        extension_coefficients.zeroize();
        for dimension in 0..extension_degree {
            scalar_pack.zeroize();
            let column = column_start
                .checked_add(dimension)
                .ok_or(SecretQuotientError::DimensionOverflow)?;
            for offset in 0..pack_width {
                let row = row_start
                    .checked_add(offset)
                    .ok_or(SecretQuotientError::DimensionOverflow)?
                    % matrix_height;
                let value = matrix
                    .get(row, column)
                    .ok_or(SecretQuotientError::InvalidShape)?;
                scalar_pack.push(value);
            }
            extension_coefficients.push(*PackedVal::<SC>::from_slice(scalar_pack.as_slice()));
        }
        let packed =
            PackedChallenge::<SC>::from_basis_coefficients_slice(extension_coefficients.as_slice())
                .ok_or(SecretQuotientError::InvalidShape)?;
        destination.push(packed);
    }
    Ok(())
}

/// Evaluate a quotient polynomial directly into the base-field layout consumed
/// by the PCS while retaining every witness-derived heap allocation under
/// zeroizing ownership.
///
/// This is the guarded counterpart of [`quotient_values`]. Selector tables,
/// periodic columns, transcript challenges, and alpha powers are public. The
/// output, packed trace rows, permutation rows, and accumulated constraints are
/// secret-derived and are guarded before their first write. Reusable row
/// buffers are explicitly erased before their lengths are reset, including the
/// spare capacity that ordinary `Vec::clear` leaves untouched.
///
/// Writing the extension-field coefficients directly avoids ever owning both
/// an extension-field quotient and a flattened copy. The returned matrix has
/// one column per extension-basis coefficient and must pass directly to a
/// guarded quotient commitment capability.
#[cfg(feature = "secret-memory")]
#[instrument(
    name = "compute guarded quotient polynomial",
    skip_all,
    level = "debug"
)]
#[allow(clippy::too_many_arguments)]
fn quotient_value_chunks_base_secret<SC, A, Mat, LG>(
    pcs: &SC::Pcs,
    air: &A,
    public_values: &[Val<SC>],
    layout: AirLayout,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    trace_on_quotient_domain: &Mat,
    opt_permutation_on_quotient_domain: Option<&Mat>,
    lookups: &[Lookup<Val<SC>>],
    permutation_vals: &[SC::Challenge],
    lookup_gadget: &LG,
    permutation_challenges: &[SC::Challenge],
    preprocessed_on_quotient_domain: Option<&Mat>,
    alpha: SC::Challenge,
    num_chunks: usize,
) -> Result<Vec<SecretProverData<RowMajorMatrix<Val<SC>>>>, SecretQuotientError>
where
    SC: SGC,
    Val<SC>: Zeroize,
    SC::Challenge: Zeroize,
    PackedVal<SC>: Zeroize,
    PackedChallenge<SC>: Zeroize,
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
    LG: LookupProtocol + Sync,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
{
    if !air.public_boundary_io().is_empty() {
        return Err(SecretQuotientError::UnsupportedPublicBoundary);
    }

    let quotient_size = quotient_domain.size();
    let trace_size = trace_domain.size();
    let main_width = trace_on_quotient_domain.width();
    if quotient_size == 0
        || trace_size == 0
        || !quotient_size.is_power_of_two()
        || !trace_size.is_power_of_two()
        || quotient_size < trace_size
        || num_chunks == 0
        || !num_chunks.is_power_of_two()
        || num_chunks > quotient_size
        || !quotient_size.is_multiple_of(num_chunks)
        || trace_on_quotient_domain.height() != quotient_size
        || main_width == 0
        || main_width != layout.main_width
        || public_values.len() != layout.num_public_values
    {
        return Err(SecretQuotientError::InvalidShape);
    }

    let extension_degree = SC::Challenge::DIMENSION;
    if extension_degree == 0 {
        return Err(SecretQuotientError::InvalidShape);
    }
    let (permutation_width, permutation_height) = opt_permutation_on_quotient_domain
        .map_or((0, 0), |matrix| (matrix.width(), matrix.height()));
    if opt_permutation_on_quotient_domain.is_some()
        && (permutation_width == 0
            || permutation_height != quotient_size
            || !permutation_width.is_multiple_of(extension_degree))
    {
        return Err(SecretQuotientError::InvalidShape);
    }
    if let Some(preprocessed) = preprocessed_on_quotient_domain
        && (preprocessed.height() != quotient_size
            || preprocessed.width() != layout.preprocessed_width)
    {
        return Err(SecretQuotientError::InvalidShape);
    }
    if preprocessed_on_quotient_domain.is_none() && layout.preprocessed_width != 0 {
        return Err(SecretQuotientError::InvalidShape);
    }

    let quotient_log = log2_strict_usize(quotient_size);
    let trace_log = log2_strict_usize(trace_size);
    let quotient_degree_bits = quotient_log
        .checked_sub(trace_log)
        .ok_or(SecretQuotientError::InvalidShape)?;
    let shift =
        u32::try_from(quotient_degree_bits).map_err(|_| SecretQuotientError::DimensionOverflow)?;
    let next_step = 1usize
        .checked_shl(shift)
        .ok_or(SecretQuotientError::DimensionOverflow)?;

    let pack_width = PackedVal::<SC>::WIDTH;
    if pack_width == 0 {
        return Err(SecretQuotientError::InvalidShape);
    }
    let remainder = quotient_size % pack_width;
    let padding = if remainder == 0 {
        0
    } else {
        pack_width
            .checked_sub(remainder)
            .ok_or(SecretQuotientError::DimensionOverflow)?
    };
    let padded_selector_len = quotient_size
        .checked_add(padding)
        .ok_or(SecretQuotientError::DimensionOverflow)?;

    let mut selectors = debug_span!("Compute Selectors")
        .in_scope(|| trace_domain.selectors_on_coset(quotient_domain));
    let pad_selector = |values: &mut Vec<Val<SC>>| -> Result<(), SecretQuotientError> {
        if values.len() != quotient_size {
            return Err(SecretQuotientError::InvalidShape);
        }
        values
            .try_reserve_exact(padding)
            .map_err(|_| SecretQuotientError::AllocationFailed)?;
        for index in quotient_size..padded_selector_len {
            values.push(values[index % quotient_size]);
        }
        Ok(())
    };
    pad_selector(&mut selectors.is_first_row)?;
    pad_selector(&mut selectors.is_last_row)?;
    pad_selector(&mut selectors.is_transition)?;
    pad_selector(&mut selectors.inv_vanishing)?;

    let constraint_layout = get_constraint_layout(air, layout, lookups, lookup_gadget);
    let (base_alpha_powers, ext_alpha_powers) = constraint_layout.decompose_alpha(alpha);

    let periodic_columns = air.periodic_columns();
    let periodic_table =
        pcs.build_periodic_lde_table(&periodic_columns, trace_domain, quotient_domain);
    let periodic_width = periodic_table.width();
    let groups_in_period = periodic_table.packed_group_period(pack_width);
    let periodic_len = groups_in_period
        .checked_mul(periodic_width)
        .ok_or(SecretQuotientError::DimensionOverflow)?;
    let mut periodic_packed = Vec::new();
    periodic_packed
        .try_reserve_exact(periodic_len)
        .map_err(|_| SecretQuotientError::AllocationFailed)?;
    if !periodic_table.is_empty() {
        for group in 0..groups_in_period {
            let row_start = group
                .checked_mul(pack_width)
                .ok_or(SecretQuotientError::DimensionOverflow)?;
            for column in 0..periodic_width {
                periodic_packed.push(PackedVal::<SC>::from_fn(|offset| {
                    *periodic_table.get(row_start + offset, column)
                }));
            }
        }
    }

    let mut packed_permutation_challenges = Vec::new();
    packed_permutation_challenges
        .try_reserve_exact(permutation_challenges.len())
        .map_err(|_| SecretQuotientError::AllocationFailed)?;
    packed_permutation_challenges.extend(
        permutation_challenges
            .iter()
            .copied()
            .map(PackedChallenge::<SC>::from),
    );
    let mut packed_permutation_values = Vec::new();
    packed_permutation_values
        .try_reserve_exact(permutation_vals.len())
        .map_err(|_| SecretQuotientError::AllocationFailed)?;
    packed_permutation_values.extend(
        permutation_vals
            .iter()
            .copied()
            .map(PackedChallenge::<SC>::from),
    );

    let base_constraint_capacity = constraint_layout.base_indices.len();
    let extension_constraint_capacity = constraint_layout.ext_indices.len();
    let constraint_count = constraint_layout.total_constraints();
    let permutation_columns = permutation_width / extension_degree;
    let permutation_capacity = permutation_columns
        .checked_mul(2)
        .ok_or(SecretQuotientError::DimensionOverflow)?;
    let main_capacity = main_width
        .checked_mul(2)
        .ok_or(SecretQuotientError::DimensionOverflow)?;
    let preprocessed_capacity = preprocessed_on_quotient_domain
        .map_or(0, Matrix::width)
        .checked_mul(2)
        .ok_or(SecretQuotientError::DimensionOverflow)?;

    let rows_per_chunk = quotient_size / num_chunks;
    let output_len = rows_per_chunk
        .checked_mul(extension_degree)
        .ok_or(SecretQuotientError::DimensionOverflow)?;
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(num_chunks)
        .map_err(|_| SecretQuotientError::AllocationFailed)?;
    for _ in 0..num_chunks {
        let mut output_values = Vec::new();
        output_values
            .try_reserve_exact(output_len)
            .map_err(|_| SecretQuotientError::AllocationFailed)?;
        output_values.resize(output_len, Val::<SC>::ZERO);
        outputs.push(SecretProverData::new(RowMajorMatrix::new(
            output_values,
            extension_degree,
        )));
    }
    let output_pointers = outputs
        .iter_mut()
        .map(|output| DisjointMutPtr::new(&mut output.values))
        .collect::<Vec<_>>();

    (0..quotient_size)
        .into_par_iter()
        .step_by(pack_width)
        .map_init(
            || {
                SecretQuotientScratch::<SC>::new(
                    permutation_capacity,
                    main_capacity,
                    preprocessed_capacity,
                    pack_width,
                    extension_degree,
                )
            },
            |scratch_result, row_start| {
                let scratch = scratch_result.as_mut().map_err(|error| *error)?;
                scratch.erase_for_reuse();

                let emitted = pack_width.min(
                    quotient_size
                        .checked_sub(row_start)
                        .ok_or(SecretQuotientError::InvalidShape)?,
                );
                let row_end = row_start
                    .checked_add(pack_width)
                    .ok_or(SecretQuotientError::DimensionOverflow)?;
                let next_row_start = row_start
                    .checked_add(next_step)
                    .ok_or(SecretQuotientError::DimensionOverflow)?;
                let row_range = row_start..row_end;
                let is_first_row =
                    *PackedVal::<SC>::from_slice(&selectors.is_first_row[row_range.clone()]);
                let is_last_row =
                    *PackedVal::<SC>::from_slice(&selectors.is_last_row[row_range.clone()]);
                let is_transition =
                    *PackedVal::<SC>::from_slice(&selectors.is_transition[row_range.clone()]);
                let inv_vanishing =
                    *PackedVal::<SC>::from_slice(&selectors.inv_vanishing[row_range]);

                scratch.main.extend(
                    trace_on_quotient_domain.vertically_packed_row::<PackedVal<SC>>(row_start),
                );
                scratch.main.extend(
                    trace_on_quotient_domain.vertically_packed_row::<PackedVal<SC>>(next_row_start),
                );
                if scratch.main.len() != main_capacity {
                    return Err(SecretQuotientError::InvalidShape);
                }
                let main = RowMajorMatrixView::new(scratch.main.as_slice(), main_width);

                let preprocessed = preprocessed_on_quotient_domain.map_or_else(
                    || Ok(RowMajorMatrixView::new(&[], 0)),
                    |matrix| {
                        scratch
                            .preprocessed
                            .extend(matrix.vertically_packed_row::<PackedVal<SC>>(row_start));
                        scratch
                            .preprocessed
                            .extend(matrix.vertically_packed_row::<PackedVal<SC>>(next_row_start));
                        if scratch.preprocessed.len() != preprocessed_capacity {
                            return Err(SecretQuotientError::InvalidShape);
                        }
                        Ok(RowMajorMatrixView::new(
                            scratch.preprocessed.as_slice(),
                            matrix.width(),
                        ))
                    },
                )?;

                if let Some(permutation_matrix) = opt_permutation_on_quotient_domain {
                    append_guarded_permutation_row::<SC, _>(
                        permutation_matrix,
                        row_start,
                        permutation_height,
                        permutation_width,
                        extension_degree,
                        pack_width,
                        &mut scratch.scalar_pack,
                        &mut scratch.extension_coefficients,
                        &mut scratch.permutation,
                    )?;
                    append_guarded_permutation_row::<SC, _>(
                        permutation_matrix,
                        next_row_start,
                        permutation_height,
                        permutation_width,
                        extension_degree,
                        pack_width,
                        &mut scratch.scalar_pack,
                        &mut scratch.extension_coefficients,
                        &mut scratch.permutation,
                    )?;
                }
                if scratch.permutation.len() != permutation_capacity {
                    return Err(SecretQuotientError::InvalidShape);
                }
                let permutation =
                    RowMajorMatrixView::new(scratch.permutation.as_slice(), permutation_columns);

                let periodic_values: &[PackedVal<SC>] = if periodic_packed.is_empty() {
                    &[]
                } else {
                    let group = (row_start / pack_width) % groups_in_period;
                    let start = group
                        .checked_mul(periodic_width)
                        .ok_or(SecretQuotientError::DimensionOverflow)?;
                    let end = start
                        .checked_add(periodic_width)
                        .ok_or(SecretQuotientError::DimensionOverflow)?;
                    periodic_packed
                        .get(start..end)
                        .ok_or(SecretQuotientError::InvalidShape)?
                };

                let mut base_constraints = Vec::new();
                base_constraints
                    .try_reserve_exact(base_constraint_capacity)
                    .map_err(|_| SecretQuotientError::AllocationFailed)?;
                let mut extension_constraints = Vec::new();
                extension_constraints
                    .try_reserve_exact(extension_constraint_capacity)
                    .map_err(|_| SecretQuotientError::AllocationFailed)?;
                let inner = ProverConstraintFolder {
                    main,
                    preprocessed,
                    preprocessed_window: RowWindow::from_view(&preprocessed),
                    periodic_values,
                    public_values,
                    is_first_row,
                    is_last_row,
                    is_transition,
                    base_alpha_powers: &base_alpha_powers,
                    ext_alpha_powers: &ext_alpha_powers,
                    base_constraints,
                    ext_constraints: extension_constraints,
                    constraint_index: 0,
                    constraint_count,
                };
                let folder = ProverConstraintFolderWithLookups {
                    inner,
                    permutation: permutation.as_view(),
                    permutation_challenges: &packed_permutation_challenges,
                    permutation_values: &packed_permutation_values,
                };
                let mut guarded_folder = SecretProverData::new(folder);
                lookup_gadget.eval_air_and_lookups(air, &mut *guarded_folder, lookups);
                let quotient = guarded_folder.inner.finalize_constraints() * inv_vanishing;

                for lane in 0..emitted {
                    let source_row = row_start
                        .checked_add(lane)
                        .ok_or(SecretQuotientError::DimensionOverflow)?;
                    let chunk_index = source_row % num_chunks;
                    let destination_row = source_row / num_chunks;
                    let destination_start = destination_row
                        .checked_mul(extension_degree)
                        .ok_or(SecretQuotientError::DimensionOverflow)?;
                    let output_pointer = output_pointers
                        .get(chunk_index)
                        .copied()
                        .ok_or(SecretQuotientError::InvalidShape)?;
                    // SAFETY: every source row is visited exactly once and the
                    // `(source_row % num_chunks, source_row / num_chunks)` map
                    // is bijective over the preallocated chunk rows. Checked
                    // arithmetic and validated geometry bound the destination.
                    let destination_row =
                        unsafe { output_pointer.slice_mut(destination_start, extension_degree) };
                    let mut value = quotient.extract(lane);
                    let coefficients = value.as_basis_coefficients_slice();
                    if coefficients.len() != extension_degree {
                        value.zeroize();
                        return Err(SecretQuotientError::InvalidShape);
                    }
                    destination_row.copy_from_slice(coefficients);
                    value.zeroize();
                }
                Ok(())
            },
        )
        .try_for_each(core::convert::identity)?;

    Ok(outputs)
}

#[cfg(feature = "secret-memory")]
#[allow(clippy::too_many_arguments)]
fn quotient_values_base_secret<SC, A, Mat, LG>(
    pcs: &SC::Pcs,
    air: &A,
    public_values: &[Val<SC>],
    layout: AirLayout,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    trace_on_quotient_domain: &Mat,
    opt_permutation_on_quotient_domain: Option<&Mat>,
    lookups: &[Lookup<Val<SC>>],
    permutation_vals: &[SC::Challenge],
    lookup_gadget: &LG,
    permutation_challenges: &[SC::Challenge],
    preprocessed_on_quotient_domain: Option<&Mat>,
    alpha: SC::Challenge,
) -> Result<SecretProverData<RowMajorMatrix<Val<SC>>>, SecretQuotientError>
where
    SC: SGC,
    Val<SC>: Zeroize,
    SC::Challenge: Zeroize,
    PackedVal<SC>: Zeroize,
    PackedChallenge<SC>: Zeroize,
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
    LG: LookupProtocol + Sync,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
{
    let mut chunks = quotient_value_chunks_base_secret::<SC, A, Mat, LG>(
        pcs,
        air,
        public_values,
        layout,
        trace_domain,
        quotient_domain,
        trace_on_quotient_domain,
        opt_permutation_on_quotient_domain,
        lookups,
        permutation_vals,
        lookup_gadget,
        permutation_challenges,
        preprocessed_on_quotient_domain,
        alpha,
        1,
    )?;
    if chunks.len() != 1 {
        return Err(SecretQuotientError::InvalidShape);
    }
    chunks.pop().ok_or(SecretQuotientError::InvalidShape)
}

/// Evaluate a guarded quotient and return its conventional extension-field
/// column.
///
/// This compatibility entry point necessarily allocates a replacement matrix.
/// The complete guarded batch prover uses [`quotient_values_base_secret`]
/// directly so its commitment path never performs this representation copy.
#[cfg(feature = "secret-memory")]
#[allow(clippy::too_many_arguments)]
pub fn quotient_values_secret<SC, A, Mat, LG>(
    pcs: &SC::Pcs,
    air: &A,
    public_values: &[Val<SC>],
    layout: AirLayout,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    trace_on_quotient_domain: &Mat,
    opt_permutation_on_quotient_domain: Option<&Mat>,
    lookups: &[Lookup<Val<SC>>],
    permutation_vals: &[SC::Challenge],
    lookup_gadget: &LG,
    permutation_challenges: &[SC::Challenge],
    preprocessed_on_quotient_domain: Option<&Mat>,
    alpha: SC::Challenge,
) -> Result<SecretProverData<RowMajorMatrix<SC::Challenge>>, SecretQuotientError>
where
    SC: SGC,
    Val<SC>: Zeroize,
    SC::Challenge: Zeroize,
    PackedVal<SC>: Zeroize,
    PackedChallenge<SC>: Zeroize,
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
    LG: LookupProtocol + Sync,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
{
    let base = quotient_values_base_secret::<SC, A, Mat, LG>(
        pcs,
        air,
        public_values,
        layout,
        trace_domain,
        quotient_domain,
        trace_on_quotient_domain,
        opt_permutation_on_quotient_domain,
        lookups,
        permutation_vals,
        lookup_gadget,
        permutation_challenges,
        preprocessed_on_quotient_domain,
        alpha,
    )?;
    recompose_base_quotient::<SC>(base)
}

#[cfg(feature = "secret-memory")]
fn recompose_base_quotient<SC>(
    mut base: SecretProverData<RowMajorMatrix<Val<SC>>>,
) -> Result<SecretProverData<RowMajorMatrix<SC::Challenge>>, SecretQuotientError>
where
    SC: SGC,
    Val<SC>: Zeroize,
    SC::Challenge: Zeroize,
{
    let extension_degree = SC::Challenge::DIMENSION;
    if extension_degree == 0
        || base.width != extension_degree
        || !base.values.len().is_multiple_of(extension_degree)
    {
        return Err(SecretQuotientError::InvalidShape);
    }
    let quotient_size = base.values.len() / extension_degree;
    let mut values = Vec::new();
    values
        .try_reserve_exact(quotient_size)
        .map_err(|_| SecretQuotientError::AllocationFailed)?;
    values.resize(quotient_size, SC::Challenge::ZERO);
    let mut output = SecretProverData::new(RowMajorMatrix::new(values, 1));
    for (coefficients, value) in base
        .values
        .chunks_exact(extension_degree)
        .zip(output.values.iter_mut())
    {
        *value = SC::Challenge::from_basis_coefficients_slice(coefficients)
            .ok_or(SecretQuotientError::InvalidShape)?;
    }
    base.zeroize();
    Ok(output)
}

/// Acquire private quotient-domain matrices from guarded PCS data and evaluate
/// the AIR quotient without opening an ordinary-memory ownership gap.
///
/// Main and optional permutation commitments remain guarded from commitment
/// through domain evaluation and quotient consumption. Preprocessed columns
/// are deliberately accepted as an ordinary matrix because they are public;
/// callers must not place witness-bearing data in that argument. This remains
/// a quotient-stage capability, not a complete guarded batch prover.
#[cfg(feature = "secret-memory")]
#[allow(clippy::too_many_arguments)]
fn quotient_value_chunks_base_from_secret_pcs<SC, A, LG>(
    pcs: &SC::Pcs,
    air: &A,
    public_values: &[Val<SC>],
    layout: AirLayout,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    main_data: &SecretPcsData<SC>,
    main_index: usize,
    permutation_data: Option<(&SecretPcsData<SC>, usize)>,
    lookups: &[Lookup<Val<SC>>],
    permutation_vals: &[SC::Challenge],
    lookup_gadget: &LG,
    permutation_challenges: &[SC::Challenge],
    preprocessed_on_quotient_domain: Option<&RowMajorMatrix<Val<SC>>>,
    alpha: SC::Challenge,
    num_chunks: usize,
) -> SecretPcsBaseQuotientChunksResult<SC>
where
    SC: SGC,
    Val<SC>: Zeroize,
    SC::Challenge: Zeroize,
    PackedVal<SC>: Zeroize,
    PackedChallenge<SC>: Zeroize,
    SC::Pcs: SecretUnivariateStarkPcsEvaluator<SC::Challenge, SC::Challenger>,
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>,
    LG: LookupProtocol + Sync,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
{
    let main = pcs
        .get_evaluations_on_domain_secret(main_data, main_index, quotient_domain)
        .map_err(SecretPcsQuotientError::MainEvaluation)?;
    let permutation = match permutation_data {
        Some((data, index)) => Some(
            pcs.get_evaluations_on_domain_secret(data, index, quotient_domain)
                .map_err(SecretPcsQuotientError::PermutationEvaluation)?,
        ),
        None => None,
    };
    let permutation_ref = permutation.as_deref();

    quotient_value_chunks_base_secret::<SC, A, RowMajorMatrix<Val<SC>>, LG>(
        pcs,
        air,
        public_values,
        layout,
        trace_domain,
        quotient_domain,
        &main,
        permutation_ref,
        lookups,
        permutation_vals,
        lookup_gadget,
        permutation_challenges,
        preprocessed_on_quotient_domain,
        alpha,
        num_chunks,
    )
    .map_err(SecretPcsQuotientError::Quotient)
}

#[cfg(feature = "secret-memory")]
#[allow(clippy::too_many_arguments)]
fn quotient_value_chunk_base_from_secret_pcs<SC, A, LG>(
    pcs: &SC::Pcs,
    air: &A,
    public_values: &[Val<SC>],
    layout: AirLayout,
    trace_domain: Domain<SC>,
    chunk_domain: Domain<SC>,
    main_data: &SecretPcsData<SC>,
    main_index: usize,
    permutation_data: Option<(&SecretPcsData<SC>, usize)>,
    lookups: &[Lookup<Val<SC>>],
    permutation_vals: &[SC::Challenge],
    lookup_gadget: &LG,
    permutation_challenges: &[SC::Challenge],
    preprocessed_on_chunk_domain: Option<&RowMajorMatrix<Val<SC>>>,
    alpha: SC::Challenge,
) -> SecretPcsBaseQuotientResult<SC>
where
    SC: SGC,
    Val<SC>: Zeroize,
    SC::Challenge: Zeroize,
    PackedVal<SC>: Zeroize,
    PackedChallenge<SC>: Zeroize,
    SC::Pcs: SecretUnivariateStarkPcsEvaluator<SC::Challenge, SC::Challenger>,
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>,
    LG: LookupProtocol + Sync,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
{
    let mut chunks = quotient_value_chunks_base_from_secret_pcs::<SC, A, LG>(
        pcs,
        air,
        public_values,
        layout,
        trace_domain,
        chunk_domain,
        main_data,
        main_index,
        permutation_data,
        lookups,
        permutation_vals,
        lookup_gadget,
        permutation_challenges,
        preprocessed_on_chunk_domain,
        alpha,
        1,
    )?;
    if chunks.len() != 1 {
        return Err(SecretPcsQuotientError::Quotient(
            SecretQuotientError::InvalidShape,
        ));
    }
    chunks.pop().ok_or(SecretPcsQuotientError::Quotient(
        SecretQuotientError::InvalidShape,
    ))
}

#[cfg(feature = "secret-memory")]
#[allow(clippy::too_many_arguments)]
fn quotient_values_base_from_secret_pcs<SC, A, LG>(
    pcs: &SC::Pcs,
    air: &A,
    public_values: &[Val<SC>],
    layout: AirLayout,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    main_data: &SecretPcsData<SC>,
    main_index: usize,
    permutation_data: Option<(&SecretPcsData<SC>, usize)>,
    lookups: &[Lookup<Val<SC>>],
    permutation_vals: &[SC::Challenge],
    lookup_gadget: &LG,
    permutation_challenges: &[SC::Challenge],
    preprocessed_on_quotient_domain: Option<&RowMajorMatrix<Val<SC>>>,
    alpha: SC::Challenge,
) -> SecretPcsBaseQuotientResult<SC>
where
    SC: SGC,
    Val<SC>: Zeroize,
    SC::Challenge: Zeroize,
    PackedVal<SC>: Zeroize,
    PackedChallenge<SC>: Zeroize,
    SC::Pcs: SecretUnivariateStarkPcsEvaluator<SC::Challenge, SC::Challenger>,
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>,
    LG: LookupProtocol + Sync,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
{
    let main = pcs
        .get_evaluations_on_domain_secret(main_data, main_index, quotient_domain)
        .map_err(SecretPcsQuotientError::MainEvaluation)?;
    let permutation = match permutation_data {
        Some((data, index)) => Some(
            pcs.get_evaluations_on_domain_secret(data, index, quotient_domain)
                .map_err(SecretPcsQuotientError::PermutationEvaluation)?,
        ),
        None => None,
    };
    let permutation_ref = permutation.as_deref();

    quotient_values_base_secret::<SC, A, RowMajorMatrix<Val<SC>>, LG>(
        pcs,
        air,
        public_values,
        layout,
        trace_domain,
        quotient_domain,
        &main,
        permutation_ref,
        lookups,
        permutation_vals,
        lookup_gadget,
        permutation_challenges,
        preprocessed_on_quotient_domain,
        alpha,
    )
    .map_err(SecretPcsQuotientError::Quotient)
}

/// Acquire guarded PCS evaluations and return the quotient in its conventional
/// extension-field representation.
///
/// The complete batch prover bypasses this compatibility representation and
/// hands the base-field quotient directly to the PCS commitment stage.
#[cfg(feature = "secret-memory")]
#[allow(clippy::too_many_arguments)]
pub fn quotient_values_from_secret_pcs<SC, A, LG>(
    pcs: &SC::Pcs,
    air: &A,
    public_values: &[Val<SC>],
    layout: AirLayout,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    main_data: &SecretPcsData<SC>,
    main_index: usize,
    permutation_data: Option<(&SecretPcsData<SC>, usize)>,
    lookups: &[Lookup<Val<SC>>],
    permutation_vals: &[SC::Challenge],
    lookup_gadget: &LG,
    permutation_challenges: &[SC::Challenge],
    preprocessed_on_quotient_domain: Option<&RowMajorMatrix<Val<SC>>>,
    alpha: SC::Challenge,
) -> SecretPcsQuotientResult<SC>
where
    SC: SGC,
    Val<SC>: Zeroize,
    SC::Challenge: Zeroize,
    PackedVal<SC>: Zeroize,
    PackedChallenge<SC>: Zeroize,
    SC::Pcs: SecretUnivariateStarkPcsEvaluator<SC::Challenge, SC::Challenger>,
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>,
    LG: LookupProtocol + Sync,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
{
    let base = quotient_values_base_from_secret_pcs::<SC, A, LG>(
        pcs,
        air,
        public_values,
        layout,
        trace_domain,
        quotient_domain,
        main_data,
        main_index,
        permutation_data,
        lookups,
        permutation_vals,
        lookup_gadget,
        permutation_challenges,
        preprocessed_on_quotient_domain,
        alpha,
    )?;
    recompose_base_quotient::<SC>(base).map_err(SecretPcsQuotientError::Quotient)
}

/// Consume a guarded extension-field quotient, flatten it without releasing
/// either representation un-erased, and transfer it to the guarded PCS
/// commitment path.
///
/// Keeping this as one consuming operation prevents callers from accidentally
/// materializing an unguarded base-field quotient between evaluation and
/// commitment.
#[cfg(feature = "secret-memory")]
pub fn commit_quotient_values_secret<SC>(
    pcs: &SC::Pcs,
    quotient_domain: Domain<SC>,
    quotient_values: SecretProverData<RowMajorMatrix<SC::Challenge>>,
    num_chunks: usize,
) -> SecretQuotientCommitResult<SC>
where
    SC: SGC,
    Val<SC>: Zeroize,
    SC::Challenge: Zeroize,
    SC::Pcs: SecretUnivariateStarkPcsCommitter<SC::Challenge, SC::Challenger>,
{
    let mut quotients = Vec::new();
    quotients
        .try_reserve_exact(1)
        .map_err(|_| SecretQuotientCommitError::AllocationFailed)?;
    quotients.push(SecretBatchQuotient::new(
        quotient_domain,
        quotient_values,
        num_chunks,
    ));
    commit_quotient_batch_values_secret::<SC>(pcs, quotients)
}

/// Commit every guarded instance quotient under the one global root expected
/// by the batch-STARK proof format.
///
/// Extension-to-base conversion consumes each guarded source and transfers the
/// new guarded representation directly into the PCS. Input and chunk order are
/// preserved so the existing verifier's quotient-opening layout is unchanged.
#[cfg(feature = "secret-memory")]
pub fn commit_quotient_batch_values_secret<SC>(
    pcs: &SC::Pcs,
    quotients: Vec<SecretBatchQuotient<SC>>,
) -> SecretQuotientCommitResult<SC>
where
    SC: SGC,
    Val<SC>: Zeroize,
    SC::Challenge: Zeroize,
    SC::Pcs: SecretUnivariateStarkPcsCommitter<SC::Challenge, SC::Challenger>,
{
    let mut inputs = Vec::new();
    inputs
        .try_reserve_exact(quotients.len())
        .map_err(|_| SecretQuotientCommitError::AllocationFailed)?;
    for quotient in quotients {
        let (domain, evaluations, num_chunks) = quotient.into_parts();
        let flattened = evaluations
            .try_flatten_extension_to_base::<Val<SC>>()
            .map_err(SecretQuotientCommitError::Matrix)?;
        inputs.push(SecretQuotientCommitmentInput::new(
            domain, flattened, num_chunks,
        ));
    }
    commit_base_quotient_batch_values_secret::<SC>(pcs, inputs)
}

#[cfg(feature = "secret-memory")]
fn commit_base_quotient_batch_values_secret<SC>(
    pcs: &SC::Pcs,
    inputs: Vec<SecretQuotientCommitmentInput<Domain<SC>, Val<SC>>>,
) -> SecretQuotientCommitResult<SC>
where
    SC: SGC,
    Val<SC>: Zeroize,
    SC::Challenge: Zeroize,
    SC::Pcs: SecretUnivariateStarkPcsCommitter<SC::Challenge, SC::Challenger>,
{
    pcs.commit_quotients_secret(inputs)
        .map_err(SecretQuotientCommitError::Pcs)
}
