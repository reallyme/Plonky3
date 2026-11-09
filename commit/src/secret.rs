// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ownership vocabulary for commitment prover data derived from secrets.
//!
//! Implementing [`ErasableProverData`] is only the inner capability. The
//! allocation is erased automatically only while it remains held by
//! [`SecretProverData`]. Protocol crates should preserve that owner across
//! every return and unwind edge before advertising a secret-prover entry point.

use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::ExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use zeroize::Zeroize;

use crate::{
    BatchOpening, Mmcs, OpenedValues, OpeningRequest, Pcs, PolynomialSpace, UnivariateStarkPcs,
};

/// Prover data whose controlled storage can be explicitly erased.
pub trait ErasableProverData: Zeroize {}

impl<T> ErasableProverData for T where T: Zeroize {}

/// Non-debuggable, non-cloneable owner that erases prover data on drop.
pub type SecretProverData<T> = p3_matrix::secret::ZeroizingOwner<T>;

/// Guarded commitment output shared by secret MMCS and PCS capabilities.
pub type SecretCommitmentResult<Commitment, Data, Error> =
    Result<(Commitment, SecretProverData<Data>), Error>;

/// Guarded commitment output after value matrices have been detached.
pub struct SecretDetachedCommitment<Commitment, Data, AuthenticationData>
where
    Data: Zeroize,
{
    /// Public commitment produced from the original value matrices.
    pub commitment: Commitment,
    /// Erased commitment-data shell retained for type-compatible ownership.
    pub prover_data: SecretProverData<Data>,
    /// Digest layers and any hiding salts required for recomputed openings.
    pub authentication_data: AuthenticationData,
}

/// Fallible guarded detached-commit transition.
pub type SecretDetachedCommitmentResult<Commitment, Data, AuthenticationData, Error> =
    Result<SecretDetachedCommitment<Commitment, Data, AuthenticationData>, Error>;

/// Multi-row opening output from a guarded MMCS.
pub type SecretMultiBatchOpeningResult<T, Proof, Error> = Result<(Vec<Vec<Vec<T>>>, Proof), Error>;

/// One evaluation matrix whose storage remains guarded during PCS commitment.
pub type SecretPcsEvaluation<Domain, Val> = (Domain, SecretProverData<RowMajorMatrix<Val>>);

/// One instance quotient awaiting split-and-commit under a shared PCS root.
///
/// The evaluation matrix remains guarded while the PCS splits it into the
/// degree-bounded chunks required by the STARK opening layout. Keeping the
/// chunk count beside its domain prevents parallel arrays from drifting.
pub struct SecretQuotientCommitmentInput<Domain, Val>
where
    Val: Zeroize,
{
    domain: Domain,
    evaluations: SecretProverData<RowMajorMatrix<Val>>,
    num_chunks: usize,
}

/// Public geometry for one quotient in a lazily generated shared commitment.
///
/// The complete plan is validated before an implementation requests the first
/// secret matrix. This is particularly important for hiding commitments:
/// malformed public geometry must fail before the PCS advances its RNG.
#[derive(Clone, Copy, Debug)]
pub struct SecretQuotientCommitmentPlan<Domain> {
    domain: Domain,
    width: usize,
    num_chunks: usize,
}

impl<Domain> SecretQuotientCommitmentPlan<Domain> {
    /// Bind one quotient domain to its expected matrix width and chunk count.
    pub const fn new(domain: Domain, width: usize, num_chunks: usize) -> Self {
        Self {
            domain,
            width,
            num_chunks,
        }
    }

    /// Return the quotient evaluation domain.
    pub const fn domain(&self) -> &Domain {
        &self.domain
    }

    /// Return the exact base-field matrix width expected from the generator.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Return the number of degree-bounded quotient chunks.
    pub const fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    /// Consume the plan into its public components.
    pub fn into_parts(self) -> (Domain, usize, usize) {
        (self.domain, self.width, self.num_chunks)
    }
}

/// Failure while lazily generating or committing a planned quotient batch.
///
/// The variants intentionally do not format their generic payloads. Prover
/// errors can contain implementation context that must not leak through an
/// outer PCS error message.
#[derive(Debug, thiserror::Error)]
pub enum SecretGeneratedQuotientCommitError<GenerationError, CommitError>
where
    GenerationError: Debug,
    CommitError: Debug,
{
    /// The caller could not construct the next guarded quotient matrix.
    #[error("guarded quotient generation failed")]
    Generation(GenerationError),
    /// The PCS rejected the public plan or guarded commitment construction.
    #[error("guarded quotient commitment failed")]
    Commitment(CommitError),
}

/// Result of lazily generating a planned quotient batch under one commitment.
pub type SecretGeneratedQuotientCommitmentResult<Commitment, Data, GenerationError, CommitError> =
    Result<
        (Commitment, SecretProverData<Data>),
        SecretGeneratedQuotientCommitError<GenerationError, CommitError>,
    >;

impl<Domain, Val> SecretQuotientCommitmentInput<Domain, Val>
where
    Val: Zeroize,
{
    /// Bind a guarded quotient matrix to its evaluation domain and chunk count.
    pub const fn new(
        domain: Domain,
        evaluations: SecretProverData<RowMajorMatrix<Val>>,
        num_chunks: usize,
    ) -> Self {
        Self {
            domain,
            evaluations,
            num_chunks,
        }
    }

    /// Consume the typed input without releasing the guarded matrix owner.
    pub fn into_parts(self) -> (Domain, SecretProverData<RowMajorMatrix<Val>>, usize) {
        (self.domain, self.evaluations, self.num_chunks)
    }

    /// Borrow the public domain metadata for fail-before-consumption checks.
    pub const fn domain(&self) -> &Domain {
        &self.domain
    }

    /// Borrow the guarded evaluations without releasing their owner.
    pub const fn evaluations(&self) -> &SecretProverData<RowMajorMatrix<Val>> {
        &self.evaluations
    }

    /// Return the requested number of degree-bounded quotient chunks.
    pub const fn num_chunks(&self) -> usize {
        self.num_chunks
    }
}

/// Mutable guarded prover data paired with the public points requested from
/// each matrix in one PCS commitment.
///
/// The mutable borrow makes the single-use lifecycle explicit: guarded
/// opening implementations may irreversibly erase retained codewords after
/// Fiat-Shamir fixes the query indices.
pub struct SecretOpeningRequest<'a, ProverData, Challenge>
where
    ProverData: Zeroize,
{
    /// Guarded prover data retained by the corresponding commitment.
    pub prover_data: &'a mut SecretProverData<ProverData>,
    /// Public opening points, one vector per committed matrix.
    pub points: Vec<Vec<Challenge>>,
}

impl<'a, ProverData, Challenge> From<(&'a mut SecretProverData<ProverData>, Vec<Vec<Challenge>>)>
    for SecretOpeningRequest<'a, ProverData, Challenge>
where
    ProverData: Zeroize,
{
    fn from(
        (prover_data, points): (&'a mut SecretProverData<ProverData>, Vec<Vec<Challenge>>),
    ) -> Self {
        Self {
            prover_data,
            points,
        }
    }
}

/// One PCS opening round whose owner model is explicit in the type.
///
/// Public preprocessing remains immutably borrowed and reusable. Private
/// prover data is mutably borrowed because a successful guarded opening may
/// irreversibly erase retained codewords. The enum preserves the caller's
/// round order without converting either owner into the other representation.
pub enum SecretPcsMixedOpeningRequest<'a, PublicData, SecretData, Challenge>
where
    SecretData: Zeroize,
{
    /// Reusable public preprocessing committed through the ordinary PCS path.
    Public(OpeningRequest<'a, PublicData, Challenge>),
    /// Single-use private data committed through the guarded PCS path.
    Private(SecretOpeningRequest<'a, SecretData, Challenge>),
}

impl<'a, PublicData, SecretData, Challenge> From<OpeningRequest<'a, PublicData, Challenge>>
    for SecretPcsMixedOpeningRequest<'a, PublicData, SecretData, Challenge>
where
    SecretData: Zeroize,
{
    fn from(request: OpeningRequest<'a, PublicData, Challenge>) -> Self {
        Self::Public(request)
    }
}

impl<'a, PublicData, SecretData, Challenge> From<SecretOpeningRequest<'a, SecretData, Challenge>>
    for SecretPcsMixedOpeningRequest<'a, PublicData, SecretData, Challenge>
where
    SecretData: Zeroize,
{
    fn from(request: SecretOpeningRequest<'a, SecretData, Challenge>) -> Self {
        Self::Private(request)
    }
}

/// Complete output of a guarded PCS opening operation.
pub type SecretPcsOpeningResult<Challenge, Proof, Error> =
    Result<(OpenedValues<Challenge>, Proof), Error>;

/// Optional guarded commitment used by a STARK randomization polynomial.
pub type SecretOptionalCommitmentResult<Commitment, Data, Error> =
    Result<Option<(Commitment, SecretProverData<Data>)>, Error>;

/// Guarded matrix produced by evaluating PCS data on a requested domain.
pub type SecretDomainEvaluationResult<Domain, Error> =
    Result<SecretProverData<RowMajorMatrix<<Domain as PolynomialSpace>::Val>>, Error>;

/// MMCS operations that preserve a zeroizing owner around every retained matrix.
///
/// This is a separate capability from [`Mmcs`]. Implementations must validate
/// caller-controlled shapes before moving inputs into code paths that can panic,
/// and must retain guarded ownership on success and every returned error.
pub trait SecretMmcs<T>: Mmcs<T>
where
    T: Send + Sync + Clone + Zeroize,
{
    /// Retained prover data for guarded input matrix type `M`.
    type SecretData<M>: ErasableProverData
    where
        M: Matrix<T> + Zeroize;

    /// Typed failure from guarded commitment or opening construction.
    type SecretError: Debug;

    /// Commit to guarded matrices without releasing their backing storage.
    fn commit_secret<M>(
        &self,
        inputs: Vec<SecretProverData<M>>,
    ) -> SecretCommitmentResult<Self::Commitment, Self::SecretData<M>, Self::SecretError>
    where
        M: Matrix<T> + Zeroize;

    /// Open one row while the retained matrices remain guarded.
    fn open_batch_secret<M>(
        &self,
        index: usize,
        prover_data: &SecretProverData<Self::SecretData<M>>,
    ) -> Result<BatchOpening<T, Self>, Self::SecretError>
    where
        M: Matrix<T> + Zeroize,
        Self: Sized;

    /// Borrow the original matrices without releasing their erasure owners.
    fn get_matrices_secret<'a, M>(
        &self,
        prover_data: &'a SecretProverData<Self::SecretData<M>>,
    ) -> Vec<&'a M>
    where
        M: Matrix<T> + Zeroize;

    /// Open several rows while the retained matrices remain guarded.
    fn open_multi_batch_secret<M>(
        &self,
        indices: &[usize],
        prover_data: &SecretProverData<Self::SecretData<M>>,
    ) -> SecretMultiBatchOpeningResult<T, Self::MultiProof, Self::SecretError>
    where
        M: Matrix<T> + Zeroize;
}

/// Guarded MMCS capability that can discard committed matrices before queries
/// are known and later authenticate independently recomputed query rows.
///
/// The transition is intentionally irreversible. Implementations must erase
/// every committed matrix before releasing its allocation, while retaining
/// only the authentication state required by the original commitment. The
/// recomputed rows must be checked against the originally committed matrix
/// count, widths, heights, and query-index semantics.
pub trait SecretRecomputableMmcs<T>: SecretMmcs<T>
where
    T: Send + Sync + Clone + Zeroize,
{
    /// Authentication state retained after committed matrices are erased.
    type AuthenticationData<M>
    where
        M: Matrix<T> + Zeroize;

    /// Commit and immediately transition to authentication-only prover state.
    ///
    /// The default implementation deliberately composes the two existing,
    /// independently testable transitions. Implementations may override this
    /// method to release value matrices during tree construction, provided the
    /// commitment and authentication state remain byte-for-byte equivalent.
    #[allow(
        clippy::type_complexity,
        reason = "the associated MMCS data and authentication types must remain linked to M"
    )]
    fn commit_detached_secret<M>(
        &self,
        inputs: Vec<SecretProverData<M>>,
    ) -> SecretDetachedCommitmentResult<
        Self::Commitment,
        Self::SecretData<M>,
        Self::AuthenticationData<M>,
        Self::SecretError,
    >
    where
        M: Matrix<T> + Zeroize,
    {
        let (commitment, mut prover_data) = self.commit_secret(inputs)?;
        let authentication_data = self.detach_authentication_data(&mut prover_data)?;
        Ok(SecretDetachedCommitment {
            commitment,
            prover_data,
            authentication_data,
        })
    }

    /// Erase every committed matrix and retain its authentication state.
    fn detach_authentication_data<M>(
        &self,
        prover_data: &mut SecretProverData<Self::SecretData<M>>,
    ) -> Result<Self::AuthenticationData<M>, Self::SecretError>
    where
        M: Matrix<T> + Zeroize;

    /// Authenticate verifier-selected rows recomputed after detachment.
    fn open_multi_batch_from_recomputed_rows<M>(
        &self,
        indices: &[usize],
        opened_rows: &[Vec<Vec<T>>],
        authentication_data: &Self::AuthenticationData<M>,
    ) -> SecretMultiBatchOpeningResult<T, Self::MultiProof, Self::SecretError>
    where
        M: Matrix<T> + Zeroize;
}

/// One in-progress MMCS commitment that accepts guarded matrices in their
/// declared order and erases each matrix before returning from absorption.
///
/// Implementations must preserve the exact commitment and authentication-data
/// semantics of [`SecretRecomputableMmcs::commit_detached_secret`]. Dropping an
/// unfinished stream must erase all retained hash state and auxiliary secrets.
pub trait SecretStreamingMmcsCommitment<T, M>
where
    T: Send + Sync + Clone + Zeroize,
    M: Matrix<T> + Zeroize,
{
    /// Public commitment type produced by the MMCS.
    type Commitment;
    /// Erased prover-data shell retained for associated-type compatibility.
    type Data: ErasableProverData;
    /// Authentication state retained for recomputed openings.
    type AuthenticationData;
    /// Typed commitment failure.
    type Error: Debug;

    /// Absorb the next matrix and erase its owner on every return path.
    fn absorb(&mut self, matrix: SecretProverData<M>) -> Result<(), Self::Error>;

    /// Finish after every predeclared matrix has been absorbed.
    fn finish(
        self,
    ) -> SecretDetachedCommitmentResult<
        Self::Commitment,
        Self::Data,
        Self::AuthenticationData,
        Self::Error,
    >;
}

/// MMCS capability for committing generated matrices without retaining the
/// complete matrix batch in memory.
///
/// The caller supplies the complete public geometry before any matrix is
/// generated. Implementations must reject invalid geometry before accepting a
/// secret matrix. A successfully absorbed matrix cannot be recovered from the
/// stream; later openings use recomputed rows and retained authentication data.
pub trait SecretStreamingRecomputableMmcs<T>: SecretRecomputableMmcs<T>
where
    T: Send + Sync + Clone + Zeroize,
{
    /// Streaming state tied to the MMCS instance that owns its hash and RNG.
    type Stream<'a, M>: SecretStreamingMmcsCommitment<
            T,
            M,
            Commitment = Self::Commitment,
            Data = Self::SecretData<M>,
            AuthenticationData = Self::AuthenticationData<M>,
            Error = Self::SecretError,
        >
    where
        Self: 'a,
        M: Matrix<T> + Zeroize;

    /// Validate the complete public geometry and begin a streaming commit.
    fn begin_streaming_commitment<M>(
        &self,
        dimensions: Vec<Dimensions>,
    ) -> Result<Self::Stream<'_, M>, Self::SecretError>
    where
        M: Matrix<T> + Zeroize;
}

/// PCS commitment construction that preserves guarded matrix ownership.
///
/// Opening and proof construction are deliberately outside this capability.
/// A protocol must not present this trait as a complete secret prover.
pub trait SecretPcsCommitter<Challenge, Challenger>: Pcs<Challenge, Challenger>
where
    Challenge: ExtensionField<<Self::Domain as PolynomialSpace>::Val>,
    <Self::Domain as PolynomialSpace>::Val: Zeroize,
{
    /// Guarded prover data retained for later opening construction.
    type SecretData: ErasableProverData;

    /// Typed failure from the guarded commitment path.
    type SecretCommitError: Debug;

    /// Commit to guarded evaluation matrices.
    fn commit_secret(
        &self,
        evaluations: Vec<SecretPcsEvaluation<Self::Domain, <Self::Domain as PolynomialSpace>::Val>>,
    ) -> SecretCommitmentResult<Self::Commitment, Self::SecretData, Self::SecretCommitError>;
}

/// PCS capability for committing guarded matrices directly into detached,
/// recomputable prover state.
///
/// This is separate from [`SecretPcsCommitter`] because not every secret PCS
/// can reconstruct verifier-selected rows after releasing its full committed
/// value matrices. Implementations must preserve the exact commitment bytes
/// and matrix ordering of `commit_secret`.
pub trait SecretDetachedPcsCommitter<Challenge, Challenger>:
    SecretPcsCommitter<Challenge, Challenger>
where
    Challenge: ExtensionField<<Self::Domain as PolynomialSpace>::Val>,
    <Self::Domain as PolynomialSpace>::Val: Zeroize,
{
    /// Commit all evaluations and return prover data with value matrices
    /// already replaced by authentication-only state.
    fn commit_detached_secret(
        &self,
        evaluations: Vec<SecretPcsEvaluation<Self::Domain, <Self::Domain as PolynomialSpace>::Val>>,
    ) -> SecretCommitmentResult<Self::Commitment, Self::SecretData, Self::SecretCommitError>;
}

/// PCS opening construction that keeps all undisclosed prover data under
/// zeroizing ownership.
///
/// This capability intentionally accepts only guarded commitment rounds.
/// Protocols that mix reusable public preprocessing with private rounds need a
/// separate mixed-opening capability rather than silently converting either
/// owner model.
pub trait SecretPcsOpener<Challenge, Challenger>:
    SecretPcsCommitter<Challenge, Challenger>
where
    Challenge: ExtensionField<<Self::Domain as PolynomialSpace>::Val>,
    <Self::Domain as PolynomialSpace>::Val: Zeroize,
{
    /// Stable typed failure from guarded opening construction.
    type SecretOpenError: Debug;

    /// Open guarded commitment rounds and irreversibly consume any retained
    /// secret state whose last protocol use has completed.
    fn open_secret(
        &self,
        requests: Vec<SecretOpeningRequest<'_, Self::SecretData, Challenge>>,
        challenger: &mut Challenger,
    ) -> SecretPcsOpeningResult<Challenge, Self::Proof, Self::SecretOpenError>;
}

/// PCS opening construction for ordered public and private commitment rounds.
///
/// Implementations must leave public prover data reusable while applying the
/// same irreversible erasure contract as [`SecretPcsOpener`] to every private
/// round. Hiding implementations must additionally reject ambiguous public
/// preprocessing layouts before consuming any private round.
pub trait SecretPcsMixedOpener<Challenge, Challenger>:
    SecretPcsOpener<Challenge, Challenger>
where
    Challenge: ExtensionField<<Self::Domain as PolynomialSpace>::Val>,
    <Self::Domain as PolynomialSpace>::Val: Zeroize,
{
    /// Open ordered public preprocessing and guarded private rounds.
    fn open_secret_with_public_preprocessing(
        &self,
        requests: Vec<
            SecretPcsMixedOpeningRequest<'_, Self::ProverData, Self::SecretData, Challenge>,
        >,
        challenger: &mut Challenger,
    ) -> SecretPcsOpeningResult<Challenge, Self::Proof, Self::SecretOpenError>;
}

/// Quotient commitment construction that preserves guarded matrix ownership.
///
/// This capability is intentionally narrower than a complete secret STARK
/// prover. Implementations must keep the quotient evaluations, split chunks,
/// LDEs, and retained commitment data guarded through every success and error
/// edge. Opening construction remains a separate capability.
pub trait SecretUnivariateStarkPcsCommitter<Challenge, Challenger>:
    SecretPcsCommitter<Challenge, Challenger> + UnivariateStarkPcs<Challenge, Challenger>
where
    Challenge: ExtensionField<<Self::Domain as PolynomialSpace>::Val>,
    <Self::Domain as PolynomialSpace>::Val: Zeroize,
{
    /// Split, extend, and commit a guarded quotient evaluation matrix.
    fn commit_quotient_secret(
        &self,
        quotient_domain: Self::Domain,
        quotient_evaluations: SecretProverData<
            RowMajorMatrix<<Self::Domain as PolynomialSpace>::Val>,
        >,
        num_chunks: usize,
    ) -> SecretCommitmentResult<Self::Commitment, Self::SecretData, Self::SecretCommitError>;

    /// Split and commit several instance quotients under one shared PCS root.
    ///
    /// Implementations must validate every input before transforming any
    /// guarded matrix or advancing any hiding RNG. Output matrices retain the
    /// input order, and chunks for each input retain their domain split order.
    fn commit_quotients_secret(
        &self,
        quotients: Vec<
            SecretQuotientCommitmentInput<Self::Domain, <Self::Domain as PolynomialSpace>::Val>,
        >,
    ) -> SecretCommitmentResult<Self::Commitment, Self::SecretData, Self::SecretCommitError>;

    /// Generate, transform, and commit quotient matrices one at a time.
    ///
    /// Implementations must validate the complete public plan before invoking
    /// `generate`. Each returned matrix must be consumed before the next call,
    /// while all transformed chunks remain ordered under one shared PCS root.
    /// A generator failure terminates the batch and drops every retained
    /// guarded intermediate.
    fn commit_generated_quotients_secret<GenerationError>(
        &self,
        plans: Vec<SecretQuotientCommitmentPlan<Self::Domain>>,
        generate: impl FnMut(
            usize,
            &SecretQuotientCommitmentPlan<Self::Domain>,
        ) -> Result<
            SecretProverData<RowMajorMatrix<<Self::Domain as PolynomialSpace>::Val>>,
            GenerationError,
        >,
    ) -> SecretGeneratedQuotientCommitmentResult<
        Self::Commitment,
        Self::SecretData,
        GenerationError,
        Self::SecretCommitError,
    >
    where
        GenerationError: Debug;

    /// Generate and commit one quotient chunk at a time.
    ///
    /// Implementations validate every public plan before invoking `generate`.
    /// The callback receives the original instance plan, the chunk index, and
    /// the exact split domain whose evaluations it must return. Its matrix
    /// height must equal the split-domain size and its width must equal the
    /// plan width. Implementations may retain the final guarded chunk owners
    /// for one instance before transforming them, but must not require the
    /// caller to allocate a full quotient-domain matrix.
    fn commit_generated_quotient_chunks_secret<GenerationError>(
        &self,
        plans: Vec<SecretQuotientCommitmentPlan<Self::Domain>>,
        generate: impl FnMut(
            usize,
            &SecretQuotientCommitmentPlan<Self::Domain>,
            usize,
            Self::Domain,
        ) -> Result<
            SecretProverData<RowMajorMatrix<<Self::Domain as PolynomialSpace>::Val>>,
            GenerationError,
        >,
    ) -> SecretGeneratedQuotientCommitmentResult<
        Self::Commitment,
        Self::SecretData,
        GenerationError,
        Self::SecretCommitError,
    >
    where
        GenerationError: Debug;

    /// Construct the optional STARK randomization-polynomial commitment while
    /// retaining generated evaluations and PCS data under erasing ownership.
    ///
    /// Non-hiding implementations return `None`. A hiding implementation must
    /// allocate its random evaluations under a guard before their first write.
    fn get_opt_randomization_poly_commitment_secret(
        &self,
        _domains: impl IntoIterator<Item = Self::Domain>,
    ) -> SecretOptionalCommitmentResult<Self::Commitment, Self::SecretData, Self::SecretCommitError>
    {
        Ok(None)
    }
}

/// Domain evaluation that preserves guarded ownership of witness-derived data.
///
/// This capability is separate from commitment and opening construction so a
/// protocol can require precisely the secret-memory operations it composes.
/// Implementations return an owned guarded matrix: borrowing an ordinary PCS
/// view would let later transformations allocate outside the erasure contract.
pub trait SecretUnivariateStarkPcsEvaluator<Challenge, Challenger>:
    SecretPcsCommitter<Challenge, Challenger> + UnivariateStarkPcs<Challenge, Challenger>
where
    Challenge: ExtensionField<<Self::Domain as PolynomialSpace>::Val>,
    <Self::Domain as PolynomialSpace>::Val: Zeroize,
{
    /// Typed failure from guarded domain evaluation.
    type SecretEvaluationError: Debug;

    /// Evaluate one guarded committed matrix on `domain`.
    fn get_evaluations_on_domain_secret(
        &self,
        prover_data: &SecretProverData<Self::SecretData>,
        idx: usize,
        domain: Self::Domain,
    ) -> SecretDomainEvaluationResult<Self::Domain, Self::SecretEvaluationError>;

    /// Recover the source evaluations consumed by a guarded commitment.
    ///
    /// `committed_domain` is the domain supplied to the commitment and
    /// `source_domain` describes the rows the caller originally provided.
    /// Non-hiding implementations normally commit those rows directly. A
    /// hiding wrapper may instead need to reverse a protocol-defined masking
    /// layout rather than merely evaluate the committed polynomial on the
    /// smaller domain.
    fn get_source_evaluations_secret(
        &self,
        prover_data: &SecretProverData<Self::SecretData>,
        idx: usize,
        source_domain: Self::Domain,
        _committed_domain: Self::Domain,
    ) -> SecretDomainEvaluationResult<Self::Domain, Self::SecretEvaluationError> {
        self.get_evaluations_on_domain_secret(prover_data, idx, source_domain)
    }

    /// Evaluate without removing columns reserved by a hiding PCS.
    ///
    /// Public preprocessing uses this layout. Non-hiding implementations have
    /// no reserved columns and therefore share the ordinary guarded operation.
    fn get_evaluations_on_domain_no_random_secret(
        &self,
        prover_data: &SecretProverData<Self::SecretData>,
        idx: usize,
        domain: Self::Domain,
    ) -> SecretDomainEvaluationResult<Self::Domain, Self::SecretEvaluationError> {
        self.get_evaluations_on_domain_secret(prover_data, idx, domain)
    }
}
