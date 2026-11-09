// Modified by ReallyMe Technologies Limited, 2026.
// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The FRI PCS protocol over two-adic fields.
//!
//! The following implements a slight variant of the usual FRI protocol. As usual we start
//! with a polynomial `F(x)` of degree `n` given as evaluations over the coset `gH` with `|H| = 2^n`.
//!
//! Now consider the polynomial `G(x) = F(gx)`. Note that `G(x)` has the same degree as `F(x)` and
//! the evaluations of `F(x)` over `gH` are identical to the evaluations of `G(x)` over `H`.
//!
//! Hence we can reinterpret our vector of evaluations as evaluations of `G(x)` over `H` and apply
//! the standard FRI protocol to this evaluation vector. This makes it easier to apply FRI to a collection
//! of polynomials defined over different cosets as we don't need to keep track of the coset shifts. We
//! can just assume that every polynomial is defined over the subgroup of the relevant size.
//!
//! If we changed our domain construction (e.g., using multiple cosets), we would need to carefully reconsider these assumptions.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::fmt::Debug;
use core::marker::PhantomData;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
#[cfg(feature = "secret-memory")]
use p3_commit::PolynomialSpace;
use p3_commit::{
    CommitmentOpening, Mmcs, OpenedValues, OpeningRequest, Pcs, PeriodicLdeTable,
    UnivariateStarkPcs,
};
#[cfg(feature = "secret-memory")]
use p3_commit::{
    SecretDetachedPcsCommitter, SecretGeneratedQuotientCommitError,
    SecretGeneratedQuotientCommitmentResult, SecretMmcs, SecretOpeningRequest, SecretPcsCommitter,
    SecretPcsMixedOpener, SecretPcsMixedOpeningRequest, SecretPcsOpener, SecretProverData,
    SecretQuotientCommitmentInput, SecretQuotientCommitmentPlan, SecretRecomputableMmcs,
    SecretStreamingMmcsCommitment, SecretStreamingRecomputableMmcs,
    SecretUnivariateStarkPcsCommitter, SecretUnivariateStarkPcsEvaluator,
};
use p3_dft::TwoAdicSubgroupDft;
#[cfg(feature = "secret-memory")]
use p3_dft::{SecretDftError, SecretTwoAdicSubgroupDft};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    ExtensionField, Field, HornerIter, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
    PrimeField64, TwoAdicField, batch_multiplicative_inverse, batch_multiplicative_inverse_map,
    dot_product,
};
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixCow, RowMajorMatrixView};
use p3_matrix::interpolation::{Interpolate, compute_adjusted_weights};
#[cfg(feature = "secret-memory")]
use p3_matrix::secret::SecretMatrixError;
#[cfg(feature = "secret-memory")]
use p3_matrix::util::reverse_matrix_index_bits;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};
use tracing::{debug_span, instrument};

#[cfg(feature = "secret-memory")]
use crate::SecretFriFoldingStrategy;
#[cfg(feature = "secret-memory")]
use crate::pcs_transcript::SecretPcsProverTranscriptError;
use crate::periodic::build_periodic_lde_table_two_adic;
use crate::verifier::{self, FriError};
use crate::{
    BatchMultiOpening, FriFoldingStrategy, FriParameters, FriProof, FriProverError,
    PcsProverTranscript, PcsShape, PcsVerifierTranscript, prover,
};

/// A polynomial commitment scheme using FRI to generate opening proofs.
///
/// We commit to a polynomial `f` via its evaluation vectors over a coset
/// `gH` where `|H| >= 2 * deg(f)`. A value `f(z)` is opened by using a FRI
/// proof to show that the evaluations of `(f(x) - f(z))/(x - z)` over
/// `gH` are low degree.
#[derive(Clone, Debug)]
pub struct TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub(crate) dft: Dft,
    pub(crate) mmcs: InputMmcs,
    pub(crate) fri: FriParameters<FriMmcs>,
    /// Folding schedule to use instead of the derived one.
    ///
    /// Only a test sets this.
    /// It lets a test forge a schedule the verifier will not derive.
    /// Every commitment and opening then agrees with the forgery.
    #[cfg(test)]
    pub(crate) forged_fold_schedule: Option<Vec<usize>>,
    _phantom: PhantomData<Val>,
}

/// Failure while constructing a guarded two-adic PCS commitment.
#[cfg(feature = "secret-memory")]
#[derive(Debug, thiserror::Error)]
pub enum SecretFriCommitError<MmcsError> {
    /// A shared quotient commitment must contain at least one instance.
    #[error("secret PCS quotient batch is empty")]
    EmptyQuotientBatch,
    /// The eager adapter did not supply the quotient described by its own plan.
    #[error("secret PCS quotient batch input count does not match its plan")]
    QuotientCountMismatch,
    /// An evaluation matrix height disagrees with its declared domain.
    #[error("secret PCS matrix height does not match its domain")]
    DomainHeightMismatch,
    /// The vector retaining guarded LDE matrices could not be reserved.
    #[error("secret PCS LDE owner allocation failed")]
    AllocationFailed,
    /// The quotient chunk count is zero, not a power of two, or larger than the domain.
    #[error("secret PCS quotient chunk count is invalid")]
    InvalidQuotientChunkCount,
    /// A guarded quotient matrix has invalid or overflowing dimensions.
    #[error("secret PCS quotient matrix is invalid")]
    Matrix(#[from] SecretMatrixError),
    /// The guarded DFT rejected the input or could not allocate its output.
    #[error("secret PCS DFT failed")]
    Dft(SecretDftError),
    /// The guarded MMCS rejected commitment construction.
    #[error("secret PCS MMCS commitment failed")]
    Mmcs(MmcsError),
    /// Retained compact data was produced for a different PCS blowup.
    #[error("secret PCS retained-data blowup does not match the PCS configuration")]
    RetainedBlowupMismatch,
    /// Retained compact data contains no coefficient matrices.
    #[error("secret PCS retained coefficient set is empty")]
    EmptyRetainedCoefficients,
    /// The requested retained coefficient matrix is absent.
    #[error("secret PCS retained matrix index is out of bounds")]
    MatrixIndexOutOfBounds { index: usize, matrix_count: usize },
    /// Committed matrices were already replaced by authentication-only state.
    #[error("secret PCS commitment data is already detached")]
    CommitmentAlreadyDetached,
}

/// Failure while evaluating guarded PCS data on another two-adic domain.
#[cfg(feature = "secret-memory")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SecretFriEvaluationError {
    /// Retained compact data was produced for a different PCS blowup.
    #[error("secret PCS retained-data blowup does not match the PCS configuration")]
    RetainedBlowupMismatch,
    /// The requested matrix is absent from the guarded commitment.
    #[error("secret PCS matrix index is out of bounds")]
    MatrixIndexOutOfBounds { index: usize, matrix_count: usize },
    /// The retained coefficient matrix has an invalid shape.
    #[error("secret PCS retained coefficient matrix is invalid")]
    Matrix(#[from] SecretMatrixError),
    /// The guarded DFT rejected the requested evaluation.
    #[error("secret PCS domain evaluation DFT failed")]
    Dft(SecretDftError),
}

/// Failure while folding a secret-derived FRI codeword.
#[cfg(feature = "secret-memory")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SecretFriFoldError {
    /// FRI folds must reduce by an arity of at least two.
    #[error("secret FRI fold arity is zero")]
    ZeroLogArity,
    /// The requested fold arity cannot be represented by `usize`.
    #[error("secret FRI fold arity overflows usize")]
    ArityOverflow,
    /// The input is not a nonempty row-major matrix of the requested arity.
    #[error("secret FRI fold matrix shape is invalid")]
    InvalidMatrixShape,
    /// The codeword domain exceeds the base field's two-adicity.
    #[error("secret FRI fold domain exceeds field two-adicity")]
    UnsupportedDomain,
    /// A guarded fold buffer could not be allocated.
    #[error("secret FRI fold allocation failed")]
    AllocationFailed,
}

/// Failure while constructing a complete guarded two-adic PCS opening proof.
#[cfg(feature = "secret-memory")]
#[derive(Debug, thiserror::Error)]
pub(crate) enum SecretFriOpenError<InputMmcsError, FriMmcsError> {
    /// Opening geometry or quotient reduction failed.
    #[error("guarded PCS opening preparation failed")]
    Preparation(FriProverError),
    /// A guarded reduced-opening allocation or shape operation failed.
    #[error("guarded PCS opening matrix operation failed")]
    Matrix(SecretMatrixError),
    /// Public proof metadata could not be reserved.
    #[error("guarded PCS opening allocation failed")]
    AllocationFailed,
    /// Full input matrices could not transition to authentication-only state.
    #[error("guarded PCS input detachment failed")]
    InputDetachment(SecretFriCommitError<InputMmcsError>),
    /// A detached input matrix could not be reconstructed for quotient reduction.
    #[error("guarded PCS input reconstruction failed")]
    InputReconstruction(SecretFriCommitError<InputMmcsError>),
    /// The guarded FRI protocol failed.
    #[error("guarded PCS FRI proof construction failed")]
    Fri(
        prover::SecretFriDriverError<
            SecretFriCommitError<InputMmcsError>,
            FriMmcsError,
            SecretFriFoldError,
        >,
    ),
    /// The outer PCS transcript did not close cleanly.
    #[error("guarded PCS transcript finalization failed")]
    Transcript(SecretPcsProverTranscriptError),
}

/// Stable public failure categories for guarded two-adic PCS opening.
///
/// The internal driver retains more detailed typed errors, but those types are
/// implementation details spanning the FRI transcript, folding strategy, and
/// MMCS. Keeping this boundary categorical prevents downstream protocols from
/// coupling their error ABI to those internals.
#[cfg(feature = "secret-memory")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SecretTwoAdicPcsOpenError {
    /// Opening geometry or quotient reduction was rejected.
    #[error("guarded PCS opening preparation failed")]
    Preparation,
    /// A guarded opening matrix operation failed.
    #[error("guarded PCS opening matrix operation failed")]
    Matrix,
    /// Public proof metadata could not be reserved.
    #[error("guarded PCS opening allocation failed")]
    AllocationFailed,
    /// Full input matrices could not transition to authentication-only state.
    #[error("guarded PCS input detachment failed")]
    InputDetachment,
    /// A detached input matrix could not be reconstructed for quotient reduction.
    #[error("guarded PCS input reconstruction failed")]
    InputReconstruction,
    /// Guarded FRI proof construction failed.
    #[error("guarded PCS FRI proof construction failed")]
    Fri,
    /// The guarded PCS transcript did not close cleanly.
    #[error("guarded PCS transcript finalization failed")]
    Transcript,
}

#[cfg(feature = "secret-memory")]
impl<InputMmcsError, FriMmcsError> From<SecretFriOpenError<InputMmcsError, FriMmcsError>>
    for SecretTwoAdicPcsOpenError
{
    fn from(error: SecretFriOpenError<InputMmcsError, FriMmcsError>) -> Self {
        match error {
            SecretFriOpenError::Preparation(_) => Self::Preparation,
            SecretFriOpenError::Matrix(_) => Self::Matrix,
            SecretFriOpenError::AllocationFailed => Self::AllocationFailed,
            SecretFriOpenError::InputDetachment(_) => Self::InputDetachment,
            SecretFriOpenError::InputReconstruction(_) => Self::InputReconstruction,
            SecretFriOpenError::Fri(_) => Self::Fri,
            SecretFriOpenError::Transcript(_) => Self::Transcript,
        }
    }
}

#[cfg(feature = "secret-memory")]
impl<InputMmcsError, FriMmcsError> From<FriProverError>
    for SecretFriOpenError<InputMmcsError, FriMmcsError>
{
    fn from(error: FriProverError) -> Self {
        Self::Preparation(error)
    }
}

#[cfg(feature = "secret-memory")]
impl<InputMmcsError, FriMmcsError> From<SecretMatrixError>
    for SecretFriOpenError<InputMmcsError, FriMmcsError>
{
    fn from(error: SecretMatrixError) -> Self {
        Self::Matrix(error)
    }
}

#[cfg(feature = "secret-memory")]
impl<InputMmcsError, FriMmcsError> From<SecretFriCommitError<InputMmcsError>>
    for SecretFriOpenError<InputMmcsError, FriMmcsError>
{
    fn from(error: SecretFriCommitError<InputMmcsError>) -> Self {
        Self::InputReconstruction(error)
    }
}

/// A guarded batch of secret-derived row-major matrices.
///
/// The nested owners let callers erase each matrix as soon as its final
/// consumer finishes without extracting any backing allocation.
#[cfg(feature = "secret-memory")]
pub type SecretMatrixBatch<F> = SecretProverData<Vec<SecretProverData<RowMajorMatrix<F>>>>;

#[cfg(feature = "secret-memory")]
type SecretTwoAdicFriProverData<Val, InputMmcs> = SecretProverData<
    SecretTwoAdicFriData<
        Val,
        <InputMmcs as SecretMmcs<Val>>::SecretData<RowMajorMatrix<Val>>,
        <InputMmcs as SecretRecomputableMmcs<Val>>::AuthenticationData<RowMajorMatrix<Val>>,
    >,
>;

/// Guarded two-adic PCS data retaining both the live commitment and the
/// compact coefficients needed to reconstruct queried LDE rows.
///
/// This type deliberately has no `Clone`, `Debug`, or serialization support.
/// The coefficient matrices are witness-derived and remain behind nested
/// zeroizing owners until opening construction is complete.
#[cfg(feature = "secret-memory")]
pub struct SecretTwoAdicFriData<F, D, A>
where
    F: zeroize::Zeroize,
    D: zeroize::Zeroize,
    A: zeroize::Zeroize,
{
    pub(crate) commitment_data: SecretProverData<D>,
    pub(crate) authentication_data: Option<A>,
    coefficients: SecretMatrixBatch<F>,
    log_blowup: usize,
}

#[cfg(feature = "secret-memory")]
impl<F, D, A> zeroize::Zeroize for SecretTwoAdicFriData<F, D, A>
where
    F: zeroize::Zeroize,
    D: zeroize::Zeroize,
    A: zeroize::Zeroize,
{
    fn zeroize(&mut self) {
        self.commitment_data.zeroize();
        if let Some(authentication_data) = self.authentication_data.as_mut() {
            authentication_data.zeroize();
        }
        self.coefficients.zeroize();
        self.log_blowup.zeroize();
    }
}

#[cfg(feature = "secret-memory")]
impl<F, D, A> SecretTwoAdicFriData<F, D, A>
where
    F: TwoAdicField + zeroize::Zeroize,
    D: zeroize::Zeroize,
    A: zeroize::Zeroize,
{
    pub(crate) fn new(
        commitment_data: SecretProverData<D>,
        coefficients: Vec<SecretProverData<RowMajorMatrix<F>>>,
        log_blowup: usize,
    ) -> Self {
        Self {
            commitment_data,
            authentication_data: None,
            coefficients: SecretProverData::new(coefficients),
            log_blowup,
        }
    }

    pub(crate) fn new_detached(
        commitment_data: SecretProverData<D>,
        authentication_data: A,
        coefficients: Vec<SecretProverData<RowMajorMatrix<F>>>,
        log_blowup: usize,
    ) -> Self {
        Self {
            commitment_data,
            authentication_data: Some(authentication_data),
            coefficients: SecretProverData::new(coefficients),
            log_blowup,
        }
    }

    fn max_lde_height(&self) -> Result<usize, SecretMatrixError> {
        let shift =
            u32::try_from(self.log_blowup).map_err(|_| SecretMatrixError::DimensionOverflow)?;
        let mut maximum = None;
        for coefficients in self.coefficients.iter() {
            let width = coefficients.width;
            if width == 0 || !coefficients.values.len().is_multiple_of(width) {
                return Err(SecretMatrixError::InvalidShape);
            }
            let coefficient_height = coefficients.values.len() / width;
            if coefficient_height == 0 || !coefficient_height.is_power_of_two() {
                return Err(SecretMatrixError::InvalidShape);
            }
            let height = coefficient_height
                .checked_shl(shift)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            maximum = Some(maximum.map_or(height, |current: usize| current.max(height)));
        }
        maximum.ok_or(SecretMatrixError::InvalidShape)
    }

    /// Return committed LDE dimensions from compact retained coefficients.
    ///
    /// These dimensions remain authoritative after Merkle value leaves have
    /// been erased. Opening preflight must therefore use this source rather
    /// than treating live MMCS matrices as persistent shape metadata.
    pub(crate) fn retained_lde_dimensions(&self) -> Result<Vec<Dimensions>, SecretMatrixError> {
        let shift =
            u32::try_from(self.log_blowup).map_err(|_| SecretMatrixError::DimensionOverflow)?;
        let mut dimensions = Vec::new();
        dimensions
            .try_reserve_exact(self.coefficients.len())
            .map_err(|_| SecretMatrixError::AllocationFailed)?;
        for coefficients in self.coefficients.iter() {
            let width = coefficients.width;
            if width == 0 || !coefficients.values.len().is_multiple_of(width) {
                return Err(SecretMatrixError::InvalidShape);
            }
            let coefficient_height = coefficients.values.len() / width;
            if coefficient_height == 0 || !coefficient_height.is_power_of_two() {
                return Err(SecretMatrixError::InvalidShape);
            }
            let height = coefficient_height
                .checked_shl(shift)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            dimensions.push(Dimensions { width, height });
        }
        if dimensions.is_empty() {
            return Err(SecretMatrixError::InvalidShape);
        }
        Ok(dimensions)
    }
}

impl<Val, Dft, InputMmcs, FriMmcs> TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub const fn new(dft: Dft, mmcs: InputMmcs, fri: FriParameters<FriMmcs>) -> Self {
        Self {
            dft,
            mmcs,
            fri,
            #[cfg(test)]
            forged_fold_schedule: None,
            _phantom: PhantomData,
        }
    }
}

impl<Val, Dft, InputMmcs, FriMmcs> TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64,
    Dft: TwoAdicSubgroupDft<Val> + Sync,
{
    /// Compute the bit-reversed LDE of every matrix, running the per-matrix LDEs concurrently.
    ///
    /// Matrices are scheduled largest first so the small ones fill idle cores, and the results
    /// are returned in the order of `evaluations`.
    ///
    /// The LDEs run as separate tasks when the cost model splits the loop, and in one task otherwise.
    fn coset_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (TwoAdicMultiplicativeCoset<Val>, RowMajorMatrix<Val>)>,
    ) -> Vec<RowMajorMatrix<Val>> {
        let mut jobs: Vec<_> = evaluations
            .into_iter()
            .enumerate()
            .map(|(idx, (domain, evals))| (idx, domain, evals))
            .collect();
        jobs.sort_by_key(|(_, _, evals)| Reverse(evals.values.len()));

        // An item reads its input and writes the blown-up output.
        let total_values: usize = jobs.iter().map(|(_, _, evals)| evals.values.len()).sum();
        let item_bytes = (total_values + (total_values << self.fri.log_blowup))
            .div_ceil(jobs.len().max(1))
            * size_of::<Val>();

        let dft = &self.dft;
        let log_blowup = self.fri.log_blowup;
        let mut ldes: Vec<_> = jobs
            .into_par_iter()
            .with_min_task_bytes(item_bytes)
            .map(|(idx, domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                // coset_lde_batch converts from evaluations over `xH` to evaluations over `shift * x * K`.
                // Hence, letting `shift = g/x` the output will be evaluations over `gK` as desired.
                // When `x = g`, we could just use the standard LDE but currently this doesn't seem
                // to give a meaningful performance boost.
                let shift = Val::GENERATOR / domain.shift();
                // Compute the LDE with blowup factor fri.log_blowup.
                // We bit reverse as this is required by our implementation of the FRI protocol.
                let lde = dft
                    .coset_lde_batch(evals, log_blowup, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix();
                (idx, lde)
            })
            .collect();
        ldes.sort_unstable_by_key(|(idx, _)| *idx);
        ldes.into_iter().map(|(_, lde)| lde).collect()
    }
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs> TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: Mmcs<Val> + SecretRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
{
    /// Rebuild one complete committed LDE under a zeroizing owner.
    ///
    /// This capability is intentionally matrix-scoped. Opening code can
    /// reconstruct, reduce, and erase each matrix in turn instead of reviving
    /// an entire commitment batch and recreating the memory peak that early
    /// detachment is meant to remove.
    pub fn recompute_lde_matrix_secret(
        &self,
        prover_data: &SecretTwoAdicFriProverData<Val, InputMmcs>,
        index: usize,
    ) -> Result<SecretProverData<RowMajorMatrix<Val>>, SecretFriCommitError<InputMmcs::SecretError>>
    {
        if prover_data.log_blowup != self.fri.log_blowup {
            return Err(SecretFriCommitError::RetainedBlowupMismatch);
        }
        let matrix_count = prover_data.coefficients.len();
        let coefficients = prover_data.coefficients.get(index).ok_or(
            SecretFriCommitError::MatrixIndexOutOfBounds {
                index,
                matrix_count,
            },
        )?;
        self.dft
            .recompute_coset_lde_secret(coefficients, self.fri.log_blowup)
            .map_err(SecretFriCommitError::Dft)
    }

    /// Recompute selected bit-reversed LDE rows from compact guarded
    /// coefficients retained by [`SecretPcsCommitter::commit_secret`].
    ///
    /// `indices` use MMCS batch semantics: they address the tallest matrix and
    /// are shifted for shorter matrices. Rows are returned matrix-major; each
    /// matrix follows `indices` order.
    /// The result stays guarded until the caller intentionally transfers the
    /// selected rows into an MMCS opening proof.
    pub fn recompute_lde_rows_secret(
        &self,
        prover_data: &SecretTwoAdicFriProverData<Val, InputMmcs>,
        indices: &[usize],
    ) -> Result<SecretMatrixBatch<Val>, SecretFriCommitError<InputMmcs::SecretError>> {
        if prover_data.log_blowup != self.fri.log_blowup {
            return Err(SecretFriCommitError::RetainedBlowupMismatch);
        }
        if prover_data.coefficients.is_empty() {
            return Err(SecretFriCommitError::EmptyRetainedCoefficients);
        }

        let blowup_shift =
            u32::try_from(self.fri.log_blowup).map_err(|_| SecretMatrixError::DimensionOverflow)?;
        let mut lde_heights = Vec::new();
        lde_heights
            .try_reserve_exact(prover_data.coefficients.len())
            .map_err(|_| SecretFriCommitError::AllocationFailed)?;
        for coefficients in prover_data.coefficients.iter() {
            let lde_height = coefficients
                .height()
                .checked_shl(blowup_shift)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            lde_heights.push(lde_height);
        }
        let max_lde_height = lde_heights
            .iter()
            .copied()
            .max()
            .ok_or(SecretFriCommitError::EmptyRetainedCoefficients)?;
        let log_max_lde_height = log2_strict_usize(max_lde_height);

        let mut rows = Vec::new();
        rows.try_reserve_exact(prover_data.coefficients.len())
            .map_err(|_| SecretFriCommitError::AllocationFailed)?;
        for (coefficients, lde_height) in prover_data.coefficients.iter().zip(lde_heights) {
            let bits_reduced = log_max_lde_height
                .checked_sub(log2_strict_usize(lde_height))
                .ok_or(SecretMatrixError::InvalidShape)?;
            let mut matrix_indices = Vec::new();
            matrix_indices
                .try_reserve_exact(indices.len())
                .map_err(|_| SecretFriCommitError::AllocationFailed)?;
            for &index in indices {
                if index >= max_lde_height {
                    return Err(SecretFriCommitError::Dft(
                        SecretDftError::QueryIndexOutOfBounds {
                            index,
                            height: max_lde_height,
                        },
                    ));
                }
                matrix_indices.push(index >> bits_reduced);
            }
            rows.push(
                self.dft
                    .recompute_coset_lde_rows_secret(
                        coefficients,
                        self.fri.log_blowup,
                        &matrix_indices,
                    )
                    .map_err(SecretFriCommitError::Dft)?,
            );
        }
        Ok(SecretProverData::new(rows))
    }

    fn evaluate_on_domain_secret(
        &self,
        prover_data: &SecretTwoAdicFriProverData<Val, InputMmcs>,
        idx: usize,
        domain: TwoAdicMultiplicativeCoset<Val>,
    ) -> Result<SecretProverData<RowMajorMatrix<Val>>, SecretFriEvaluationError> {
        if prover_data.log_blowup != self.fri.log_blowup {
            return Err(SecretFriEvaluationError::RetainedBlowupMismatch);
        }
        let matrix_count = prover_data.coefficients.len();
        let coefficients = prover_data.coefficients.get(idx).ok_or(
            SecretFriEvaluationError::MatrixIndexOutOfBounds {
                index: idx,
                matrix_count,
            },
        )?;
        let width = coefficients.width;
        if width == 0 || !coefficients.values.len().is_multiple_of(width) {
            return Err(SecretMatrixError::InvalidShape.into());
        }
        let coefficient_height = coefficients.values.len() / width;
        if coefficient_height == 0 || !coefficient_height.is_power_of_two() {
            return Err(SecretMatrixError::InvalidShape.into());
        }

        let target_height = domain.size();
        if target_height == 0 || !target_height.is_power_of_two() {
            return Err(SecretMatrixError::InvalidShape.into());
        }
        let output_len = target_height
            .checked_mul(width)
            .ok_or(SecretMatrixError::DimensionOverflow)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(output_len)
            .map_err(|_| SecretMatrixError::AllocationFailed)?;
        values.resize(output_len, Val::ZERO);
        let mut evaluations = SecretProverData::new(RowMajorMatrix::new(values, width));

        let blowup_shift =
            u32::try_from(self.fri.log_blowup).map_err(|_| SecretMatrixError::DimensionOverflow)?;
        let committed_lde_height = coefficient_height
            .checked_shl(blowup_shift)
            .ok_or(SecretMatrixError::DimensionOverflow)?;
        if domain.shift() == Val::GENERATOR && committed_lde_height >= target_height {
            // The ordinary PCS takes a bit-reversed prefix in this case. In
            // coefficient form that is evaluation on a smaller subgroup, so
            // degrees congruent modulo the target height must be added rather
            // than discarded.
            for row in 0..coefficient_height {
                let source_start = row
                    .checked_mul(width)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let source_end = source_start
                    .checked_add(width)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let destination_row = row % target_height;
                let destination_start = destination_row
                    .checked_mul(width)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let destination_end = destination_start
                    .checked_add(width)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let destination = evaluations
                    .values
                    .get_mut(destination_start..destination_end)
                    .ok_or(SecretMatrixError::InvalidShape)?;
                let source = coefficients
                    .values
                    .get(source_start..source_end)
                    .ok_or(SecretMatrixError::InvalidShape)?;
                for (output, coefficient) in destination.iter_mut().zip(source) {
                    *output += *coefficient;
                }
            }
        } else {
            // Retained coefficients describe f(GENERATOR * X). Multiplying
            // degree j by (domain.shift / GENERATOR)^j converts them to
            // f(domain.shift * X) before the subgroup DFT. When the target
            // subgroup is smaller than the retained coefficient domain,
            // degrees congruent modulo the target height evaluate identically
            // and therefore must be added. Truncation is not valid here: it
            // silently drops the hiding polynomial's upper coefficients on
            // every shifted quotient chunk except the generator coset.
            let relative_shift = domain.shift() / Val::GENERATOR;
            let mut row_scale = Val::ONE;
            for row in 0..coefficient_height {
                let source_start = row
                    .checked_mul(width)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let source_end = source_start
                    .checked_add(width)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let destination_row = row % target_height;
                let destination_start = destination_row
                    .checked_mul(width)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let destination_end = destination_start
                    .checked_add(width)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let destination = evaluations
                    .values
                    .get_mut(destination_start..destination_end)
                    .ok_or(SecretMatrixError::InvalidShape)?;
                let source = coefficients
                    .values
                    .get(source_start..source_end)
                    .ok_or(SecretMatrixError::InvalidShape)?;
                for (output, coefficient) in destination.iter_mut().zip(source) {
                    *output += *coefficient * row_scale;
                }
                row_scale *= relative_shift;
            }
        }
        self.dft
            .dft_batch_secret(&mut evaluations)
            .map_err(SecretFriEvaluationError::Dft)?;
        Ok(evaluations)
    }

    /// Evaluate retained coefficients at one disclosed opening point without
    /// reconstructing the committed LDE.
    ///
    /// The retained matrix contains coefficients of `f(GENERATOR * X)` in
    /// natural degree order. Evaluating it at `point / GENERATOR` therefore
    /// yields `f(point)`. Horner accumulators remain behind a zeroizing owner;
    /// only the completed PCS claims are copied into ordinary proof metadata.
    fn evaluate_coefficients_at_point_secret<Challenge>(
        coefficients: &SecretProverData<RowMajorMatrix<Val>>,
        point: Challenge,
    ) -> Result<Vec<Challenge>, SecretMatrixError>
    where
        Challenge: ExtensionField<Val> + zeroize::Zeroize,
    {
        let width = coefficients.width;
        if width == 0 || !coefficients.values.len().is_multiple_of(width) || point.is_zero() {
            return Err(SecretMatrixError::InvalidShape);
        }
        let height = coefficients.values.len() / width;
        if height == 0 || !height.is_power_of_two() {
            return Err(SecretMatrixError::InvalidShape);
        }

        let mut accumulator_values = Vec::new();
        accumulator_values
            .try_reserve_exact(width)
            .map_err(|_| SecretMatrixError::AllocationFailed)?;
        accumulator_values.resize(width, Challenge::ZERO);
        let mut accumulators = SecretProverData::new(accumulator_values);
        let evaluation_point = point * Val::GENERATOR.inverse();

        for row_index in (0..height).rev() {
            let row = coefficients
                .row_slice(row_index)
                .ok_or(SecretMatrixError::InvalidShape)?;
            for (accumulator, coefficient) in accumulators.iter_mut().zip(row.iter()) {
                *accumulator = *accumulator * evaluation_point + *coefficient;
            }
        }

        let mut disclosed = Vec::new();
        disclosed
            .try_reserve_exact(width)
            .map_err(|_| SecretMatrixError::AllocationFailed)?;
        disclosed.extend_from_slice(&accumulators);
        zeroize::Zeroize::zeroize(&mut accumulators);
        Ok(disclosed)
    }
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs> TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: Mmcs<Val> + SecretRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
{
    /// Erase full committed matrices while retaining Merkle authentication
    /// state and compact coefficients for the eventual queried-row opening.
    pub(crate) fn detach_commitment_data_secret(
        &self,
        prover_data: &mut SecretTwoAdicFriProverData<Val, InputMmcs>,
    ) -> Result<(), SecretFriCommitError<InputMmcs::SecretError>> {
        if prover_data.authentication_data.is_some() {
            return Err(SecretFriCommitError::CommitmentAlreadyDetached);
        }
        let authentication_data = self
            .mmcs
            .detach_authentication_data(&mut prover_data.commitment_data)
            .map_err(SecretFriCommitError::Mmcs)?;
        prover_data.authentication_data = Some(authentication_data);
        Ok(())
    }

    /// Recompute selected LDE rows, erase the full committed matrices, and
    /// authenticate the rows against the original commitment.
    ///
    /// Reconstruction deliberately precedes the irreversible detachment so an
    /// invalid query cannot destroy otherwise usable prover data. The temporary
    /// overlap is bounded by `query_count * total_committed_width`, not by an
    /// additional full LDE. On success the retained MMCS data contains no value
    /// leaves and cannot service an ordinary opening.
    pub fn detach_and_open_multi_batch_secret(
        &self,
        indices: &[usize],
        prover_data: &mut SecretTwoAdicFriProverData<Val, InputMmcs>,
    ) -> Result<BatchMultiOpening<Val, InputMmcs>, SecretFriCommitError<InputMmcs::SecretError>>
    {
        let recomputed = self.recompute_lde_rows_secret(prover_data, indices)?;
        let mut query_rows = Vec::new();
        query_rows
            .try_reserve_exact(indices.len())
            .map_err(|_| SecretFriCommitError::AllocationFailed)?;
        for query_index in 0..indices.len() {
            let mut rows_at_index = Vec::new();
            rows_at_index
                .try_reserve_exact(recomputed.len())
                .map_err(|_| SecretFriCommitError::AllocationFailed)?;
            for matrix in recomputed.iter() {
                let row = matrix
                    .row_slice(query_index)
                    .ok_or(SecretMatrixError::InvalidShape)?;
                let mut owned_row = Vec::new();
                owned_row
                    .try_reserve_exact(row.len())
                    .map_err(|_| SecretFriCommitError::AllocationFailed)?;
                owned_row.extend_from_slice(&row);
                rows_at_index.push(owned_row);
            }
            query_rows.push(rows_at_index);
        }

        if prover_data.authentication_data.is_none() {
            self.detach_commitment_data_secret(prover_data)?;
        }
        let authentication_data = prover_data
            .authentication_data
            .as_ref()
            .ok_or(SecretFriCommitError::CommitmentAlreadyDetached)?;
        let (opened_values, opening_proof) = self
            .mmcs
            .open_multi_batch_from_recomputed_rows::<RowMajorMatrix<Val>>(
                indices,
                &query_rows,
                authentication_data,
            )
            .map_err(SecretFriCommitError::Mmcs)?;
        Ok(BatchMultiOpening {
            opened_values,
            opening_proof,
        })
    }
}

/// The Prover Data associated to a commitment to a collection of matrices
/// and a list of points to open each matrix at.
pub type ProverDataWithOpeningPoints<'a, EF, ProverData> = OpeningRequest<'a, ProverData, EF>;

#[cfg(feature = "secret-memory")]
type SecretTwoAdicMixedOpeningRequests<
    'a,
    Val,
    PublicData,
    SecretInputData,
    SecretAuthenticationData,
    Challenge,
> = Vec<
    SecretPcsMixedOpeningRequest<
        'a,
        PublicData,
        SecretTwoAdicFriData<Val, SecretInputData, SecretAuthenticationData>,
        Challenge,
    >,
>;

#[cfg(feature = "secret-memory")]
type SecretTwoAdicPcsMixedOpeningRequests<'a, Val, InputMmcs, Challenge> =
    SecretTwoAdicMixedOpeningRequests<
        'a,
        Val,
        <InputMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>,
        <InputMmcs as SecretMmcs<Val>>::SecretData<RowMajorMatrix<Val>>,
        <InputMmcs as SecretRecomputableMmcs<Val>>::AuthenticationData<RowMajorMatrix<Val>>,
        Challenge,
    >;

#[cfg(feature = "secret-memory")]
type SecretTwoAdicOpeningResult<Challenge, FriMmcs, Val, InputMmcs, InputMmcsError, FriMmcsError> =
    Result<
        (
            OpenedValues<Challenge>,
            FriProof<Challenge, FriMmcs, Val, Vec<BatchMultiOpening<Val, InputMmcs>>>,
        ),
        SecretFriOpenError<InputMmcsError, FriMmcsError>,
    >;

// Re-exported so `p3_fri::CommitmentWithOpeningPoints` keeps naming the shape this PCS
// verifies against; it is defined in `p3-commit` alongside `Pcs` so crates that build an
// opening argument without depending on FRI can name it too.
pub use p3_commit::CommitmentWithOpeningPoints;

pub struct TwoAdicFriFolding<InputProof, InputError>(pub PhantomData<(InputProof, InputError)>);

pub type TwoAdicFriFoldingForMmcs<F, M> =
    TwoAdicFriFolding<Vec<BatchMultiOpening<F, M>>, <M as Mmcs<F>>::Error>;

impl<F: TwoAdicField, InputProof: Sync, InputError: Debug + Sync, EF: ExtensionField<F>>
    FriFoldingStrategy<F, EF> for TwoAdicFriFolding<InputProof, InputError>
{
    type InputProof = InputProof;
    type InputError = InputError;

    fn extra_query_index_bits(&self) -> usize {
        0
    }

    fn fold_row(
        &self,
        index: usize,
        log_height: usize,
        log_arity: usize,
        beta: EF,
        mut evals: impl Iterator<Item = EF>,
    ) -> EF {
        let arity = 1 << log_arity;

        // Compute this row's subgroup generator and bit-reversed exponent.
        let subgroup_generator = F::two_adic_generator(log_height + log_arity);
        let subgroup_power = reverse_bits_len(index, log_height) as u64;

        if log_arity == 1 {
            let subgroup_start = subgroup_generator.exp_u64(subgroup_power);
            // The two interpolation points are `{s, -s}` for `s = subgroup_start`, so
            //     f(beta) = (y_0 + y_1) / 2 + (y_0 - y_1) * beta / (2 s).
            // This is the closed form used by the arity-2 kernel of `fold_matrix`.
            let (lo, hi) = evals.next_tuple().expect("Expected 2 evaluations");
            assert!(evals.next().is_none(), "Expected 2 evaluations");

            let halve_inv_power = subgroup_start.double().inverse();
            return (lo + hi).halve() + (lo - hi) * beta * halve_inv_power;
        }

        if matches!(log_arity, 2 | 3) {
            // A subgroup order wider than u64 needs the original inversion path.
            let subgroup_start_inv = 1u64
                .checked_shl((log_height + log_arity) as u32)
                .map_or_else(
                    || subgroup_generator.exp_u64(subgroup_power).inverse(),
                    |order| subgroup_generator.exp_u64(order - subgroup_power),
                );

            // The inverse FFT returns the coefficients of arity * p(s X), where s is the subgroup start.
            // Horner evaluation at beta / s followed by division by arity therefore yields p(beta), with no allocation or extension-field inversion.
            if log_arity == 2 {
                let row = core::array::from_fn(|_| evals.next().expect("Expected 4 evaluations"));
                assert!(evals.next().is_none(), "Expected 4 evaluations");
                let [c0, c1, c2, c3] = ifft4(row, -F::two_adic_generator(2));
                let x = beta * subgroup_start_inv;
                return [c0, c1, c2].into_iter().horner_acc(c3, x) * F::ONE.halve().halve();
            }

            let row = core::array::from_fn(|_| evals.next().expect("Expected 8 evaluations"));
            assert!(evals.next().is_none(), "Expected 8 evaluations");
            let [c0, c1, c2, c3, c4, c5, c6, c7] = ifft8::<F, EF>(row);
            let x = beta * subgroup_start_inv;
            return [c0, c1, c2, c3, c4, c5, c6].into_iter().horner_acc(c7, x)
                * F::ONE.halve().halve().halve();
        }

        let evals: Vec<_> = evals.collect();
        assert_eq!(evals.len(), arity, "Expected {} evaluations", arity);

        let subgroup_start = subgroup_generator.exp_u64(subgroup_power);
        let mut xs: Vec<F> = F::two_adic_generator(log_arity)
            .shifted_powers(subgroup_start)
            .take(arity)
            .collect();
        reverse_slice_index_bits(&mut xs);

        // Lagrange interpolation at beta
        lagrange_interpolate_at(&xs, &evals, beta)
    }

    #[instrument(skip_all, level = "debug")]
    fn fold_matrix<M: Matrix<EF>>(&self, beta: EF, log_arity: usize, m: M) -> Vec<EF> {
        if use_packed_fold(F::Packing::WIDTH, m.height(), F::bits()) {
            return fold_matrix_packed(beta, log_arity, &m);
        }

        if log_arity == 1 {
            // Optimized path for arity 2
            // We use the fact that
            //     p_e(x^2) = (p(x) + p(-x)) / 2
            //     p_o(x^2) = (p(x) - p(-x)) / (2 x)
            // that is,
            //     p_e(g^(2i)) = (p(g^i) + p(g^(n/2 + i))) / 2
            //     p_o(g^(2i)) = (p(g^i) - p(g^(n/2 + i))) / (2 g^i)
            // so
            //     result(g^(2i)) = p_e(g^(2i)) + beta p_o(g^(2i))
            //
            // As p_e, p_o will be in the extension field we want to find ways to avoid extension multiplications.
            // We should only need a single one (namely multiplication by beta).
            let g_inv = F::two_adic_generator(log2_strict_usize(m.height()) + 1).inverse();

            // As beta is in the extension field, we want to avoid multiplying by it
            // for as long as possible. Here we precompute the powers  `g_inv^i / 2` in the base field.
            let mut halve_inv_powers = g_inv.shifted_powers(F::ONE.halve()).collect_n(m.height());
            reverse_slice_index_bits(&mut halve_inv_powers);

            m.par_rows()
                .zip(halve_inv_powers)
                .map(|(mut row, halve_inv_power)| {
                    let (lo, hi) = row.next_tuple().unwrap();
                    (lo + hi).halve() + (lo - hi) * beta * halve_inv_power
                })
                .collect()
        } else {
            // Decompose arity-2^k fold into k sequential arity-2 folds.
            // This way, an arity-2^k fold with a single challenge beta is equivalent to
            // k arity-2 folds with challenges beta, beta^2, beta^4, ..., beta^{2^{k-1}}.
            //
            // For arity 4 with evaluation points {s, -s, si, -si}:
            //   Step 1 (beta):   fold pairs → g(s^2), g(-s^2) where g = f_e + beta*f_o
            //   Step 2 (beta^2): fold pair  → g(beta^2) = f(beta)

            // Every row splits into whole `(lo, hi)` pairs.
            debug_assert!(m.width().is_multiple_of(2));
            let pairs_per_row = m.width() / 2;
            let initial_height = m.height() * pairs_per_row;
            let g_inv = F::two_adic_generator(log2_strict_usize(initial_height) + 1).inverse();
            let mut halve_inv_powers = g_inv
                .shifted_powers(F::ONE.halve())
                .collect_n(initial_height);
            reverse_slice_index_bits(&mut halve_inv_powers);

            // The first fold reads the borrowed matrix row by row, so `m` is never copied.
            let mut data = EF::zero_vec(initial_height);
            data.par_chunks_exact_mut(pairs_per_row)
                .zip(m.par_rows())
                .zip(halve_inv_powers.par_chunks_exact(pairs_per_row))
                .for_each(|((out, row), inv_powers)| {
                    for (o, (lo, hi), &halve_inv_power) in
                        izip!(out.iter_mut(), row.tuples::<(EF, EF)>(), inv_powers)
                    {
                        *o = (lo + hi).halve() + (lo - hi) * beta * halve_inv_power;
                    }
                });

            let mut current_beta = beta.square();
            let mut next_data = EF::zero_vec(initial_height / 2);

            for _ in 1..log_arity {
                let height = data.len() / 2;
                // For a bit-reversed table of `g_inv^i / 2` over `n` elements, the first `n/2`
                // entries already equal the bit-reversed table of `(g_inv^2)^i / 2` over `n/2`
                // elements, i.e. the table this round's halving factors need. So each round
                // reuses a shrinking prefix of the same table.
                next_data[..height]
                    .par_iter_mut()
                    .zip(data.par_chunks_exact(2))
                    .zip(&halve_inv_powers[..height])
                    .for_each(|((out, chunk), &halve_inv_power)| {
                        // chunk is guaranteed to be size 2 by par_chunks_exact
                        let lo = chunk[0];
                        let hi = chunk[1];

                        *out = (lo + hi).halve() + (lo - hi) * current_beta * halve_inv_power;
                    });
                current_beta = current_beta.square();

                // Ping-pong the two buffers; the tail of the new `data` is stale and dropped.
                core::mem::swap(&mut data, &mut next_data);
                data.truncate(height);
            }

            data
        }
    }
}

/// Unnormalized inverse FFT of four evaluations in bit-reversed order.
/// The supplied inverse fourth root must belong to this transform's domain.
#[inline]
fn ifft4<F: TwoAdicField, EF: ExtensionField<F>>(
    [y0, y2, y1, y3]: [EF; 4],
    root_inv: F,
) -> [EF; 4] {
    let a0 = y0 + y2;
    let a1 = y0 - y2;
    let a2 = y1 + y3;
    let a3 = (y1 - y3) * root_inv;
    [a0 + a2, a1 + a3, a0 - a2, a1 - a3]
}

/// Unnormalized inverse FFT of eight evaluations in bit-reversed order.
#[inline]
fn ifft8<F: TwoAdicField, EF: ExtensionField<F>>(
    [y0, y4, y2, y6, y1, y5, y3, y7]: [EF; 8],
) -> [EF; 8] {
    let root = F::two_adic_generator(3);
    let root_squared = root.square();
    // Derive every twiddle from this domain's root.
    // A field need not choose its fourth root to be the square of its eighth root.
    let fourth_root_inv = -root_squared;
    let [b0, b1, b2, b3] = ifft4([y0, y4, y2, y6], fourth_root_inv);
    let [b4, b5, b6, b7] = ifft4([y1, y5, y3, y7], fourth_root_inv);
    let t1 = b5 * -(root_squared * root);
    let t2 = b6 * fourth_root_inv;
    let t3 = b7 * -root;
    [
        b0 + b4,
        b1 + t1,
        b2 + t2,
        b3 + t3,
        b0 - b4,
        b1 - t1,
        b2 - t2,
        b3 - t3,
    ]
}

#[cfg(feature = "secret-memory")]
fn allocate_secret_fold_matrix<EF>(
    height: usize,
) -> Result<SecretProverData<RowMajorMatrix<EF>>, SecretFriFoldError>
where
    EF: Field + zeroize::Zeroize,
{
    let mut values = Vec::new();
    values
        .try_reserve_exact(height)
        .map_err(|_| SecretFriFoldError::AllocationFailed)?;
    values.resize(height, EF::ZERO);
    Ok(SecretProverData::new(RowMajorMatrix::new(values, 1)))
}

#[cfg(feature = "secret-memory")]
impl<
    F: TwoAdicField,
    InputProof: Sync,
    InputError: Debug + Sync,
    EF: ExtensionField<F> + zeroize::Zeroize,
> SecretFriFoldingStrategy<F, EF> for TwoAdicFriFolding<InputProof, InputError>
{
    type SecretFoldError = SecretFriFoldError;

    fn fold_matrix_secret(
        &self,
        beta: EF,
        log_arity: usize,
        matrix: RowMajorMatrixView<'_, EF>,
    ) -> Result<SecretProverData<RowMajorMatrix<EF>>, Self::SecretFoldError> {
        if log_arity == 0 {
            return Err(SecretFriFoldError::ZeroLogArity);
        }
        let arity_shift =
            u32::try_from(log_arity).map_err(|_| SecretFriFoldError::ArityOverflow)?;
        let arity = 1_usize
            .checked_shl(arity_shift)
            .ok_or(SecretFriFoldError::ArityOverflow)?;
        if matrix.width != arity
            || matrix.values.is_empty()
            || !matrix.values.len().is_multiple_of(arity)
        {
            return Err(SecretFriFoldError::InvalidMatrixShape);
        }
        let height = matrix.values.len() / arity;
        if !height.is_power_of_two() {
            return Err(SecretFriFoldError::InvalidMatrixShape);
        }
        let log_domain = log2_strict_usize(height)
            .checked_add(log_arity)
            .ok_or(SecretFriFoldError::UnsupportedDomain)?;
        if log_domain > F::TWO_ADICITY {
            return Err(SecretFriFoldError::UnsupportedDomain);
        }

        let g_inv = F::two_adic_generator(log_domain).inverse();
        let initial_height = if log_arity == 1 {
            height
        } else {
            height
                .checked_mul(arity / 2)
                .ok_or(SecretFriFoldError::ArityOverflow)?
        };
        let mut halve_inv_powers = g_inv
            .shifted_powers(F::ONE.halve())
            .collect_n(initial_height);
        reverse_slice_index_bits(&mut halve_inv_powers);

        if log_arity == 1 {
            let mut output = allocate_secret_fold_matrix(height)?;
            output
                .values
                .par_iter_mut()
                .zip(matrix.values.par_chunks_exact(2))
                .zip(halve_inv_powers)
                .for_each(|((out, row), halve_inv_power)| {
                    if let [lo, hi] = row {
                        *out = (*lo + *hi).halve() + (*lo - *hi) * beta * halve_inv_power;
                    }
                });
            return Ok(output);
        }

        let pairs_per_row = arity / 2;
        let mut data = allocate_secret_fold_matrix(initial_height)?;
        data.values
            .par_chunks_exact_mut(pairs_per_row)
            .zip(matrix.values.par_chunks_exact(arity))
            .zip(halve_inv_powers.par_chunks_exact(pairs_per_row))
            .for_each(|((output_row, input_row), inverse_powers)| {
                let (input_pairs, remainder) = input_row.as_chunks::<2>();
                debug_assert!(remainder.is_empty());
                for ((out, pair), &halve_inv_power) in
                    output_row.iter_mut().zip(input_pairs).zip(inverse_powers)
                {
                    let [lo, hi] = pair;
                    *out = (*lo + *hi).halve() + (*lo - *hi) * beta * halve_inv_power;
                }
            });

        let mut current_beta = beta.square();
        let mut next_data = allocate_secret_fold_matrix(initial_height / 2)?;
        for _ in 1..log_arity {
            let next_height = data.values.len() / 2;
            let output = next_data
                .values
                .get_mut(..next_height)
                .ok_or(SecretFriFoldError::InvalidMatrixShape)?;
            let inverse_powers = halve_inv_powers
                .get(..next_height)
                .ok_or(SecretFriFoldError::InvalidMatrixShape)?;
            output
                .par_iter_mut()
                .zip(data.values.par_chunks_exact(2))
                .zip(inverse_powers)
                .for_each(|((out, pair), &halve_inv_power)| {
                    if let [lo, hi] = pair {
                        *out = (*lo + *hi).halve() + (*lo - *hi) * current_beta * halve_inv_power;
                    }
                });
            current_beta = current_beta.square();

            core::mem::swap(&mut data, &mut next_data);
            if let Some(stale_tail) = data.values.get_mut(next_height..) {
                for value in stale_tail {
                    zeroize::Zeroize::zeroize(value);
                }
            }
            data.values.truncate(next_height);
        }

        Ok(data)
    }
}

/// Whether `fold_matrix` folds a matrix of `height` rows over a base field of `field_bits`
/// bits with `fold_matrix_packed`, given a packing of `packing_width` lanes.
///
/// Packed arithmetic pays off where SIMD lanes multiply field elements natively, which
/// holds for fields of at most 32 bits. Wider fields emulate the lane product on common
/// targets, so they keep the scalar paths, as do width-1 packings. `PackedField` does not
/// bound the packing width, and the packed kernel bit-reverses lane indices and splits the
/// power-of-two height into whole blocks, so the width must also be a power of two no
/// larger than the height.
const fn use_packed_fold(packing_width: usize, height: usize, field_bits: usize) -> bool {
    packing_width > 1
        && packing_width.is_power_of_two()
        && height >= packing_width
        && field_bits <= 32
}

/// Fold `F::Packing::WIDTH` rows of `m` at a time, one row per SIMD lane.
///
/// Each row goes through the `log_arity` arity-2 rounds of `fold_matrix` without leaving
/// packed form. Round `s` halves pair `k` of row `t` by the base-field factor
///
/// ```text
///     g_inv^(2^s * rev(t)) * w_inv^(2^s * rev(k)) / 2
/// ```
///
/// where `g` generates the subgroup of order `height * arity`, `w = g^height` generates the
/// subgroup of order `arity`, and `rev` reverses the bits of a row or pair index. This is
/// the entry at `t * (arity >> (s + 1)) + k` of the bit-reversed `g_inv^i / 2` table the
/// other paths use, split into a packed per-row factor, squared once per round, and a
/// per-pair scalar that is merged with `beta^(2^s)`.
///
/// `F::Packing::WIDTH` must be a power of two, and `m.height()` a nonzero multiple of it.
fn fold_matrix_packed<F, EF, M>(beta: EF, log_arity: usize, m: &M) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    M: Matrix<EF>,
{
    let width = F::Packing::WIDTH;
    let height = m.height();
    let pairs_per_row = 1 << (log_arity - 1);
    debug_assert!(width.is_power_of_two() && height >= width && height.is_multiple_of(width));
    debug_assert_eq!(m.width(), 2 * pairs_per_row);

    // Row `t = b * width + l` of block `b` has the factor
    //     g_inv^rev(t) = g_inv^rev(b) * (g_inv^num_blocks)^rev(l)
    // with `rev(b)` over the block index bits and `rev(l)` over the lane bits.
    let g_inv = F::two_adic_generator(log2_strict_usize(height) + log_arity).inverse();
    let num_blocks = height / width;
    let mut block_factors = g_inv.powers().collect_n(num_blocks);
    reverse_slice_index_bits(&mut block_factors);
    let lane_root = g_inv.exp_u64(num_blocks as u64);
    let log_width = log2_strict_usize(width);
    let lane_factors =
        F::Packing::from_fn(|lane| lane_root.exp_u64(reverse_bits_len(lane, log_width) as u64));

    // Pair factors `w_inv^rev(k)`, in bit-reversed order. Round `s` reads the prefix of
    // length `pairs_per_row >> s`, which holds `w_inv^(2^s * rev(k))` with `rev` taken over
    // that prefix's index bits.
    let w_inv = F::two_adic_generator(log_arity).inverse();
    let mut pair_factors = w_inv.powers().collect_n(pairs_per_row);
    reverse_slice_index_bits(&mut pair_factors);

    // Round `s` multiplies pair `k` by `beta^(2^s) * w_inv^(2^s * rev(k)) / 2`.
    let mut round_beta = beta;
    let round_scalars: Vec<Vec<EF>> = (0..log_arity)
        .map(|round| {
            let half_beta = round_beta.halve();
            round_beta = round_beta.square();
            pair_factors[..pairs_per_row >> round]
                .iter()
                .map(|&w| half_beta * w)
                .collect()
        })
        .collect();

    let fold =
        |lo: EF::ExtensionPacking, hi: EF::ExtensionPacking, scalar: EF, row_factor: F::Packing| {
            (lo + hi).halve() + (lo - hi) * scalar * row_factor
        };

    let mut folded = EF::zero_vec(height);
    folded
        .par_chunks_exact_mut(width)
        .zip(block_factors.par_iter())
        .enumerate()
        .for_each_init(
            || EF::ExtensionPacking::zero_vec(pairs_per_row),
            |pairs, (block, (out, &block_factor))| {
                let first_row = block * width;
                let column = |c| {
                    EF::ExtensionPacking::from_ext_fn(|lane| m.get(first_row + lane, c).unwrap())
                };

                let mut row_factor = lane_factors * block_factor;
                for (k, (pair, &scalar)) in pairs.iter_mut().zip(&round_scalars[0]).enumerate() {
                    *pair = fold(column(2 * k), column(2 * k + 1), scalar, row_factor);
                }

                // Later rounds fold adjacent packed pairs in place.
                for scalars in &round_scalars[1..] {
                    row_factor = row_factor.square();
                    for (k, &scalar) in scalars.iter().enumerate() {
                        pairs[k] = fold(pairs[2 * k], pairs[2 * k + 1], scalar, row_factor);
                    }
                }

                pairs[0].to_ext_slice(out);
            },
        );
    folded
}

/// Lagrange interpolation: given points `(xs[i], ys[i])`, evaluate at z.
///
/// Uses the barycentric formula for efficiency when xs are roots of unity.
fn lagrange_interpolate_at<F: TwoAdicField, EF: ExtensionField<F>>(
    xs: &[F],
    ys: &[EF],
    z: EF,
) -> EF {
    debug_assert_eq!(xs.len(), ys.len());
    let n = xs.len();

    if n == 0 {
        return EF::ZERO;
    }

    // If z equals one of the interpolation points, return early.
    for i in 0..n {
        if (z - xs[i]).is_zero() {
            return ys[i];
        }
    }

    let log_n = log2_strict_usize(n);

    // All xs lie in a coset of the 2^log_n roots of unity.
    let coset_power = xs[0].exp_power_of_2(log_n);
    let weight_scale = (F::from_usize(n) * coset_power).inverse();

    // Compute (z - x_i)^{-1} as a batch inversion
    let diffs: Vec<_> = xs.iter().map(|&x| z - x).collect();
    let diff_invs = batch_multiplicative_inverse(&diffs);

    // Compute L(z) = prod_i (z - x_i)
    let l_z = diffs.iter().copied().product::<EF>();

    // Barycentric formula: sum_i (w_i * y_i / (z - x_i))
    // where w_i = 1 / prod_{j != i} (x_i - x_j) = x_i * weight_scale.
    let mut result = EF::ZERO;
    for ((&x, &y), &diff_inv) in xs.iter().zip(ys).zip(diff_invs.iter()) {
        let weight = x * weight_scale;
        result += y * weight * diff_inv;
    }
    result * l_z
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger> SecretPcsCommitter<Challenge, Challenger>
    for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: Mmcs<Val, MultiProof: Sync, Error: Sync> + SecretRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
    FriMmcs: Mmcs<Challenge>,
    Challenge: ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    type SecretData = SecretTwoAdicFriData<
        Val,
        InputMmcs::SecretData<RowMajorMatrix<Val>>,
        InputMmcs::AuthenticationData<RowMajorMatrix<Val>>,
    >;
    type SecretCommitError = SecretFriCommitError<InputMmcs::SecretError>;

    fn commit_secret(
        &self,
        evaluations: Vec<(Self::Domain, SecretProverData<RowMajorMatrix<Val>>)>,
    ) -> Result<(Self::Commitment, SecretProverData<Self::SecretData>), Self::SecretCommitError>
    {
        for (domain, matrix) in &evaluations {
            if matrix.width == 0 || !matrix.values.len().is_multiple_of(matrix.width) {
                return Err(SecretMatrixError::InvalidShape.into());
            }
            if domain.size() != matrix.height() {
                return Err(SecretFriCommitError::DomainHeightMismatch);
            }
        }

        let mut ldes = Vec::new();
        ldes.try_reserve_exact(evaluations.len())
            .map_err(|_| SecretFriCommitError::AllocationFailed)?;
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(evaluations.len())
            .map_err(|_| SecretFriCommitError::AllocationFailed)?;
        for (domain, mut matrix) in evaluations {
            let shift = Val::GENERATOR / domain.shift();
            let matrix_coefficients = self
                .dft
                .coset_lde_batch_secret_with_coefficients(&mut matrix, self.fri.log_blowup, shift)
                .map_err(SecretFriCommitError::Dft)?;
            reverse_matrix_index_bits(&mut matrix);
            ldes.push(matrix);
            coefficients.push(matrix_coefficients);
        }

        let (commitment, commitment_data) = self
            .mmcs
            .commit_secret(ldes)
            .map_err(SecretFriCommitError::Mmcs)?;
        Ok((
            commitment,
            SecretProverData::new(SecretTwoAdicFriData::new(
                commitment_data,
                coefficients,
                self.fri.log_blowup,
            )),
        ))
    }
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger>
    SecretDetachedPcsCommitter<Challenge, Challenger>
    for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: Mmcs<Val, MultiProof: Sync, Error: Sync>
        + SecretRecomputableMmcs<Val>
        + SecretStreamingRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
    FriMmcs: Mmcs<Challenge>,
    Challenge: ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    fn commit_detached_secret(
        &self,
        evaluations: Vec<(Self::Domain, SecretProverData<RowMajorMatrix<Val>>)>,
    ) -> Result<(Self::Commitment, SecretProverData<Self::SecretData>), Self::SecretCommitError>
    {
        let blowup_shift = u32::try_from(self.fri.log_blowup)
            .map_err(|_| SecretFriCommitError::Dft(SecretDftError::SizeOverflow))?;
        let mut dimensions = Vec::new();
        dimensions
            .try_reserve_exact(evaluations.len())
            .map_err(|_| SecretFriCommitError::AllocationFailed)?;
        for (domain, matrix) in &evaluations {
            if matrix.width == 0 || !matrix.values.len().is_multiple_of(matrix.width) {
                return Err(SecretMatrixError::InvalidShape.into());
            }
            if domain.size() != matrix.height() {
                return Err(SecretFriCommitError::DomainHeightMismatch);
            }
            let height = matrix
                .height()
                .checked_shl(blowup_shift)
                .ok_or(SecretFriCommitError::Dft(SecretDftError::SizeOverflow))?;
            dimensions.push(Dimensions {
                width: matrix.width,
                height,
            });
        }

        let mut stream =
            <InputMmcs as SecretStreamingRecomputableMmcs<Val>>::begin_streaming_commitment::<
                RowMajorMatrix<Val>,
            >(&self.mmcs, dimensions)
            .map_err(SecretFriCommitError::Mmcs)?;
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(evaluations.len())
            .map_err(|_| SecretFriCommitError::AllocationFailed)?;
        for (domain, mut matrix) in evaluations {
            let shift = Val::GENERATOR / domain.shift();
            let matrix_coefficients = self
                .dft
                .coset_lde_batch_secret_with_coefficients(&mut matrix, self.fri.log_blowup, shift)
                .map_err(SecretFriCommitError::Dft)?;
            reverse_matrix_index_bits(&mut matrix);
            stream.absorb(matrix).map_err(SecretFriCommitError::Mmcs)?;
            coefficients.push(matrix_coefficients);
        }

        let detached = stream.finish().map_err(SecretFriCommitError::Mmcs)?;
        Ok((
            detached.commitment,
            SecretProverData::new(SecretTwoAdicFriData::new_detached(
                detached.prover_data,
                detached.authentication_data,
                coefficients,
                self.fri.log_blowup,
            )),
        ))
    }
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger>
    SecretUnivariateStarkPcsCommitter<Challenge, Challenger>
    for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: Mmcs<Val, MultiProof: Sync, Error: Sync>
        + SecretRecomputableMmcs<Val>
        + SecretStreamingRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
    FriMmcs: Mmcs<Challenge>,
    Challenge: ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    fn commit_quotient_secret(
        &self,
        quotient_domain: Self::Domain,
        quotient_evaluations: SecretProverData<RowMajorMatrix<Val>>,
        num_chunks: usize,
    ) -> Result<(Self::Commitment, SecretProverData<Self::SecretData>), Self::SecretCommitError>
    {
        let mut quotients = Vec::new();
        quotients
            .try_reserve_exact(1)
            .map_err(|_| SecretFriCommitError::AllocationFailed)?;
        quotients.push(SecretQuotientCommitmentInput::new(
            quotient_domain,
            quotient_evaluations,
            num_chunks,
        ));
        <Self as SecretUnivariateStarkPcsCommitter<Challenge, Challenger>>::commit_quotients_secret(
            self, quotients,
        )
    }

    fn commit_quotients_secret(
        &self,
        quotients: Vec<SecretQuotientCommitmentInput<Self::Domain, Val>>,
    ) -> Result<(Self::Commitment, SecretProverData<Self::SecretData>), Self::SecretCommitError>
    {
        let mut plans = Vec::new();
        plans
            .try_reserve_exact(quotients.len())
            .map_err(|_| SecretFriCommitError::AllocationFailed)?;
        for quotient in &quotients {
            let domain_size = quotient.domain().size();
            let evaluations = quotient.evaluations();
            let width = evaluations.width;
            let num_chunks = quotient.num_chunks();
            if width == 0 || !evaluations.values.len().is_multiple_of(width) {
                return Err(SecretMatrixError::InvalidShape.into());
            }
            if evaluations.values.len() / width != domain_size {
                return Err(SecretFriCommitError::DomainHeightMismatch);
            }
            if num_chunks == 0
                || !num_chunks.is_power_of_two()
                || num_chunks > domain_size
                || !domain_size.is_multiple_of(num_chunks)
            {
                return Err(SecretFriCommitError::InvalidQuotientChunkCount);
            }
            let expected_len = domain_size
                .checked_mul(width)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            if evaluations.values.len() != expected_len {
                return Err(SecretMatrixError::InvalidShape.into());
            }
            plans.push(SecretQuotientCommitmentPlan::new(
                *quotient.domain(),
                width,
                num_chunks,
            ));
        }

        let mut quotients = quotients.into_iter();
        match <Self as SecretUnivariateStarkPcsCommitter<Challenge, Challenger>>::commit_generated_quotients_secret(
            self,
            plans,
            |_, _| {
                quotients
                    .next()
                    .map(|quotient| quotient.into_parts().1)
                    .ok_or(SecretFriCommitError::QuotientCountMismatch)
            },
        ) {
            Ok(output) => Ok(output),
            Err(SecretGeneratedQuotientCommitError::Generation(error))
            | Err(SecretGeneratedQuotientCommitError::Commitment(error)) => Err(error),
        }
    }

    fn commit_generated_quotients_secret<GenerationError>(
        &self,
        plans: Vec<SecretQuotientCommitmentPlan<Self::Domain>>,
        mut generate: impl FnMut(
            usize,
            &SecretQuotientCommitmentPlan<Self::Domain>,
        )
            -> Result<SecretProverData<RowMajorMatrix<Val>>, GenerationError>,
    ) -> SecretGeneratedQuotientCommitmentResult<
        Self::Commitment,
        Self::SecretData,
        GenerationError,
        Self::SecretCommitError,
    >
    where
        GenerationError: Debug,
    {
        if plans.is_empty() {
            return Err(SecretGeneratedQuotientCommitError::Commitment(
                SecretFriCommitError::EmptyQuotientBatch,
            ));
        }

        let mut total_chunks = 0_usize;
        for plan in &plans {
            let domain_size = plan.domain().size();
            let width = plan.width();
            let num_chunks = plan.num_chunks();
            if width == 0 {
                return Err(SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::InvalidShape.into(),
                ));
            }
            if num_chunks == 0
                || !num_chunks.is_power_of_two()
                || num_chunks > domain_size
                || !domain_size.is_multiple_of(num_chunks)
            {
                return Err(SecretGeneratedQuotientCommitError::Commitment(
                    SecretFriCommitError::InvalidQuotientChunkCount,
                ));
            }
            domain_size.checked_mul(width).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::DimensionOverflow.into(),
                )
            })?;
            total_chunks = total_chunks.checked_add(num_chunks).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::DimensionOverflow.into(),
                )
            })?;
        }

        let log_blowup = u32::try_from(self.fri.log_blowup).map_err(|_| {
            SecretGeneratedQuotientCommitError::Commitment(SecretFriCommitError::Dft(
                SecretDftError::SizeOverflow,
            ))
        })?;
        let mut lde_dimensions = Vec::new();
        lde_dimensions
            .try_reserve_exact(total_chunks)
            .map_err(|_| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretFriCommitError::AllocationFailed,
                )
            })?;
        for plan in &plans {
            let rows_per_chunk = plan.domain().size() / plan.num_chunks();
            let lde_height = rows_per_chunk.checked_shl(log_blowup).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(SecretFriCommitError::Dft(
                    SecretDftError::SizeOverflow,
                ))
            })?;
            for _ in 0..plan.num_chunks() {
                lde_dimensions.push(Dimensions {
                    width: plan.width(),
                    height: lde_height,
                });
            }
        }
        let mut lde_stream =
            <InputMmcs as SecretStreamingRecomputableMmcs<Val>>::begin_streaming_commitment::<
                RowMajorMatrix<Val>,
            >(&self.mmcs, lde_dimensions)
            .map_err(|error| {
                SecretGeneratedQuotientCommitError::Commitment(SecretFriCommitError::Mmcs(error))
            })?;
        let mut coefficients = Vec::new();
        coefficients.try_reserve_exact(total_chunks).map_err(|_| {
            SecretGeneratedQuotientCommitError::Commitment(SecretFriCommitError::AllocationFailed)
        })?;

        for (index, plan) in plans.into_iter().enumerate() {
            let quotient_evaluations =
                generate(index, &plan).map_err(SecretGeneratedQuotientCommitError::Generation)?;
            let (quotient_domain, width, num_chunks) = plan.into_parts();
            let domain_size = quotient_domain.size();
            let expected_len = domain_size.checked_mul(width).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::DimensionOverflow.into(),
                )
            })?;
            if quotient_evaluations.width != width
                || quotient_evaluations.values.len() != expected_len
            {
                return Err(SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::InvalidShape.into(),
                ));
            }
            let rows_per_chunk = domain_size / num_chunks;
            let values_per_chunk = rows_per_chunk.checked_mul(width).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::DimensionOverflow.into(),
                )
            })?;
            let domains = quotient_domain.split_domains(num_chunks);
            if domains.len() != num_chunks {
                return Err(SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::InvalidShape.into(),
                ));
            }

            let mut chunks = Vec::new();
            chunks.try_reserve_exact(num_chunks).map_err(|_| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretFriCommitError::AllocationFailed,
                )
            })?;
            for _ in 0..num_chunks {
                let mut values = Vec::new();
                values.try_reserve_exact(values_per_chunk).map_err(|_| {
                    SecretGeneratedQuotientCommitError::Commitment(
                        SecretMatrixError::AllocationFailed.into(),
                    )
                })?;
                values.resize(values_per_chunk, Val::ZERO);
                chunks.push(SecretProverData::new(RowMajorMatrix::new(values, width)));
            }

            for source_row_index in 0..domain_size {
                let chunk_index = source_row_index % num_chunks;
                let destination_row_index = source_row_index / num_chunks;
                let source_start = source_row_index.checked_mul(width).ok_or_else(|| {
                    SecretGeneratedQuotientCommitError::Commitment(
                        SecretMatrixError::DimensionOverflow.into(),
                    )
                })?;
                let source_end = source_start.checked_add(width).ok_or_else(|| {
                    SecretGeneratedQuotientCommitError::Commitment(
                        SecretMatrixError::DimensionOverflow.into(),
                    )
                })?;
                let destination_start =
                    destination_row_index.checked_mul(width).ok_or_else(|| {
                        SecretGeneratedQuotientCommitError::Commitment(
                            SecretMatrixError::DimensionOverflow.into(),
                        )
                    })?;
                let destination_end = destination_start.checked_add(width).ok_or_else(|| {
                    SecretGeneratedQuotientCommitError::Commitment(
                        SecretMatrixError::DimensionOverflow.into(),
                    )
                })?;
                let source = quotient_evaluations
                    .values
                    .get(source_start..source_end)
                    .ok_or_else(|| {
                        SecretGeneratedQuotientCommitError::Commitment(
                            SecretMatrixError::InvalidShape.into(),
                        )
                    })?;
                let destination = chunks
                    .get_mut(chunk_index)
                    .and_then(|chunk| chunk.values.get_mut(destination_start..destination_end))
                    .ok_or_else(|| {
                        SecretGeneratedQuotientCommitError::Commitment(
                            SecretMatrixError::InvalidShape.into(),
                        )
                    })?;
                destination.copy_from_slice(source);
            }

            drop(quotient_evaluations);

            for (domain, mut chunk) in domains.into_iter().zip(chunks) {
                let shift = Val::GENERATOR / domain.shift();
                let chunk_coefficients = self
                    .dft
                    .coset_lde_batch_secret_with_coefficients(
                        &mut chunk,
                        self.fri.log_blowup,
                        shift,
                    )
                    .map_err(|error| {
                        SecretGeneratedQuotientCommitError::Commitment(SecretFriCommitError::Dft(
                            error,
                        ))
                    })?;
                reverse_matrix_index_bits(&mut chunk);
                lde_stream.absorb(chunk).map_err(|error| {
                    SecretGeneratedQuotientCommitError::Commitment(SecretFriCommitError::Mmcs(
                        error,
                    ))
                })?;
                coefficients.push(chunk_coefficients);
            }
        }

        let detached = lde_stream.finish().map_err(|error| {
            SecretGeneratedQuotientCommitError::Commitment(SecretFriCommitError::Mmcs(error))
        })?;
        let prover_data = SecretProverData::new(SecretTwoAdicFriData::new_detached(
            detached.prover_data,
            detached.authentication_data,
            coefficients,
            self.fri.log_blowup,
        ));
        Ok((detached.commitment, prover_data))
    }

    fn commit_generated_quotient_chunks_secret<GenerationError>(
        &self,
        plans: Vec<SecretQuotientCommitmentPlan<Self::Domain>>,
        mut generate: impl FnMut(
            usize,
            &SecretQuotientCommitmentPlan<Self::Domain>,
            usize,
            Self::Domain,
        )
            -> Result<SecretProverData<RowMajorMatrix<Val>>, GenerationError>,
    ) -> SecretGeneratedQuotientCommitmentResult<
        Self::Commitment,
        Self::SecretData,
        GenerationError,
        Self::SecretCommitError,
    >
    where
        GenerationError: Debug,
    {
        if plans.is_empty() {
            return Err(SecretGeneratedQuotientCommitError::Commitment(
                SecretFriCommitError::EmptyQuotientBatch,
            ));
        }

        let mut total_chunks = 0_usize;
        for plan in &plans {
            let domain_size = plan.domain().size();
            let width = plan.width();
            let num_chunks = plan.num_chunks();
            if width == 0 {
                return Err(SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::InvalidShape.into(),
                ));
            }
            if num_chunks == 0
                || !num_chunks.is_power_of_two()
                || num_chunks > domain_size
                || !domain_size.is_multiple_of(num_chunks)
            {
                return Err(SecretGeneratedQuotientCommitError::Commitment(
                    SecretFriCommitError::InvalidQuotientChunkCount,
                ));
            }
            domain_size.checked_mul(width).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::DimensionOverflow.into(),
                )
            })?;
            total_chunks = total_chunks.checked_add(num_chunks).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::DimensionOverflow.into(),
                )
            })?;
        }

        let log_blowup = u32::try_from(self.fri.log_blowup).map_err(|_| {
            SecretGeneratedQuotientCommitError::Commitment(SecretFriCommitError::Dft(
                SecretDftError::SizeOverflow,
            ))
        })?;
        let mut lde_dimensions = Vec::new();
        lde_dimensions
            .try_reserve_exact(total_chunks)
            .map_err(|_| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretFriCommitError::AllocationFailed,
                )
            })?;
        for plan in &plans {
            let rows_per_chunk = plan.domain().size() / plan.num_chunks();
            let lde_height = rows_per_chunk.checked_shl(log_blowup).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(SecretFriCommitError::Dft(
                    SecretDftError::SizeOverflow,
                ))
            })?;
            for _ in 0..plan.num_chunks() {
                lde_dimensions.push(Dimensions {
                    width: plan.width(),
                    height: lde_height,
                });
            }
        }
        let mut lde_stream =
            <InputMmcs as SecretStreamingRecomputableMmcs<Val>>::begin_streaming_commitment::<
                RowMajorMatrix<Val>,
            >(&self.mmcs, lde_dimensions)
            .map_err(|error| {
                SecretGeneratedQuotientCommitError::Commitment(SecretFriCommitError::Mmcs(error))
            })?;
        let mut coefficients = Vec::new();
        coefficients.try_reserve_exact(total_chunks).map_err(|_| {
            SecretGeneratedQuotientCommitError::Commitment(SecretFriCommitError::AllocationFailed)
        })?;

        for (instance_index, plan) in plans.into_iter().enumerate() {
            let width = plan.width();
            let num_chunks = plan.num_chunks();
            let rows_per_chunk = plan.domain().size() / num_chunks;
            let expected_len = rows_per_chunk.checked_mul(width).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::DimensionOverflow.into(),
                )
            })?;
            let domains = plan.domain().split_domains(num_chunks);
            if domains.len() != num_chunks {
                return Err(SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::InvalidShape.into(),
                ));
            }

            for (chunk_index, domain) in domains.into_iter().enumerate() {
                let mut chunk = generate(instance_index, &plan, chunk_index, domain)
                    .map_err(SecretGeneratedQuotientCommitError::Generation)?;
                if chunk.width != width || chunk.values.len() != expected_len {
                    return Err(SecretGeneratedQuotientCommitError::Commitment(
                        SecretMatrixError::InvalidShape.into(),
                    ));
                }
                let shift = Val::GENERATOR / domain.shift();
                let chunk_coefficients = self
                    .dft
                    .coset_lde_batch_secret_with_coefficients(
                        &mut chunk,
                        self.fri.log_blowup,
                        shift,
                    )
                    .map_err(|error| {
                        SecretGeneratedQuotientCommitError::Commitment(SecretFriCommitError::Dft(
                            error,
                        ))
                    })?;
                reverse_matrix_index_bits(&mut chunk);
                lde_stream.absorb(chunk).map_err(|error| {
                    SecretGeneratedQuotientCommitError::Commitment(SecretFriCommitError::Mmcs(
                        error,
                    ))
                })?;
                coefficients.push(chunk_coefficients);
            }
        }

        let detached = lde_stream.finish().map_err(|error| {
            SecretGeneratedQuotientCommitError::Commitment(SecretFriCommitError::Mmcs(error))
        })?;
        let prover_data = SecretProverData::new(SecretTwoAdicFriData::new_detached(
            detached.prover_data,
            detached.authentication_data,
            coefficients,
            self.fri.log_blowup,
        ));
        Ok((detached.commitment, prover_data))
    }
}

struct PreparedPcsOpening<'a, Challenger, Val: PrimeField64, Challenge, ReducedOpening> {
    all_opened_values: OpenedValues<Challenge>,
    fri_input: Vec<ReducedOpening>,
    log_global_max_height: usize,
    batch_pow_witness: Val,
    transcript: PcsProverTranscript<'a, Challenger, Val, Challenge>,
}

struct PreparedPcsClaims<'a, Challenger, Val: PrimeField64, Challenge> {
    all_opened_values: OpenedValues<Challenge>,
    inverse_denominators: LinearMap<Challenge, Vec<Challenge>>,
    global_max_width: usize,
    log_global_max_height: usize,
    alpha: Challenge,
    batch_pow_witness: Val,
    transcript: PcsProverTranscript<'a, Challenger, Val, Challenge>,
}

trait ReducedOpeningBuffer<F> {
    fn len(&self) -> usize;
    fn values_mut(&mut self) -> &mut [F];
}

impl<F> ReducedOpeningBuffer<F> for Vec<F> {
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn values_mut(&mut self) -> &mut [F] {
        self.as_mut_slice()
    }
}

struct OpeningReductionState<'a, Val, Challenge, ReducedOpening, PreparationError, MakeReduced>
where
    Val: PrimeField64 + TwoAdicField,
    Challenge: ExtensionField<Val>,
{
    alpha: Challenge,
    inverse_denominators: &'a LinearMap<Challenge, Vec<Challenge>>,
    packed_alpha_powers: Vec<Challenge::ExtensionPacking>,
    num_reduced: Vec<usize>,
    reduced_openings: Vec<Option<ReducedOpening>>,
    make_reduced: MakeReduced,
    _error: PhantomData<PreparationError>,
}

impl<'a, Val, Challenge, ReducedOpening, PreparationError, MakeReduced>
    OpeningReductionState<'a, Val, Challenge, ReducedOpening, PreparationError, MakeReduced>
where
    Val: PrimeField64 + TwoAdicField,
    Challenge: ExtensionField<Val>,
    ReducedOpening: ReducedOpeningBuffer<Challenge>,
    PreparationError: From<FriProverError>,
    MakeReduced: FnMut(usize) -> Result<ReducedOpening, PreparationError>,
{
    fn new(
        alpha: Challenge,
        global_max_width: usize,
        inverse_denominators: &'a LinearMap<Challenge, Vec<Challenge>>,
        make_reduced: MakeReduced,
    ) -> Result<Self, PreparationError> {
        let height_bucket_count = Val::TWO_ADICITY
            .checked_add(1)
            .ok_or(FriProverError::OpeningHeightUnsupported)?;
        let packed_alpha_powers =
            Challenge::ExtensionPacking::packed_ext_powers_capped(alpha, global_max_width)
                .collect_vec();
        Ok(Self {
            alpha,
            inverse_denominators,
            packed_alpha_powers,
            num_reduced: alloc::vec![0_usize; height_bucket_count],
            reduced_openings: core::iter::repeat_with(|| None)
                .take(height_bucket_count)
                .collect_vec(),
            make_reduced,
            _error: PhantomData,
        })
    }

    fn reduce_matrix(
        &mut self,
        matrix: RowMajorMatrixView<'_, Val>,
        points: &[Challenge],
        openings: &[Vec<Challenge>],
    ) -> Result<(), PreparationError> {
        if points.len() != openings.len() {
            return Err(FriProverError::OpeningRequestShapeMismatch.into());
        }
        let _guard = debug_span!("reduce matrix quotient", dims = %matrix.dimensions()).entered();
        let log_height = log2_strict_usize(matrix.height());
        let reduced_opening_slot = self
            .reduced_openings
            .get_mut(log_height)
            .ok_or(FriProverError::OpeningHeightUnsupported)?;
        if reduced_opening_slot.is_none() {
            *reduced_opening_slot = Some((self.make_reduced)(matrix.height())?);
        }
        let reduced_opening_for_log_height = reduced_opening_slot
            .as_mut()
            .ok_or(FriProverError::OpeningRequestShapeMismatch)?;
        if reduced_opening_for_log_height.len() != matrix.height() {
            return Err(FriProverError::OpeningRequestShapeMismatch.into());
        }
        let num_reduced_at_height = self
            .num_reduced
            .get_mut(log_height)
            .ok_or(FriProverError::OpeningHeightUnsupported)?;

        let first_alpha_exponent = u64::try_from(*num_reduced_at_height)
            .map_err(|_| FriProverError::OpeningBatchWidthOverflow)?;
        let first_alpha_pow_offset = self.alpha.exp_u64(first_alpha_exponent);
        let packed_width = matrix.width().div_ceil(Val::Packing::WIDTH);
        let offset_alpha_powers = self
            .packed_alpha_powers
            .get(..packed_width)
            .ok_or(FriProverError::OpeningRequestShapeMismatch)?
            .iter()
            .map(|&power| power * first_alpha_pow_offset)
            .collect_vec();

        let mut point_reductions = Vec::with_capacity(points.len());
        for (point_index, (&point, opened_values)) in points.iter().zip(openings).enumerate() {
            let alpha_exponent = u64::try_from(*num_reduced_at_height)
                .map_err(|_| FriProverError::OpeningBatchWidthOverflow)?;
            let alpha_pow_offset = self.alpha.exp_u64(alpha_exponent);
            let reduced_opening: Challenge =
                dot_product(self.alpha.powers(), opened_values.iter().copied());
            let weighted_opening = alpha_pow_offset * reduced_opening;
            let point_offset = point_index
                .checked_mul(matrix.width())
                .ok_or(FriProverError::OpeningBatchWidthOverflow)?;
            let point_exponent = u64::try_from(point_offset)
                .map_err(|_| FriProverError::OpeningBatchWidthOverflow)?;
            let rescale = self.alpha.exp_u64(point_exponent);
            let denominators = self
                .inverse_denominators
                .get(&point)
                .ok_or(FriProverError::OpeningDenominatorTableMismatch)?;
            if denominators.len() < matrix.height() {
                return Err(FriProverError::OpeningDenominatorTableMismatch.into());
            }
            *num_reduced_at_height = num_reduced_at_height
                .checked_add(matrix.width())
                .ok_or(FriProverError::OpeningBatchWidthOverflow)?;
            point_reductions.push((weighted_opening, rescale, denominators.as_slice()));
        }

        debug_span!("compress and reduce mat").in_scope(|| {
            matrix
                .rowwise_packed_dot_product::<Challenge>(&offset_alpha_powers)
                .zip(reduced_opening_for_log_height.values_mut().par_iter_mut())
                .enumerate()
                .for_each(|(row_index, (reduced_row, reduced_opening))| {
                    let Some((first, remaining)) = point_reductions.split_first() else {
                        return;
                    };
                    let first_denominator =
                        first.2.get(row_index).copied().unwrap_or(Challenge::ZERO);
                    *reduced_opening += (first.0 - reduced_row) * first_denominator;

                    for &(weighted_opening, rescale, denominators) in remaining {
                        let denominator = denominators
                            .get(row_index)
                            .copied()
                            .unwrap_or(Challenge::ZERO);
                        *reduced_opening +=
                            (weighted_opening - rescale * reduced_row) * denominator;
                    }
                });
        });
        Ok(())
    }

    fn finish(self) -> Vec<ReducedOpening> {
        self.reduced_openings
            .into_iter()
            .rev()
            .flatten()
            .collect_vec()
    }
}

#[cfg(feature = "secret-memory")]
impl<F> ReducedOpeningBuffer<F> for SecretProverData<RowMajorMatrix<F>>
where
    F: zeroize::Zeroize,
{
    fn len(&self) -> usize {
        self.values.len()
    }

    fn values_mut(&mut self) -> &mut [F] {
        self.values.as_mut_slice()
    }
}

type OpeningMatricesAndPoints<'a, Val, Challenge> =
    (Vec<RowMajorMatrixView<'a, Val>>, &'a [Vec<Challenge>]);

type OpeningDimensionsAndPoints<'a, Challenge> = (Vec<Dimensions>, &'a [Vec<Challenge>]);

struct OpeningClaimGeometry<Challenge> {
    inverse_denominators: LinearMap<Challenge, Vec<Challenge>>,
    adjusted_weights: LinearMap<Challenge, Vec<Challenge>>,
    global_max_width: usize,
    log_global_max_height: usize,
}

impl<Val, Dft, InputMmcs, FriMmcs> TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    fn prepare_opening_geometry<Challenge>(
        &self,
        dimensions_and_points: &[OpeningDimensionsAndPoints<'_, Challenge>],
    ) -> Result<OpeningClaimGeometry<Challenge>, FriProverError>
    where
        Val: TwoAdicField + PrimeField64,
        Challenge: ExtensionField<Val>,
    {
        for (batch, (dimensions, points)) in dimensions_and_points.iter().enumerate() {
            if dimensions.len() != points.len() {
                return Err(FriProverError::OpeningRequestShapeMismatch);
            }
            for (matrix, (matrix_dimensions, points_for_matrix)) in
                dimensions.iter().zip(*points).enumerate()
            {
                if matrix_dimensions.height == 0
                    || !matrix_dimensions.height.is_power_of_two()
                    || matrix_dimensions.width == 0
                {
                    return Err(FriProverError::OpeningRequestShapeMismatch);
                }
                self.fri
                    .validate_input_height(log2_strict_usize(matrix_dimensions.height))?;
                // The verifier cannot bind a matrix width when that matrix has
                // no claim. Reject this before transcript state or guarded
                // prover data can be consumed.
                if points_for_matrix.is_empty() {
                    return Err(FriProverError::MatrixWithoutOpeningPoints { batch, matrix });
                }
                if points_for_matrix.iter().any(Field::is_zero) {
                    return Err(FriProverError::OpeningPointUnsupported);
                }
            }
        }

        let (global_max_height, global_max_width) = dimensions_and_points
            .iter()
            .flat_map(|(dimensions, _)| {
                dimensions
                    .iter()
                    .map(|dimensions| (dimensions.height, dimensions.width))
            })
            .reduce(|(height_max, width_max), (height, width)| {
                (height_max.max(height), width_max.max(width))
            })
            .ok_or(FriProverError::EmptyOpeningBatch)?;
        let log_global_max_height = log2_strict_usize(global_max_height);
        let coset = {
            let domain = TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_global_max_height)
                .ok_or(FriProverError::OpeningDomainUnavailable)?;
            let mut points = Vec::new();
            points
                .try_reserve_exact(global_max_height)
                .map_err(|_| FriProverError::AllocationFailed)?;
            points.extend(domain.iter());
            reverse_slice_index_bits(&mut points);
            points
        };
        let inverse_denominators =
            compute_inverse_denominators_for_dimensions(dimensions_and_points, &coset);
        drop(coset);

        let mut adjusted_weights = LinearMap::new();
        for (point, denominators) in &inverse_denominators {
            let prefix_len = denominators.len() >> self.fri.log_blowup;
            let prefix = denominators
                .get(..prefix_len)
                .ok_or(FriProverError::OpeningWeightTableMismatch)?;
            adjusted_weights.insert(*point, compute_adjusted_weights(*point, prefix));
        }

        Ok(OpeningClaimGeometry {
            inverse_denominators,
            adjusted_weights,
            global_max_width,
            log_global_max_height,
        })
    }

    /// Interpolate claims and reduce every quotient into owned FRI codewords.
    ///
    /// The returned state deliberately contains no view into the committed matrices. This is the
    /// ownership boundary required by secret-memory proving: a guarded caller can release or detach
    /// the commitment data before FRI selects and authenticates query rows.
    fn prepare_opening<'challenger, Challenge, Challenger>(
        &self,
        mats_and_points: &[OpeningMatricesAndPoints<'_, Val, Challenge>],
        challenger: &'challenger mut Challenger,
    ) -> Result<
        PreparedPcsOpening<'challenger, Challenger, Val, Challenge, Vec<Challenge>>,
        FriProverError,
    >
    where
        Val: TwoAdicField + PrimeField64,
        Challenge: ExtensionField<Val>,
        FriMmcs: Mmcs<Challenge>,
        Challenger: FieldChallenger<Val>
            + CanObserve<FriMmcs::Commitment>
            + GrindingChallenger<Witness = Val>,
    {
        self.prepare_opening_with_reduced_buffer(
            mats_and_points,
            challenger,
            |height| Ok(Challenge::zero_vec(height)),
            |low_coset, point, adjusted| {
                Ok(
                    low_coset.interpolate_coset_with_precomputation(
                        Val::GENERATOR,
                        point,
                        adjusted,
                    ),
                )
            },
        )
    }

    fn prepare_opening_claims_with_interpolator<
        'challenger,
        Challenge,
        Challenger,
        PreparationError,
        InterpolateOpening,
    >(
        &self,
        mats_and_points: &[OpeningMatricesAndPoints<'_, Val, Challenge>],
        challenger: &'challenger mut Challenger,
        mut interpolate_opening: InterpolateOpening,
    ) -> Result<PreparedPcsClaims<'challenger, Challenger, Val, Challenge>, PreparationError>
    where
        Val: TwoAdicField + PrimeField64,
        Challenge: ExtensionField<Val>,
        FriMmcs: Mmcs<Challenge>,
        Challenger: FieldChallenger<Val>
            + CanObserve<FriMmcs::Commitment>
            + GrindingChallenger<Witness = Val>,
        PreparationError: From<FriProverError>,
        InterpolateOpening: for<'matrix> FnMut(
            RowMajorMatrixView<'matrix, Val>,
            Challenge,
            &[Challenge],
        ) -> Result<Vec<Challenge>, PreparationError>,
    {
        let mut dimensions_and_points = Vec::new();
        dimensions_and_points
            .try_reserve_exact(mats_and_points.len())
            .map_err(|_| FriProverError::AllocationFailed)?;
        for (matrices, points) in mats_and_points {
            let mut dimensions = Vec::new();
            dimensions
                .try_reserve_exact(matrices.len())
                .map_err(|_| FriProverError::AllocationFailed)?;
            dimensions.extend(matrices.iter().map(|matrix| matrix.dimensions()));
            dimensions_and_points.push((dimensions, *points));
        }
        let OpeningClaimGeometry {
            inverse_denominators,
            adjusted_weights,
            global_max_width,
            log_global_max_height,
        } = self.prepare_opening_geometry(&dimensions_and_points)?;
        drop(dimensions_and_points);

        let mut all_opened_values = Vec::new();
        all_opened_values
            .try_reserve_exact(mats_and_points.len())
            .map_err(|_| FriProverError::AllocationFailed)?;
        for (matrices, points) in mats_and_points {
            let mut openings_for_round = Vec::new();
            openings_for_round
                .try_reserve_exact(matrices.len())
                .map_err(|_| FriProverError::AllocationFailed)?;
            for (matrix, points_for_matrix) in izip!(matrices, *points) {
                let interpolation_height = matrix.height() >> self.fri.log_blowup;
                let (low_coset, _) = matrix.split_rows(interpolation_height);
                let mut openings_for_matrix = Vec::new();
                openings_for_matrix
                    .try_reserve_exact(points_for_matrix.len())
                    .map_err(|_| FriProverError::AllocationFailed)?;
                for &point in points_for_matrix {
                    let _guard =
                        debug_span!("evaluate matrix", dims = %matrix.dimensions()).entered();
                    let weights = adjusted_weights
                        .get(&point)
                        .ok_or(FriProverError::OpeningWeightTableMismatch)?;
                    let adjusted = weights
                        .get(..interpolation_height)
                        .ok_or(FriProverError::OpeningWeightTableMismatch)?;
                    let opening = debug_span!("compute opened values with Lagrange interpolation")
                        .in_scope(|| interpolate_opening(low_coset, point, adjusted))?;
                    openings_for_matrix.push(opening);
                }
                openings_for_round.push(openings_for_matrix);
            }
            all_opened_values.push(openings_for_round);
        }
        drop(adjusted_weights);

        let mut transcript = PcsProverTranscript::<Challenger, Val, Challenge>::new(
            challenger,
            PcsShape::from_opened_values(&self.fri, &all_opened_values),
        );
        transcript.claimed_openings(&all_opened_values);
        let (alpha, batch_pow_witness) = transcript.batch_phase();

        Ok(PreparedPcsClaims {
            all_opened_values,
            inverse_denominators,
            global_max_width,
            log_global_max_height,
            alpha,
            batch_pow_witness: batch_pow_witness.unwrap_or(Val::ZERO),
            transcript,
        })
    }

    /// Shared opening reduction whose storage policy is selected by the caller.
    ///
    /// The ordinary path supplies `Vec`; guarded proving supplies a matrix that
    /// is already under a zeroizing owner before quotient values are written.
    fn prepare_opening_with_reduced_buffer<
        'challenger,
        Challenge,
        Challenger,
        ReducedOpening,
        PreparationError,
        MakeReduced,
        InterpolateOpening,
    >(
        &self,
        mats_and_points: &[OpeningMatricesAndPoints<'_, Val, Challenge>],
        challenger: &'challenger mut Challenger,
        make_reduced: MakeReduced,
        interpolate_opening: InterpolateOpening,
    ) -> Result<
        PreparedPcsOpening<'challenger, Challenger, Val, Challenge, ReducedOpening>,
        PreparationError,
    >
    where
        Val: TwoAdicField + PrimeField64,
        Challenge: ExtensionField<Val>,
        FriMmcs: Mmcs<Challenge>,
        Challenger: FieldChallenger<Val>
            + CanObserve<FriMmcs::Commitment>
            + GrindingChallenger<Witness = Val>,
        ReducedOpening: ReducedOpeningBuffer<Challenge>,
        PreparationError: From<FriProverError>,
        MakeReduced: FnMut(usize) -> Result<ReducedOpening, PreparationError>,
        InterpolateOpening: for<'matrix> FnMut(
            RowMajorMatrixView<'matrix, Val>,
            Challenge,
            &[Challenge],
        ) -> Result<Vec<Challenge>, PreparationError>,
    {
        let PreparedPcsClaims {
            all_opened_values,
            inverse_denominators,
            global_max_width,
            log_global_max_height,
            alpha,
            batch_pow_witness,
            mut transcript,
        } = self.prepare_opening_claims_with_interpolator(
            mats_and_points,
            challenger,
            interpolate_opening,
        )?;

        // Now that we have sent the openings to the verifier, it remains to prove
        // that those openings are correct.

        // Given a low degree polynomial `f(x)` with claimed evaluation `f(zeta)`, we can check
        // that `f(zeta)` is correct by doing a low degree test on `(f(zeta) - f(x))/(zeta - x)`.
        // We will use `alpha` to batch together both different claimed openings `zeta` and
        // different polynomials `f` whose evaluation vectors have the same height.

        // TODO: If we allow different polynomials to have different blow_up factors
        // we may need to revisit this and to ensure it is safe to batch them together.

        // Every operation below is fallible while the PCS transcript is live.
        // Keep it in one result boundary so an error can release the transcript
        // completeness check before guarded reduced-opening owners unwind.
        let reduction_result = (|| -> Result<Vec<ReducedOpening>, PreparationError> {
            let mut reduction = OpeningReductionState::new(
                alpha,
                global_max_width,
                &inverse_denominators,
                make_reduced,
            )?;

            for ((mats, points), openings_for_round) in
                mats_and_points.iter().zip(all_opened_values.iter())
            {
                for (mat, points_for_mat, openings_for_mat) in
                    izip!(mats.iter(), points.iter(), openings_for_round.iter())
                {
                    reduction.reduce_matrix(*mat, points_for_mat, openings_for_mat)?;
                }
            }

            Ok(reduction.finish())
        })();
        let fri_input = match reduction_result {
            Ok(fri_input) => fri_input,
            Err(error) => {
                transcript.abort();
                return Err(error);
            }
        };

        Ok(PreparedPcsOpening {
            all_opened_values,
            fri_input,
            log_global_max_height,
            batch_pow_witness,
            transcript,
        })
    }
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs> TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: Mmcs<Val, MultiProof: Sync, Error: Sync> + SecretRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
{
    /// Construct a complete PCS opening proof without releasing a
    /// secret-derived codeword from zeroizing ownership.
    ///
    /// Opening geometry is validated from public dimensions first. The prover
    /// then detaches full private value leaves before evaluating disclosed
    /// claims directly from compact coefficients. Quotient reduction
    /// reconstructs one private matrix under a temporary zeroizing owner and
    /// erases it before reconstructing the next. The FRI driver later fixes
    /// its query indices before selected rows are authenticated.
    fn open_secret_mixed_guarded_inner<Challenge, Challenger>(
        &self,
        mut requests: SecretTwoAdicPcsMixedOpeningRequests<'_, Val, InputMmcs, Challenge>,
        challenger: &mut Challenger,
    ) -> SecretTwoAdicOpeningResult<
        Challenge,
        FriMmcs,
        Val,
        InputMmcs,
        InputMmcs::SecretError,
        FriMmcs::SecretError,
    >
    where
        Challenge: ExtensionField<Val> + zeroize::Zeroize,
        FriMmcs: Mmcs<Challenge> + SecretMmcs<Challenge>,
        Challenger: FieldChallenger<Val>
            + CanObserve<FriMmcs::Commitment>
            + GrindingChallenger<Witness = Val>,
    {
        let mut dimensions_and_points = Vec::new();
        dimensions_and_points
            .try_reserve_exact(requests.len())
            .map_err(|_| SecretFriOpenError::AllocationFailed)?;
        for request in &requests {
            let (dimensions, points) = match request {
                SecretPcsMixedOpeningRequest::Public(request) => {
                    let matrices = self.mmcs.get_matrices(request.prover_data);
                    let mut dimensions = Vec::new();
                    dimensions
                        .try_reserve_exact(matrices.len())
                        .map_err(|_| SecretFriOpenError::AllocationFailed)?;
                    dimensions.extend(matrices.iter().map(|matrix| matrix.dimensions()));
                    (dimensions, request.points.as_slice())
                }
                SecretPcsMixedOpeningRequest::Private(request) => {
                    if request.prover_data.log_blowup != self.fri.log_blowup {
                        return Err(FriProverError::OpeningRequestShapeMismatch.into());
                    }
                    let dimensions = request.prover_data.retained_lde_dimensions()?;
                    (dimensions, request.points.as_slice())
                }
            };
            dimensions_and_points.push((dimensions, points));
        }

        let OpeningClaimGeometry {
            inverse_denominators,
            adjusted_weights,
            global_max_width,
            log_global_max_height,
        } = self.prepare_opening_geometry(&dimensions_and_points)?;
        drop(dimensions_and_points);

        // Geometry and every opening request have been validated without
        // touching prover state. Compact coefficients plus authentication
        // nodes are now sufficient, so erase full private leaves before any
        // witness-derived claim interpolation or FRI reduction allocation.
        for request in &mut requests {
            if let SecretPcsMixedOpeningRequest::Private(request) = request
                && request.prover_data.authentication_data.is_none()
            {
                self.detach_commitment_data_secret(&mut *request.prover_data)
                    .map_err(SecretFriOpenError::InputDetachment)?;
            }
        }

        let mut all_opened_values = Vec::new();
        all_opened_values
            .try_reserve_exact(requests.len())
            .map_err(|_| SecretFriOpenError::AllocationFailed)?;
        for request in &requests {
            let matrix_count = match request {
                SecretPcsMixedOpeningRequest::Public(request) => request.points.len(),
                SecretPcsMixedOpeningRequest::Private(request) => request.points.len(),
            };
            let mut openings_for_round = Vec::new();
            openings_for_round
                .try_reserve_exact(matrix_count)
                .map_err(|_| SecretFriOpenError::AllocationFailed)?;

            match request {
                SecretPcsMixedOpeningRequest::Public(request) => {
                    let matrices = self.mmcs.get_matrices(request.prover_data);
                    if matrices.len() != request.points.len() {
                        return Err(FriProverError::OpeningRequestShapeMismatch.into());
                    }
                    for (matrix, points_for_matrix) in izip!(matrices, &request.points) {
                        let interpolation_height = matrix.height() >> self.fri.log_blowup;
                        let (low_coset, _) = matrix.split_rows(interpolation_height);
                        let mut openings_for_matrix = Vec::new();
                        openings_for_matrix
                            .try_reserve_exact(points_for_matrix.len())
                            .map_err(|_| SecretFriOpenError::AllocationFailed)?;
                        for &point in points_for_matrix {
                            let weights = adjusted_weights
                                .get(&point)
                                .ok_or(FriProverError::OpeningWeightTableMismatch)?;
                            let adjusted = weights
                                .get(..interpolation_height)
                                .ok_or(FriProverError::OpeningWeightTableMismatch)?;
                            let opening = low_coset
                                .interpolate_coset_with_precomputation_secret(
                                    Val::GENERATOR,
                                    point,
                                    adjusted,
                                )
                                .map_err(SecretFriOpenError::Matrix)?;
                            openings_for_matrix.push(opening);
                        }
                        openings_for_round.push(openings_for_matrix);
                    }
                }
                SecretPcsMixedOpeningRequest::Private(request) => {
                    if request.prover_data.coefficients.len() != request.points.len() {
                        return Err(FriProverError::OpeningRequestShapeMismatch.into());
                    }
                    for (coefficients, points_for_matrix) in
                        request.prover_data.coefficients.iter().zip(&request.points)
                    {
                        let mut openings_for_matrix = Vec::new();
                        openings_for_matrix
                            .try_reserve_exact(points_for_matrix.len())
                            .map_err(|_| SecretFriOpenError::AllocationFailed)?;
                        for &point in points_for_matrix {
                            openings_for_matrix.push(
                                Self::evaluate_coefficients_at_point_secret(coefficients, point)
                                    .map_err(SecretFriOpenError::Matrix)?,
                            );
                        }
                        openings_for_round.push(openings_for_matrix);
                    }
                }
            }
            all_opened_values.push(openings_for_round);
        }
        drop(adjusted_weights);

        let mut transcript = PcsProverTranscript::<Challenger, Val, Challenge>::new(
            challenger,
            PcsShape::from_opened_values(&self.fri, &all_opened_values),
        );
        transcript.claimed_openings(&all_opened_values);
        let (alpha, batch_pow_witness) = transcript.batch_phase();
        let batch_pow_witness = batch_pow_witness.unwrap_or(Val::ZERO);

        let reduction_result = (|| {
            let mut reduction = OpeningReductionState::new(
                alpha,
                global_max_width,
                &inverse_denominators,
                |height| {
                    let mut values = Vec::new();
                    values
                        .try_reserve_exact(height)
                        .map_err(|_| SecretMatrixError::AllocationFailed)?;
                    values.resize(height, Challenge::ZERO);
                    Ok::<_, SecretFriOpenError<InputMmcs::SecretError, FriMmcs::SecretError>>(
                        SecretProverData::new(RowMajorMatrix::new(values, 1)),
                    )
                },
            )?;

            for (round_index, request) in requests.iter().enumerate() {
                let openings_for_round = all_opened_values
                    .get(round_index)
                    .ok_or(FriProverError::OpeningRequestShapeMismatch)?;
                match request {
                    SecretPcsMixedOpeningRequest::Public(request) => {
                        let matrices = self.mmcs.get_matrices(request.prover_data);
                        if matrices.len() != request.points.len()
                            || matrices.len() != openings_for_round.len()
                        {
                            return Err(FriProverError::OpeningRequestShapeMismatch.into());
                        }
                        for (matrix, points, openings) in
                            izip!(matrices, &request.points, openings_for_round)
                        {
                            reduction.reduce_matrix(matrix.as_view(), points, openings)?;
                        }
                    }
                    SecretPcsMixedOpeningRequest::Private(request) => {
                        if request.points.len() != openings_for_round.len() {
                            return Err(FriProverError::OpeningRequestShapeMismatch.into());
                        }
                        for (matrix_index, (points, openings)) in
                            request.points.iter().zip(openings_for_round).enumerate()
                        {
                            let matrix = self
                                .recompute_lde_matrix_secret(&*request.prover_data, matrix_index)?;
                            reduction.reduce_matrix(matrix.as_view(), points, openings)?;
                        }
                    }
                }
            }
            Ok::<_, SecretFriOpenError<InputMmcs::SecretError, FriMmcs::SecretError>>(
                reduction.finish(),
            )
        })();
        let fri_input = match reduction_result {
            Ok(fri_input) => fri_input,
            Err(error) => {
                transcript.abort();
                return Err(error);
            }
        };

        let folding: TwoAdicFriFoldingForMmcs<Val, InputMmcs> = TwoAdicFriFolding(PhantomData);
        let fri_result = transcript.delegate(|challenger| {
            prover::prove_fri_secret_with_input_opener(
                &folding,
                &self.fri,
                fri_input,
                challenger,
                log_global_max_height,
                batch_pow_witness,
                |indices| {
                    let mut input_openings = Vec::new();
                    input_openings
                        .try_reserve_exact(requests.len())
                        .map_err(|_| SecretFriCommitError::AllocationFailed)?;

                    for request in &mut requests {
                        let opening = match request {
                            SecretPcsMixedOpeningRequest::Public(request) => {
                                let local_max_height = self
                                    .mmcs
                                    .get_matrices(request.prover_data)
                                    .iter()
                                    .map(|matrix| matrix.height())
                                    .max()
                                    .ok_or(SecretFriCommitError::EmptyRetainedCoefficients)?;
                                let local_log_height = log2_strict_usize(local_max_height);
                                let bits_reduced = log_global_max_height
                                    .checked_sub(local_log_height)
                                    .ok_or(SecretMatrixError::InvalidShape)?;
                                let mut reduced_indices = Vec::new();
                                reduced_indices
                                    .try_reserve_exact(indices.len())
                                    .map_err(|_| SecretFriCommitError::AllocationFailed)?;
                                reduced_indices
                                    .extend(indices.iter().map(|&index| index >> bits_reduced));
                                let (opened_values, opening_proof) = self
                                    .mmcs
                                    .open_multi_batch(&reduced_indices, request.prover_data);
                                BatchMultiOpening {
                                    opened_values,
                                    opening_proof,
                                }
                            }
                            SecretPcsMixedOpeningRequest::Private(request) => {
                                let local_max_height = request
                                    .prover_data
                                    .max_lde_height()
                                    .map_err(SecretFriCommitError::Matrix)?;
                                let local_log_height = log2_strict_usize(local_max_height);
                                let bits_reduced = log_global_max_height
                                    .checked_sub(local_log_height)
                                    .ok_or(SecretMatrixError::InvalidShape)?;
                                let mut reduced_indices = Vec::new();
                                reduced_indices
                                    .try_reserve_exact(indices.len())
                                    .map_err(|_| SecretFriCommitError::AllocationFailed)?;
                                reduced_indices
                                    .extend(indices.iter().map(|&index| index >> bits_reduced));
                                let opening = self.detach_and_open_multi_batch_secret(
                                    &reduced_indices,
                                    &mut *request.prover_data,
                                )?;
                                // Selected rows and their authentication proof are now
                                // public proof material. No later phase needs the compact
                                // coefficients, so erase them before retaining the output.
                                zeroize::Zeroize::zeroize(&mut request.prover_data.coefficients);
                                opening
                            }
                        };
                        input_openings.push(opening);
                    }

                    Ok(input_openings)
                },
            )
        });

        let fri_proof = match fri_result {
            Ok(proof) => proof,
            Err(error) => {
                transcript.abort();
                return Err(SecretFriOpenError::Fri(error));
            }
        };
        transcript
            .finish_secret()
            .map_err(SecretFriOpenError::Transcript)?;

        Ok((all_opened_values, fri_proof))
    }
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger> SecretPcsOpener<Challenge, Challenger>
    for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: Mmcs<Val, MultiProof: Sync, Error: Sync> + SecretRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
    FriMmcs: Mmcs<Challenge> + SecretMmcs<Challenge>,
    Challenge: ExtensionField<Val> + zeroize::Zeroize,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    type SecretOpenError = SecretTwoAdicPcsOpenError;

    fn open_secret(
        &self,
        requests: Vec<SecretOpeningRequest<'_, Self::SecretData, Challenge>>,
        challenger: &mut Challenger,
    ) -> Result<(OpenedValues<Challenge>, Self::Proof), Self::SecretOpenError> {
        let mut mixed_requests = Vec::new();
        mixed_requests
            .try_reserve_exact(requests.len())
            .map_err(|_| SecretTwoAdicPcsOpenError::AllocationFailed)?;
        mixed_requests.extend(
            requests
                .into_iter()
                .map(SecretPcsMixedOpeningRequest::Private),
        );
        self.open_secret_mixed_guarded_inner(mixed_requests, challenger)
            .map_err(SecretTwoAdicPcsOpenError::from)
    }
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger>
    SecretPcsMixedOpener<Challenge, Challenger> for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: Mmcs<Val, MultiProof: Sync, Error: Sync> + SecretRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
    FriMmcs: Mmcs<Challenge> + SecretMmcs<Challenge>,
    Challenge: ExtensionField<Val> + zeroize::Zeroize,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    fn open_secret_with_public_preprocessing(
        &self,
        requests: Vec<
            SecretPcsMixedOpeningRequest<'_, Self::ProverData, Self::SecretData, Challenge>,
        >,
        challenger: &mut Challenger,
    ) -> Result<(OpenedValues<Challenge>, Self::Proof), Self::SecretOpenError> {
        self.open_secret_mixed_guarded_inner(requests, challenger)
            .map_err(SecretTwoAdicPcsOpenError::from)
    }
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger>
    SecretUnivariateStarkPcsEvaluator<Challenge, Challenger>
    for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: Mmcs<Val, MultiProof: Sync, Error: Sync> + SecretRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
    FriMmcs: Mmcs<Challenge>,
    Challenge: ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    type SecretEvaluationError = SecretFriEvaluationError;

    fn get_evaluations_on_domain_secret(
        &self,
        prover_data: &SecretProverData<Self::SecretData>,
        idx: usize,
        domain: Self::Domain,
    ) -> Result<SecretProverData<RowMajorMatrix<Val>>, Self::SecretEvaluationError> {
        self.evaluate_on_domain_secret(prover_data, idx, domain)
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64,
    Dft: TwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: Mmcs<Val, MultiProof: Sync, Error: Sync>,
    FriMmcs: Mmcs<Challenge>,
    Challenge: ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;
    type Proof = FriProof<Challenge, FriMmcs, Val, Vec<BatchMultiOpening<Val, InputMmcs>>>;
    type Error = FriError<FriMmcs::Error, InputMmcs::Error>;
    type ProverError = FriProverError;

    /// Get the unique subgroup `H` of size `|H| = degree`.
    ///
    /// # Panics:
    /// This function will panic if `degree` is not a power of 2 or `degree > (1 << Val::TWO_ADICITY)`.
    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        TwoAdicMultiplicativeCoset::new(Val::ONE, log2_strict_usize(degree)).unwrap()
    }

    /// Commit to a collection of evaluation matrices.
    ///
    /// Each element of `evaluations` contains a coset `shift * H` and a matrix `mat` with `mat.height() = |H|`.
    /// Interpreting each column of `mat` as the evaluations of a polynomial `p_i(x)` over `shift * H`,
    /// this computes the evaluations of `p_i` over `gK` where `g` is the chosen generator of the multiplicative group
    /// of `Val` and `K` is the unique subgroup of order `|H| << self.fri.log_blowup`.
    ///
    /// This then outputs a Merkle commitment to these evaluations.
    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
        let ldes = self.coset_ldes(evaluations);

        Ok(
            // Commit to the bit-reversed LDEs.
            self.mmcs.commit(ldes),
        )
    }

    /// Open a batch of matrices at a collection of points.
    ///
    /// Returns the opened values along with a proof.
    ///
    /// This function assumes that all matrices correspond to evaluations over the
    /// coset `gH` where `g = Val::GENERATOR` and `H` is a subgroup of appropriate size depending on the
    /// matrix.
    fn open(
        &self,
        // For each multi-matrix commitment,
        commitment_data_with_opening_points: Vec<OpeningRequest<'_, Self::ProverData, Challenge>>,
        challenger: &mut Challenger,
    ) -> Result<(OpenedValues<Challenge>, Self::Proof), Self::ProverError> {
        // Contained in each `Self::ProverData` is a list of matrices which have been committed to.
        // We extract those matrices to be able to refer to them directly.
        let mats_and_points = commitment_data_with_opening_points
            .iter()
            .map(
                |OpeningRequest {
                     prover_data: data,
                     points,
                 }| {
                    let mats = self
                        .mmcs
                        .get_matrices(data)
                        .into_iter()
                        .map(|m| m.as_view())
                        .collect_vec();
                    (mats, points.as_slice())
                },
            )
            .collect_vec();

        let PreparedPcsOpening {
            all_opened_values,
            fri_input,
            log_global_max_height,
            batch_pow_witness,
            mut transcript,
        } = self.prepare_opening(&mats_and_points, challenger)?;

        // Quotient reduction owns everything FRI needs. End every immutable matrix borrow before
        // query selection so the guarded path can detach and erase commitment data at this point.
        drop(mats_and_points);

        let folding: TwoAdicFriFoldingForMmcs<Val, InputMmcs> = TwoAdicFriFolding(PhantomData);

        // Produce the FRI proof, bracketed as a sub-protocol of this transcript.
        //
        // FRI seeds its own description from the sponge state reached here.
        let fri_proof = transcript.delegate(|challenger| {
            #[cfg(test)]
            {
                prover::prove_fri_with_schedule(
                    &folding,
                    &self.fri,
                    fri_input,
                    challenger,
                    log_global_max_height,
                    &commitment_data_with_opening_points,
                    &self.mmcs,
                    batch_pow_witness,
                    self.forged_fold_schedule.clone(),
                )
            }
            #[cfg(not(test))]
            {
                prover::prove_fri(
                    &folding,
                    &self.fri,
                    fri_input,
                    challenger,
                    log_global_max_height,
                    &commitment_data_with_opening_points,
                    &self.mmcs,
                    batch_pow_witness,
                )
            }
        });

        // A rejected run played fewer steps than it described.
        //
        // Release the driver rather than close it, since no proof is emitted.
        let fri_proof = match fri_proof {
            Ok(fri_proof) => fri_proof,
            Err(error) => {
                transcript.abort();
                return Err(error);
            }
        };

        // Every described step has now been played.
        transcript.finish();

        Ok((all_opened_values, fri_proof))
    }

    fn verify(
        &self,
        // For each commitment:
        commitments_with_opening_points: Vec<
            CommitmentOpening<Challenge, Self::Commitment, Self::Domain>,
        >,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        // A zero grinding budget leaves the witness unread, so every field is pinned here
        // rather than by its grind.
        //
        //     bits = 0 -> prover emits zero, verifier reads nothing -> pin the field here
        //     bits > 0 -> prover grinds,     verifier resamples     -> the grind pins it
        //
        // The FRI verifier repeats it, since it can also be called on its own.
        verifier::check_canonical_pow_witnesses(&self.fri, proof)?;

        // Reject a grinding difficulty the base field cannot sample, before the batching grind.
        //
        // The FRI verifier repeats it, since it can also be called on its own.
        self.fri.check_pow_bits::<Challenger, _, _>()?;

        // Describe the transcript from the claims, which are this verifier's own input.
        //
        // The prover built the identical description from the matrices it opened.
        let mut transcript = PcsVerifierTranscript::<Challenger, Val, Challenge>::new(
            challenger,
            PcsShape::from_claims(&self.fri, &commitments_with_opening_points),
        );

        // Replay every claimed evaluation, in commitment, matrix and point order.
        //
        // A failure poisons the driver on its way out, so dropping it here is silent.
        transcript.claimed_openings(&commitments_with_opening_points)?;

        // Replay the grind and redraw the challenge that batches the claims.
        let alpha = transcript.batch_phase(Some(proof.batch_pow_witness))?;

        let folding: TwoAdicFriFoldingForMmcs<Val, InputMmcs> = TwoAdicFriFolding(PhantomData);

        // Run the low-degree test inside the bracket, lending it the sponge.
        //
        // Invariant: the bracket closes whichever way the delegated run goes.
        //
        //     delegate  -> Begin, run, End      (End is recorded on any outcome)
        //     finish    -> every described step replayed
        //     rejection -> propagated afterwards, never across an unfinished driver
        let result = transcript.delegate(|challenger| {
            verifier::verify_fri(
                &folding,
                &self.fri,
                proof,
                challenger,
                &commitments_with_opening_points,
                &self.mmcs,
                alpha,
            )
        });

        // Every described step has now been replayed.
        transcript.finish();

        result
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger> UnivariateStarkPcs<Challenge, Challenger>
    for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64,
    Dft: TwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: Mmcs<Val, MultiProof: Sync, Error: Sync>,
    FriMmcs: Mmcs<Challenge>,
    Challenge: ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixCow<'a, Val>>;

    const ZK: bool = false;

    fn log_max_trace_height(&self) -> usize {
        Val::TWO_ADICITY
    }

    fn log_min_trace_height(&self) -> usize {
        // Multiplicative coset selectors are defined at every height, down to a single row.
        0
    }

    fn get_quotient_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
        _num_chunks: usize,
    ) -> Result<Vec<RowMajorMatrix<Val>>, Self::ProverError> {
        Ok(self.coset_ldes(evaluations))
    }

    fn commit_ldes(
        &self,
        ldes: Vec<RowMajorMatrix<Val>>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
        // Opening assumes every committed matrix is an LDE at `self.fri.log_blowup` and recovers the
        // underlying polynomial degree as `height >> log_blowup`. A matrix shorter than the blowup
        // factor would silently yield a zero-height degree and a malformed proof, so reject it here.
        let min_height = 1 << self.fri.log_blowup;
        for lde in &ldes {
            assert!(
                lde.height() >= min_height,
                "committed LDE height {} is smaller than the blowup factor {min_height}",
                lde.height()
            );
        }
        Ok(self.mmcs.commit(ldes))
    }

    /// Given the evaluations on a domain `gH`, return the evaluations on a different domain `g'K`.
    ///
    /// Arguments:
    /// - `prover_data`: The prover data containing all committed evaluation matrices.
    /// - `idx`: The index of the matrix containing the evaluations we want. These evaluations
    ///   are assumed to be over the coset `gH` where `g = Val::GENERATOR`.
    /// - `domain`: The domain `g'K` on which to get evaluations on.
    ///
    /// When `g' = g` (i.e. `Val::GENERATOR`) and `K` is a subgroup of `H`, this is a simple
    /// truncation of the bit-reversed LDE. Otherwise, we recover the polynomial coefficients
    /// from the committed LDE and re-evaluate on the requested domain.
    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let lde = self.mmcs.get_matrices(prover_data)[idx];
        if domain.shift() == Val::GENERATOR && lde.height() >= domain.size() {
            return lde.split_rows(domain.size()).0.as_cow().bit_reverse_rows();
        }

        // The committed LDE contains bit-reversed evaluations over `gH`. Its first
        // `poly_height` rows are exactly the bit-reversed evaluations over the
        // order-`poly_height` subgroup `gK` (bit-reversing an index below `poly_height`
        // within the larger domain equals bit-reversing it within the smaller one and
        // shifting the result up by `log_blowup` zero bits). Un-bit-reversing that prefix
        // and taking a `poly_height`-sized coset iDFT therefore recovers the polynomial's
        // coefficients directly, without touching the rest of the LDE.
        let poly_height = lde.height() >> self.fri.log_blowup;
        let lde_mat = lde
            .split_rows(poly_height)
            .0
            .bit_reverse_rows()
            .to_row_major_matrix();
        let mut coeffs = self.dft.coset_idft_batch(lde_mat, Val::GENERATOR);
        let width = coeffs.width();
        coeffs.values.resize(domain.size() * width, Val::ZERO);
        let result = self
            .dft
            .coset_dft_batch(coeffs, domain.shift())
            .bit_reverse_rows()
            .to_row_major_matrix();
        let result_width = result.width();

        RowMajorMatrixCow::new(Cow::Owned(result.values), result_width).bit_reverse_rows()
    }

    fn build_periodic_lde_table(
        &self,
        periodic_cols: &[Vec<Val>],
        trace_domain: Self::Domain,
        quotient_domain: Self::Domain,
    ) -> PeriodicLdeTable<Val> {
        build_periodic_lde_table_two_adic(&self.dft, periodic_cols, &trace_domain, &quotient_domain)
    }
}

/// Compute vectors of inverse denominators for each unique opening point.
///
/// Arguments:
/// - `dimensions_and_points` is a list of matrix dimensions and for each matrix a list of points. We assume that
///    the total number of distinct points is very small as several methods contained herein are `O(n^2)`
///    in the number of points.
/// - `coset` is the set of points `gH` where `H` a two-adic subgroup such that `|H|` is greater
///     than or equal to the largest requested matrix height. The values
///     in `coset` must be in bit-reversed order.
///
/// For each point `z`, let `M` be the matrix of largest height which opens at `z`.
/// let `H_z` be the unique subgroup of order `M.height()`. Compute the vector of
/// `1/(z - x)` for `x` in `gH_z`.
///
/// Return a LinearMap which allows us to recover the computed vectors for each `z`.
#[instrument(skip_all)]
fn compute_inverse_denominators_for_dimensions<F: TwoAdicField, EF: ExtensionField<F>>(
    dimensions_and_points: &[OpeningDimensionsAndPoints<'_, EF>],
    coset: &[F],
) -> LinearMap<EF, Vec<EF>> {
    // For each `z`, find the maximal height of any matrix which we need to
    // open at `z`.
    let mut max_log_height_for_point: LinearMap<EF, usize> = LinearMap::new();
    for (dimensions, points) in dimensions_and_points {
        for (matrix_dimensions, points_for_mat) in izip!(dimensions, *points) {
            let log_height = log2_strict_usize(matrix_dimensions.height);
            for &z in points_for_mat {
                if let Some(lh) = max_log_height_for_point.get_mut(&z) {
                    *lh = core::cmp::max(*lh, log_height);
                } else {
                    max_log_height_for_point.insert(z, log_height);
                }
            }
        }
    }

    // Compute the inverse denominators for each point `z`.
    max_log_height_for_point
        .into_iter()
        .map(|(z, log_height)| {
            // As coset is stored in bit-reversed order, the prefix is the right subgroup.
            let xs = &coset[..(1 << log_height)];
            let mut inverses = EF::zero_vec(xs.len());
            batch_multiplicative_inverse_map(xs, |&x| z - x, &mut inverses);
            (z, inverses)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    #[cfg(feature = "secret-memory")]
    use p3_commit::{
        SecretDetachedPcsCommitter, SecretMmcs, SecretPcsCommitter, SecretProverData,
        SecretUnivariateStarkPcsCommitter, SecretUnivariateStarkPcsEvaluator,
    };
    #[cfg(feature = "secret-memory")]
    use p3_dft::Radix2Bowers;
    use p3_dft::Radix2Dit;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrixView;
    #[cfg(feature = "secret-memory")]
    use p3_merkle_tree::MerkleTreeHidingMmcs;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    #[cfg(feature = "secret-memory")]
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::verifier::PowPhase;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;
    type MyPcs = TwoAdicFriPcs<F, Radix2Dit<F>, ValMmcs, ChallengeMmcs>;
    type Commitment = <ValMmcs as Mmcs<F>>::Commitment;
    type Domain = TwoAdicMultiplicativeCoset<F>;
    type Claims = Vec<CommitmentWithOpeningPoints<EF, Commitment, Domain>>;
    type MyProof = <MyPcs as Pcs<EF, Challenger>>::Proof;
    type TestError = FriError<<ChallengeMmcs as Mmcs<EF>>::Error, <ValMmcs as Mmcs<F>>::Error>;

    // A wrong coset prefix, reused block offset, sign, or per-point height must disagree
    // with independently inverted elements of the point's own subgroup.
    #[test]
    fn inverse_denominators_match_individual_inverses() {
        fn check<Base: TwoAdicField, Ext: ExtensionField<Base>>()
        where
            StandardUniform: Distribution<Ext>,
        {
            let mut rng = SmallRng::seed_from_u64(23);
            let log_heights = [0, 1, 2, 3, 9, 10, 11, 12];
            for shift in [Base::ONE, Base::GENERATOR, Base::GENERATOR.square()] {
                let mut coset = Base::two_adic_generator(12)
                    .shifted_powers(shift)
                    .take(1 << 12)
                    .collect_vec();
                reverse_slice_index_bits(&mut coset);
                let points = (0..=log_heights.len())
                    .map(|_| {
                        loop {
                            let z: Ext = rng.random();
                            if coset.iter().all(|&x| z != Ext::from(x)) {
                                break z;
                            }
                        }
                    })
                    .collect_vec();
                let point_lists = log_heights
                    .iter()
                    .enumerate()
                    .map(|(i, _)| vec![points[i], points[8], points[8]])
                    .collect_vec();
                let first_points = point_lists[..4].to_vec();
                let last_points = point_lists[4..].to_vec();
                let dimensions = log_heights
                    .iter()
                    .map(|&log_height| Dimensions {
                        width: 1,
                        height: 1 << log_height,
                    })
                    .collect_vec();
                let rounds = vec![
                    (dimensions[..4].to_vec(), first_points.as_slice()),
                    (dimensions[4..].to_vec(), last_points.as_slice()),
                ];
                let inverses = compute_inverse_denominators_for_dimensions(&rounds, &coset);
                for (i, &z) in points.iter().enumerate() {
                    let log_height = log_heights[i.min(7)];
                    let expected = (0..1 << log_height)
                        .map(|j| {
                            let x = shift
                                * Base::two_adic_generator(log_height)
                                    .exp_u64(reverse_bits_len(j, log_height) as u64);
                            (z - x).inverse()
                        })
                        .collect_vec();
                    assert_eq!(inverses.get(&z), Some(&expected));
                }
                let keys = inverses.into_iter().map(|(z, _)| z).collect_vec();
                let expected_keys = [points[0], points[8]]
                    .into_iter()
                    .chain(points[1..8].iter().copied())
                    .collect_vec();
                assert_eq!(keys, expected_keys);
            }
        }
        check::<F, EF>();
        check::<F, F>();
        check::<Goldilocks, BinomialExtensionField<Goldilocks, 2>>();
    }

    #[test]
    #[should_panic(expected = "Tried to invert zero")]
    fn inverse_denominators_reject_zero_in_a_small_domain() {
        let points = vec![vec![EF::ONE]];
        let rounds = vec![(
            vec![Dimensions {
                width: 1,
                height: 1,
            }],
            points.as_slice(),
        )];
        compute_inverse_denominators_for_dimensions(&rounds, &[F::ONE]);
    }

    #[test]
    #[should_panic(expected = "Tried to invert zero")]
    fn inverse_denominators_reject_zero_in_a_later_block() {
        let coset = F::two_adic_generator(11)
            .shifted_powers(F::GENERATOR)
            .take(1 << 11)
            .collect_vec();
        let points = vec![vec![EF::from(coset[1025])]];
        let rounds = vec![(
            vec![Dimensions {
                width: 1,
                height: 1 << 11,
            }],
            points.as_slice(),
        )];
        compute_inverse_denominators_for_dimensions(&rounds, &coset);
    }

    #[cfg(feature = "secret-memory")]
    type SecretCommitPcs = TwoAdicFriPcs<F, Radix2Bowers, ValMmcs, ChallengeMmcs>;

    #[cfg(feature = "secret-memory")]
    type HidingValMmcs = MerkleTreeHidingMmcs<
        <F as Field>::Packing,
        <F as Field>::Packing,
        MyHash,
        MyCompress,
        StdRng,
        2,
        8,
        4,
    >;

    #[cfg(feature = "secret-memory")]
    type SecretHidingMerklePcs = TwoAdicFriPcs<F, Radix2Bowers, HidingValMmcs, ChallengeMmcs>;

    #[cfg(feature = "secret-memory")]
    fn make_secret_commit_pcs() -> SecretCommitPcs {
        let mut rng = SmallRng::seed_from_u64(0x5345_4352_4554_5043);
        let permutation = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(permutation.clone());
        let compress = MyCompress::new(permutation);
        let val_mmcs = ValMmcs::new(hash.clone(), compress.clone(), 0);
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress, 0));
        SecretCommitPcs::new(
            Radix2Bowers,
            val_mmcs,
            FriParameters {
                log_blowup: 1,
                log_final_poly_len: 0,
                max_log_arity: 1,
                num_queries: 2,
                batch_proof_of_work_bits: 0,
                commit_proof_of_work_bits: 0,
                query_proof_of_work_bits: 0,
                mmcs: challenge_mmcs,
            },
        )
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_detached_commit_matches_retained_commit_and_opens() {
        const DOMAIN_SIZE: usize = 8;
        let pcs = make_secret_commit_pcs();
        let mut permutation_rng = SmallRng::seed_from_u64(0x5345_4352_4554_5043);
        let permutation = Perm::new_from_rng_128(&mut permutation_rng);
        let domain =
            <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, DOMAIN_SIZE);
        let matrices = [2, 3].map(|width| {
            let value_count = DOMAIN_SIZE
                .checked_mul(width)
                .expect("fixed test dimensions must fit");
            RowMajorMatrix::new(
                (0..value_count)
                    .map(|index| F::from_usize(index + 1))
                    .collect_vec(),
                width,
            )
        });

        let retained_evaluations = matrices
            .clone()
            .into_iter()
            .map(|matrix| (domain, SecretProverData::new(matrix)))
            .collect();
        let (retained_commitment, mut retained_data) =
            <SecretCommitPcs as SecretPcsCommitter<EF, Challenger>>::commit_secret(
                &pcs,
                retained_evaluations,
            )
            .expect("valid retained commitment must succeed");
        let detached_evaluations = matrices
            .into_iter()
            .map(|matrix| (domain, SecretProverData::new(matrix)))
            .collect();
        let (detached_commitment, mut detached_data) =
            <SecretCommitPcs as SecretDetachedPcsCommitter<EF, Challenger>>::commit_detached_secret(
                &pcs,
                detached_evaluations,
            )
            .expect("valid detached commitment must succeed");

        assert_eq!(detached_commitment, retained_commitment);
        assert!(retained_data.authentication_data.is_none());
        assert!(detached_data.authentication_data.is_some());
        assert!(
            pcs.mmcs
                .get_matrices_secret(&detached_data.commitment_data)
                .is_empty()
        );

        let mut retained_challenger = Challenger::new(permutation.clone());
        retained_challenger.observe(&retained_commitment);
        let retained_zeta: EF = retained_challenger.sample_algebra_element();
        let retained_opening = <SecretCommitPcs as SecretPcsOpener<EF, Challenger>>::open_secret(
            &pcs,
            vec![
                (
                    &mut retained_data,
                    vec![vec![retained_zeta], vec![retained_zeta]],
                )
                    .into(),
            ],
            &mut retained_challenger,
        )
        .expect("retained commitment opening must succeed");

        let mut detached_challenger = Challenger::new(permutation.clone());
        detached_challenger.observe(&detached_commitment);
        let detached_zeta: EF = detached_challenger.sample_algebra_element();
        assert_eq!(detached_zeta, retained_zeta);
        let detached_opening = <SecretCommitPcs as SecretPcsOpener<EF, Challenger>>::open_secret(
            &pcs,
            vec![
                (
                    &mut detached_data,
                    vec![vec![detached_zeta], vec![detached_zeta]],
                )
                    .into(),
            ],
            &mut detached_challenger,
        )
        .expect("detached commitment opening must succeed");
        assert_eq!(detached_opening.0, retained_opening.0);
        assert_eq!(
            detached_challenger.sample_algebra_element::<EF>(),
            retained_challenger.sample_algebra_element::<EF>()
        );

        let claims: Claims = vec![
            (
                detached_commitment,
                detached_opening.0[0]
                    .iter()
                    .map(|matrix| (domain, vec![(detached_zeta, matrix[0].clone())]))
                    .collect(),
            )
                .into(),
        ];
        let mut verifier_challenger = Challenger::new(permutation);
        verifier_challenger.observe(&claims[0].commitment);
        let verifier_zeta: EF = verifier_challenger.sample_algebra_element();
        assert_eq!(verifier_zeta, detached_zeta);
        <SecretCommitPcs as Pcs<EF, Challenger>>::verify(
            &pcs,
            claims,
            &detached_opening.1,
            &mut verifier_challenger,
        )
        .expect("ordinary PCS verifier must accept the detached opening");
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_domain_evaluation_matches_ordinary_pcs_on_generator_and_disjoint_cosets() {
        const DOMAIN_SIZE: usize = 8;
        const WIDTH: usize = 2;
        let pcs = make_secret_commit_pcs();
        let domain =
            <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, DOMAIN_SIZE);
        let matrix = RowMajorMatrix::new((1_u8..=16).map(F::from_u8).collect::<Vec<_>>(), WIDTH);
        let (_, ordinary_data) =
            <SecretCommitPcs as Pcs<EF, Challenger>>::commit(&pcs, [(domain, matrix.clone())])
                .expect("valid ordinary commitment must succeed");
        let (_, guarded_data) =
            <SecretCommitPcs as SecretPcsCommitter<EF, Challenger>>::commit_secret(
                &pcs,
                vec![(domain, SecretProverData::new(matrix))],
            )
            .expect("valid guarded commitment must succeed");

        let smaller_generator_domain = TwoAdicMultiplicativeCoset::new(F::GENERATOR, 2)
            .expect("test smaller generator domain must be valid");
        let generator_domain = TwoAdicMultiplicativeCoset::new(F::GENERATOR, 3)
            .expect("test generator domain must be valid");
        let disjoint_domain = domain.create_disjoint_domain(
            DOMAIN_SIZE
                .checked_mul(4)
                .expect("fixed test domain size must fit"),
        );
        for target_domain in [smaller_generator_domain, generator_domain, disjoint_domain] {
            let expected =
                <SecretCommitPcs as UnivariateStarkPcs<EF, Challenger>>::get_evaluations_on_domain(
                    &pcs,
                    &ordinary_data,
                    0,
                    target_domain,
                )
                .to_row_major_matrix();
            let actual = <SecretCommitPcs as SecretUnivariateStarkPcsEvaluator<
                EF,
                Challenger,
            >>::get_evaluations_on_domain_secret(&pcs, &guarded_data, 0, target_domain)
            .expect("valid guarded domain evaluation must succeed");

            assert_eq!(actual.width, expected.width);
            assert_eq!(actual.values, expected.values);
        }
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_split_coset_evaluations_are_exact_interleaved_full_domain_rows() {
        const SOURCE_SIZE: usize = 16;
        const TARGET_SIZE: usize = 64;
        const NUM_CHUNKS: usize = 8;
        const WIDTH: usize = 3;

        let pcs = make_secret_commit_pcs();
        let source_domain =
            <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, SOURCE_SIZE);
        let source_len = SOURCE_SIZE
            .checked_mul(WIDTH)
            .expect("fixed test dimensions must fit");
        let source = RowMajorMatrix::new(
            (0..source_len)
                .map(|index| F::from_usize(index + 1))
                .collect_vec(),
            WIDTH,
        );
        let (_, guarded_data) =
            <SecretCommitPcs as SecretPcsCommitter<EF, Challenger>>::commit_secret(
                &pcs,
                vec![(source_domain, SecretProverData::new(source))],
            )
            .expect("valid guarded commitment must succeed");
        let full_domain = source_domain.create_disjoint_domain(TARGET_SIZE);
        let full = <SecretCommitPcs as SecretUnivariateStarkPcsEvaluator<EF, Challenger>>::
            get_evaluations_on_domain_secret(&pcs, &guarded_data, 0, full_domain)
            .expect("full guarded domain evaluation must succeed");

        for (chunk_index, chunk_domain) in full_domain
            .split_domains(NUM_CHUNKS)
            .into_iter()
            .enumerate()
        {
            let chunk = <SecretCommitPcs as SecretUnivariateStarkPcsEvaluator<
                EF,
                Challenger,
            >>::get_evaluations_on_domain_secret(&pcs, &guarded_data, 0, chunk_domain)
            .expect("split guarded domain evaluation must succeed");
            assert_eq!(chunk.width, WIDTH);
            assert_eq!(chunk.height(), TARGET_SIZE / NUM_CHUNKS);

            for local_row in 0..chunk.height() {
                let full_row = local_row
                    .checked_mul(NUM_CHUNKS)
                    .and_then(|row| row.checked_add(chunk_index))
                    .expect("fixed test row mapping must fit");
                for column in 0..WIDTH {
                    assert_eq!(chunk.get(local_row, column), full.get(full_row, column));
                }
            }
        }
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_domain_evaluation_rejects_invalid_index_and_mismatched_blowup() {
        const DOMAIN_SIZE: usize = 8;
        let pcs = make_secret_commit_pcs();
        let domain =
            <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, DOMAIN_SIZE);
        let matrix = RowMajorMatrix::new(vec![F::ONE; DOMAIN_SIZE], 1);
        let (_, mut guarded_data) =
            <SecretCommitPcs as SecretPcsCommitter<EF, Challenger>>::commit_secret(
                &pcs,
                vec![(domain, SecretProverData::new(matrix))],
            )
            .expect("valid guarded commitment must succeed");

        let invalid_index = <SecretCommitPcs as SecretUnivariateStarkPcsEvaluator<
            EF,
            Challenger,
        >>::get_evaluations_on_domain_secret(
            &pcs, &guarded_data, 1, domain
        );
        assert!(matches!(
            invalid_index,
            Err(SecretFriEvaluationError::MatrixIndexOutOfBounds {
                index: 1,
                matrix_count: 1
            })
        ));

        guarded_data.log_blowup = guarded_data
            .log_blowup
            .checked_add(1)
            .expect("fixed test blowup increment must fit");
        let mismatched_blowup = <SecretCommitPcs as SecretUnivariateStarkPcsEvaluator<
            EF,
            Challenger,
        >>::get_evaluations_on_domain_secret(
            &pcs, &guarded_data, 0, domain
        );
        assert!(matches!(
            mismatched_blowup,
            Err(SecretFriEvaluationError::RetainedBlowupMismatch)
        ));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn retained_lde_dimensions_rejects_zero_width_without_panicking() {
        const DOMAIN_SIZE: usize = 8;
        let pcs = make_secret_commit_pcs();
        let domain =
            <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, DOMAIN_SIZE);
        let matrix = RowMajorMatrix::new(vec![F::ONE; DOMAIN_SIZE], 1);
        let (_, mut prover_data) =
            <SecretCommitPcs as SecretPcsCommitter<EF, Challenger>>::commit_secret(
                &pcs,
                vec![(domain, SecretProverData::new(matrix))],
            )
            .expect("valid guarded commitment must succeed");
        prover_data.coefficients[0].width = 0;

        assert!(matches!(
            prover_data.retained_lde_dimensions(),
            Err(SecretMatrixError::InvalidShape)
        ));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_coefficient_opening_matches_independent_coset_interpolation() {
        const DOMAIN_SIZE: usize = 8;
        const WIDTH: usize = 3;

        let pcs = make_secret_commit_pcs();
        let domain =
            <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, DOMAIN_SIZE);
        let matrix = RowMajorMatrix::new(
            (0..DOMAIN_SIZE * WIDTH)
                .map(|value| F::from_usize(value + 1))
                .collect_vec(),
            WIDTH,
        );
        let (_, guarded_data) =
            <SecretCommitPcs as SecretPcsCommitter<EF, Challenger>>::commit_secret(
                &pcs,
                vec![(domain, SecretProverData::new(matrix.clone()))],
            )
            .expect("valid guarded commitment must succeed");
        let coefficients = guarded_data
            .coefficients
            .first()
            .expect("test commitment must retain one coefficient matrix");

        for point_value in [31_u16, 127_u16, 1021_u16] {
            let point = EF::from(F::from_u16(point_value));
            let expected = matrix.interpolate_coset(domain.shift(), point);
            let actual =
                SecretCommitPcs::evaluate_coefficients_at_point_secret(coefficients, point)
                    .expect("valid retained coefficients must evaluate");
            assert_eq!(actual, expected);
        }
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_pcs_opening_constructs_a_proof_accepted_by_the_ordinary_verifier() {
        const DOMAIN_SIZE: usize = 8;
        let pcs = make_secret_commit_pcs();
        let mut permutation_rng = SmallRng::seed_from_u64(0x5345_4352_4554_5043);
        let permutation = Perm::new_from_rng_128(&mut permutation_rng);
        let domain =
            <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, DOMAIN_SIZE);

        let mut trace_rng = SmallRng::seed_from_u64(0x4755_4152_4445_4450);
        let matrices = [2, 3]
            .map(|width| RowMajorMatrix::<F>::rand_nonzero(&mut trace_rng, DOMAIN_SIZE, width));
        let (ordinary_commitment, ordinary_data) =
            <SecretCommitPcs as Pcs<EF, Challenger>>::commit(
                &pcs,
                matrices.clone().into_iter().map(|matrix| (domain, matrix)),
            )
            .expect("valid ordinary commitment must succeed");
        let guarded_evaluations = matrices
            .into_iter()
            .map(|matrix| (domain, SecretProverData::new(matrix)))
            .collect();
        let (commitment, mut prover_data) = <SecretCommitPcs as SecretPcsCommitter<
            EF,
            Challenger,
        >>::commit_secret(&pcs, guarded_evaluations)
        .expect("valid guarded commitment must succeed");
        assert_eq!(commitment, ordinary_commitment);

        let mut ordinary_challenger = Challenger::new(permutation.clone());
        ordinary_challenger.observe(&ordinary_commitment);
        let ordinary_zeta: EF = ordinary_challenger.sample_algebra_element();
        let (ordinary_opened_values, ordinary_proof) =
            <SecretCommitPcs as Pcs<EF, Challenger>>::open(
                &pcs,
                vec![
                    (
                        &ordinary_data,
                        vec![vec![ordinary_zeta], vec![ordinary_zeta]],
                    )
                        .into(),
                ],
                &mut ordinary_challenger,
            )
            .expect("valid ordinary opening must succeed");

        let mut prover_challenger = Challenger::new(permutation.clone());
        prover_challenger.observe(&commitment);
        let zeta: EF = prover_challenger.sample_algebra_element();
        assert_eq!(zeta, ordinary_zeta);
        let (opened_values, proof) =
            <SecretCommitPcs as SecretPcsOpener<EF, Challenger>>::open_secret(
                &pcs,
                vec![(&mut prover_data, vec![vec![zeta], vec![zeta]]).into()],
                &mut prover_challenger,
            )
            .expect("guarded PCS opening must succeed");
        assert_eq!(opened_values, ordinary_opened_values);
        assert_eq!(proof.batch_pow_witness, ordinary_proof.batch_pow_witness);
        assert_eq!(
            proof.commit_phase_commits,
            ordinary_proof.commit_phase_commits
        );
        assert_eq!(
            proof.commit_pow_witnesses,
            ordinary_proof.commit_pow_witnesses
        );
        assert_eq!(proof.final_poly, ordinary_proof.final_poly);
        assert_eq!(proof.query_pow_witness, ordinary_proof.query_pow_witness);
        assert_eq!(
            proof.input_openings.len(),
            ordinary_proof.input_openings.len()
        );
        for (guarded, ordinary) in proof
            .input_openings
            .iter()
            .zip(&ordinary_proof.input_openings)
        {
            assert_eq!(guarded.opened_values, ordinary.opened_values);
            assert_eq!(
                guarded.opening_proof.sibling_hashes,
                ordinary.opening_proof.sibling_hashes
            );
        }
        assert_eq!(
            proof.commit_phase_openings.len(),
            ordinary_proof.commit_phase_openings.len()
        );
        for (guarded, ordinary) in proof
            .commit_phase_openings
            .iter()
            .zip(&ordinary_proof.commit_phase_openings)
        {
            assert_eq!(guarded.sibling_values, ordinary.sibling_values);
            assert_eq!(
                guarded.opening_proof.sibling_hashes,
                ordinary.opening_proof.sibling_hashes
            );
        }
        assert_eq!(
            prover_challenger.sample_algebra_element::<EF>(),
            ordinary_challenger.sample_algebra_element::<EF>()
        );

        let retained_matrices = pcs.mmcs.get_matrices_secret(&prover_data.commitment_data);
        assert!(retained_matrices.iter().all(|matrix| {
            matrix.width == 0 && matrix.values.iter().all(|value| *value == F::ZERO)
        }));
        assert!(prover_data.coefficients.is_empty());

        let claims: Claims = vec![
            (
                commitment,
                opened_values[0]
                    .iter()
                    .map(|matrix| (domain, vec![(zeta, matrix[0].clone())]))
                    .collect(),
            )
                .into(),
        ];
        let mut verifier_challenger = Challenger::new(permutation);
        verifier_challenger.observe(&claims[0].commitment);
        let verifier_zeta: EF = verifier_challenger.sample_algebra_element();
        assert_eq!(verifier_zeta, zeta);

        <SecretCommitPcs as Pcs<EF, Challenger>>::verify(
            &pcs,
            claims,
            &proof,
            &mut verifier_challenger,
        )
        .expect("ordinary PCS verification must accept the guarded proof");
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_pcs_opening_rejects_invalid_retained_geometry_before_detachment() {
        const DOMAIN_SIZE: usize = 8;
        let pcs = make_secret_commit_pcs();
        let mut permutation_rng = SmallRng::seed_from_u64(0x5345_4352_4554_5043);
        let permutation = Perm::new_from_rng_128(&mut permutation_rng);
        let domain =
            <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, DOMAIN_SIZE);
        let matrix = RowMajorMatrix::new(vec![F::ONE; DOMAIN_SIZE], 1);
        let (commitment, mut prover_data) =
            <SecretCommitPcs as SecretPcsCommitter<EF, Challenger>>::commit_secret(
                &pcs,
                vec![(domain, SecretProverData::new(matrix))],
            )
            .expect("valid guarded commitment must succeed");
        prover_data.log_blowup = prover_data
            .log_blowup
            .checked_add(1)
            .expect("test blowup increment must fit");

        let mut challenger = Challenger::new(permutation);
        challenger.observe(&commitment);
        let zeta: EF = challenger.sample_algebra_element();
        let result = <SecretCommitPcs as SecretPcsOpener<EF, Challenger>>::open_secret(
            &pcs,
            vec![(&mut prover_data, vec![vec![zeta]]).into()],
            &mut challenger,
        );

        assert!(matches!(
            result,
            Err(SecretTwoAdicPcsOpenError::Preparation)
        ));
        let matrices = pcs.mmcs.get_matrices_secret(&prover_data.commitment_data);
        assert_eq!(matrices.len(), 1);
        assert!(prover_data.authentication_data.is_none());
        assert!(!prover_data.coefficients.is_empty());
    }

    #[cfg(feature = "secret-memory")]
    fn make_secret_hiding_merkle_pcs() -> SecretHidingMerklePcs {
        let mut rng = SmallRng::seed_from_u64(0x5345_4352_4554_484d);
        let permutation = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(permutation.clone());
        let compress = MyCompress::new(permutation);
        let val_mmcs = HidingValMmcs::new(
            hash.clone(),
            compress.clone(),
            0,
            StdRng::seed_from_u64(0x4d45_524b_4c45_5341),
        );
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress, 0));
        SecretHidingMerklePcs::new(
            Radix2Bowers,
            val_mmcs,
            FriParameters {
                log_blowup: 1,
                log_final_poly_len: 0,
                max_log_arity: 1,
                num_queries: 2,
                batch_proof_of_work_bits: 0,
                commit_proof_of_work_bits: 0,
                query_proof_of_work_bits: 0,
                mmcs: challenge_mmcs,
            },
        )
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_pcs_recomputes_hiding_opening_after_erasing_full_lde() {
        const DOMAIN_SIZE: usize = 8;
        const WIDTH: usize = 2;
        const LDE_HEIGHT: usize = DOMAIN_SIZE * 2;

        let pcs = make_secret_hiding_merkle_pcs();
        let domain = <SecretHidingMerklePcs as Pcs<EF, Challenger>>::natural_domain_for_degree(
            &pcs,
            DOMAIN_SIZE,
        );
        let matrix = RowMajorMatrix::new(
            (0..DOMAIN_SIZE * WIDTH)
                .map(|value| F::from_usize(value + 1))
                .collect_vec(),
            WIDTH,
        );
        let (commitment, mut prover_data) =
            <SecretHidingMerklePcs as SecretPcsCommitter<EF, Challenger>>::commit_secret(
                &pcs,
                vec![(domain, SecretProverData::new(matrix))],
            )
            .expect("valid guarded commitment must succeed");

        let indices = [0, 3, LDE_HEIGHT - 1];
        let baseline = pcs
            .mmcs
            .open_multi_batch_secret(&indices, &prover_data.commitment_data)
            .expect("live guarded LDE must open");
        let original = pcs.mmcs.get_matrices_secret(&prover_data.commitment_data);
        let original_matrix = original
            .first()
            .expect("fixture commitment must retain one matrix");
        let original_values = original_matrix.values.clone();
        let reconstructed_full = pcs
            .recompute_lde_matrix_secret(&prover_data, 0)
            .expect("complete guarded LDE must be reconstructable");
        assert_eq!(reconstructed_full.width, original_matrix.width);
        assert_eq!(reconstructed_full.values, original_matrix.values);
        drop(reconstructed_full);
        drop(original);
        assert!(matches!(
            pcs.recompute_lde_matrix_secret(&prover_data, 1),
            Err(SecretFriCommitError::MatrixIndexOutOfBounds {
                index: 1,
                matrix_count: 1
            })
        ));
        let reconstructed = pcs
            .detach_and_open_multi_batch_secret(&indices, &mut prover_data)
            .expect("guarded PCS must reconstruct and authenticate selected rows");
        assert!(
            pcs.mmcs
                .get_matrices_secret(&prover_data.commitment_data)
                .is_empty()
        );
        assert!(prover_data.authentication_data.is_some());
        let reconstructed_after_detachment = pcs
            .recompute_lde_matrix_secret(&prover_data, 0)
            .expect("detached commitment must retain enough data to rebuild its LDE");
        assert_eq!(reconstructed_after_detachment.width, WIDTH);
        assert_eq!(reconstructed_after_detachment.values, original_values);
        assert!(matches!(
            pcs.detach_commitment_data_secret(&mut prover_data),
            Err(SecretFriCommitError::CommitmentAlreadyDetached)
        ));
        assert_eq!(reconstructed.opened_values, baseline.0);
        assert_eq!(reconstructed.opening_proof.0, baseline.1.0);
        pcs.mmcs
            .verify_multi_batch(
                &commitment,
                &[p3_matrix::Dimensions {
                    width: WIDTH,
                    height: LDE_HEIGHT,
                }],
                &indices,
                &reconstructed.opened_values,
                &reconstructed.opening_proof,
            )
            .expect("reconstructed hiding opening must verify against the original cap");

        let out_of_bounds = pcs.recompute_lde_rows_secret(&prover_data, &[LDE_HEIGHT]);
        assert!(matches!(
            out_of_bounds,
            Err(SecretFriCommitError::Dft(
                SecretDftError::QueryIndexOutOfBounds {
                    index: LDE_HEIGHT,
                    height: LDE_HEIGHT
                }
            ))
        ));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_pcs_detached_opening_preserves_mixed_height_index_semantics() {
        const LARGE_DOMAIN_SIZE: usize = 8;
        const SMALL_DOMAIN_SIZE: usize = 4;
        const LARGE_WIDTH: usize = 2;
        const SMALL_WIDTH: usize = 1;
        const LARGE_LDE_HEIGHT: usize = LARGE_DOMAIN_SIZE * 2;
        const SMALL_LDE_HEIGHT: usize = SMALL_DOMAIN_SIZE * 2;

        let pcs = make_secret_hiding_merkle_pcs();
        let large_domain =
            <SecretHidingMerklePcs as Pcs<EF, Challenger>>::natural_domain_for_degree(
                &pcs,
                LARGE_DOMAIN_SIZE,
            );
        let small_domain =
            <SecretHidingMerklePcs as Pcs<EF, Challenger>>::natural_domain_for_degree(
                &pcs,
                SMALL_DOMAIN_SIZE,
            );
        let large_matrix = RowMajorMatrix::new(
            (0..LARGE_DOMAIN_SIZE * LARGE_WIDTH)
                .map(|value| F::from_usize(value + 1))
                .collect_vec(),
            LARGE_WIDTH,
        );
        let small_matrix = RowMajorMatrix::new(
            (0..SMALL_DOMAIN_SIZE * SMALL_WIDTH)
                .map(|value| F::from_usize(value + 101))
                .collect_vec(),
            SMALL_WIDTH,
        );
        let (commitment, mut prover_data) =
            <SecretHidingMerklePcs as SecretPcsCommitter<EF, Challenger>>::commit_secret(
                &pcs,
                vec![
                    (large_domain, SecretProverData::new(large_matrix)),
                    (small_domain, SecretProverData::new(small_matrix)),
                ],
            )
            .expect("valid mixed-height guarded commitment must succeed");

        let invalid = pcs.detach_and_open_multi_batch_secret(&[LARGE_LDE_HEIGHT], &mut prover_data);
        assert!(matches!(
            invalid,
            Err(SecretFriCommitError::Dft(
                SecretDftError::QueryIndexOutOfBounds {
                    index: LARGE_LDE_HEIGHT,
                    height: LARGE_LDE_HEIGHT
                }
            ))
        ));
        assert_eq!(
            pcs.mmcs
                .get_matrices_secret(&prover_data.commitment_data)
                .len(),
            2,
            "an invalid query must not irreversibly detach valid prover data"
        );

        let indices = [0, 3, LARGE_LDE_HEIGHT - 1];
        let baseline = pcs
            .mmcs
            .open_multi_batch_secret(&indices, &prover_data.commitment_data)
            .expect("live mixed-height guarded LDEs must open");
        let detached = pcs
            .detach_and_open_multi_batch_secret(&indices, &mut prover_data)
            .expect("mixed-height rows must reconstruct with MMCS index reduction");
        assert_eq!(detached.opened_values, baseline.0);
        assert_eq!(detached.opening_proof.0, baseline.1.0);
        assert!(
            pcs.mmcs
                .get_matrices_secret(&prover_data.commitment_data)
                .is_empty()
        );

        pcs.mmcs
            .verify_multi_batch(
                &commitment,
                &[
                    p3_matrix::Dimensions {
                        width: LARGE_WIDTH,
                        height: LARGE_LDE_HEIGHT,
                    },
                    p3_matrix::Dimensions {
                        width: SMALL_WIDTH,
                        height: SMALL_LDE_HEIGHT,
                    },
                ],
                &indices,
                &detached.opened_values,
                &detached.opening_proof,
            )
            .expect("mixed-height detached opening must verify against the original cap");
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_pcs_commit_matches_standard_commit_and_erases_retained_lde() {
        let pcs = make_secret_commit_pcs();
        let domain = <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, 8);
        let matrix = RowMajorMatrix::new((1_u8..=8).map(F::from_u8).collect_vec(), 1);
        let (expected_commitment, _standard_data) =
            <SecretCommitPcs as Pcs<EF, Challenger>>::commit(&pcs, [(domain, matrix.clone())])
                .expect("valid standard commitment must succeed");

        let (commitment, mut prover_data) =
            <SecretCommitPcs as SecretPcsCommitter<EF, Challenger>>::commit_secret(
                &pcs,
                vec![(domain, SecretProverData::new(matrix))],
            )
            .expect("valid guarded commitment must succeed");
        assert_eq!(commitment, expected_commitment);

        zeroize::Zeroize::zeroize(&mut prover_data);
        let retained = pcs.mmcs.get_matrices_secret(&prover_data.commitment_data);
        assert_eq!(retained.len(), 1);
        assert!(retained[0].values.iter().all(|value| *value == F::ZERO));
        assert_eq!(retained[0].width, 0);
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_pcs_commit_rejects_domain_height_mismatch() {
        let pcs = make_secret_commit_pcs();
        let domain = <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, 8);
        let matrix = SecretProverData::new(RowMajorMatrix::new(vec![F::ONE; 4], 1));

        let result = <SecretCommitPcs as SecretPcsCommitter<EF, Challenger>>::commit_secret(
            &pcs,
            vec![(domain, matrix)],
        );

        assert!(matches!(
            result,
            Err(SecretFriCommitError::DomainHeightMismatch)
        ));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_quotient_commit_matches_standard_and_returns_detached() {
        const DOMAIN_SIZE: usize = 16;
        const NUM_CHUNKS: usize = 4;
        const WIDTH: usize = 2;
        const SECRET_CANARY: [u8; DOMAIN_SIZE * WIDTH] = *b"SECRET-PROVER-CANARY-0123456789!";

        let pcs = make_secret_commit_pcs();
        let domain =
            <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, DOMAIN_SIZE);
        let values = SECRET_CANARY.into_iter().map(F::from_u8).collect_vec();
        let matrix = RowMajorMatrix::new(values, WIDTH);

        let (expected_commitment, _standard_data) = <SecretCommitPcs as UnivariateStarkPcs<
            EF,
            Challenger,
        >>::commit_quotient(
            &pcs, domain, matrix.clone(), NUM_CHUNKS
        )
        .expect("valid standard quotient commitment must succeed");

        let (commitment, mut prover_data) = <SecretCommitPcs as
            SecretUnivariateStarkPcsCommitter<EF, Challenger>>::commit_quotient_secret(
            &pcs,
            domain,
            SecretProverData::new(matrix),
            NUM_CHUNKS,
        )
        .expect("valid guarded quotient commitment must succeed");

        assert_eq!(commitment, expected_commitment);
        assert!(
            pcs.mmcs
                .get_matrices_secret(&prover_data.commitment_data)
                .is_empty()
        );
        assert!(prover_data.authentication_data.is_some());
        assert_eq!(
            prover_data
                .retained_lde_dimensions()
                .expect("valid retained quotient geometry must remain available")
                .len(),
            NUM_CHUNKS
        );

        zeroize::Zeroize::zeroize(&mut prover_data);
        let retained = pcs.mmcs.get_matrices_secret(&prover_data.commitment_data);
        assert!(retained.is_empty());
        assert!(prover_data.coefficients.is_empty());
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_quotient_batch_matches_one_standard_global_commitment() {
        const FIRST_SIZE: usize = 16;
        const FIRST_CHUNKS: usize = 4;
        const SECOND_SIZE: usize = 8;
        const SECOND_CHUNKS: usize = 2;

        let pcs = make_secret_commit_pcs();
        let first_domain =
            <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, FIRST_SIZE);
        let second_domain =
            <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, SECOND_SIZE);
        let first = RowMajorMatrix::new((0..FIRST_SIZE).map(F::from_usize).collect_vec(), 1);
        let second = RowMajorMatrix::new(
            (FIRST_SIZE..FIRST_SIZE + SECOND_SIZE * 2)
                .map(F::from_usize)
                .collect_vec(),
            2,
        );

        let mut ordinary_ldes = Vec::new();
        for (domain, matrix, num_chunks) in [
            (first_domain, first.clone(), FIRST_CHUNKS),
            (second_domain, second.clone(), SECOND_CHUNKS),
        ] {
            let chunk_domains = domain.split_domains(num_chunks);
            let chunk_matrices = domain.split_evals(num_chunks, matrix);
            let chunk_ldes =
                <SecretCommitPcs as UnivariateStarkPcs<EF, Challenger>>::get_quotient_ldes(
                    &pcs,
                    chunk_domains.into_iter().zip(chunk_matrices),
                    num_chunks,
                )
                .expect("valid ordinary quotient LDE construction must succeed");
            ordinary_ldes.extend(chunk_ldes);
        }
        let (expected_commitment, _ordinary_data) = <SecretCommitPcs as UnivariateStarkPcs<
            EF,
            Challenger,
        >>::commit_ldes(&pcs, ordinary_ldes)
        .expect("valid ordinary global quotient commitment must succeed");

        let (generated_commitment, mut generated_data) = <SecretCommitPcs as
            SecretUnivariateStarkPcsCommitter<EF, Challenger>>::commit_generated_quotients_secret(
            &pcs,
            vec![
                SecretQuotientCommitmentPlan::new(first_domain, 1, FIRST_CHUNKS),
                SecretQuotientCommitmentPlan::new(second_domain, 2, SECOND_CHUNKS),
            ],
            |index, _| {
                let matrix = if index == 0 {
                    first.clone()
                } else {
                    second.clone()
                };
                Ok::<_, SecretMatrixError>(SecretProverData::new(matrix))
            },
        )
        .expect("valid generated quotient batch must succeed");
        assert_eq!(generated_commitment, expected_commitment);
        zeroize::Zeroize::zeroize(&mut generated_data);

        let (commitment, mut prover_data) = <SecretCommitPcs as
            SecretUnivariateStarkPcsCommitter<EF, Challenger>>::commit_quotients_secret(
            &pcs,
            vec![
                SecretQuotientCommitmentInput::new(
                    first_domain,
                    SecretProverData::new(first),
                    FIRST_CHUNKS,
                ),
                SecretQuotientCommitmentInput::new(
                    second_domain,
                    SecretProverData::new(second),
                    SECOND_CHUNKS,
                ),
            ],
        )
        .expect("valid guarded quotient batch must succeed");

        assert_eq!(commitment, expected_commitment);
        assert!(
            pcs.mmcs
                .get_matrices_secret(&prover_data.commitment_data)
                .is_empty()
        );
        assert!(prover_data.authentication_data.is_some());
        assert_eq!(
            prover_data
                .retained_lde_dimensions()
                .expect("valid retained quotient geometry must remain available")
                .len(),
            FIRST_CHUNKS + SECOND_CHUNKS
        );
        zeroize::Zeroize::zeroize(&mut prover_data);
        let retained = pcs.mmcs.get_matrices_secret(&prover_data.commitment_data);
        assert!(retained.is_empty());
        assert!(prover_data.coefficients.is_empty());
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_quotient_batch_rejects_empty_input() {
        let pcs = make_secret_commit_pcs();
        let result = <SecretCommitPcs as SecretUnivariateStarkPcsCommitter<
            EF,
            Challenger,
        >>::commit_quotients_secret(&pcs, Vec::new());

        assert!(matches!(
            result,
            Err(SecretFriCommitError::EmptyQuotientBatch)
        ));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn generated_quotient_batch_rejects_complete_plan_before_generation() {
        const DOMAIN_SIZE: usize = 16;

        let pcs = make_secret_commit_pcs();
        let domain =
            <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, DOMAIN_SIZE);
        let mut generation_calls = 0_usize;
        let result = <SecretCommitPcs as SecretUnivariateStarkPcsCommitter<EF, Challenger>>::commit_generated_quotients_secret(
            &pcs,
            vec![
                SecretQuotientCommitmentPlan::new(domain, 1, 2),
                SecretQuotientCommitmentPlan::new(domain, 1, 3),
            ],
            |_, _| {
                generation_calls += 1;
                Ok::<_, SecretMatrixError>(SecretProverData::new(RowMajorMatrix::new(
                    vec![F::ONE; DOMAIN_SIZE],
                    1,
                )))
            },
        );

        assert!(matches!(
            result,
            Err(SecretGeneratedQuotientCommitError::Commitment(
                SecretFriCommitError::InvalidQuotientChunkCount
            ))
        ));
        assert_eq!(generation_calls, 0);
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_quotient_commit_rejects_invalid_chunk_counts() {
        const DOMAIN_SIZE: usize = 16;

        let pcs = make_secret_commit_pcs();
        let domain =
            <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, DOMAIN_SIZE);

        for num_chunks in [0, 3, DOMAIN_SIZE * 2] {
            let matrix = SecretProverData::new(RowMajorMatrix::new(vec![F::ONE; DOMAIN_SIZE], 1));
            let result = <SecretCommitPcs as
                SecretUnivariateStarkPcsCommitter<EF, Challenger>>::commit_quotient_secret(
                &pcs,
                domain,
                matrix,
                num_chunks,
            );

            assert!(matches!(
                result,
                Err(SecretFriCommitError::InvalidQuotientChunkCount)
            ));
        }
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_commitment_paths_reject_zero_width_without_panicking() {
        const DOMAIN_SIZE: usize = 8;

        let pcs = make_secret_commit_pcs();
        let domain =
            <SecretCommitPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, DOMAIN_SIZE);
        let mut malformed = RowMajorMatrix::new(vec![F::ONE; DOMAIN_SIZE], 1);
        malformed.width = 0;

        let commit_result = <SecretCommitPcs as SecretPcsCommitter<EF, Challenger>>::commit_secret(
            &pcs,
            vec![(domain, SecretProverData::new(malformed))],
        );
        assert!(matches!(
            commit_result,
            Err(SecretFriCommitError::Matrix(
                SecretMatrixError::InvalidShape
            ))
        ));

        let mut malformed = RowMajorMatrix::new(vec![F::ONE; DOMAIN_SIZE], 1);
        malformed.width = 0;
        let quotient_result = <SecretCommitPcs as SecretUnivariateStarkPcsCommitter<
            EF,
            Challenger,
        >>::commit_quotient_secret(
            &pcs, domain, SecretProverData::new(malformed), 2
        );
        assert!(matches!(
            quotient_result,
            Err(SecretFriCommitError::Matrix(
                SecretMatrixError::InvalidShape
            ))
        ));
    }

    /// Grinding difficulty the roundtrip fixture proves at.
    ///
    /// Positive so the witness is genuinely absorbed.
    /// A zero-bit check short-circuits without reading the witness at all.
    const BATCH_POW_BITS: usize = 1;

    /// Run a real prover roundtrip and return everything a verifier needs.
    ///
    /// One commitment, two matrices, one opening point each.
    ///
    /// The claimed openings therefore nest as:
    ///
    /// ```text
    ///     claims           = [commitment_0]         // 1 commitment
    ///     claims[0].1      = [matrix_0, matrix_1]   // 2 matrices
    ///     claims[0].1[i].1 = [point_0]              // 1 point each
    /// ```
    ///
    /// The challenger is advanced past the commitment, ready for `Pcs::verify`.
    fn make_pcs_fixture() -> (MyPcs, Claims, MyProof, Challenger) {
        make_pcs_fixture_at(BATCH_POW_BITS)
    }

    /// The same roundtrip at a caller-chosen batch difficulty.
    ///
    /// The difficulty reaches the transcript's description, so prover and verifier must
    /// be built from the same number for the replay to line up.
    fn make_pcs_fixture_at(batch_pow_bits: usize) -> (MyPcs, Claims, MyProof, Challenger) {
        // Fixed seed keeps the roundtrip deterministic.
        let mut rng = SmallRng::seed_from_u64(42);

        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());

        let val_mmcs = ValMmcs::new(hash.clone(), compress.clone(), 0);
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress, 0));

        // Minimal sound parameters: blowup 2, binary folding, 2 queries.
        let fri_params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            batch_proof_of_work_bits: batch_pow_bits,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: challenge_mmcs,
        };
        let pcs = MyPcs::new(Radix2Dit::default(), val_mmcs, fri_params);

        // Two matrices of different widths, so moving a value between them is observable.
        let log_degree = 3;
        let domain =
            <MyPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_degree);
        let traces = [2, 3].map(|width| {
            (
                domain,
                RowMajorMatrix::<F>::rand_nonzero(&mut rng, 1 << log_degree, width),
            )
        });
        let (commitment, prover_data) =
            <MyPcs as Pcs<EF, Challenger>>::commit(&pcs, traces).unwrap();

        // Prover: observe the commitment, sample the point, open.
        let mut p_challenger = Challenger::new(perm.clone());
        p_challenger.observe(&commitment);
        let zeta: EF = p_challenger.sample_algebra_element();
        let (opened_values, proof) = pcs
            .open(
                vec![(&prover_data, vec![vec![zeta], vec![zeta]]).into()],
                &mut p_challenger,
            )
            .unwrap();

        // Verifier: replay up to the point sample so a valid proof must pass.
        let mut v_challenger = Challenger::new(perm);
        v_challenger.observe(&commitment);
        let v_zeta: EF = v_challenger.sample_algebra_element();
        assert_eq!(
            v_zeta, zeta,
            "prover and verifier must sample the same point"
        );

        let claims = vec![
            (
                commitment,
                opened_values[0]
                    .iter()
                    .map(|matrix| (domain, vec![(zeta, matrix[0].clone())]))
                    .collect(),
            )
                .into(),
        ];

        (pcs, claims, proof, v_challenger)
    }

    #[test]
    fn open_rejects_a_matrix_without_opening_points() {
        // Fixture state: one commitment of two 8-row matrices, the second opened at no point.
        //
        //     points = [[zeta], []]
        //
        // Invariant: the verifier pins each width to a claim, so the prover refuses up front.
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash.clone(), compress.clone(), 0);
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress, 0));
        let pcs = MyPcs::new(
            Radix2Dit::default(),
            val_mmcs,
            FriParameters::new_testing(challenge_mmcs, 0),
        );
        let domain = <MyPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, 8);
        let traces = [2, 3].map(|width| {
            (
                domain,
                RowMajorMatrix::<F>::rand_nonzero(&mut rng, 8, width),
            )
        });
        let (commitment, prover_data) =
            <MyPcs as Pcs<EF, Challenger>>::commit(&pcs, traces).unwrap();

        let mut challenger = Challenger::new(perm);
        challenger.observe(&commitment);
        let zeta: EF = challenger.sample_algebra_element();

        // Mutation: dropping the guard lets the prover emit a proof the verifier rejects
        // with `FriError::MatrixWithoutOpeningPoints { batch: 0, matrix: 1 }`.
        let result = pcs.open(
            vec![(&prover_data, vec![vec![zeta], vec![]]).into()],
            &mut challenger,
        );
        assert_eq!(
            result.err(),
            Some(FriProverError::MatrixWithoutOpeningPoints {
                batch: 0,
                matrix: 1
            })
        );
    }

    /// Verify with fully qualified syntax so the type parameters are unambiguous.
    fn run_pcs_verify(
        pcs: &MyPcs,
        claims: Claims,
        proof: &MyProof,
        challenger: &mut Challenger,
    ) -> Result<(), TestError> {
        <MyPcs as Pcs<EF, Challenger>>::verify(pcs, claims, proof, challenger)
    }

    #[test]
    fn empty_opening_batch_is_a_typed_error_and_preserves_challenger() {
        let (pcs, _, _, mut challenger) = make_pcs_fixture();
        let mut before = challenger.clone();

        let result = pcs.open(Vec::new(), &mut challenger);

        assert!(matches!(result, Err(FriProverError::EmptyOpeningBatch)));
        assert_eq!(
            challenger.sample_algebra_element::<EF>(),
            before.sample_algebra_element::<EF>()
        );
    }

    #[test]
    fn opening_request_shape_mismatch_is_a_typed_error_and_preserves_challenger() {
        let (pcs, _, _, mut challenger) = make_pcs_fixture();
        let domain = <MyPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, 8);
        let (_, prover_data) = <MyPcs as Pcs<EF, Challenger>>::commit(
            &pcs,
            [(domain, RowMajorMatrix::new(vec![F::ONE; 8], 1))],
        )
        .expect("valid test commitment must succeed");
        let mut before = challenger.clone();

        let result = pcs.open(vec![(&prover_data, Vec::new()).into()], &mut challenger);

        assert!(matches!(
            result,
            Err(FriProverError::OpeningRequestShapeMismatch)
        ));
        assert_eq!(
            challenger.sample_algebra_element::<EF>(),
            before.sample_algebra_element::<EF>()
        );
    }

    #[test]
    fn terminal_height_rejection_preserves_pcs_challenger() {
        for (heights, expected_log_height) in [(vec![2], 2), (vec![1], 1), (vec![8, 2], 2)] {
            let (mut pcs, _, _, mut challenger) = make_pcs_fixture();
            pcs.fri.log_final_poly_len = 1;
            let evaluations = heights
                .iter()
                .map(|&height| {
                    let domain =
                        <MyPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, height);
                    (domain, RowMajorMatrix::new(vec![F::ONE; height], 1))
                })
                .collect_vec();
            let (_, data) = <MyPcs as Pcs<EF, Challenger>>::commit(&pcs, evaluations).unwrap();
            let mut before = challenger.clone();
            let result = pcs.open(
                vec![(&data, vec![vec![EF::TWO]; heights.len()]).into()],
                &mut challenger,
            );
            assert!(matches!(result, Err(FriProverError::InputHeightTooSmall {
                log_input_height, log_final_height: 2,
            }) if log_input_height == expected_log_height));
            assert_eq!(
                challenger.sample_algebra_element::<EF>(),
                before.sample_algebra_element::<EF>()
            );
        }
    }

    #[test]
    fn terminal_height_rejection_preserves_core_challenger() {
        for (heights, max_log_height, min_log_height) in
            [(vec![4], 2, 2), (vec![2], 1, 1), (vec![16, 4], 4, 2)]
        {
            let (mut pcs, _, _, mut challenger) = make_pcs_fixture();
            pcs.fri.log_final_poly_len = 1;
            let mut before = challenger.clone();
            let folding: TwoAdicFriFoldingForMmcs<F, ValMmcs> = TwoAdicFriFolding(PhantomData);
            let proof = prover::prove_fri(
                &folding,
                &pcs.fri,
                heights
                    .into_iter()
                    .map(|height| vec![EF::ZERO; height])
                    .collect(),
                &mut challenger,
                max_log_height,
                &[],
                &pcs.mmcs,
                F::ZERO,
            );
            assert!(matches!(proof, Err(FriProverError::InputHeightTooSmall {
                log_input_height, log_final_height: 2,
            }) if log_input_height == min_log_height));
            assert_eq!(
                challenger.sample_algebra_element::<EF>(),
                before.sample_algebra_element::<EF>()
            );
        }
    }

    #[test]
    fn terminal_height_valid_boundaries_roundtrip() {
        for (height, log_final_poly_len) in [(4, 1), (1, 0)] {
            let (mut pcs, _, _, mut challenger) = make_pcs_fixture();
            pcs.fri.log_final_poly_len = log_final_poly_len;
            let domain = <MyPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, height);
            let (commitment, data) = <MyPcs as Pcs<EF, Challenger>>::commit(
                &pcs,
                [(domain, RowMajorMatrix::new(vec![F::ONE; height], 1))],
            )
            .unwrap();
            challenger.observe(&commitment);
            let mut verifier = challenger.clone();
            let (values, proof) = pcs
                .open(vec![(&data, vec![vec![EF::TWO]]).into()], &mut challenger)
                .unwrap();
            pcs.verify(
                vec![
                    (
                        commitment,
                        vec![(domain, vec![(EF::TWO, values[0][0][0].clone())])],
                    )
                        .into(),
                ],
                &proof,
                &mut verifier,
            )
            .unwrap();
        }
    }

    #[test]
    fn terminal_height_overflow_is_recoverable() {
        let (mut pcs, _, _, mut challenger) = make_pcs_fixture();
        pcs.fri.log_final_poly_len = usize::MAX;
        let mut before = challenger.clone();
        let folding: TwoAdicFriFoldingForMmcs<F, ValMmcs> = TwoAdicFriFolding(PhantomData);
        let proof = prover::prove_fri(
            &folding,
            &pcs.fri,
            vec![vec![EF::ZERO; 4]],
            &mut challenger,
            2,
            &[],
            &pcs.mmcs,
            F::ZERO,
        );
        assert!(matches!(proof, Err(FriProverError::FinalHeightOverflow)));
        assert_eq!(
            challenger.sample_algebra_element::<EF>(),
            before.sample_algebra_element::<EF>()
        );
    }

    #[test]
    fn a_valid_opening_proof_verifies() {
        // Baseline: the mutation tests below start from a genuinely valid proof.
        let (pcs, claims, proof, mut challenger) = make_pcs_fixture();

        run_pcs_verify(&pcs, claims, &proof, &mut challenger)
            .expect("an untouched opening proof must verify");
    }

    #[test]
    fn a_perturbed_claimed_evaluation_is_rejected() {
        // Every claimed evaluation is absorbed before `alpha` is drawn.
        //
        // Moving one moves the seed of the low-degree test that follows.
        // The reduced opening the verifier rebuilds then misses the committed codeword.
        let (pcs, mut claims, proof, mut challenger) = make_pcs_fixture();
        claims[0].matrices[0].points[0].values[0] += EF::ONE;

        run_pcs_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("a perturbed claimed evaluation must be rejected");
    }

    #[test]
    fn a_claimed_evaluation_moved_between_matrices_is_rejected() {
        // Invariant: the transcript sees one step per opening, of that opening's width.
        //
        //     described:  [2, 3]
        //     supplied:   [1, 4]   -> a different seed and a different step sequence
        //
        // Both sides derive that description from their own inputs, so the two disagree.
        let (pcs, mut claims, proof, mut challenger) = make_pcs_fixture();
        let moved = claims[0].matrices[0].points[0].values.pop().unwrap();
        claims[0].matrices[1].points[0].values.push(moved);

        run_pcs_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("a reshaped set of claims must be rejected");
    }

    #[test]
    fn a_tampered_batch_grinding_witness_is_rejected() {
        // Invariant: the grind is binding, not decorative.
        //
        // The witness must satisfy the proof-of-work predicate at the difficulty
        // the run was described with, even though the proof was produced at it.
        let (pcs, claims, mut proof, mut challenger) = make_pcs_fixture();
        // Fixed invalid candidate for this transcript seed. Adding one to a valid
        // 1-bit witness can produce another valid witness.
        proof.batch_pow_witness = F::TWO;

        let err = run_pcs_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("a tampered batch grinding witness must be rejected");

        match err {
            FriError::InvalidPowWitness(PowPhase::Batch) => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn a_batch_grinding_witness_below_the_configured_difficulty_is_rejected() {
        // A verifier demanding more work than the prover did must reject, and must
        // name the batch phase rather than one of the two witnesses FRI also carries.
        let (mut pcs, claims, proof, mut challenger) = make_pcs_fixture();
        pcs.fri.batch_proof_of_work_bits = 20;

        let err = run_pcs_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("a batch grinding witness below the difficulty must be rejected");

        match err {
            FriError::InvalidPowWitness(PowPhase::Batch) => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn a_batch_difficulty_past_the_field_order_is_rejected() {
        // Invariant: a batching difficulty the field cannot sample is a rejection, not a panic.
        //
        // Fixture state: BabyBear, p = 15 * 2^27 + 1 < 2^31.
        //
        // Mutation: verify under a batching difficulty of 31 bits.
        // The batching grind runs before the low-degree test, so it is checked first.
        let (mut pcs, claims, proof, mut challenger) = make_pcs_fixture();
        pcs.fri.batch_proof_of_work_bits = 31;

        let err = run_pcs_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("a 31-bit batching difficulty must be rejected over BabyBear");

        match err {
            FriError::UnsupportedPowBits {
                phase: PowPhase::Batch,
                bits: 31,
            } => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn a_noncanonical_batch_grinding_witness_at_zero_difficulty_is_rejected() {
        // Invariant: at zero difficulty only a canonical-value check binds the witness.
        //
        //     bits = 0 -> check_witness returns true, the step is elided -> unbound
        //     bits > 0 -> absorbed, bits resampled                       -> grind binds
        let (pcs, claims, mut proof, mut challenger) = make_pcs_fixture_at(0);

        // The honest prover writes zero when it pays no work.
        assert_eq!(proof.batch_pow_witness, F::ZERO);
        proof.batch_pow_witness = F::ONE;

        let err = run_pcs_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("a rewritten batch grinding witness must be rejected");

        match err {
            FriError::NonCanonicalPowWitness {
                phase: PowPhase::Batch,
            } => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    /// `fold_row`'s arity-2 closed form must agree with the generic barycentric path.
    #[test]
    fn fold_row_arity_two_matches_lagrange_interpolation() {
        let mut rng = SmallRng::seed_from_u64(0);
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);

        for log_height in 0..6 {
            for index in 0..(1 << log_height) {
                let beta: EF = rng.random();
                let evals: [EF; 2] = [rng.random(), rng.random()];

                // The interpolation points the generic path would build for this row.
                let subgroup_start = F::two_adic_generator(log_height + 1)
                    .exp_u64(reverse_bits_len(index, log_height) as u64);
                let xs = [subgroup_start, -subgroup_start];
                let expected = lagrange_interpolate_at(&xs, &evals, beta);

                let folded = FriFoldingStrategy::<F, EF>::fold_row(
                    &folding,
                    index,
                    log_height,
                    1,
                    beta,
                    evals.into_iter(),
                );
                assert_eq!(folded, expected);
            }
        }
    }

    /// Catch incorrect input permutations, inverse twiddles, coset shifts or normalization.
    #[test]
    fn fold_row_matches_direct_interpolation() {
        fn check<Base: TwoAdicField, Ext: ExtensionField<Base>>()
        where
            StandardUniform: Distribution<Ext>,
        {
            let mut rng = SmallRng::seed_from_u64(3);
            let folding = TwoAdicFriFolding::<(), ()>(PhantomData);
            for log_arity in 1..=4 {
                let arity = 1 << log_arity;
                for log_height in 0..=Base::TWO_ADICITY - log_arity {
                    let height = 1usize << log_height;
                    let indices: Vec<_> = if log_height <= 5 {
                        (0..height).collect()
                    } else {
                        vec![0, 1, 2, height / 2, height - 1]
                    };
                    for index in indices {
                        let s = Base::two_adic_generator(log_height + log_arity)
                            .exp_u64(reverse_bits_len(index, log_height) as u64);
                        let root = Base::two_adic_generator(log_arity);
                        let xs: Vec<_> = (0..arity)
                            .map(|i| s * root.exp_u64(reverse_bits_len(i, log_arity) as u64))
                            .collect();
                        let evals: Vec<Ext> = (0..arity).map(|_| rng.random()).collect();
                        for beta in [Ext::ZERO, rng.random()]
                            .into_iter()
                            .chain(xs.iter().copied().map(Ext::from))
                        {
                            // Direct Lagrange basis, independent of the barycentric and FFT paths.
                            let expected: Ext = (0..arity)
                                .map(|i| {
                                    let mut numerator = Ext::ONE;
                                    let mut denominator = Base::ONE;
                                    for j in 0..arity {
                                        if i != j {
                                            numerator *= beta - xs[j];
                                            denominator *= xs[i] - xs[j];
                                        }
                                    }
                                    evals[i] * numerator * denominator.inverse()
                                })
                                .sum();
                            assert_eq!(
                                FriFoldingStrategy::<Base, Ext>::fold_row(
                                    &folding,
                                    index,
                                    log_height,
                                    log_arity,
                                    beta,
                                    evals.iter().copied(),
                                ),
                                expected,
                                "log_arity={log_arity}, log_height={log_height}, index={index}",
                            );
                        }
                    }
                }
            }
        }
        check::<F, EF>();
        check::<Goldilocks, BinomialExtensionField<Goldilocks, 2>>();
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn fold_row_rejects_missing_and_excess_evaluations() {
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);
        for log_arity in 1..=4 {
            let arity = 1 << log_arity;
            for len in (0..=arity + 2).filter(|&len| len != arity) {
                // Filter hides the exact size, so validation cannot rely on a size hint.
                let result = std::panic::catch_unwind(|| {
                    FriFoldingStrategy::<F, EF>::fold_row(
                        &folding,
                        0,
                        0,
                        log_arity,
                        EF::ZERO,
                        (0..len).filter(|_| true).map(|_| EF::ONE),
                    )
                });
                assert!(result.is_err(), "log_arity={log_arity}, len={len}");
            }
        }
    }

    /// `fold_matrix` folds row `i` of its input exactly as `fold_row` folds that row on its own,
    /// below, at and above the packing width, both for fields that take the packed path and for
    /// fields that keep the scalar one.
    #[test]
    fn fold_matrix_matches_fold_row() {
        fn check<Base: TwoAdicField, Ext: ExtensionField<Base>>()
        where
            StandardUniform: Distribution<Ext>,
        {
            let folding = TwoAdicFriFolding::<(), ()>(PhantomData);
            check_fold_against_fold_row::<Base, Ext>(0, |beta, log_arity, m| {
                FriFoldingStrategy::<Base, Ext>::fold_matrix(&folding, beta, log_arity, m)
            });
        }

        check::<F, EF>();
        check::<F, F>();
        check::<Goldilocks, BinomialExtensionField<Goldilocks, 2>>();
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_fold_matches_ordinary_at_every_supported_test_arity_and_erases() {
        let mut rng = SmallRng::seed_from_u64(0x4652_492d_464f_4c44);
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);

        for log_arity in 1..=4 {
            let beta: EF = rng.random();
            let matrix = RowMajorMatrix::<EF>::rand(&mut rng, 32, 1 << log_arity);
            let expected = FriFoldingStrategy::<F, EF>::fold_matrix(
                &folding,
                beta,
                log_arity,
                matrix.as_view(),
            );
            let mut guarded = SecretFriFoldingStrategy::<F, EF>::fold_matrix_secret(
                &folding,
                beta,
                log_arity,
                matrix.as_view(),
            )
            .expect("valid guarded fold must succeed");

            assert_eq!(guarded.width, 1);
            assert_eq!(guarded.values, expected);
            zeroize::Zeroize::zeroize(&mut guarded);
            assert_eq!(guarded.width, 0);
            assert!(guarded.values.iter().all(|value| *value == EF::ZERO));
        }
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_fold_rejects_invalid_shapes_without_panicking() {
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);
        let matrix = RowMajorMatrix::new(vec![EF::ONE; 8], 2);

        let zero_arity = SecretFriFoldingStrategy::<F, EF>::fold_matrix_secret(
            &folding,
            EF::TWO,
            0,
            matrix.as_view(),
        );
        assert!(matches!(zero_arity, Err(SecretFriFoldError::ZeroLogArity)));

        let wrong_width = SecretFriFoldingStrategy::<F, EF>::fold_matrix_secret(
            &folding,
            EF::TWO,
            2,
            matrix.as_view(),
        );
        assert!(matches!(
            wrong_width,
            Err(SecretFriFoldError::InvalidMatrixShape)
        ));
    }

    /// Check that `fold` folds each row of random matrices exactly as `fold_row` folds that
    /// row on its own, for arities 2 to 16 and heights from `2^min_log_height` up to eight
    /// packing widths.
    fn check_fold_against_fold_row<Base, Ext>(
        min_log_height: usize,
        fold: impl Fn(Ext, usize, RowMajorMatrixView<'_, Ext>) -> Vec<Ext>,
    ) where
        Base: TwoAdicField,
        Ext: ExtensionField<Base>,
        StandardUniform: Distribution<Ext>,
    {
        let mut rng = SmallRng::seed_from_u64(2);
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);
        let log_width = log2_strict_usize(Base::Packing::WIDTH);

        for log_arity in 1..=4 {
            for log_height in min_log_height..=log_width + 3 {
                let beta: Ext = rng.random();
                let m = RowMajorMatrix::<Ext>::rand(&mut rng, 1 << log_height, 1 << log_arity);

                let folded = fold(beta, log_arity, m.as_view());
                assert_eq!(folded.len(), m.height());

                for (index, &expected) in folded.iter().enumerate() {
                    let row = m.row(index).unwrap().into_iter();
                    let folded_row = FriFoldingStrategy::<Base, Ext>::fold_row(
                        &folding, index, log_height, log_arity, beta, row,
                    );
                    assert_eq!(folded_row, expected);
                }
            }
        }
    }

    /// The packed kernel on its own matches `fold_row`, including for fields that
    /// `fold_matrix` keeps on the scalar path.
    #[test]
    fn fold_matrix_packed_matches_fold_row() {
        fn check<Base: TwoAdicField, Ext: ExtensionField<Base>>()
        where
            StandardUniform: Distribution<Ext>,
        {
            let log_width = log2_strict_usize(Base::Packing::WIDTH);
            check_fold_against_fold_row::<Base, Ext>(log_width, |beta, log_arity, m| {
                fold_matrix_packed(beta, log_arity, &m)
            });
        }

        check::<F, EF>();
        check::<F, F>();
        check::<Goldilocks, BinomialExtensionField<Goldilocks, 2>>();
    }

    /// Packings of a width the packed kernel cannot split into bit-reversed lanes keep the
    /// scalar paths, as do width-1 packings, heights below one packing width and wide fields.
    #[test]
    fn packed_fold_gate() {
        assert!(use_packed_fold(4, 4, 31));
        assert!(use_packed_fold(8, 1 << 10, 32));

        assert!(!use_packed_fold(3, 4, 31));
        assert!(!use_packed_fold(3, 8, 31));
        assert!(!use_packed_fold(6, 8, 31));
        assert!(!use_packed_fold(1, 8, 31));
        assert!(!use_packed_fold(8, 4, 31));
        assert!(!use_packed_fold(4, 8, 64));
    }

    /// Run the per-matrix LDEs one after another, in input order.
    fn sequential_ldes(
        pcs: &MyPcs,
        inputs: &[(Domain, RowMajorMatrix<F>)],
    ) -> Vec<RowMajorMatrix<F>> {
        inputs
            .iter()
            .map(|(domain, evals)| {
                pcs.dft
                    .coset_lde_batch(
                        evals.clone(),
                        pcs.fri.log_blowup,
                        F::GENERATOR / domain.shift(),
                    )
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect()
    }

    /// Concurrent LDEs must hand the committer the same matrices in the same order as a
    /// sequential pass.
    ///
    /// The commitment depends on that order for matrices of equal height, and openings
    /// address matrices by their input position.
    /// Heights are neither sorted nor distinct, so the largest-first schedule has to be undone.
    #[test]
    fn concurrent_ldes_match_sequential_reference_in_input_order() {
        let mut rng = SmallRng::seed_from_u64(7);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let val_mmcs = ValMmcs::new(hash.clone(), compress.clone(), 0);
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress, 0));
        let fri_params = FriParameters {
            log_blowup: 2,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            batch_proof_of_work_bits: 0,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: challenge_mmcs,
        };
        let pcs = MyPcs::new(Radix2Dit::default(), val_mmcs, fri_params);

        // Repeated heights with distinct widths: a swap between equal-height matrices
        // changes the matrices handed to the committer.
        let inputs: Vec<_> = [(10, 5), (11, 4), (10, 8), (11, 7), (10, 4), (11, 6)]
            .into_iter()
            .map(|(log_height, width)| {
                let domain = <MyPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(
                    &pcs,
                    1 << log_height,
                );
                (
                    domain,
                    RowMajorMatrix::<F>::rand_nonzero(&mut rng, 1 << log_height, width),
                )
            })
            .collect();
        let expected = sequential_ldes(&pcs, &inputs);

        let (commitment, data) =
            <MyPcs as Pcs<EF, Challenger>>::commit(&pcs, inputs.clone()).unwrap();
        let (expected_commitment, expected_data) =
            <MyPcs as UnivariateStarkPcs<EF, Challenger>>::commit_ldes(&pcs, expected.clone())
                .unwrap();
        assert_eq!(commitment, expected_commitment);
        assert_eq!(
            pcs.mmcs.get_matrices(&data),
            pcs.mmcs.get_matrices(&expected_data)
        );

        let quotient_ldes = <MyPcs as UnivariateStarkPcs<EF, Challenger>>::get_quotient_ldes(
            &pcs,
            inputs,
            expected.len(),
        )
        .unwrap();
        assert_eq!(quotient_ldes, expected);
    }
}
