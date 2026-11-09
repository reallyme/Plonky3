// Modified by ReallyMe Technologies Limited, 2026.
// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{
    CommitmentOpening, HidingMmcs, Mmcs, OpenedValues, OpeningRequest, Pcs, PolynomialSpace,
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
#[cfg(feature = "secret-memory")]
use p3_dft::SecretTwoAdicSubgroupDft;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, PrimeField64, TwoAdicField, batch_multiplicative_inverse};
#[cfg(feature = "secret-memory")]
use p3_matrix::Dimensions;
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversalPerm, BitReversibleMatrix};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixCow};
use p3_matrix::horizontally_truncated::HorizontallyTruncated;
use p3_matrix::row_index_mapped::RowIndexMappedView;
#[cfg(feature = "secret-memory")]
use p3_matrix::secret::SecretMatrixError;
#[cfg(feature = "secret-memory")]
use p3_matrix::util::reverse_matrix_index_bits;
use p3_util::log2_strict_usize;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;
use rand::{CryptoRng, RngExt, SeedableRng};
use spin::Mutex;
use tracing::info_span;

#[cfg(feature = "secret-memory")]
use crate::SecretTwoAdicPcsOpenError;
use crate::verifier::FriError;
use crate::{BatchMultiOpening, FriParameters, FriProof, FriProverError, TwoAdicFriPcs};

/// A hiding commitment cannot safely support the requested disclosure budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum HidingFriProverError {
    /// The inner FRI parameters cannot fold every committed input.
    #[error(transparent)]
    Fri(#[from] FriProverError),
    /// Too few independent random codewords for the challenge extension.
    #[error("hiding FRI requires {required} random codewords, got {got}")]
    InsufficientRandomCodewords { required: usize, got: usize },
    /// The requested disclosure count does not fit the budget counter.
    #[error("hiding budget arithmetic overflow")]
    BudgetOverflow,
    /// The mask cannot cover all requested field evaluations and FRI queries.
    #[error(
        "insufficient hiding budget: {mask_height} mask values, {num_queries} queries, {num_opening_points} opening points, extension degree {extension_degree}"
    )]
    HidingBudgetExceeded {
        mask_height: usize,
        num_queries: usize,
        num_opening_points: usize,
        extension_degree: usize,
    },
    /// The number of quotient evaluations differs from the chunk count.
    #[error("expected {expected} quotient chunks, got {got}")]
    QuotientChunkCountMismatch { expected: usize, got: usize },
}

/// Failure while constructing a guarded hiding-FRI commitment.
#[cfg(feature = "secret-memory")]
#[derive(Debug, thiserror::Error)]
pub enum SecretHidingFriCommitError<InnerError> {
    /// The hiding parameters cannot safely cover the configured disclosure budget.
    #[error("guarded hiding-FRI budget validation failed")]
    Hiding(#[from] HidingFriProverError),
    /// Randomized matrix construction rejected a shape or allocation.
    #[error("guarded hiding-FRI matrix randomization failed")]
    Matrix(#[from] SecretMatrixError),
    /// The vector retaining randomized matrix owners could not be reserved.
    #[error("guarded hiding-FRI owner allocation failed")]
    AllocationFailed,
    /// The guarded inner PCS rejected commitment construction.
    #[error("guarded hiding-FRI inner commitment failed")]
    Inner(InnerError),
}

/// Failure while evaluating guarded hiding-PCS data on another domain.
#[cfg(feature = "secret-memory")]
#[derive(Debug, thiserror::Error)]
pub enum SecretHidingFriEvaluationError<InnerError> {
    /// Prover data was created with a different random-codeword count.
    #[error("guarded hiding-FRI random-codeword configuration mismatch")]
    RandomCodewordCountMismatch,
    /// Prover data was created with a different FRI blowup.
    #[error("guarded hiding-FRI blowup configuration mismatch")]
    BlowupMismatch,
    /// The inner evaluation matrix cannot contain the configured hiding columns.
    #[error("guarded hiding-FRI evaluation width is invalid")]
    InvalidRandomizedWidth,
    /// Removing hiding columns failed without releasing their allocation.
    #[error("guarded hiding-FRI evaluation truncation failed")]
    Matrix(#[from] SecretMatrixError),
    /// The inner PCS rejected guarded domain evaluation.
    #[error("guarded hiding-FRI inner domain evaluation failed")]
    Inner(InnerError),
}

/// Stable failures from guarded hiding-PCS opening construction.
#[cfg(feature = "secret-memory")]
#[derive(Debug, thiserror::Error)]
pub enum SecretHidingFriOpenError {
    /// Prover data was created with a different random-codeword count.
    #[error("guarded hiding-FRI random-codeword configuration mismatch")]
    RandomCodewordCountMismatch,
    /// Prover data was created with a different FRI blowup.
    #[error("guarded hiding-FRI blowup configuration mismatch")]
    BlowupMismatch,
    /// The guarded capability accepts private commitment rounds only.
    #[error("guarded hiding-FRI opening does not accept preprocessing data")]
    PreprocessedData,
    /// Public preprocessing requests are duplicated or mislabeled.
    #[error("guarded hiding-FRI public preprocessing layout is invalid")]
    InvalidPublicPreprocessingLayout,
    /// A commitment's retained matrix count or opening-value shape is invalid.
    #[error("guarded hiding-FRI opening shape is invalid")]
    InvalidShape,
    /// The hiding parameters cannot safely cover the requested disclosures.
    #[error("guarded hiding-FRI opening budget validation failed")]
    Hiding(#[from] HidingFriProverError),
    /// A private commitment has already been consumed by an opening attempt.
    #[error("guarded hiding-FRI commitment was already opened")]
    AlreadyOpened,
    /// Public proof metadata could not be reserved.
    #[error("guarded hiding-FRI opening allocation failed")]
    AllocationFailed,
    /// The inner guarded two-adic PCS opening failed.
    #[error("guarded hiding-FRI inner opening failed")]
    Inner(#[from] SecretTwoAdicPcsOpenError),
}

/// A hiding FRI PCS.
///
/// Both commitment schemes must be hiding, or zero-knowledge is lost.
///
/// Each call that draws masks seeds a private [`StdRng`] from the caller-supplied `R`, then draws every mask from it.
///
/// The shared generator is locked only for that seed, so the parallel work runs without the lock.
///
/// Every mask thus depends on `R`, which is why `R` is bounded by [`CryptoRng`].
///
/// That rules out generators known to be unsuitable for cryptographic use, but it does not replace proper seeding.
///
/// A caller who seeds `R` from a predictable source lets an observer reproduce the stream and strip the masks.
///
/// # Hiding requires a large enough trace relative to the query budget
///
/// [`Self::commit`] masks each committed column with `N` uniform base-field values, where `N`
/// is the trace height. The commitment and opening entry points enforce the conservative budget
///
/// ```text
/// N >= 2 * (num_queries + Challenge::DIMENSION * number_of_opening_points)
/// ```
///
/// following the Lagrange quotient masking bound in Section 4.2 of
/// <https://eprint.iacr.org/2024/1037>. This counts every supplied point, including translated or
/// repeated points, conservatively. Commitments reserve at least one opening point; opening and
/// verification check the actual number. Public preprocessing uses the same size bound because
/// the generic verifier does not receive a trusted preprocessing index.
///
/// Private prover data may be opened only once, including across clones. Recommit with fresh
/// randomness for another proof. This restriction does not apply to public preprocessing.
/// Prover data must be used with the same random-codeword count and blowup as its commitment.
/// At least `Challenge::DIMENSION` random codewords are required to mask extension-field batching.
/// These checks do not replace the hiding-MMCS and unpredictable-seeding requirements above,
/// or the STARK protocol's randomization polynomial and out-of-domain challenge sampling.
#[derive(Debug)]
pub struct HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R> {
    inner: TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>,
    num_random_codewords: usize,
    rng: Mutex<R>,
}

/// Prover data whose private masks can be used in only one opening proof.
///
/// In the standard memory capability, clones share the consumed flag and
/// public preprocessing remains reusable. The `secret-memory` capability
/// deliberately removes `Clone` and `Debug`; copying or formatting this owner
/// would undermine its secret-lifecycle contract.
#[cfg_attr(not(feature = "secret-memory"), derive(Clone, Debug))]
pub struct HidingFriPcsProverData<T> {
    inner: T,
    preprocessed: bool,
    num_random_codewords: usize,
    log_blowup: usize,
    opened: Arc<AtomicBool>,
}

#[cfg(feature = "secret-memory")]
impl<T> zeroize::Zeroize for HidingFriPcsProverData<T>
where
    T: zeroize::Zeroize,
{
    fn zeroize(&mut self) {
        self.inner.zeroize();
        self.preprocessed = false;
        self.num_random_codewords.zeroize();
        self.log_blowup.zeroize();
        // A wiped private commitment must remain unusable. Marking it opened is
        // fail-closed if a caller retains the shell after explicit erasure.
        self.opened.store(true, Ordering::SeqCst);
    }
}

impl<T> HidingFriPcsProverData<T> {
    fn new(inner: T, preprocessed: bool, num_random_codewords: usize, log_blowup: usize) -> Self {
        Self {
            inner,
            preprocessed,
            num_random_codewords,
            log_blowup,
            opened: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Cloning forks the RNG stream by drawing a fresh seed from the source RNG,
/// so the clone and the original never produce the same sequence of masks.
impl<Val, Dft, InputMmcs, FriMmcs, R> Clone for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: Clone,
    Dft: Clone,
    InputMmcs: Clone,
    FriMmcs: Clone,
    R: CryptoRng + SeedableRng,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            num_random_codewords: self.num_random_codewords,
            rng: Mutex::new(R::from_rng(&mut *self.rng.lock())),
        }
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, R> HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: Send + Sync + Clone,
    InputMmcs: HidingMmcs<Val>,
{
    /// Rejects a commitment scheme that is not hiding.
    ///
    /// The challenge field is a parameter only so the bound can name it, and is usually inferred.
    pub const fn new<Challenge: Send + Sync + Clone>(
        dft: Dft,
        mmcs: InputMmcs,
        params: FriParameters<FriMmcs>,
        num_random_codewords: usize,
        rng: R,
    ) -> Self
    where
        FriMmcs: HidingMmcs<Challenge>,
    {
        let inner = TwoAdicFriPcs::new(dft, mmcs, params);
        Self {
            inner,
            num_random_codewords,
            rng: Mutex::new(rng),
        }
    }

    fn hiding_budget(&self, extension_degree: usize, num_points: usize) -> Option<usize> {
        extension_degree
            .checked_mul(num_points.max(1))?
            .checked_add(self.inner.fri.num_queries)?
            .checked_mul(2)
    }

    fn check_hiding_budget(
        &self,
        mask_height: usize,
        extension_degree: usize,
        num_points: usize,
    ) -> Result<(), HidingFriProverError> {
        if self.num_random_codewords < extension_degree {
            return Err(HidingFriProverError::InsufficientRandomCodewords {
                required: extension_degree,
                got: self.num_random_codewords,
            });
        }
        let required = self
            .hiding_budget(extension_degree, num_points)
            .ok_or(HidingFriProverError::BudgetOverflow)?;
        if required > mask_height {
            return Err(HidingFriProverError::HidingBudgetExceeded {
                mask_height,
                num_queries: self.inner.fri.num_queries,
                num_opening_points: num_points,
                extension_degree,
            });
        }
        Ok(())
    }

    fn lde_mask_height(&self, height: usize) -> usize {
        height
            .checked_shr(u32::try_from(self.inner.fri.log_blowup).unwrap_or(u32::MAX))
            .unwrap_or(0)
            / 2
    }

    fn assert_prover_data_config<T>(&self, prover_data: &HidingFriPcsProverData<T>) {
        assert_eq!(
            prover_data.num_random_codewords, self.num_random_codewords,
            "hiding prover data random-codeword count does not match PCS configuration"
        );
        assert_eq!(
            prover_data.log_blowup, self.inner.fri.log_blowup,
            "hiding prover data blowup does not match PCS configuration"
        );
    }

    #[cfg(feature = "secret-memory")]
    const fn check_secret_prover_data_config<T, InnerError>(
        &self,
        prover_data: &HidingFriPcsProverData<T>,
    ) -> Result<(), SecretHidingFriEvaluationError<InnerError>> {
        if prover_data.num_random_codewords != self.num_random_codewords {
            return Err(SecretHidingFriEvaluationError::RandomCodewordCountMismatch);
        }
        if prover_data.log_blowup != self.inner.fri.log_blowup {
            return Err(SecretHidingFriEvaluationError::BlowupMismatch);
        }
        Ok(())
    }

    #[cfg(feature = "secret-memory")]
    const fn check_secret_open_config<T>(
        &self,
        prover_data: &HidingFriPcsProverData<T>,
    ) -> Result<(), SecretHidingFriOpenError> {
        if prover_data.num_random_codewords != self.num_random_codewords {
            return Err(SecretHidingFriOpenError::RandomCodewordCountMismatch);
        }
        if prover_data.log_blowup != self.inner.fri.log_blowup {
            return Err(SecretHidingFriOpenError::BlowupMismatch);
        }
        Ok(())
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger, R> Pcs<Challenge, Challenger>
    for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: TwoAdicField + PrimeField64,
    StandardUniform: Distribution<Val>,
    Dft: TwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: HidingMmcs<Val> + Mmcs<Val, MultiProof: Sync, Error: Sync>,
    FriMmcs: HidingMmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
    R: CryptoRng + Send + Sync,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = HidingFriPcsProverData<InputMmcs::ProverData<RowMajorMatrix<Val>>>;

    /// The first item contains the openings of the random polynomials added by this wrapper.
    /// The second item is the usual FRI proof.
    type Proof = (
        OpenedValues<Challenge>,
        FriProof<Challenge, FriMmcs, Val, Vec<BatchMultiOpening<Val, InputMmcs>>>,
    );

    type Error = FriError<FriMmcs::Error, InputMmcs::Error>;
    type ProverError = HidingFriProverError;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &self.inner, degree)
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
        let evaluations: Vec<_> = evaluations.into_iter().collect();
        for (_, mat) in &evaluations {
            self.check_hiding_budget(mat.height(), Challenge::DIMENSION, 1)?;
        }
        // Fork a private RNG so the shared lock is released before any parallel work.
        //
        // Why: a waiting rayon worker may run a queued call on this same PCS.
        // That call would spin forever on a lock its own thread still holds.
        let mut rng = StdRng::from_rng(&mut *self.rng.lock());
        let randomized_evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)> =
            info_span!("randomize polys").in_scope(|| {
                evaluations
                    .into_iter()
                    .map(|(domain, mat)| {
                        let mat_width = mat.width();
                        // Let `w` and `h` be the width and height of the original matrix. The randomized matrix should have height `2h` and width `w + num_random_codewords`.
                        // To generate it, we add `w + 2 * num_random_codewords` columns to the original matrix, then reshape it by setting the width to `w + num_random_codewords`.
                        // All columns are added on the right hand side so, after reshaping, this has the net effect of adding `num_random_codewords` random columns on the right and interleaving the original trace with random rows.

                        let mut random_evaluation = mat
                            .with_random_cols(mat_width + 2 * self.num_random_codewords, &mut rng);
                        random_evaluation.width = mat_width + self.num_random_codewords;

                        (domain, random_evaluation)
                    })
                    .collect()
            });

        let (commitment, data) =
            Pcs::<Challenge, Challenger>::commit(&self.inner, randomized_evaluations)?;
        Ok((
            commitment,
            HidingFriPcsProverData::new(
                data,
                false,
                self.num_random_codewords,
                self.inner.fri.log_blowup,
            ),
        ))
    }

    fn open(
        &self,
        // For each round,
        rounds: Vec<OpeningRequest<'_, Self::ProverData, Challenge>>,
        challenger: &mut Challenger,
    ) -> Result<(OpenedValues<Challenge>, Self::Proof), Self::ProverError> {
        self.open_with_preprocessing(rounds, challenger, None)
    }

    fn verify(
        &self,
        rounds: Vec<CommitmentOpening<Challenge, Self::Commitment, Self::Domain>>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        self.verify_with_preprocessing(rounds, proof, challenger, None)
    }
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger, R>
    SecretPcsCommitter<Challenge, Challenger> for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    StandardUniform: Distribution<Val>,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs:
        HidingMmcs<Val> + Mmcs<Val, MultiProof: Sync, Error: Sync> + SecretRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
    FriMmcs: HidingMmcs<Challenge> + Mmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
    R: CryptoRng + Send + Sync,
{
    type SecretData = HidingFriPcsProverData<
        SecretProverData<
            crate::SecretTwoAdicFriData<
                Val,
                InputMmcs::SecretData<RowMajorMatrix<Val>>,
                InputMmcs::AuthenticationData<RowMajorMatrix<Val>>,
            >,
        >,
    >;
    type SecretCommitError =
        SecretHidingFriCommitError<crate::SecretFriCommitError<InputMmcs::SecretError>>;

    fn commit_secret(
        &self,
        evaluations: Vec<(Self::Domain, SecretProverData<RowMajorMatrix<Val>>)>,
    ) -> Result<(Self::Commitment, SecretProverData<Self::SecretData>), Self::SecretCommitError>
    {
        for (_, matrix) in &evaluations {
            if matrix.width == 0 || !matrix.values.len().is_multiple_of(matrix.width) {
                return Err(SecretMatrixError::InvalidShape.into());
            }
            self.check_hiding_budget(matrix.height(), Challenge::DIMENSION, 1)?;
        }

        let mut randomized_evaluations = Vec::new();
        randomized_evaluations
            .try_reserve_exact(evaluations.len())
            .map_err(|_| SecretHidingFriCommitError::AllocationFailed)?;

        // Mirror the ordinary hiding PCS: advance the shared generator once,
        // then keep all witness-derived mask generation on the private stream.
        // This preserves commitment parity while releasing the shared lock
        // before the guarded DFT and MMCS work begins.
        let mut rng = StdRng::from_rng(&mut *self.rng.lock());
        for (domain, mut matrix) in evaluations {
            let matrix_width = matrix.width();
            let original_height = matrix.height();
            let twice_random_codewords = self
                .num_random_codewords
                .checked_mul(2)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            let additional_columns = matrix_width
                .checked_add(twice_random_codewords)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            let randomized_width = matrix_width
                .checked_add(self.num_random_codewords)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            let randomized_height = original_height
                .checked_mul(2)
                .ok_or(SecretMatrixError::DimensionOverflow)?;

            matrix.try_widen_with_random_cols(additional_columns, &mut rng)?;
            if randomized_width == 0
                || !matrix.values.len().is_multiple_of(randomized_width)
                || matrix.values.len() / randomized_width != randomized_height
            {
                return Err(SecretMatrixError::InvalidShape.into());
            }
            matrix.width = randomized_width;
            randomized_evaluations.push((domain, matrix));
        }
        let (commitment, inner_data) =
            <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as SecretPcsCommitter<
                Challenge,
                Challenger,
            >>::commit_secret(&self.inner, randomized_evaluations)
            .map_err(SecretHidingFriCommitError::Inner)?;

        Ok((
            commitment,
            SecretProverData::new(HidingFriPcsProverData::new(
                inner_data,
                false,
                self.num_random_codewords,
                self.inner.fri.log_blowup,
            )),
        ))
    }
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger, R>
    SecretDetachedPcsCommitter<Challenge, Challenger>
    for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    StandardUniform: Distribution<Val>,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: HidingMmcs<Val>
        + Mmcs<Val, MultiProof: Sync, Error: Sync>
        + SecretRecomputableMmcs<Val>
        + SecretStreamingRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
    FriMmcs: HidingMmcs<Challenge> + Mmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
    R: CryptoRng + Send + Sync,
{
    fn commit_detached_secret(
        &self,
        evaluations: Vec<(Self::Domain, SecretProverData<RowMajorMatrix<Val>>)>,
    ) -> Result<(Self::Commitment, SecretProverData<Self::SecretData>), Self::SecretCommitError>
    {
        let blowup_shift = u32::try_from(self.inner.fri.log_blowup).map_err(|_| {
            SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Dft(
                p3_dft::SecretDftError::SizeOverflow,
            ))
        })?;
        let twice_random_codewords = self
            .num_random_codewords
            .checked_mul(2)
            .ok_or(SecretMatrixError::DimensionOverflow)?;
        let mut lde_dimensions = Vec::new();
        lde_dimensions
            .try_reserve_exact(evaluations.len())
            .map_err(|_| SecretHidingFriCommitError::AllocationFailed)?;
        for (domain, matrix) in &evaluations {
            if matrix.width == 0 || !matrix.values.len().is_multiple_of(matrix.width) {
                return Err(SecretMatrixError::InvalidShape.into());
            }
            let matrix_width = matrix.width();
            let original_height = matrix.height();
            let additional_columns = matrix_width
                .checked_add(twice_random_codewords)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            let physical_width = matrix_width
                .checked_add(additional_columns)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            original_height
                .checked_mul(physical_width)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            let randomized_width = matrix_width
                .checked_add(self.num_random_codewords)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            let randomized_height = original_height
                .checked_mul(2)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            if domain.size() != randomized_height {
                return Err(SecretHidingFriCommitError::Inner(
                    crate::SecretFriCommitError::DomainHeightMismatch,
                ));
            }
            self.check_hiding_budget(original_height, Challenge::DIMENSION, 1)?;
            let lde_height = randomized_height.checked_shl(blowup_shift).ok_or(
                SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Dft(
                    p3_dft::SecretDftError::SizeOverflow,
                )),
            )?;
            lde_dimensions.push(Dimensions {
                width: randomized_width,
                height: lde_height,
            });
        }

        let mut lde_stream =
            <InputMmcs as SecretStreamingRecomputableMmcs<Val>>::begin_streaming_commitment::<
                RowMajorMatrix<Val>,
            >(&self.inner.mmcs, lde_dimensions)
            .map_err(|error| {
                SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Mmcs(error))
            })?;
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(evaluations.len())
            .map_err(|_| SecretHidingFriCommitError::AllocationFailed)?;

        // Mirror the ordinary hiding PCS by deriving one private mask stream
        // for the complete batch. The shared generator is not held while DFTs
        // or commitment construction run.
        let mut rng = StdRng::from_rng(&mut *self.rng.lock());
        for (domain, mut matrix) in evaluations {
            let matrix_width = matrix.width();
            let original_height = matrix.height();
            let additional_columns = matrix_width
                .checked_add(twice_random_codewords)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            let randomized_width = matrix_width
                .checked_add(self.num_random_codewords)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            let randomized_height = original_height
                .checked_mul(2)
                .ok_or(SecretMatrixError::DimensionOverflow)?;

            matrix.try_widen_with_random_cols(additional_columns, &mut rng)?;
            if randomized_width == 0
                || !matrix.values.len().is_multiple_of(randomized_width)
                || matrix.values.len() / randomized_width != randomized_height
            {
                return Err(SecretMatrixError::InvalidShape.into());
            }
            matrix.width = randomized_width;
            let shift = Val::GENERATOR / domain.shift();
            let matrix_coefficients = self
                .inner
                .dft
                .coset_lde_batch_secret_with_coefficients(
                    &mut matrix,
                    self.inner.fri.log_blowup,
                    shift,
                )
                .map_err(|error| {
                    SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Dft(error))
                })?;
            reverse_matrix_index_bits(&mut matrix);
            lde_stream.absorb(matrix).map_err(|error| {
                SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Mmcs(error))
            })?;
            coefficients.push(matrix_coefficients);
        }
        let detached = lde_stream.finish().map_err(|error| {
            SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Mmcs(error))
        })?;
        let inner_data = SecretProverData::new(crate::SecretTwoAdicFriData::new_detached(
            detached.prover_data,
            detached.authentication_data,
            coefficients,
            self.inner.fri.log_blowup,
        ));

        Ok((
            detached.commitment,
            SecretProverData::new(HidingFriPcsProverData::new(
                inner_data,
                false,
                self.num_random_codewords,
                self.inner.fri.log_blowup,
            )),
        ))
    }
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger, R> SecretPcsOpener<Challenge, Challenger>
    for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    StandardUniform: Distribution<Val>,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs:
        HidingMmcs<Val> + Mmcs<Val, MultiProof: Sync, Error: Sync> + SecretRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
    FriMmcs: HidingMmcs<Challenge> + Mmcs<Challenge> + SecretMmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val> + zeroize::Zeroize,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
    R: CryptoRng + Send + Sync,
{
    type SecretOpenError = SecretHidingFriOpenError;

    fn open_secret(
        &self,
        requests: Vec<SecretOpeningRequest<'_, Self::SecretData, Challenge>>,
        challenger: &mut Challenger,
    ) -> Result<(OpenedValues<Challenge>, Self::Proof), Self::SecretOpenError> {
        let mut mixed_requests = Vec::new();
        mixed_requests
            .try_reserve_exact(requests.len())
            .map_err(|_| SecretHidingFriOpenError::AllocationFailed)?;
        mixed_requests.extend(
            requests
                .into_iter()
                .map(SecretPcsMixedOpeningRequest::Private),
        );
        <Self as SecretPcsMixedOpener<Challenge, Challenger>>::open_secret_with_public_preprocessing(
            self,
            mixed_requests,
            challenger,
        )
    }
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger, R>
    SecretPcsMixedOpener<Challenge, Challenger> for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    StandardUniform: Distribution<Val>,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs:
        HidingMmcs<Val> + Mmcs<Val, MultiProof: Sync, Error: Sync> + SecretRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
    FriMmcs: HidingMmcs<Challenge> + Mmcs<Challenge> + SecretMmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val> + zeroize::Zeroize,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
    R: CryptoRng + Send + Sync,
{
    fn open_secret_with_public_preprocessing(
        &self,
        mut requests: Vec<
            SecretPcsMixedOpeningRequest<'_, Self::ProverData, Self::SecretData, Challenge>,
        >,
        challenger: &mut Challenger,
    ) -> Result<(OpenedValues<Challenge>, Self::Proof), Self::SecretOpenError> {
        let mut public_count = 0_usize;
        for request in &requests {
            match request {
                SecretPcsMixedOpeningRequest::Public(request) => {
                    public_count = public_count
                        .checked_add(1)
                        .ok_or(SecretHidingFriOpenError::InvalidPublicPreprocessingLayout)?;
                    if public_count > 1 || !request.prover_data.preprocessed {
                        return Err(SecretHidingFriOpenError::InvalidPublicPreprocessingLayout);
                    }
                    self.check_secret_open_config(request.prover_data)?;
                    let matrices = self.inner.mmcs.get_matrices(&request.prover_data.inner);
                    if matrices.len() != request.points.len() {
                        return Err(SecretHidingFriOpenError::InvalidShape);
                    }
                    for (matrix, opening_points) in matrices.iter().zip(&request.points) {
                        let height = matrix.height();
                        if height == 0 || !height.is_power_of_two() {
                            return Err(SecretHidingFriOpenError::InvalidShape);
                        }
                        self.inner
                            .fri
                            .validate_input_height(log2_strict_usize(height))
                            .map_err(HidingFriProverError::from)?;
                        self.check_hiding_budget(
                            self.lde_mask_height(height),
                            Challenge::DIMENSION,
                            opening_points.len(),
                        )?;
                    }
                }
                SecretPcsMixedOpeningRequest::Private(request) => {
                    let prover_data = &**request.prover_data;
                    if prover_data.preprocessed {
                        return Err(SecretHidingFriOpenError::PreprocessedData);
                    }
                    if prover_data.opened.load(Ordering::Acquire) {
                        return Err(SecretHidingFriOpenError::AlreadyOpened);
                    }
                    self.check_secret_open_config(prover_data)?;
                    let dimensions = prover_data.inner.retained_lde_dimensions().map_err(
                        |error| match error {
                            SecretMatrixError::AllocationFailed => {
                                SecretHidingFriOpenError::AllocationFailed
                            }
                            SecretMatrixError::DimensionOverflow
                            | SecretMatrixError::InvalidShape => {
                                SecretHidingFriOpenError::InvalidShape
                            }
                        },
                    )?;
                    if dimensions.len() != request.points.len() {
                        return Err(SecretHidingFriOpenError::InvalidShape);
                    }
                    for (dimensions, opening_points) in dimensions.iter().zip(&request.points) {
                        self.inner
                            .fri
                            .validate_input_height(log2_strict_usize(dimensions.height))
                            .map_err(HidingFriProverError::from)?;
                        self.check_hiding_budget(
                            self.lde_mask_height(dimensions.height),
                            Challenge::DIMENSION,
                            opening_points.len(),
                        )?;
                    }
                }
            }
        }

        // All public layout, profile, geometry, and disclosure checks finish
        // before any private commitment becomes single-use.
        for request in &requests {
            if let SecretPcsMixedOpeningRequest::Private(request) = request {
                request
                    .prover_data
                    .opened
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .map_err(|_| SecretHidingFriOpenError::AlreadyOpened)?;
            }
        }

        let mut public_rounds = Vec::new();
        public_rounds
            .try_reserve_exact(requests.len())
            .map_err(|_| SecretHidingFriOpenError::AllocationFailed)?;
        let mut inner_requests = Vec::new();
        inner_requests
            .try_reserve_exact(requests.len())
            .map_err(|_| SecretHidingFriOpenError::AllocationFailed)?;
        for request in &mut requests {
            match request {
                SecretPcsMixedOpeningRequest::Public(request) => {
                    public_rounds.push(true);
                    inner_requests.push(SecretPcsMixedOpeningRequest::Public(OpeningRequest {
                        prover_data: &request.prover_data.inner,
                        points: core::mem::take(&mut request.points),
                    }));
                }
                SecretPcsMixedOpeningRequest::Private(request) => {
                    public_rounds.push(false);
                    inner_requests.push(SecretPcsMixedOpeningRequest::Private(
                        SecretOpeningRequest {
                            prover_data: &mut request.prover_data.inner,
                            points: core::mem::take(&mut request.points),
                        },
                    ));
                }
            }
        }

        let (mut opened_values, inner_proof) =
            <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as SecretPcsMixedOpener<
                Challenge,
                Challenger,
            >>::open_secret_with_public_preprocessing(
                &self.inner, inner_requests, challenger
            )?;
        if opened_values.len() != public_rounds.len() {
            return Err(SecretHidingFriOpenError::InvalidShape);
        }

        let mut random_opened_values = Vec::new();
        random_opened_values
            .try_reserve_exact(opened_values.len())
            .map_err(|_| SecretHidingFriOpenError::AllocationFailed)?;
        for (opened_round, is_public) in opened_values.iter_mut().zip(public_rounds) {
            let random_width = if is_public {
                0
            } else {
                self.num_random_codewords
            };
            let mut random_round = Vec::new();
            random_round
                .try_reserve_exact(opened_round.len())
                .map_err(|_| SecretHidingFriOpenError::AllocationFailed)?;
            for opened_matrix in opened_round {
                let mut random_matrix = Vec::new();
                random_matrix
                    .try_reserve_exact(opened_matrix.len())
                    .map_err(|_| SecretHidingFriOpenError::AllocationFailed)?;
                for opened_point in opened_matrix {
                    let split = opened_point
                        .len()
                        .checked_sub(random_width)
                        .ok_or(SecretHidingFriOpenError::InvalidShape)?;
                    let mut random_point = Vec::new();
                    random_point
                        .try_reserve_exact(random_width)
                        .map_err(|_| SecretHidingFriOpenError::AllocationFailed)?;
                    random_point.extend(opened_point.drain(split..));
                    random_matrix.push(random_point);
                }
                random_round.push(random_matrix);
            }
            random_opened_values.push(random_round);
        }

        Ok((opened_values, (random_opened_values, inner_proof)))
    }
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger, R>
    SecretUnivariateStarkPcsCommitter<Challenge, Challenger>
    for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    StandardUniform: Distribution<Val>,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: HidingMmcs<Val>
        + Mmcs<Val, MultiProof: Sync, Error: Sync>
        + SecretRecomputableMmcs<Val>
        + SecretStreamingRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
    FriMmcs: HidingMmcs<Challenge> + Mmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
    R: CryptoRng + Send + Sync,
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
            .map_err(|_| SecretHidingFriCommitError::AllocationFailed)?;
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
            .map_err(|_| SecretHidingFriCommitError::AllocationFailed)?;
        for quotient in &quotients {
            let domain_size = quotient.domain().size();
            let evaluations = quotient.evaluations();
            let width = evaluations.width;
            let num_chunks = quotient.num_chunks();
            if width == 0 || !evaluations.values.len().is_multiple_of(width) {
                return Err(SecretMatrixError::InvalidShape.into());
            }
            if evaluations.values.len() / width != domain_size {
                return Err(SecretHidingFriCommitError::Inner(
                    crate::SecretFriCommitError::DomainHeightMismatch,
                ));
            }
            if num_chunks <= 1
                || !num_chunks.is_power_of_two()
                || num_chunks > domain_size
                || !domain_size.is_multiple_of(num_chunks)
            {
                return Err(SecretHidingFriCommitError::Inner(
                    crate::SecretFriCommitError::InvalidQuotientChunkCount,
                ));
            }
            let expected_len = domain_size
                .checked_mul(width)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            if evaluations.values.len() != expected_len {
                return Err(SecretMatrixError::InvalidShape.into());
            }
            self.check_hiding_budget(domain_size / num_chunks, Challenge::DIMENSION, 1)?;
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
                quotients.next().map(|quotient| quotient.into_parts().1).ok_or(
                    SecretHidingFriCommitError::Inner(
                        crate::SecretFriCommitError::QuotientCountMismatch,
                    ),
                )
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
        GenerationError: core::fmt::Debug,
    {
        if plans.is_empty() {
            return Err(SecretGeneratedQuotientCommitError::Commitment(
                SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::EmptyQuotientBatch),
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
            if num_chunks <= 1
                || !num_chunks.is_power_of_two()
                || num_chunks > domain_size
                || !domain_size.is_multiple_of(num_chunks)
            {
                return Err(SecretGeneratedQuotientCommitError::Commitment(
                    SecretHidingFriCommitError::Inner(
                        crate::SecretFriCommitError::InvalidQuotientChunkCount,
                    ),
                ));
            }
            domain_size.checked_mul(width).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::DimensionOverflow.into(),
                )
            })?;
            self.check_hiding_budget(domain_size / num_chunks, Challenge::DIMENSION, 1)
                .map_err(|error| {
                    SecretGeneratedQuotientCommitError::Commitment(
                        SecretHidingFriCommitError::Hiding(error),
                    )
                })?;
            total_chunks = total_chunks.checked_add(num_chunks).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::DimensionOverflow.into(),
                )
            })?;
        }

        let added_bits = self.inner.fri.log_blowup.checked_add(1).ok_or_else(|| {
            SecretGeneratedQuotientCommitError::Commitment(SecretHidingFriCommitError::Inner(
                crate::SecretFriCommitError::Dft(p3_dft::SecretDftError::SizeOverflow),
            ))
        })?;
        let added_bits_u32 = u32::try_from(added_bits).map_err(|_| {
            SecretGeneratedQuotientCommitError::Commitment(SecretHidingFriCommitError::Inner(
                crate::SecretFriCommitError::Dft(p3_dft::SecretDftError::SizeOverflow),
            ))
        })?;
        let mut lde_dimensions = Vec::new();
        lde_dimensions
            .try_reserve_exact(total_chunks)
            .map_err(|_| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretHidingFriCommitError::AllocationFailed,
                )
            })?;
        for plan in &plans {
            let randomized_width = plan
                .width()
                .checked_add(self.num_random_codewords)
                .ok_or_else(|| {
                    SecretGeneratedQuotientCommitError::Commitment(
                        SecretMatrixError::DimensionOverflow.into(),
                    )
                })?;
            let rows_per_chunk = plan.domain().size() / plan.num_chunks();
            let lde_height = rows_per_chunk.checked_shl(added_bits_u32).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(SecretHidingFriCommitError::Inner(
                    crate::SecretFriCommitError::Dft(p3_dft::SecretDftError::SizeOverflow),
                ))
            })?;
            for _ in 0..plan.num_chunks() {
                lde_dimensions.push(Dimensions {
                    width: randomized_width,
                    height: lde_height,
                });
            }
        }
        let mut lde_stream =
            <InputMmcs as SecretStreamingRecomputableMmcs<Val>>::begin_streaming_commitment::<
                RowMajorMatrix<Val>,
            >(&self.inner.mmcs, lde_dimensions)
            .map_err(|error| {
                SecretGeneratedQuotientCommitError::Commitment(SecretHidingFriCommitError::Inner(
                    crate::SecretFriCommitError::Mmcs(error),
                ))
            })?;
        let mut coefficients = Vec::new();
        coefficients.try_reserve_exact(total_chunks).map_err(|_| {
            SecretGeneratedQuotientCommitError::Commitment(
                SecretHidingFriCommitError::AllocationFailed,
            )
        })?;

        for (index, plan) in plans.into_iter().enumerate() {
            let quotient_evaluations =
                generate(index, &plan).map_err(SecretGeneratedQuotientCommitError::Generation)?;
            let (quotient_domain, planned_width, num_chunks) = plan.into_parts();
            let preparation = (|| -> Result<(), Self::SecretCommitError> {
                let domain_size = quotient_domain.size();
                let width = quotient_evaluations.width;
                if width != planned_width {
                    return Err(SecretMatrixError::InvalidShape.into());
                }
                if width == 0 || !quotient_evaluations.values.len().is_multiple_of(width) {
                    return Err(SecretMatrixError::InvalidShape.into());
                }
                let height = quotient_evaluations.values.len() / width;
                if height != domain_size {
                    return Err(SecretHidingFriCommitError::Inner(
                        crate::SecretFriCommitError::DomainHeightMismatch,
                    ));
                }
                if num_chunks <= 1
                    || !num_chunks.is_power_of_two()
                    || num_chunks > domain_size
                    || !domain_size.is_multiple_of(num_chunks)
                {
                    return Err(SecretHidingFriCommitError::Inner(
                        crate::SecretFriCommitError::InvalidQuotientChunkCount,
                    ));
                }

                let expected_len = domain_size
                    .checked_mul(width)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                if quotient_evaluations.values.len() != expected_len {
                    return Err(SecretMatrixError::InvalidShape.into());
                }
                let rows_per_chunk = domain_size / num_chunks;
                self.check_hiding_budget(rows_per_chunk, Challenge::DIMENSION, 1)?;
                let values_per_chunk = rows_per_chunk
                    .checked_mul(width)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let domains = quotient_domain.split_domains(num_chunks);
                if domains.len() != num_chunks {
                    return Err(SecretMatrixError::InvalidShape.into());
                }

                let mut chunks = Vec::new();
                chunks
                    .try_reserve_exact(num_chunks)
                    .map_err(|_| SecretHidingFriCommitError::AllocationFailed)?;
                for _ in 0..num_chunks {
                    let mut values = Vec::new();
                    values
                        .try_reserve_exact(values_per_chunk)
                        .map_err(|_| SecretMatrixError::AllocationFailed)?;
                    values.resize(values_per_chunk, Val::ZERO);
                    chunks.push(SecretProverData::new(RowMajorMatrix::new(values, width)));
                }

                for source_row_index in 0..domain_size {
                    let chunk_index = source_row_index % num_chunks;
                    let destination_row_index = source_row_index / num_chunks;
                    let source_start = source_row_index
                        .checked_mul(width)
                        .ok_or(SecretMatrixError::DimensionOverflow)?;
                    let source_end = source_start
                        .checked_add(width)
                        .ok_or(SecretMatrixError::DimensionOverflow)?;
                    let destination_start = destination_row_index
                        .checked_mul(width)
                        .ok_or(SecretMatrixError::DimensionOverflow)?;
                    let destination_end = destination_start
                        .checked_add(width)
                        .ok_or(SecretMatrixError::DimensionOverflow)?;
                    let source = quotient_evaluations
                        .values
                        .get(source_start..source_end)
                        .ok_or(SecretMatrixError::InvalidShape)?;
                    let destination = chunks
                        .get_mut(chunk_index)
                        .and_then(|chunk| chunk.values.get_mut(destination_start..destination_end))
                        .ok_or(SecretMatrixError::InvalidShape)?;
                    destination.copy_from_slice(source);
                }

                drop(quotient_evaluations);

                let cis = get_zp_cis(&domains);
                let last_chunk = num_chunks
                    .checked_sub(1)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let last_chunk_ci_inv = cis
                    .get(last_chunk)
                    .copied()
                    .ok_or(SecretMatrixError::InvalidShape)?
                    .inverse();
                let mut mul_coeffs = Vec::new();
                mul_coeffs
                    .try_reserve_exact(last_chunk)
                    .map_err(|_| SecretHidingFriCommitError::AllocationFailed)?;
                for ci in cis.iter().take(last_chunk) {
                    mul_coeffs.push(*ci * last_chunk_ci_inv);
                }

                let randomized_width = width
                    .checked_add(self.num_random_codewords)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let random_values_per_chunk = rows_per_chunk
                    .checked_mul(randomized_width)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let mut final_random_values = Vec::new();
                final_random_values
                    .try_reserve_exact(random_values_per_chunk)
                    .map_err(|_| SecretMatrixError::AllocationFailed)?;
                final_random_values.resize(random_values_per_chunk, Val::ZERO);
                let mut final_random_values = Some(SecretProverData::new(RowMajorMatrix::new(
                    final_random_values,
                    randomized_width,
                )));

                // One private stream per quotient instance matches the
                // ordinary `get_quotient_ldes` transition exactly.
                let mut rng = StdRng::from_rng(&mut *self.rng.lock());
                for chunk in &mut chunks {
                    chunk.try_widen_with_random_cols(self.num_random_codewords, &mut rng)?;
                }

                let added_bits = self.inner.fri.log_blowup.checked_add(1).ok_or(
                    SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Dft(
                        p3_dft::SecretDftError::SizeOverflow,
                    )),
                )?;
                let added_bits_u32 = u32::try_from(added_bits).map_err(|_| {
                    SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Dft(
                        p3_dft::SecretDftError::SizeOverflow,
                    ))
                })?;
                let lde_height = rows_per_chunk.checked_shl(added_bits_u32).ok_or(
                    SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Dft(
                        p3_dft::SecretDftError::SizeOverflow,
                    )),
                )?;
                let rows_per_chunk_u64 = u64::try_from(rows_per_chunk).map_err(|_| {
                    SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Dft(
                        p3_dft::SecretDftError::SizeOverflow,
                    ))
                })?;

                for (chunk_index, (domain, mut chunk)) in
                    domains.into_iter().zip(chunks).enumerate()
                {
                    let chunk_random_values = if chunk_index == last_chunk {
                        final_random_values
                            .take()
                            .ok_or(SecretMatrixError::InvalidShape)?
                    } else {
                        let random_values = SecretProverData::try_random(
                            rows_per_chunk,
                            randomized_width,
                            &mut rng,
                        )?;
                        let mul_coeff = mul_coeffs
                            .get(chunk_index)
                            .copied()
                            .ok_or(SecretMatrixError::InvalidShape)?;
                        let final_values = final_random_values
                            .as_mut()
                            .ok_or(SecretMatrixError::InvalidShape)?;
                        if final_values.values.len() != random_values.values.len() {
                            return Err(SecretMatrixError::InvalidShape.into());
                        }
                        for (final_value, source_value) in
                            final_values.values.iter_mut().zip(&random_values.values)
                        {
                            *final_value -= *source_value * mul_coeff;
                        }
                        random_values
                    };
                    let shift = Val::GENERATOR / domain.shift();
                    self.inner
                        .dft
                        .coset_idft_batch_secret(&mut chunk, shift.inverse())
                        .map_err(|error| {
                            SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Dft(
                                error,
                            ))
                        })?;
                    chunk.try_grow_zeroed_rows(lde_height)?;

                    if chunk_random_values.width != randomized_width
                        || chunk_random_values.values.len() != random_values_per_chunk
                    {
                        return Err(SecretMatrixError::InvalidShape.into());
                    }
                    let upper_row_offset = rows_per_chunk
                        .checked_mul(randomized_width)
                        .ok_or(SecretMatrixError::DimensionOverflow)?;
                    let p = shift.exp_u64(rows_per_chunk_u64);

                    for (row, p_i) in Val::GENERATOR.powers().take(rows_per_chunk).enumerate() {
                        let lower_start = row
                            .checked_mul(randomized_width)
                            .ok_or(SecretMatrixError::DimensionOverflow)?;
                        let lower_end = lower_start
                            .checked_add(randomized_width)
                            .ok_or(SecretMatrixError::DimensionOverflow)?;
                        let upper_start = upper_row_offset
                            .checked_add(lower_start)
                            .ok_or(SecretMatrixError::DimensionOverflow)?;
                        let upper_end = upper_start
                            .checked_add(randomized_width)
                            .ok_or(SecretMatrixError::DimensionOverflow)?;
                        let random_row = chunk_random_values
                            .values
                            .get(lower_start..lower_end)
                            .ok_or(SecretMatrixError::InvalidShape)?;
                        if lower_end > upper_start || upper_end > chunk.values.len() {
                            return Err(SecretMatrixError::InvalidShape.into());
                        }
                        let (lower_region, upper_region) = chunk.values.split_at_mut(upper_start);
                        let lower_row = lower_region
                            .get_mut(lower_start..lower_end)
                            .ok_or(SecretMatrixError::InvalidShape)?;
                        let upper_row = upper_region
                            .get_mut(..randomized_width)
                            .ok_or(SecretMatrixError::InvalidShape)?;
                        for ((lower_value, upper_value), random_value) in
                            lower_row.iter_mut().zip(upper_row).zip(random_row)
                        {
                            let mask = p_i * *random_value;
                            *lower_value -= mask;
                            *upper_value += p * mask;
                        }
                    }

                    let coefficient_height = rows_per_chunk
                        .checked_mul(2)
                        .ok_or(SecretMatrixError::DimensionOverflow)?;
                    let coefficient_len = coefficient_height
                        .checked_mul(randomized_width)
                        .ok_or(SecretMatrixError::DimensionOverflow)?;
                    let coefficient_source = chunk
                        .values
                        .get(..coefficient_len)
                        .ok_or(SecretMatrixError::InvalidShape)?;
                    let mut coefficient_values = Vec::new();
                    coefficient_values
                        .try_reserve_exact(coefficient_len)
                        .map_err(|_| SecretMatrixError::AllocationFailed)?;
                    coefficient_values.extend_from_slice(coefficient_source);
                    coefficients.push(SecretProverData::new(RowMajorMatrix::new(
                        coefficient_values,
                        randomized_width,
                    )));

                    self.inner
                        .dft
                        .dft_batch_secret(&mut chunk)
                        .map_err(|error| {
                            SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Dft(
                                error,
                            ))
                        })?;
                    reverse_matrix_index_bits(&mut chunk);
                    lde_stream.absorb(chunk).map_err(|error| {
                        SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Mmcs(error))
                    })?;
                }
                Ok(())
            })();
            preparation.map_err(SecretGeneratedQuotientCommitError::Commitment)?;
        }

        let detached = lde_stream.finish().map_err(|error| {
            SecretGeneratedQuotientCommitError::Commitment(SecretHidingFriCommitError::Inner(
                crate::SecretFriCommitError::Mmcs(error),
            ))
        })?;
        let inner_prover_data = SecretProverData::new(crate::SecretTwoAdicFriData::new_detached(
            detached.prover_data,
            detached.authentication_data,
            coefficients,
            self.inner.fri.log_blowup,
        ));
        Ok((
            detached.commitment,
            SecretProverData::new(HidingFriPcsProverData::new(
                inner_prover_data,
                false,
                self.num_random_codewords,
                self.inner.fri.log_blowup,
            )),
        ))
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
        GenerationError: core::fmt::Debug,
    {
        if plans.is_empty() {
            return Err(SecretGeneratedQuotientCommitError::Commitment(
                SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::EmptyQuotientBatch),
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
            if num_chunks <= 1
                || !num_chunks.is_power_of_two()
                || num_chunks > domain_size
                || !domain_size.is_multiple_of(num_chunks)
            {
                return Err(SecretGeneratedQuotientCommitError::Commitment(
                    SecretHidingFriCommitError::Inner(
                        crate::SecretFriCommitError::InvalidQuotientChunkCount,
                    ),
                ));
            }
            domain_size.checked_mul(width).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::DimensionOverflow.into(),
                )
            })?;
            self.check_hiding_budget(domain_size / num_chunks, Challenge::DIMENSION, 1)
                .map_err(|error| {
                    SecretGeneratedQuotientCommitError::Commitment(
                        SecretHidingFriCommitError::Hiding(error),
                    )
                })?;
            total_chunks = total_chunks.checked_add(num_chunks).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretMatrixError::DimensionOverflow.into(),
                )
            })?;
        }

        let added_bits = self.inner.fri.log_blowup.checked_add(1).ok_or_else(|| {
            SecretGeneratedQuotientCommitError::Commitment(SecretHidingFriCommitError::Inner(
                crate::SecretFriCommitError::Dft(p3_dft::SecretDftError::SizeOverflow),
            ))
        })?;
        let added_bits_u32 = u32::try_from(added_bits).map_err(|_| {
            SecretGeneratedQuotientCommitError::Commitment(SecretHidingFriCommitError::Inner(
                crate::SecretFriCommitError::Dft(p3_dft::SecretDftError::SizeOverflow),
            ))
        })?;
        let mut lde_dimensions = Vec::new();
        lde_dimensions
            .try_reserve_exact(total_chunks)
            .map_err(|_| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretHidingFriCommitError::AllocationFailed,
                )
            })?;
        for plan in &plans {
            let randomized_width = plan
                .width()
                .checked_add(self.num_random_codewords)
                .ok_or_else(|| {
                    SecretGeneratedQuotientCommitError::Commitment(
                        SecretMatrixError::DimensionOverflow.into(),
                    )
                })?;
            let rows_per_chunk = plan.domain().size() / plan.num_chunks();
            let lde_height = rows_per_chunk.checked_shl(added_bits_u32).ok_or_else(|| {
                SecretGeneratedQuotientCommitError::Commitment(SecretHidingFriCommitError::Inner(
                    crate::SecretFriCommitError::Dft(p3_dft::SecretDftError::SizeOverflow),
                ))
            })?;
            for _ in 0..plan.num_chunks() {
                lde_dimensions.push(Dimensions {
                    width: randomized_width,
                    height: lde_height,
                });
            }
        }
        let mut lde_stream =
            <InputMmcs as SecretStreamingRecomputableMmcs<Val>>::begin_streaming_commitment::<
                RowMajorMatrix<Val>,
            >(&self.inner.mmcs, lde_dimensions)
            .map_err(|error| {
                SecretGeneratedQuotientCommitError::Commitment(SecretHidingFriCommitError::Inner(
                    crate::SecretFriCommitError::Mmcs(error),
                ))
            })?;
        let mut coefficients = Vec::new();
        coefficients.try_reserve_exact(total_chunks).map_err(|_| {
            SecretGeneratedQuotientCommitError::Commitment(
                SecretHidingFriCommitError::AllocationFailed,
            )
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

            // Acquire the complete instance before advancing the hiding RNG.
            // The batch generator writes directly into these final chunk
            // owners, so this does not recreate a full quotient allocation.
            let mut chunks = Vec::new();
            chunks.try_reserve_exact(num_chunks).map_err(|_| {
                SecretGeneratedQuotientCommitError::Commitment(
                    SecretHidingFriCommitError::AllocationFailed,
                )
            })?;
            for (chunk_index, domain) in domains.iter().copied().enumerate() {
                let chunk = generate(instance_index, &plan, chunk_index, domain)
                    .map_err(SecretGeneratedQuotientCommitError::Generation)?;
                if chunk.width != width || chunk.values.len() != expected_len {
                    return Err(SecretGeneratedQuotientCommitError::Commitment(
                        SecretMatrixError::InvalidShape.into(),
                    ));
                }
                chunks.push(chunk);
            }

            let preparation = (|| -> Result<(), Self::SecretCommitError> {
                let cis = get_zp_cis(&domains);
                let last_chunk = num_chunks
                    .checked_sub(1)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let last_chunk_ci_inv = cis
                    .get(last_chunk)
                    .copied()
                    .ok_or(SecretMatrixError::InvalidShape)?
                    .inverse();
                let mut mul_coeffs = Vec::new();
                mul_coeffs
                    .try_reserve_exact(last_chunk)
                    .map_err(|_| SecretHidingFriCommitError::AllocationFailed)?;
                for ci in cis.iter().take(last_chunk) {
                    mul_coeffs.push(*ci * last_chunk_ci_inv);
                }

                let randomized_width = width
                    .checked_add(self.num_random_codewords)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let random_values_per_chunk = rows_per_chunk
                    .checked_mul(randomized_width)
                    .ok_or(SecretMatrixError::DimensionOverflow)?;
                let mut final_random_values = Vec::new();
                final_random_values
                    .try_reserve_exact(random_values_per_chunk)
                    .map_err(|_| SecretMatrixError::AllocationFailed)?;
                final_random_values.resize(random_values_per_chunk, Val::ZERO);
                let mut final_random_values = Some(SecretProverData::new(RowMajorMatrix::new(
                    final_random_values,
                    randomized_width,
                )));

                // One private stream per quotient instance matches the
                // ordinary `get_quotient_ldes` transition exactly.
                let mut rng = StdRng::from_rng(&mut *self.rng.lock());
                for chunk in &mut chunks {
                    chunk.try_widen_with_random_cols(self.num_random_codewords, &mut rng)?;
                }

                let lde_height = rows_per_chunk.checked_shl(added_bits_u32).ok_or(
                    SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Dft(
                        p3_dft::SecretDftError::SizeOverflow,
                    )),
                )?;
                let rows_per_chunk_u64 = u64::try_from(rows_per_chunk).map_err(|_| {
                    SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Dft(
                        p3_dft::SecretDftError::SizeOverflow,
                    ))
                })?;

                for (chunk_index, (domain, mut chunk)) in
                    domains.into_iter().zip(chunks).enumerate()
                {
                    let chunk_random_values = if chunk_index == last_chunk {
                        final_random_values
                            .take()
                            .ok_or(SecretMatrixError::InvalidShape)?
                    } else {
                        let random_values = SecretProverData::try_random(
                            rows_per_chunk,
                            randomized_width,
                            &mut rng,
                        )?;
                        let mul_coeff = mul_coeffs
                            .get(chunk_index)
                            .copied()
                            .ok_or(SecretMatrixError::InvalidShape)?;
                        let final_values = final_random_values
                            .as_mut()
                            .ok_or(SecretMatrixError::InvalidShape)?;
                        if final_values.values.len() != random_values.values.len() {
                            return Err(SecretMatrixError::InvalidShape.into());
                        }
                        for (final_value, source_value) in
                            final_values.values.iter_mut().zip(&random_values.values)
                        {
                            *final_value -= *source_value * mul_coeff;
                        }
                        random_values
                    };
                    let shift = Val::GENERATOR / domain.shift();
                    self.inner
                        .dft
                        .coset_idft_batch_secret(&mut chunk, shift.inverse())
                        .map_err(|error| {
                            SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Dft(
                                error,
                            ))
                        })?;
                    chunk.try_grow_zeroed_rows(lde_height)?;

                    if chunk_random_values.width != randomized_width
                        || chunk_random_values.values.len() != random_values_per_chunk
                    {
                        return Err(SecretMatrixError::InvalidShape.into());
                    }
                    let upper_row_offset = rows_per_chunk
                        .checked_mul(randomized_width)
                        .ok_or(SecretMatrixError::DimensionOverflow)?;
                    let p = shift.exp_u64(rows_per_chunk_u64);

                    for (row, p_i) in Val::GENERATOR.powers().take(rows_per_chunk).enumerate() {
                        let lower_start = row
                            .checked_mul(randomized_width)
                            .ok_or(SecretMatrixError::DimensionOverflow)?;
                        let lower_end = lower_start
                            .checked_add(randomized_width)
                            .ok_or(SecretMatrixError::DimensionOverflow)?;
                        let upper_start = upper_row_offset
                            .checked_add(lower_start)
                            .ok_or(SecretMatrixError::DimensionOverflow)?;
                        let upper_end = upper_start
                            .checked_add(randomized_width)
                            .ok_or(SecretMatrixError::DimensionOverflow)?;
                        let random_row = chunk_random_values
                            .values
                            .get(lower_start..lower_end)
                            .ok_or(SecretMatrixError::InvalidShape)?;
                        if lower_end > upper_start || upper_end > chunk.values.len() {
                            return Err(SecretMatrixError::InvalidShape.into());
                        }
                        let (lower_region, upper_region) = chunk.values.split_at_mut(upper_start);
                        let lower_row = lower_region
                            .get_mut(lower_start..lower_end)
                            .ok_or(SecretMatrixError::InvalidShape)?;
                        let upper_row = upper_region
                            .get_mut(..randomized_width)
                            .ok_or(SecretMatrixError::InvalidShape)?;
                        for ((lower_value, upper_value), random_value) in
                            lower_row.iter_mut().zip(upper_row).zip(random_row)
                        {
                            let mask = p_i * *random_value;
                            *lower_value -= mask;
                            *upper_value += p * mask;
                        }
                    }

                    let coefficient_height = rows_per_chunk
                        .checked_mul(2)
                        .ok_or(SecretMatrixError::DimensionOverflow)?;
                    let coefficient_len = coefficient_height
                        .checked_mul(randomized_width)
                        .ok_or(SecretMatrixError::DimensionOverflow)?;
                    let coefficient_source = chunk
                        .values
                        .get(..coefficient_len)
                        .ok_or(SecretMatrixError::InvalidShape)?;
                    let mut coefficient_values = Vec::new();
                    coefficient_values
                        .try_reserve_exact(coefficient_len)
                        .map_err(|_| SecretMatrixError::AllocationFailed)?;
                    coefficient_values.extend_from_slice(coefficient_source);
                    coefficients.push(SecretProverData::new(RowMajorMatrix::new(
                        coefficient_values,
                        randomized_width,
                    )));

                    self.inner
                        .dft
                        .dft_batch_secret(&mut chunk)
                        .map_err(|error| {
                            SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Dft(
                                error,
                            ))
                        })?;
                    reverse_matrix_index_bits(&mut chunk);
                    lde_stream.absorb(chunk).map_err(|error| {
                        SecretHidingFriCommitError::Inner(crate::SecretFriCommitError::Mmcs(error))
                    })?;
                }
                Ok(())
            })();
            preparation.map_err(SecretGeneratedQuotientCommitError::Commitment)?;
        }

        let detached = lde_stream.finish().map_err(|error| {
            SecretGeneratedQuotientCommitError::Commitment(SecretHidingFriCommitError::Inner(
                crate::SecretFriCommitError::Mmcs(error),
            ))
        })?;
        let inner_prover_data = SecretProverData::new(crate::SecretTwoAdicFriData::new_detached(
            detached.prover_data,
            detached.authentication_data,
            coefficients,
            self.inner.fri.log_blowup,
        ));
        Ok((
            detached.commitment,
            SecretProverData::new(HidingFriPcsProverData::new(
                inner_prover_data,
                false,
                self.num_random_codewords,
                self.inner.fri.log_blowup,
            )),
        ))
    }

    fn get_opt_randomization_poly_commitment_secret(
        &self,
        ext_trace_domains: impl IntoIterator<Item = Self::Domain>,
    ) -> Result<
        Option<(Self::Commitment, SecretProverData<Self::SecretData>)>,
        Self::SecretCommitError,
    > {
        let mut domains = Vec::new();
        for domain in ext_trace_domains {
            domains
                .try_reserve(1)
                .map_err(|_| SecretHidingFriCommitError::AllocationFailed)?;
            domains.push(domain);
        }
        for domain in &domains {
            self.check_hiding_budget(domain.size() / 2, Challenge::DIMENSION, 1)?;
        }

        // Finish every profile and shape calculation before advancing the
        // prover RNG. A rejected batch must not perturb later commitments.
        let width = self
            .num_random_codewords
            .checked_add(Challenge::DIMENSION)
            .ok_or(SecretMatrixError::DimensionOverflow)?;
        let mut evaluations = Vec::new();
        evaluations
            .try_reserve_exact(domains.len())
            .map_err(|_| SecretHidingFriCommitError::AllocationFailed)?;

        let mut rng = self.rng.lock();
        for domain in domains {
            // The allocation is guarded while it still contains only public
            // zeroes; random values are written only after ownership exists.
            let matrix = SecretProverData::try_random(domain.size(), width, &mut *rng)?;
            evaluations.push((domain, matrix));
        }
        drop(rng);

        let (commitment, inner_data) =
            <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as SecretPcsCommitter<
                Challenge,
                Challenger,
            >>::commit_secret(&self.inner, evaluations)
            .map_err(SecretHidingFriCommitError::Inner)?;

        Ok(Some((
            commitment,
            SecretProverData::new(HidingFriPcsProverData::new(
                inner_data,
                false,
                self.num_random_codewords,
                self.inner.fri.log_blowup,
            )),
        )))
    }
}

#[cfg(feature = "secret-memory")]
impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger, R>
    SecretUnivariateStarkPcsEvaluator<Challenge, Challenger>
    for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    StandardUniform: Distribution<Val>,
    Dft: TwoAdicSubgroupDft<Val> + SecretTwoAdicSubgroupDft<Val> + Sync,
    InputMmcs:
        HidingMmcs<Val> + Mmcs<Val, MultiProof: Sync, Error: Sync> + SecretRecomputableMmcs<Val>,
    InputMmcs::AuthenticationData<RowMajorMatrix<Val>>: zeroize::Zeroize,
    FriMmcs: HidingMmcs<Challenge> + Mmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
    R: CryptoRng + Send + Sync,
{
    type SecretEvaluationError = SecretHidingFriEvaluationError<crate::SecretFriEvaluationError>;

    fn get_evaluations_on_domain_secret(
        &self,
        prover_data: &SecretProverData<Self::SecretData>,
        idx: usize,
        domain: Self::Domain,
    ) -> Result<SecretProverData<RowMajorMatrix<Val>>, Self::SecretEvaluationError> {
        self.check_secret_prover_data_config(prover_data)?;
        let mut evaluations =
            <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as SecretUnivariateStarkPcsEvaluator<
                Challenge,
                Challenger,
            >>::get_evaluations_on_domain_secret(
                &self.inner, &prover_data.inner, idx, domain
            )
            .map_err(SecretHidingFriEvaluationError::Inner)?;
        let public_width = evaluations
            .width
            .checked_sub(self.num_random_codewords)
            .ok_or(SecretHidingFriEvaluationError::InvalidRandomizedWidth)?;
        if public_width == 0 {
            return Err(SecretHidingFriEvaluationError::InvalidRandomizedWidth);
        }
        evaluations.try_truncate_columns(public_width)?;
        Ok(evaluations)
    }

    fn get_source_evaluations_secret(
        &self,
        prover_data: &SecretProverData<Self::SecretData>,
        idx: usize,
        source_domain: Self::Domain,
        committed_domain: Self::Domain,
    ) -> Result<SecretProverData<RowMajorMatrix<Val>>, Self::SecretEvaluationError> {
        self.check_secret_prover_data_config(prover_data)?;
        let expected_committed_size = source_domain
            .size()
            .checked_mul(2)
            .ok_or(SecretMatrixError::DimensionOverflow)?;
        if committed_domain.size() != expected_committed_size
            || committed_domain.shift() != source_domain.shift()
        {
            return Err(SecretMatrixError::InvalidShape.into());
        }

        let mut randomized =
            <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as SecretUnivariateStarkPcsEvaluator<
                Challenge,
                Challenger,
            >>::get_evaluations_on_domain_secret(
                &self.inner,
                &prover_data.inner,
                idx,
                committed_domain,
            )
            .map_err(SecretHidingFriEvaluationError::Inner)?;
        let public_width = randomized
            .width
            .checked_sub(self.num_random_codewords)
            .ok_or(SecretHidingFriEvaluationError::InvalidRandomizedWidth)?;
        if public_width == 0 {
            return Err(SecretHidingFriEvaluationError::InvalidRandomizedWidth);
        }
        randomized.try_truncate_columns(public_width)?;
        randomized
            .try_select_strided_rows(0, 2, source_domain.size())
            .map_err(SecretHidingFriEvaluationError::Matrix)
    }

    fn get_evaluations_on_domain_no_random_secret(
        &self,
        prover_data: &SecretProverData<Self::SecretData>,
        idx: usize,
        domain: Self::Domain,
    ) -> Result<SecretProverData<RowMajorMatrix<Val>>, Self::SecretEvaluationError> {
        self.check_secret_prover_data_config(prover_data)?;
        <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as SecretUnivariateStarkPcsEvaluator<
            Challenge,
            Challenger,
        >>::get_evaluations_on_domain_secret(&self.inner, &prover_data.inner, idx, domain)
        .map_err(SecretHidingFriEvaluationError::Inner)
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger, R>
    UnivariateStarkPcs<Challenge, Challenger> for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: TwoAdicField + PrimeField64,
    StandardUniform: Distribution<Val>,
    Dft: TwoAdicSubgroupDft<Val> + Sync,
    InputMmcs: HidingMmcs<Val> + Mmcs<Val, MultiProof: Sync, Error: Sync>,
    FriMmcs: HidingMmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
    R: CryptoRng + Send + Sync,
{
    type EvaluationsOnDomain<'a> =
        HorizontallyTruncated<Val, RowIndexMappedView<BitReversalPerm, RowMajorMatrixCow<'a, Val>>>;

    const ZK: bool = true;

    fn verify_with_preprocessing(
        &self,
        // For each round:
        mut rounds: Vec<CommitmentOpening<Challenge, Self::Commitment, Self::Domain>>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
        preprocessed_commitment: Option<usize>,
    ) -> Result<(), Self::Error> {
        let (opened_values_for_rand_cws, inner_proof) = proof;
        if let Some(index) = preprocessed_commitment
            && index >= rounds.len()
        {
            return Err(FriError::HidingPreprocessedCommitmentOutOfBounds {
                index,
                num_rounds: rounds.len(),
            });
        }
        if self.num_random_codewords < Challenge::DIMENSION {
            return Err(FriError::InsufficientHidingRandomCodewords {
                required: Challenge::DIMENSION,
                got: self.num_random_codewords,
            });
        }

        // Proving split each opening into a public half and a hidden half.
        // - The public half lives here.
        // - The hidden half travels beside the proof as random codewords.
        //
        // Re-joining them gives the inner verifier the full openings it committed to.
        //
        //     public (per point):  [v_0, .., v_k]
        //     hidden (per point):                [r_0, .., r_m]
        //     merged:              [v_0, .., v_k,  r_0, .., r_m]
        //
        // Invariant: the halves nest identically by round, then matrix, then point.
        // Each level's length is checked before merging.
        // A mismatch returns a precise error instead of being truncated silently.

        // Level 1: one set of random openings per round.
        if opened_values_for_rand_cws.len() != rounds.len() {
            return Err(FriError::HidingRandomOpeningRoundCountMismatch {
                expected: rounds.len(),
                got: opened_values_for_rand_cws.len(),
            });
        }
        for (round_idx, (round, rand_round)) in rounds
            .iter_mut()
            .zip(opened_values_for_rand_cws.iter())
            .enumerate()
        {
            // Level 2: one set per matrix in this round.
            if rand_round.len() != round.matrices.len() {
                return Err(FriError::HidingRandomOpeningMatrixCountMismatch {
                    round: round_idx,
                    expected: round.matrices.len(),
                    got: rand_round.len(),
                });
            }
            for (matrix_idx, (mat, rand_mat)) in
                round.matrices.iter_mut().zip(rand_round.iter()).enumerate()
            {
                let mask_height = mat.domain.size() / 2;
                if !self
                    .hiding_budget(Challenge::DIMENSION, mat.points.len())
                    .is_some_and(|required| mask_height >= required)
                {
                    return Err(FriError::HidingBudgetExceeded {
                        round: round_idx,
                        matrix: matrix_idx,
                        mask_height,
                        num_queries: self.inner.fri.num_queries,
                        num_opening_points: mat.points.len(),
                        extension_degree: Challenge::DIMENSION,
                    });
                }
                // Level 3: one set per opening point of this matrix.
                if rand_mat.len() != mat.points.len() {
                    return Err(FriError::HidingRandomOpeningPointCountMismatch {
                        round: round_idx,
                        matrix: matrix_idx,
                        expected: mat.points.len(),
                        got: rand_mat.len(),
                    });
                }
                // Only trusted preprocessing metadata exempts a request from random values.
                let expected = if preprocessed_commitment == Some(round_idx) {
                    0
                } else {
                    self.num_random_codewords
                };
                for (point_idx, (point, rand_point)) in
                    mat.points.iter_mut().zip(rand_mat.iter()).enumerate()
                {
                    if rand_point.len() != expected {
                        return Err(FriError::HidingRandomOpeningValueCountMismatch {
                            round: round_idx,
                            matrix: matrix_idx,
                            point: point_idx,
                            expected,
                            got: rand_point.len(),
                        });
                    }
                    point.values.extend(rand_point);
                }
            }
        }
        self.inner.verify(rounds, inner_proof, challenger)
    }

    fn log_max_trace_height(&self) -> usize {
        <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as UnivariateStarkPcs<Challenge, Challenger>>::log_max_trace_height(
            &self.inner)
    }

    fn log_min_trace_height(&self) -> usize {
        // Hiding only adds randomness rows, so the bound is the wrapped scheme's.
        <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as UnivariateStarkPcs<Challenge, Challenger>>::log_min_trace_height(
            &self.inner)
    }

    fn commit_preprocessing(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
        let evaluations: Vec<_> = evaluations.into_iter().collect();
        for (_, mat) in &evaluations {
            self.check_hiding_budget(mat.height(), Challenge::DIMENSION, 1)?;
        }
        // Pad values with zero columns instead of random columns.
        let padded_evals = evaluations
            .into_iter()
            .map(|(domain, mat)| {
                let mat_width = mat.width();
                // Let `w` and `h` be the width and height of the original matrix. The padded matrix should have height `2h` and width `w`.
                // To generate it, we add `w` zero columns to the original matrix, then reshape it by setting the width to `w`.
                // All columns are added on the right hand side so, after reshaping, this has the net effect of adding interleaving the original trace with zero rows.
                let mut padded_evaluation = mat.with_zero_cols(mat_width);
                padded_evaluation.width = mat_width;
                (domain, padded_evaluation)
            })
            .collect::<Vec<_>>();

        let (commitment, data) = Pcs::<Challenge, Challenger>::commit(&self.inner, padded_evals)?;
        Ok((
            commitment,
            HidingFriPcsProverData::new(
                data,
                true,
                self.num_random_codewords,
                self.inner.fri.log_blowup,
            ),
        ))
    }

    /// Get the quotient polynomial LDEs. We first decompose the quotient polynomial into
    /// `num_chunks` many smaller polynomials each of degree `degree / num_chunks`.
    /// These quotient polynomials are then randomized as explained in Section 4.2 of
    /// <https://eprint.iacr.org/2024/1037.pdf>.
    ///
    /// ### Arguments
    /// - `quotient_domain` the domain of the quotient polynomial.
    /// - `quotient_evaluations` the evaluations of the quotient polynomial over the domain. This should be in
    ///   standard (not bit-reversed) order.
    /// - `num_chunks` the number of smaller polynomials to decompose the quotient polynomial into.
    ///
    /// # Panics
    /// This function panics if `num_chunks` is either `0` or `1`. The first case makes no logical
    /// sense and in the second case, the resulting commitment would not be hiding.
    fn get_quotient_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
        num_chunks: usize,
    ) -> Result<Vec<RowMajorMatrix<Val>>, Self::ProverError> {
        assert!(
            num_chunks > 1,
            "num_chunks must be > 1 to preserve hiding (got {num_chunks})"
        );
        let (domains, evaluations): (Vec<_>, Vec<_>) = evaluations.into_iter().unzip();
        // The mask of chunk `num_chunks - 1` balances the masks of the chunks before it.
        // The random draws end with a zero block for the last evaluation.
        // Both describe one chunk only when the two counts agree.
        //
        // Why: with fewer chunks than evaluations, the masks no longer cancel.
        // With more, the indexing panics.
        if domains.len() != num_chunks {
            return Err(HidingFriProverError::QuotientChunkCountMismatch {
                expected: num_chunks,
                got: domains.len(),
            });
        }
        for mat in &evaluations {
            self.check_hiding_budget(mat.height(), Challenge::DIMENSION, 1)?;
        }
        let cis = get_zp_cis(&domains);
        let last_chunk = num_chunks - 1;
        let last_chunk_ci_inv = cis[last_chunk].inverse();
        let mul_coeffs = (0..last_chunk)
            .map(|i| cis[i] * last_chunk_ci_inv)
            .collect_vec();

        // Fork a private RNG so the shared lock is released before the parallel DFTs.
        //
        // Why: this runs inside the batch prover's parallel loop over AIR instances.
        // A waiting rayon worker may run another instance's call on this same PCS.
        // That call would spin forever on a lock its own thread still holds.
        let mut rng = StdRng::from_rng(&mut *self.rng.lock());
        let randomized_evaluations: Vec<RowMajorMatrix<Val>> = evaluations
            .into_iter()
            .map(|mat| mat.with_random_cols(self.num_random_codewords, &mut rng))
            .collect();
        // Add random values to the LDE evaluations as described in https://eprint.iacr.org/2024/1037.pdf.
        // If we have `d` chunks, let q'_i(X) = q_i(X) + v_H_i(X) * t_i(X) where t(X) is random, for 1 <= i < d.
        // q'_d(X) = q_d(X) - v_H_d(X) c_i \sum t_i(X) where c_i is a Lagrange normalization constant.
        let h = randomized_evaluations[0].height();
        let w = randomized_evaluations[0].width();
        let mut all_random_values = (0..(randomized_evaluations.len() - 1) * h * w)
            .map(|_| rng.random())
            .chain(core::iter::repeat_n(Val::ZERO, h * w))
            .collect::<Vec<_>>();

        // Set the random values for the final chunk accordingly
        for j in 0..last_chunk {
            let mul_coeff = mul_coeffs[j];
            for k in 0..h * w {
                let t = all_random_values[j * h * w + k] * mul_coeff;
                all_random_values[last_chunk * h * w + k] -= t;
            }
        }

        Ok(domains
            .into_iter()
            .zip(randomized_evaluations)
            .enumerate()
            .map(|(i, (domain, evals))| {
                assert_eq!(domain.size(), evals.height());
                let shift = Val::GENERATOR / domain.shift();
                let random_values = &all_random_values[i * h * w..(i + 1) * h * w];

                // The quotient chunk and its mask are both extended onto the same coset, and the
                // DFT is linear, so they share a single forward transform. Recovering the chunk's
                // coefficients with `shift.inverse()` scales coefficient `j` by `shift^j`, placing
                // them on the coset exactly as `coset_lde_batch(evals, log_blowup + 1, shift)` would.
                //
                // Why the coset LDE's own coefficient hook cannot host this:
                //
                // - That hook runs before the buffer is zero-extended, so the mask's upper half
                //   has no room.
                // - The coset scaling applied afterwards would scale the mask by `shift^j` too.
                let mut lde_coeffs = self
                    .inner
                    .dft
                    .coset_idft_batch(evals, shift.inverse())
                    .values;
                lde_coeffs.resize((h * w) << (self.inner.fri.log_blowup + 1), Val::ZERO);

                // Add in the coefficients of `v_H(X) * r(X)`, where:
                // - `v_H` is the coset vanishing polynomial, here equal to (GENERATOR * X / domain.shift)^n - 1,
                // - and `r` is a random polynomial.
                // These are already coset-adjusted, so they are not scaled by `shift^j`.
                let p = shift.exp_u64(h as u64);
                Val::GENERATOR
                    .powers()
                    .take(h)
                    .enumerate()
                    .for_each(|(i, p_i)| {
                        for j in 0..w {
                            let mul_coeff = p_i * random_values[i * w + j];
                            lde_coeffs[i * w + j] -= mul_coeff;
                            lde_coeffs[(h + i) * w + j] += p * mul_coeff;
                        }
                    });

                // Commit to the bit-reversed LDE.
                self.inner
                    .dft
                    .dft_batch(DenseMatrix::new(lde_coeffs, w))
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect())
    }

    /// The LDEs must contain fresh masks, as produced by [`Self::get_quotient_ldes`].
    /// Recommitting copies of the same masked LDEs does not refresh their hiding budget.
    fn commit_ldes(
        &self,
        ldes: Vec<RowMajorMatrix<Val>>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
        for lde in &ldes {
            self.check_hiding_budget(self.lde_mask_height(lde.height()), Challenge::DIMENSION, 1)?;
        }
        let (commitment, data) =
            UnivariateStarkPcs::<Challenge, Challenger>::commit_ldes(&self.inner, ldes)?;
        Ok((
            commitment,
            HidingFriPcsProverData::new(
                data,
                false,
                self.num_random_codewords,
                self.inner.fri.log_blowup,
            ),
        ))
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        self.assert_prover_data_config(prover_data);
        let inner_evals =
            <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as UnivariateStarkPcs<
                Challenge,
                Challenger,
            >>::get_evaluations_on_domain(&self.inner, &prover_data.inner, idx, domain);
        let inner_width = inner_evals.width();
        // Truncate off the columns representing random codewords we added in `commit` above.
        // The unwrap is safe as inner_width - self.num_random_codewords <= inner_width.
        HorizontallyTruncated::new(inner_evals, inner_width - self.num_random_codewords).unwrap()
    }

    fn get_evaluations_on_domain_no_random<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        self.assert_prover_data_config(prover_data);
        let inner_evals =
            <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as UnivariateStarkPcs<
                Challenge,
                Challenger,
            >>::get_evaluations_on_domain(&self.inner, &prover_data.inner, idx, domain);
        let inner_width = inner_evals.width();

        HorizontallyTruncated::new(inner_evals, inner_width).unwrap()
    }

    fn open_with_preprocessing(
        &self,
        // For each round,
        rounds: Vec<OpeningRequest<'_, Self::ProverData, Challenge>>,
        challenger: &mut Challenger,
        preprocessed_commitment: Option<usize>,
    ) -> Result<(OpenedValues<Challenge>, Self::Proof), Self::ProverError> {
        for (round_idx, round) in rounds.iter().enumerate() {
            self.assert_prover_data_config(round.prover_data);
            assert_eq!(
                round.prover_data.preprocessed,
                preprocessed_commitment == Some(round_idx),
                "preprocessing index must identify public preprocessing data"
            );
            let matrices = self.inner.mmcs.get_matrices(&round.prover_data.inner);
            assert_eq!(matrices.len(), round.points.len());
            for (matrix_idx, (matrix, points)) in matrices.iter().zip(&round.points).enumerate() {
                self.inner
                    .fri
                    .validate_input_height(log2_strict_usize(matrix.height()))?;
                // A matrix with no claim has no proof that verifies.
                // Rejecting it here leaves every single-use mask unclaimed.
                if points.is_empty() {
                    return Err(FriProverError::MatrixWithoutOpeningPoints {
                        batch: round_idx,
                        matrix: matrix_idx,
                    }
                    .into());
                }
                self.check_hiding_budget(
                    self.lde_mask_height(matrix.height()),
                    Challenge::DIMENSION,
                    points.len(),
                )?;
            }
        }
        // Consume before producing any openings. Repeated requests and concurrent calls must
        // not turn a per-proof budget into multiple disclosures of the same mask.
        for round in &rounds {
            if !round.prover_data.preprocessed {
                assert!(
                    !round.prover_data.opened.swap(true, Ordering::Relaxed),
                    "private hiding commitment was already opened; recommit with fresh masks"
                );
            }
        }
        let rounds = rounds
            .into_iter()
            .map(|round| OpeningRequest {
                prover_data: &round.prover_data.inner,
                points: round.points,
            })
            .collect();
        let (mut inner_opened_values, inner_proof) =
            self.inner
                .open_with_preprocessing(rounds, challenger, preprocessed_commitment)?;
        // inner_opened_values includes opened values for the random codewords. Those should be
        // hidden from our caller, so we split them off and store them in the proof.
        let opened_values_rand = inner_opened_values
            .iter_mut()
            .enumerate()
            .map(|(idx, opened_values_for_round)| {
                opened_values_for_round
                    .iter_mut()
                    .map(|opened_values_for_mat| {
                        opened_values_for_mat
                            .iter_mut()
                            .map(|opened_values_for_point| {
                                let num_random_codewords = if preprocessed_commitment == Some(idx) {
                                    0
                                } else {
                                    self.num_random_codewords
                                };
                                let split = opened_values_for_point.len() - num_random_codewords;
                                opened_values_for_point.drain(split..).collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        Ok((inner_opened_values, (opened_values_rand, inner_proof)))
    }

    fn get_opt_randomization_poly_commitment(
        &self,
        ext_trace_domains: impl IntoIterator<Item = Self::Domain>,
    ) -> Result<Option<(Self::Commitment, Self::ProverData)>, Self::ProverError> {
        let ext_trace_domains: Vec<_> = ext_trace_domains.into_iter().collect();
        for domain in &ext_trace_domains {
            self.check_hiding_budget(domain.size() / 2, Challenge::DIMENSION, 1)?;
        }
        let random_input_vals = ext_trace_domains
            .into_iter()
            .map(|domain| {
                let m = DenseMatrix::rand(
                    &mut *self.rng.lock(),
                    domain.size(),
                    self.num_random_codewords + Challenge::DIMENSION,
                );

                (domain, m)
            })
            .collect::<Vec<_>>();

        let r_commit_and_data =
            Pcs::<Challenge, Challenger>::commit(&self.inner, random_input_vals)?;
        Ok(Some((
            r_commit_and_data.0,
            HidingFriPcsProverData::new(
                r_commit_and_data.1,
                false,
                self.num_random_codewords,
                self.inner.fri.log_blowup,
            ),
        )))
    }

    fn build_periodic_lde_table(
        &self,
        periodic_cols: &[Vec<Val>],
        trace_domain: Self::Domain,
        quotient_domain: Self::Domain,
    ) -> p3_commit::PeriodicLdeTable<Val> {
        UnivariateStarkPcs::<Challenge, Challenger>::build_periodic_lde_table(
            &self.inner,
            periodic_cols,
            trace_domain,
            quotient_domain,
        )
    }
}

/// Compute the normalizing constants for the Langrange selectors of the provided domains.
/// See Section 4.2 of <https://eprint.iacr.org/2024/1037.pdf> for more details.
fn get_zp_cis<D: PolynomialSpace>(qc_domains: &[D]) -> Vec<p3_commit::Val<D>> {
    batch_multiplicative_inverse(
        &qc_domains
            .iter()
            .enumerate()
            .map(|(i, domain)| {
                qc_domains
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, other_domain)| {
                        other_domain.vanishing_poly_at_point(domain.first_point())
                    })
                    .product()
            })
            .collect::<Vec<_>>(),
    )
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::vec;
    use core::convert::Infallible;
    use core::sync::atomic::AtomicUsize;
    #[cfg(feature = "secret-memory")]
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    #[cfg(feature = "secret-memory")]
    use p3_commit::{
        SecretDetachedPcsCommitter, SecretPcsCommitter, SecretPcsMixedOpener,
        SecretPcsMixedOpeningRequest, SecretPcsOpener, SecretProverData,
        SecretUnivariateStarkPcsCommitter, SecretUnivariateStarkPcsEvaluator,
    };
    use p3_dft::{Radix2Bowers, Radix2DFTSmallBatch, Radix2Dit, Radix2DitParallel};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_merkle_tree::MerkleTreeHidingMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::rngs::{SmallRng, StdRng};
    use rand::{RngExt, SeedableRng, TryCryptoRng, TryRng};
    use spin::Once;

    #[cfg(feature = "secret-memory")]
    struct BorrowedCanary<'a>(&'a mut [u32]);

    #[cfg(feature = "secret-memory")]
    impl zeroize::Zeroize for BorrowedCanary<'_> {
        fn zeroize(&mut self) {
            self.0.fill(0);
        }
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn hiding_prover_data_zeroize_erases_inner_data_and_fails_closed() {
        let mut canary = [0xdead_beef_u32, 0x5245_414c, 0x4c59_4d45, 0x0102_0304];
        {
            let mut prover_data =
                HidingFriPcsProverData::new(BorrowedCanary(&mut canary), false, 8, 3);

            zeroize::Zeroize::zeroize(&mut prover_data);

            assert!(prover_data.opened.load(Ordering::SeqCst));
            assert_eq!(prover_data.num_random_codewords, 0);
            assert_eq!(prover_data.log_blowup, 0);
        }
        assert_eq!(canary, [0_u32; 4]);
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn hiding_prover_data_owner_erases_inner_data_on_normal_drop() {
        let mut canary = [0xdead_beef_u32, 0x5245_414c, 0x4c59_4d45, 0x0102_0304];
        {
            let prover_data = HidingFriPcsProverData::new(
                BorrowedCanary(&mut canary),
                false,
                NUM_RANDOM_CODEWORDS,
                3,
            );
            let _owner = SecretProverData::new(prover_data);
        }

        assert_eq!(canary, [0_u32; 4]);
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn repeated_hiding_prover_data_owners_erase_independently() {
        for marker in [0x1111_1111_u32, 0x2222_2222, 0x3333_3333] {
            let mut canary = [marker, 0xdead_beef, 0x5245_414c, 0x4c59_4d45];
            {
                let prover_data = HidingFriPcsProverData::new(
                    BorrowedCanary(&mut canary),
                    false,
                    NUM_RANDOM_CODEWORDS,
                    3,
                );
                let _owner = SecretProverData::new(prover_data);
            }
            assert_eq!(canary, [0_u32; 4]);
        }
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn hiding_prover_data_owner_erases_inner_data_during_unwind() {
        let mut canary = [0xdead_beef_u32, 0x5245_414c, 0x4c59_4d45, 0x0102_0304];
        let result = catch_unwind(AssertUnwindSafe(|| {
            let prover_data = HidingFriPcsProverData::new(BorrowedCanary(&mut canary), false, 8, 3);
            let _owner = SecretProverData::new(prover_data);
            panic!("exercise hiding prover-data unwind cleanup");
        }));

        assert!(result.is_err());
        assert_eq!(canary, [0_u32; 4]);
    }

    use super::*;

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    /// Salted from a fixed seed, so the hiding tests stay deterministic.
    type ValMmcs = MerkleTreeHidingMmcs<
        <Val as Field>::Packing,
        <Val as Field>::Packing,
        MyHash,
        MyCompress,
        StdRng,
        2,
        8,
        4,
    >;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Dft = Radix2Dit<Val>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type MyPcs = HidingFriPcs<Val, Dft, ValMmcs, ChallengeMmcs, StdRng>;
    #[cfg(feature = "secret-memory")]
    type SecretHidingPcs = HidingFriPcs<Val, Radix2Bowers, ValMmcs, ChallengeMmcs, StdRng>;
    /// Same wrapper, left generic over the DFT backend.
    type HidingPcs<D> = HidingFriPcs<Val, D, ValMmcs, ChallengeMmcs, StdRng>;
    type Commitment = <ValMmcs as Mmcs<Val>>::Commitment;
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    /// Public opening claims (the `rounds` argument): per matrix, its domain and
    /// the `(point, values)` pairs.
    type Claims = Vec<(Domain, Vec<(Challenge, Vec<Challenge>)>)>;
    type Proof = <MyPcs as Pcs<Challenge, Challenger>>::Proof;
    type TestError =
        FriError<<ChallengeMmcs as Mmcs<Challenge>>::Error, <ValMmcs as Mmcs<Val>>::Error>;

    /// Random codewords appended per matrix.
    ///
    /// Must be `> 0` so each opening splits into a public part (`rounds`) and a
    /// hidden part (`proof.0`) — the split `verify` re-merges and whose shape the
    /// new error variants guard.
    const NUM_RANDOM_CODEWORDS: usize = 4;

    #[cfg(feature = "secret-memory")]
    fn make_secret_hiding_pcs(num_random_codewords: usize) -> SecretHidingPcs {
        let mut parameter_rng = SmallRng::seed_from_u64(0x5345_4352_4554_4849);
        let permutation = Perm::new_from_rng_128(&mut parameter_rng);
        let hash = MyHash::new(permutation.clone());
        let compress = MyCompress::new(permutation);
        let val_mmcs = ValMmcs::new(
            hash.clone(),
            compress.clone(),
            0,
            StdRng::seed_from_u64(0x5641_4c5f_4d4d_4353),
        );
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(
            hash,
            compress,
            0,
            StdRng::seed_from_u64(0x4348_414c_4d4d_4353),
        ));
        SecretHidingPcs::new(
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
            num_random_codewords,
            StdRng::seed_from_u64(0x4d41_534b_5f52_4e47),
        )
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_commit_matches_standard_commit_and_erases_retained_lde() {
        let standard_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let guarded_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let domain = <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &standard_pcs,
            32,
        );
        let matrix = RowMajorMatrix::new((1_u8..=32).map(Val::from_u8).collect_vec(), 2);

        let (expected_commitment, _standard_data) =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::commit(
                &standard_pcs,
                [(domain, matrix.clone())],
            )
            .expect("valid standard hiding commitment must succeed");
        let (commitment, mut prover_data) =
            <SecretHidingPcs as SecretPcsCommitter<Challenge, Challenger>>::commit_secret(
                &guarded_pcs,
                vec![(domain, SecretProverData::new(matrix))],
            )
            .expect("valid guarded hiding commitment must succeed");
        assert_eq!(commitment, expected_commitment);

        zeroize::Zeroize::zeroize(&mut prover_data);
        let retained = guarded_pcs
            .inner
            .mmcs
            .get_matrices_secret(&prover_data.inner.commitment_data);
        assert_eq!(retained.len(), 1);
        assert!(retained[0].values.iter().all(|value| *value == Val::ZERO));
        assert_eq!(retained[0].width, 0);
        assert!(prover_data.opened.load(Ordering::SeqCst));
        assert_eq!(commitment, expected_commitment);
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_detached_commit_matches_retained_commit_and_rng_state() {
        let retained_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let detached_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let domain = <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &retained_pcs,
            32,
        );
        let matrices = [2, 3].map(|width| {
            let value_count = 16_usize
                .checked_mul(width)
                .expect("fixed test dimensions must fit");
            RowMajorMatrix::new(
                (0..value_count)
                    .map(|index| Val::from_usize(index + 1))
                    .collect_vec(),
                width,
            )
        });

        let retained_evaluations = matrices
            .clone()
            .into_iter()
            .map(|matrix| (domain, SecretProverData::new(matrix)))
            .collect();
        let (retained_commitment, retained_data) =
            <SecretHidingPcs as SecretPcsCommitter<Challenge, Challenger>>::commit_secret(
                &retained_pcs,
                retained_evaluations,
            )
            .expect("valid retained hiding commitment must succeed");
        let detached_evaluations = matrices
            .into_iter()
            .map(|matrix| (domain, SecretProverData::new(matrix)))
            .collect();
        let (detached_commitment, detached_data) = <SecretHidingPcs as SecretDetachedPcsCommitter<
            Challenge,
            Challenger,
        >>::commit_detached_secret(&detached_pcs, detached_evaluations)
        .expect("valid detached hiding commitment must succeed");

        assert_eq!(detached_commitment, retained_commitment);
        assert!(retained_data.inner.authentication_data.is_none());
        assert!(detached_data.inner.authentication_data.is_some());
        assert!(
            detached_pcs
                .inner
                .mmcs
                .get_matrices_secret(&detached_data.inner.commitment_data)
                .is_empty()
        );
        assert_eq!(
            detached_pcs.rng.lock().random::<u64>(),
            retained_pcs.rng.lock().random::<u64>()
        );
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_detached_commit_preflights_every_domain_before_rng() {
        const SOURCE_HEIGHT: usize = 16;
        const WIDTH: usize = 2;
        let pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let control = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let valid_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 32);
        let invalid_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 16);
        let value_count = SOURCE_HEIGHT
            .checked_mul(WIDTH)
            .expect("fixed test dimensions must fit");
        let first = RowMajorMatrix::new(vec![Val::ONE; value_count], WIDTH);
        let second = RowMajorMatrix::new(vec![Val::TWO; value_count], WIDTH);

        let result = <SecretHidingPcs as SecretDetachedPcsCommitter<Challenge, Challenger>>::
            commit_detached_secret(
                &pcs,
                vec![
                    (valid_domain, SecretProverData::new(first)),
                    (invalid_domain, SecretProverData::new(second)),
                ],
            );

        assert!(matches!(
            result,
            Err(SecretHidingFriCommitError::Inner(
                crate::SecretFriCommitError::DomainHeightMismatch
            ))
        ));
        assert_eq!(
            pcs.rng.lock().random::<u64>(),
            control.rng.lock().random::<u64>()
        );
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_domain_evaluation_matches_ordinary_width_semantics() {
        const COMMITTED_DOMAIN_SIZE: usize = 32;
        const TARGET_DOMAIN_SIZE: usize = 64;
        const PUBLIC_WIDTH: usize = 2;
        let standard_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let guarded_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let domain = <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &standard_pcs,
            COMMITTED_DOMAIN_SIZE,
        );
        let matrix = RowMajorMatrix::new((1_u8..=32).map(Val::from_u8).collect_vec(), PUBLIC_WIDTH);
        let (ordinary_commitment, ordinary_data) =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::commit(
                &standard_pcs,
                [(domain, matrix.clone())],
            )
            .expect("valid ordinary hiding commitment must succeed");
        let (guarded_commitment, guarded_data) =
            <SecretHidingPcs as SecretPcsCommitter<Challenge, Challenger>>::commit_secret(
                &guarded_pcs,
                vec![(domain, SecretProverData::new(matrix))],
            )
            .expect("valid guarded hiding commitment must succeed");
        assert_eq!(guarded_commitment, ordinary_commitment);

        let target_domain = domain.create_disjoint_domain(TARGET_DOMAIN_SIZE);
        let expected_public = <SecretHidingPcs as UnivariateStarkPcs<
            Challenge,
            Challenger,
        >>::get_evaluations_on_domain(&standard_pcs, &ordinary_data, 0, target_domain)
        .to_row_major_matrix();
        let expected_full = <SecretHidingPcs as UnivariateStarkPcs<Challenge, Challenger>>::get_evaluations_on_domain_no_random(
            &standard_pcs,
            &ordinary_data,
            0,
            target_domain,
        )
        .to_row_major_matrix();

        let actual_public = <SecretHidingPcs as SecretUnivariateStarkPcsEvaluator<
            Challenge,
            Challenger,
        >>::get_evaluations_on_domain_secret(
            &guarded_pcs, &guarded_data, 0, target_domain
        )
        .expect("guarded hiding evaluation must succeed");
        let actual_full = <SecretHidingPcs as SecretUnivariateStarkPcsEvaluator<
            Challenge,
            Challenger,
        >>::get_evaluations_on_domain_no_random_secret(
            &guarded_pcs,
            &guarded_data,
            0,
            target_domain,
        )
        .expect("guarded untruncated hiding evaluation must succeed");

        assert_eq!(actual_public.width, PUBLIC_WIDTH);
        assert_eq!(actual_public.values, expected_public.values);
        assert_eq!(
            actual_full.width,
            PUBLIC_WIDTH
                .checked_add(NUM_RANDOM_CODEWORDS)
                .expect("fixed test width must fit")
        );
        assert_eq!(actual_full.values, expected_full.values);
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_split_cosets_are_exact_interleaved_full_domain_rows() {
        const SOURCE_SIZE: usize = 32;
        const TARGET_SIZE: usize = 128;
        const NUM_CHUNKS: usize = 8;
        const PUBLIC_WIDTH: usize = 2;

        let pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let source_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &pcs,
                SOURCE_SIZE,
            );
        let source_len = (SOURCE_SIZE / 2)
            .checked_mul(PUBLIC_WIDTH)
            .expect("fixed test dimensions must fit");
        let source = RowMajorMatrix::new(
            (0..source_len)
                .map(|index| Val::from_usize(index + 1))
                .collect_vec(),
            PUBLIC_WIDTH,
        );
        let (_, guarded_data) =
            <SecretHidingPcs as SecretPcsCommitter<Challenge, Challenger>>::commit_secret(
                &pcs,
                vec![(source_domain, SecretProverData::new(source))],
            )
            .expect("valid guarded hiding commitment must succeed");
        let full_domain = source_domain.create_disjoint_domain(TARGET_SIZE);
        let full = <SecretHidingPcs as SecretUnivariateStarkPcsEvaluator<
            Challenge,
            Challenger,
        >>::get_evaluations_on_domain_secret(&pcs, &guarded_data, 0, full_domain)
        .expect("full guarded hiding evaluation must succeed");

        for (chunk_index, chunk_domain) in full_domain
            .split_domains(NUM_CHUNKS)
            .into_iter()
            .enumerate()
        {
            let chunk = <SecretHidingPcs as SecretUnivariateStarkPcsEvaluator<
                Challenge,
                Challenger,
            >>::get_evaluations_on_domain_secret(
                &pcs, &guarded_data, 0, chunk_domain
            )
            .expect("split guarded hiding evaluation must succeed");
            assert_eq!(chunk.width, PUBLIC_WIDTH);
            assert_eq!(chunk.height(), TARGET_SIZE / NUM_CHUNKS);

            for local_row in 0..chunk.height() {
                let full_row = local_row
                    .checked_mul(NUM_CHUNKS)
                    .and_then(|row| row.checked_add(chunk_index))
                    .expect("fixed test row mapping must fit");
                for column in 0..PUBLIC_WIDTH {
                    assert_eq!(chunk.get(local_row, column), full.get(full_row, column));
                }
            }
        }
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_source_reconstruction_removes_interleaved_mask_rows() {
        const SOURCE_DOMAIN_SIZE: usize = 16;
        const COMMITTED_DOMAIN_SIZE: usize = SOURCE_DOMAIN_SIZE * 2;
        const PUBLIC_WIDTH: usize = 2;
        let pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let source_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &pcs,
                SOURCE_DOMAIN_SIZE,
            );
        let committed_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &pcs,
                COMMITTED_DOMAIN_SIZE,
            );
        let matrix = RowMajorMatrix::new((1_u8..=32).map(Val::from_u8).collect_vec(), PUBLIC_WIDTH);
        let expected = matrix.values.clone();
        let (_, guarded_data) =
            <SecretHidingPcs as SecretPcsCommitter<Challenge, Challenger>>::commit_secret(
                &pcs,
                vec![(committed_domain, SecretProverData::new(matrix))],
            )
            .expect("valid guarded hiding commitment must succeed");

        let reconstructed = <SecretHidingPcs as SecretUnivariateStarkPcsEvaluator<
            Challenge,
            Challenger,
        >>::get_source_evaluations_secret(
            &pcs, &guarded_data, 0, source_domain, committed_domain
        )
        .expect("guarded hiding source reconstruction must succeed");

        assert_eq!(reconstructed.width, PUBLIC_WIDTH);
        assert_eq!(reconstructed.height(), SOURCE_DOMAIN_SIZE);
        assert_eq!(reconstructed.values, expected);
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_source_reconstruction_rejects_domain_mismatch() {
        const SOURCE_DOMAIN_SIZE: usize = 16;
        const COMMITTED_DOMAIN_SIZE: usize = SOURCE_DOMAIN_SIZE * 2;
        let pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let source_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &pcs,
                SOURCE_DOMAIN_SIZE,
            );
        let committed_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &pcs,
                COMMITTED_DOMAIN_SIZE,
            );
        let wrong_committed_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &pcs,
                COMMITTED_DOMAIN_SIZE * 2,
            );
        let matrix = RowMajorMatrix::new(vec![Val::ONE; SOURCE_DOMAIN_SIZE], 1);
        let (_, guarded_data) =
            <SecretHidingPcs as SecretPcsCommitter<Challenge, Challenger>>::commit_secret(
                &pcs,
                vec![(committed_domain, SecretProverData::new(matrix))],
            )
            .expect("valid guarded hiding commitment must succeed");

        let result = <SecretHidingPcs as SecretUnivariateStarkPcsEvaluator<
            Challenge,
            Challenger,
        >>::get_source_evaluations_secret(
            &pcs,
            &guarded_data,
            0,
            source_domain,
            wrong_committed_domain,
        );

        assert!(matches!(
            result,
            Err(SecretHidingFriEvaluationError::Matrix(
                SecretMatrixError::InvalidShape
            ))
        ));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_domain_evaluation_rejects_profile_drift() {
        let pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 32);
        let matrix = RowMajorMatrix::new((1_u8..=32).map(Val::from_u8).collect_vec(), 2);
        let (_, mut guarded_data) =
            <SecretHidingPcs as SecretPcsCommitter<Challenge, Challenger>>::commit_secret(
                &pcs,
                vec![(domain, SecretProverData::new(matrix))],
            )
            .expect("valid guarded hiding commitment must succeed");
        guarded_data.num_random_codewords = guarded_data
            .num_random_codewords
            .checked_add(1)
            .expect("fixed test random-codeword increment must fit");

        let result = <SecretHidingPcs as SecretUnivariateStarkPcsEvaluator<
            Challenge,
            Challenger,
        >>::get_evaluations_on_domain_secret(&pcs, &guarded_data, 0, domain);

        assert!(matches!(
            result,
            Err(SecretHidingFriEvaluationError::RandomCodewordCountMismatch)
        ));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_commit_rejects_randomization_dimension_overflow() {
        let pcs = make_secret_hiding_pcs(usize::MAX);
        let domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 32);
        let matrix = SecretProverData::new(RowMajorMatrix::new(vec![Val::ONE; 16], 1));

        let result = <SecretHidingPcs as SecretPcsCommitter<Challenge, Challenger>>::commit_secret(
            &pcs,
            vec![(domain, matrix)],
        );

        assert!(matches!(
            result,
            Err(SecretHidingFriCommitError::Matrix(
                SecretMatrixError::DimensionOverflow
            ))
        ));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_commit_erases_randomized_state_after_inner_error() {
        let pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let oversized_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 64);
        let matrix = SecretProverData::new(RowMajorMatrix::new(vec![Val::ONE; 16], 1));

        let result = <SecretHidingPcs as SecretPcsCommitter<Challenge, Challenger>>::commit_secret(
            &pcs,
            vec![(oversized_domain, matrix)],
        );

        assert!(matches!(
            result,
            Err(SecretHidingFriCommitError::Inner(
                crate::SecretFriCommitError::DomainHeightMismatch
            ))
        ));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_quotient_commit_matches_standard_and_returns_detached() {
        const DOMAIN_SIZE: usize = 64;
        const NUM_CHUNKS: usize = 2;
        const WIDTH: usize = 2;

        let standard_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let guarded_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let domain = <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &standard_pcs,
            DOMAIN_SIZE,
        );
        let values = (0..DOMAIN_SIZE * WIDTH).map(Val::from_usize).collect_vec();
        let matrix = RowMajorMatrix::new(values, WIDTH);

        let (expected_commitment, _standard_data) =
            <SecretHidingPcs as UnivariateStarkPcs<Challenge, Challenger>>::commit_quotient(
                &standard_pcs,
                domain,
                matrix.clone(),
                NUM_CHUNKS,
            )
            .expect("valid standard hiding quotient commitment must succeed");
        let (commitment, mut prover_data) = <SecretHidingPcs as
            SecretUnivariateStarkPcsCommitter<Challenge, Challenger>>::commit_quotient_secret(
            &guarded_pcs,
            domain,
            SecretProverData::new(matrix),
            NUM_CHUNKS,
        )
        .expect("valid guarded hiding quotient commitment must succeed");

        assert_eq!(commitment, expected_commitment);
        assert!(
            guarded_pcs
                .inner
                .mmcs
                .get_matrices_secret(&prover_data.inner.commitment_data)
                .is_empty()
        );
        assert!(prover_data.inner.authentication_data.is_some());
        assert_eq!(
            prover_data
                .inner
                .retained_lde_dimensions()
                .expect("valid retained hiding quotient geometry must remain available")
                .len(),
            NUM_CHUNKS
        );

        zeroize::Zeroize::zeroize(&mut prover_data);
        let retained = guarded_pcs
            .inner
            .mmcs
            .get_matrices_secret(&prover_data.inner.commitment_data);
        assert!(retained.is_empty());
        assert!(prover_data.inner.retained_lde_dimensions().is_err());
        assert!(prover_data.opened.load(Ordering::SeqCst));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_quotient_batch_matches_one_standard_global_commitment() {
        const FIRST_SIZE: usize = 64;
        const FIRST_CHUNKS: usize = 2;
        const SECOND_SIZE: usize = 128;
        const SECOND_CHUNKS: usize = 4;

        let standard_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let guarded_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let first_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &standard_pcs,
                FIRST_SIZE,
            );
        let second_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &standard_pcs,
                SECOND_SIZE,
            );
        let first = RowMajorMatrix::new((0..FIRST_SIZE).map(Val::from_usize).collect_vec(), 1);
        let second = RowMajorMatrix::new(
            (FIRST_SIZE..FIRST_SIZE + SECOND_SIZE * 2)
                .map(Val::from_usize)
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
                <SecretHidingPcs as UnivariateStarkPcs<Challenge, Challenger>>::get_quotient_ldes(
                    &standard_pcs,
                    chunk_domains.into_iter().zip(chunk_matrices),
                    num_chunks,
                )
                .expect("valid ordinary hiding quotient LDE construction must succeed");
            ordinary_ldes.extend(chunk_ldes);
        }
        let (expected_commitment, _ordinary_data) =
            <SecretHidingPcs as UnivariateStarkPcs<Challenge, Challenger>>::commit_ldes(
                &standard_pcs,
                ordinary_ldes,
            )
            .expect("valid ordinary hiding global quotient commitment must succeed");

        let (commitment, mut prover_data) = <SecretHidingPcs as
            SecretUnivariateStarkPcsCommitter<Challenge, Challenger>>::commit_quotients_secret(
            &guarded_pcs,
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
        .expect("valid guarded hiding quotient batch must succeed");

        assert_eq!(commitment, expected_commitment);
        let next_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &standard_pcs,
                32,
            );
        let next_matrix = RowMajorMatrix::new(
            (0..16)
                .map(|value| Val::from_usize(value + 7))
                .collect_vec(),
            1,
        );
        let (expected_next_commitment, _expected_next_data) =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::commit(
                &standard_pcs,
                [(next_domain, next_matrix.clone())],
            )
            .expect("valid standard commitment after quotient batch must succeed");
        let (next_commitment, _next_data) =
            <SecretHidingPcs as SecretPcsCommitter<Challenge, Challenger>>::commit_secret(
                &guarded_pcs,
                vec![(next_domain, SecretProverData::new(next_matrix))],
            )
            .expect("valid guarded commitment after quotient batch must succeed");
        assert_eq!(next_commitment, expected_next_commitment);
        assert!(
            guarded_pcs
                .inner
                .mmcs
                .get_matrices_secret(&prover_data.inner.commitment_data)
                .is_empty()
        );
        assert!(prover_data.inner.authentication_data.is_some());
        assert_eq!(
            prover_data
                .inner
                .retained_lde_dimensions()
                .expect("valid retained hiding quotient geometry must remain available")
                .len(),
            FIRST_CHUNKS + SECOND_CHUNKS
        );
        zeroize::Zeroize::zeroize(&mut prover_data);
        let retained = guarded_pcs
            .inner
            .mmcs
            .get_matrices_secret(&prover_data.inner.commitment_data);
        assert!(retained.is_empty());
        assert!(prover_data.inner.retained_lde_dimensions().is_err());
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_quotient_batch_validates_every_input_before_rng() {
        const DOMAIN_SIZE: usize = 64;
        const NUM_CHUNKS: usize = 2;

        let rejected_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let control_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let domain = <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &rejected_pcs,
            DOMAIN_SIZE,
        );
        let valid = RowMajorMatrix::new((0..DOMAIN_SIZE).map(Val::from_usize).collect_vec(), 1);
        let malformed = RowMajorMatrix::new(vec![Val::ONE; DOMAIN_SIZE / 2], 1);

        let mut generation_calls = 0_usize;
        let invalid_plan = <SecretHidingPcs as SecretUnivariateStarkPcsCommitter<
            Challenge,
            Challenger,
        >>::commit_generated_quotients_secret(
            &rejected_pcs,
            vec![
                SecretQuotientCommitmentPlan::new(domain, 1, NUM_CHUNKS),
                SecretQuotientCommitmentPlan::new(domain, 1, 1),
            ],
            |_, _| {
                generation_calls += 1;
                Ok::<_, SecretMatrixError>(SecretProverData::new(valid.clone()))
            },
        );
        assert!(matches!(
            invalid_plan,
            Err(SecretGeneratedQuotientCommitError::Commitment(
                SecretHidingFriCommitError::Inner(
                    crate::SecretFriCommitError::InvalidQuotientChunkCount
                )
            ))
        ));
        assert_eq!(generation_calls, 0);

        let rejected = <SecretHidingPcs as SecretUnivariateStarkPcsCommitter<
            Challenge,
            Challenger,
        >>::commit_quotients_secret(
            &rejected_pcs,
            vec![
                SecretQuotientCommitmentInput::new(
                    domain,
                    SecretProverData::new(valid.clone()),
                    NUM_CHUNKS,
                ),
                SecretQuotientCommitmentInput::new(
                    domain,
                    SecretProverData::new(malformed),
                    NUM_CHUNKS,
                ),
            ],
        );
        assert!(matches!(
            rejected,
            Err(SecretHidingFriCommitError::Inner(
                crate::SecretFriCommitError::DomainHeightMismatch
            ))
        ));

        let (after_rejection, _after_data) = <SecretHidingPcs as
            SecretUnivariateStarkPcsCommitter<Challenge, Challenger>>::commit_quotient_secret(
            &rejected_pcs,
            domain,
            SecretProverData::new(valid.clone()),
            NUM_CHUNKS,
        )
        .expect("valid guarded quotient after rejection must succeed");
        let (control, _control_data) = <SecretHidingPcs as SecretUnivariateStarkPcsCommitter<
            Challenge,
            Challenger,
        >>::commit_quotient_secret(
            &control_pcs,
            domain,
            SecretProverData::new(valid),
            NUM_CHUNKS,
        )
        .expect("valid guarded control quotient must succeed");
        assert_eq!(after_rejection, control);
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_randomization_commitment_matches_standard_and_verifies() {
        let standard_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let guarded_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let first_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &standard_pcs,
                32,
            );
        let second_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &standard_pcs,
                64,
            );

        let (ordinary_commitment, ordinary_data) = <SecretHidingPcs as UnivariateStarkPcs<
            Challenge,
            Challenger,
        >>::get_opt_randomization_poly_commitment(
            &standard_pcs, [first_domain, second_domain]
        )
        .expect("valid standard randomization commitment must succeed")
        .expect("hiding PCS must produce a randomization commitment");
        let (commitment, mut guarded_data) = <SecretHidingPcs as
            SecretUnivariateStarkPcsCommitter<Challenge, Challenger>>::get_opt_randomization_poly_commitment_secret(
            &guarded_pcs,
            [first_domain, second_domain],
        )
        .expect("valid guarded randomization commitment must succeed")
        .expect("guarded hiding PCS must produce a randomization commitment");
        assert_eq!(commitment, ordinary_commitment);

        let mut challenger_rng = SmallRng::seed_from_u64(0x5241_4e44_5f50_4152);
        let challenger_permutation = Perm::new_from_rng_128(&mut challenger_rng);
        let mut ordinary_challenger = Challenger::new(challenger_permutation.clone());
        ordinary_challenger.observe(&ordinary_commitment);
        let ordinary_zeta: Challenge = ordinary_challenger.sample_algebra_element();
        let (ordinary_values, ordinary_proof) =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::open(
                &standard_pcs,
                vec![
                    (
                        &ordinary_data,
                        vec![vec![ordinary_zeta], vec![ordinary_zeta]],
                    )
                        .into(),
                ],
                &mut ordinary_challenger,
            )
            .expect("valid standard randomization opening must succeed");

        let mut guarded_challenger = Challenger::new(challenger_permutation.clone());
        guarded_challenger.observe(&commitment);
        let zeta: Challenge = guarded_challenger.sample_algebra_element();
        assert_eq!(zeta, ordinary_zeta);
        let (opened_values, proof) =
            <SecretHidingPcs as SecretPcsOpener<Challenge, Challenger>>::open_secret(
                &guarded_pcs,
                vec![(&mut guarded_data, vec![vec![zeta], vec![zeta]]).into()],
                &mut guarded_challenger,
            )
            .expect("valid guarded randomization opening must succeed");

        assert_eq!(opened_values, ordinary_values);
        assert_eq!(proof.0, ordinary_proof.0);
        assert_eq!(
            proof.1.batch_pow_witness,
            ordinary_proof.1.batch_pow_witness
        );
        assert_eq!(
            proof.1.commit_phase_commits,
            ordinary_proof.1.commit_phase_commits
        );
        assert_eq!(proof.1.final_poly, ordinary_proof.1.final_poly);
        assert_eq!(
            guarded_challenger.sample_algebra_element::<Challenge>(),
            ordinary_challenger.sample_algebra_element::<Challenge>()
        );

        let claims: Vec<CommitmentOpening<Challenge, Commitment, Domain>> = vec![
            (
                commitment,
                vec![
                    (first_domain, vec![(zeta, opened_values[0][0][0].clone())]),
                    (second_domain, vec![(zeta, opened_values[0][1][0].clone())]),
                ],
            )
                .into(),
        ];
        let mut verifier_challenger = Challenger::new(challenger_permutation);
        verifier_challenger.observe(&claims[0].commitment);
        let verifier_zeta: Challenge = verifier_challenger.sample_algebra_element();
        assert_eq!(verifier_zeta, zeta);
        <SecretHidingPcs as Pcs<Challenge, Challenger>>::verify(
            &guarded_pcs,
            claims,
            &proof,
            &mut verifier_challenger,
        )
        .expect("ordinary hiding verifier must accept guarded randomization proof");
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_randomization_rejects_all_domains_before_using_rng() {
        let rejected_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let control_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let valid_domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &rejected_pcs,
                32,
            );
        let short_domain = Domain::new(Val::ONE, 2).expect("fixed test domain must be valid");

        let rejected = <SecretHidingPcs as SecretUnivariateStarkPcsCommitter<
            Challenge,
            Challenger,
        >>::get_opt_randomization_poly_commitment_secret(
            &rejected_pcs,
            [valid_domain, short_domain],
        );
        assert!(matches!(
            rejected,
            Err(SecretHidingFriCommitError::Hiding(
                HidingFriProverError::HidingBudgetExceeded { .. }
            ))
        ));

        let (after_rejection, _after_data) = <SecretHidingPcs as
            SecretUnivariateStarkPcsCommitter<Challenge, Challenger>>::get_opt_randomization_poly_commitment_secret(
            &rejected_pcs,
            [valid_domain],
        )
        .expect("valid guarded randomization after rejection must succeed")
        .expect("guarded hiding PCS must produce a randomization commitment");
        let (control, _control_data) = <SecretHidingPcs as SecretUnivariateStarkPcsCommitter<
            Challenge,
            Challenger,
        >>::get_opt_randomization_poly_commitment_secret(
            &control_pcs, [valid_domain]
        )
        .expect("valid guarded control randomization must succeed")
        .expect("guarded hiding PCS must produce a randomization commitment");

        assert_eq!(after_rejection, control);
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_randomization_rejects_width_overflow() {
        let pcs = make_secret_hiding_pcs(usize::MAX);
        let domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 32);

        let result = <SecretHidingPcs as SecretUnivariateStarkPcsCommitter<
            Challenge,
            Challenger,
        >>::get_opt_randomization_poly_commitment_secret(&pcs, [domain]);

        assert!(matches!(
            result,
            Err(SecretHidingFriCommitError::Matrix(
                SecretMatrixError::DimensionOverflow
            ))
        ));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_opening_matches_standard_and_is_accepted_by_verifier() {
        let standard_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let guarded_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let domain = <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &standard_pcs,
            32,
        );
        let matrix = RowMajorMatrix::new((1_u8..=32).map(Val::from_u8).collect_vec(), 2);

        let (ordinary_commitment, ordinary_data) =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::commit(
                &standard_pcs,
                [(domain, matrix.clone())],
            )
            .expect("valid standard hiding commitment must succeed");
        let (commitment, mut guarded_data) =
            <SecretHidingPcs as SecretPcsCommitter<Challenge, Challenger>>::commit_secret(
                &guarded_pcs,
                vec![(domain, SecretProverData::new(matrix))],
            )
            .expect("valid guarded hiding commitment must succeed");
        assert_eq!(commitment, ordinary_commitment);

        let mut challenger_rng = SmallRng::seed_from_u64(0x4f50_454e_5f50_4152);
        let challenger_permutation = Perm::new_from_rng_128(&mut challenger_rng);
        let mut ordinary_challenger = Challenger::new(challenger_permutation.clone());
        ordinary_challenger.observe(&ordinary_commitment);
        let ordinary_zeta: Challenge = ordinary_challenger.sample_algebra_element();
        let (ordinary_values, ordinary_proof) =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::open(
                &standard_pcs,
                vec![(&ordinary_data, vec![vec![ordinary_zeta]]).into()],
                &mut ordinary_challenger,
            )
            .expect("valid standard hiding opening must succeed");

        let mut guarded_challenger = Challenger::new(challenger_permutation.clone());
        guarded_challenger.observe(&commitment);
        let zeta: Challenge = guarded_challenger.sample_algebra_element();
        assert_eq!(zeta, ordinary_zeta);
        let (opened_values, proof) =
            <SecretHidingPcs as SecretPcsOpener<Challenge, Challenger>>::open_secret(
                &guarded_pcs,
                vec![(&mut guarded_data, vec![vec![zeta]]).into()],
                &mut guarded_challenger,
            )
            .expect("valid guarded hiding opening must succeed");

        assert_eq!(opened_values, ordinary_values);
        assert_eq!(proof.0, ordinary_proof.0);
        assert_eq!(
            proof.1.batch_pow_witness,
            ordinary_proof.1.batch_pow_witness
        );
        assert_eq!(
            proof.1.commit_phase_commits,
            ordinary_proof.1.commit_phase_commits
        );
        assert_eq!(proof.1.final_poly, ordinary_proof.1.final_poly);
        assert_eq!(
            guarded_challenger.sample_algebra_element::<Challenge>(),
            ordinary_challenger.sample_algebra_element::<Challenge>()
        );

        let claims: Vec<CommitmentOpening<Challenge, Commitment, Domain>> = vec![
            (
                commitment,
                vec![(domain, vec![(zeta, opened_values[0][0][0].clone())])],
            )
                .into(),
        ];
        let mut verifier_challenger = Challenger::new(challenger_permutation);
        verifier_challenger.observe(&claims[0].commitment);
        let verifier_zeta: Challenge = verifier_challenger.sample_algebra_element();
        assert_eq!(verifier_zeta, zeta);
        <SecretHidingPcs as Pcs<Challenge, Challenger>>::verify(
            &guarded_pcs,
            claims,
            &proof,
            &mut verifier_challenger,
        )
        .expect("ordinary hiding verifier must accept the guarded proof");

        let repeated = <SecretHidingPcs as SecretPcsOpener<Challenge, Challenger>>::open_secret(
            &guarded_pcs,
            vec![(&mut guarded_data, vec![vec![zeta]]).into()],
            &mut guarded_challenger,
        );
        assert!(matches!(
            repeated,
            Err(SecretHidingFriOpenError::AlreadyOpened)
        ));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_mixed_opening_matches_standard_verifies_and_preserves_public_data() {
        let standard_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let guarded_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let domain = <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &standard_pcs,
            32,
        );
        let private_matrix = RowMajorMatrix::new((1_u8..=32).map(Val::from_u8).collect_vec(), 2);
        let public_matrix = RowMajorMatrix::new((33_u8..=48).map(Val::from_u8).collect_vec(), 1);

        let (ordinary_private_commitment, ordinary_private_data) =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::commit(
                &standard_pcs,
                [(domain, private_matrix.clone())],
            )
            .expect("valid standard private commitment must succeed");
        let (private_commitment, mut guarded_private_data) =
            <SecretHidingPcs as SecretPcsCommitter<Challenge, Challenger>>::commit_secret(
                &guarded_pcs,
                vec![(domain, SecretProverData::new(private_matrix))],
            )
            .expect("valid guarded private commitment must succeed");
        assert_eq!(private_commitment, ordinary_private_commitment);

        let (ordinary_public_commitment, ordinary_public_data) =
            <SecretHidingPcs as UnivariateStarkPcs<Challenge, Challenger>>::commit_preprocessing(
                &standard_pcs,
                [(domain, public_matrix.clone())],
            )
            .expect("valid standard preprocessing commitment must succeed");
        let (public_commitment, public_data) =
            <SecretHidingPcs as UnivariateStarkPcs<Challenge, Challenger>>::commit_preprocessing(
                &guarded_pcs,
                [(domain, public_matrix)],
            )
            .expect("valid guarded-path preprocessing commitment must succeed");
        assert_eq!(public_commitment, ordinary_public_commitment);

        let mut challenger_rng = SmallRng::seed_from_u64(0x4d49_5845_445f_5043);
        let challenger_permutation = Perm::new_from_rng_128(&mut challenger_rng);
        let mut ordinary_challenger = Challenger::new(challenger_permutation.clone());
        ordinary_challenger.observe(&ordinary_private_commitment);
        ordinary_challenger.observe(&ordinary_public_commitment);
        let ordinary_zeta: Challenge = ordinary_challenger.sample_algebra_element();
        let (ordinary_values, ordinary_proof) = <SecretHidingPcs as UnivariateStarkPcs<
            Challenge,
            Challenger,
        >>::open_with_preprocessing(
            &standard_pcs,
            vec![
                (&ordinary_private_data, vec![vec![ordinary_zeta]]).into(),
                (&ordinary_public_data, vec![vec![ordinary_zeta]]).into(),
            ],
            &mut ordinary_challenger,
            Some(1),
        )
        .expect("valid standard mixed opening must succeed");

        let mut guarded_challenger = Challenger::new(challenger_permutation.clone());
        guarded_challenger.observe(&private_commitment);
        guarded_challenger.observe(&public_commitment);
        let zeta: Challenge = guarded_challenger.sample_algebra_element();
        assert_eq!(zeta, ordinary_zeta);
        let (opened_values, proof) = <SecretHidingPcs as SecretPcsMixedOpener<
            Challenge,
            Challenger,
        >>::open_secret_with_public_preprocessing(
            &guarded_pcs,
            vec![
                SecretPcsMixedOpeningRequest::Private(
                    (&mut guarded_private_data, vec![vec![zeta]]).into(),
                ),
                SecretPcsMixedOpeningRequest::Public((&public_data, vec![vec![zeta]]).into()),
            ],
            &mut guarded_challenger,
        )
        .expect("valid guarded mixed opening must succeed");

        assert_eq!(opened_values, ordinary_values);
        assert_eq!(proof.0, ordinary_proof.0);
        assert_eq!(
            proof.1.batch_pow_witness,
            ordinary_proof.1.batch_pow_witness
        );
        assert_eq!(
            proof.1.commit_phase_commits,
            ordinary_proof.1.commit_phase_commits
        );
        assert_eq!(proof.1.final_poly, ordinary_proof.1.final_poly);
        assert_eq!(
            guarded_challenger.sample_algebra_element::<Challenge>(),
            ordinary_challenger.sample_algebra_element::<Challenge>()
        );

        let claims: Vec<CommitmentOpening<Challenge, Commitment, Domain>> = vec![
            (
                private_commitment,
                vec![(domain, vec![(zeta, opened_values[0][0][0].clone())])],
            )
                .into(),
            (
                public_commitment.clone(),
                vec![(domain, vec![(zeta, opened_values[1][0][0].clone())])],
            )
                .into(),
        ];
        let mut verifier_challenger = Challenger::new(challenger_permutation.clone());
        verifier_challenger.observe(&claims[0].commitment);
        verifier_challenger.observe(&claims[1].commitment);
        let verifier_zeta: Challenge = verifier_challenger.sample_algebra_element();
        assert_eq!(verifier_zeta, zeta);
        <SecretHidingPcs as UnivariateStarkPcs<Challenge, Challenger>>::verify_with_preprocessing(
            &guarded_pcs,
            claims,
            &proof,
            &mut verifier_challenger,
            Some(1),
        )
        .expect("ordinary verifier must accept guarded mixed opening proof");

        assert!(!public_data.opened.load(Ordering::Acquire));
        let mut reuse_challenger = Challenger::new(challenger_permutation);
        reuse_challenger.observe(&public_commitment);
        let reuse_zeta: Challenge = reuse_challenger.sample_algebra_element();
        <SecretHidingPcs as UnivariateStarkPcs<Challenge, Challenger>>::open_with_preprocessing(
            &guarded_pcs,
            vec![(&public_data, vec![vec![reuse_zeta]]).into()],
            &mut reuse_challenger,
            Some(0),
        )
        .expect("public preprocessing must remain reusable after guarded mixed opening");
        assert!(!public_data.opened.load(Ordering::Acquire));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_mixed_opening_rejects_public_layout_before_private_consumption() {
        let pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 32);
        let private_matrix = RowMajorMatrix::new((1_u8..=32).map(Val::from_u8).collect_vec(), 2);
        let public_matrix = RowMajorMatrix::new(vec![Val::ONE; 16], 1);
        let (_, mut private_data) =
            <SecretHidingPcs as SecretPcsCommitter<Challenge, Challenger>>::commit_secret(
                &pcs,
                vec![(domain, SecretProverData::new(private_matrix))],
            )
            .expect("valid guarded private commitment must succeed");
        let (_, mislabeled_public_data) = <SecretHidingPcs as Pcs<Challenge, Challenger>>::commit(
            &pcs,
            [(domain, public_matrix.clone())],
        )
        .expect("valid ordinary private commitment must succeed");
        let (_, public_data) =
            <SecretHidingPcs as UnivariateStarkPcs<Challenge, Challenger>>::commit_preprocessing(
                &pcs,
                [(domain, public_matrix)],
            )
            .expect("valid public preprocessing commitment must succeed");

        let mut first_rng = SmallRng::seed_from_u64(0x4d49_5845_445f_4241);
        let mut first_challenger = Challenger::new(Perm::new_from_rng_128(&mut first_rng));
        let mislabeled = <SecretHidingPcs as SecretPcsMixedOpener<
            Challenge,
            Challenger,
        >>::open_secret_with_public_preprocessing(
            &pcs,
            vec![
                SecretPcsMixedOpeningRequest::Private(
                    (&mut private_data, vec![vec![Challenge::ONE]]).into(),
                ),
                SecretPcsMixedOpeningRequest::Public(
                    (&mislabeled_public_data, vec![vec![Challenge::ONE]]).into(),
                ),
            ],
            &mut first_challenger,
        );
        assert!(matches!(
            mislabeled,
            Err(SecretHidingFriOpenError::InvalidPublicPreprocessingLayout)
        ));
        assert!(!private_data.opened.load(Ordering::Acquire));

        let mut second_rng = SmallRng::seed_from_u64(0x4d49_5845_445f_4242);
        let mut second_challenger = Challenger::new(Perm::new_from_rng_128(&mut second_rng));
        let duplicated = <SecretHidingPcs as SecretPcsMixedOpener<
            Challenge,
            Challenger,
        >>::open_secret_with_public_preprocessing(
            &pcs,
            vec![
                SecretPcsMixedOpeningRequest::Private(
                    (&mut private_data, vec![vec![Challenge::ONE]]).into(),
                ),
                SecretPcsMixedOpeningRequest::Public(
                    (&public_data, vec![vec![Challenge::ONE]]).into(),
                ),
                SecretPcsMixedOpeningRequest::Public(
                    (&public_data, vec![vec![Challenge::ONE]]).into(),
                ),
            ],
            &mut second_challenger,
        );
        assert!(matches!(
            duplicated,
            Err(SecretHidingFriOpenError::InvalidPublicPreprocessingLayout)
        ));
        assert!(!private_data.opened.load(Ordering::Acquire));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_opening_rejects_shape_and_profile_before_consumption() {
        let pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let domain =
            <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 32);
        let matrix = RowMajorMatrix::new((1_u8..=32).map(Val::from_u8).collect_vec(), 2);
        let (commitment, mut prover_data) =
            <SecretHidingPcs as SecretPcsCommitter<Challenge, Challenger>>::commit_secret(
                &pcs,
                vec![(domain, SecretProverData::new(matrix))],
            )
            .expect("valid guarded hiding commitment must succeed");

        let mut challenger_rng = SmallRng::seed_from_u64(0x5052_4546_4c49_4748);
        let mut challenger = Challenger::new(Perm::new_from_rng_128(&mut challenger_rng));
        challenger.observe(&commitment);
        let shape_error = <SecretHidingPcs as SecretPcsOpener<Challenge, Challenger>>::open_secret(
            &pcs,
            vec![(&mut prover_data, Vec::new()).into()],
            &mut challenger,
        );
        assert!(matches!(
            shape_error,
            Err(SecretHidingFriOpenError::InvalidShape)
        ));
        assert!(!prover_data.opened.load(Ordering::Acquire));

        prover_data.num_random_codewords = prover_data
            .num_random_codewords
            .checked_add(1)
            .expect("fixed test profile increment must fit");
        let profile_error =
            <SecretHidingPcs as SecretPcsOpener<Challenge, Challenger>>::open_secret(
                &pcs,
                vec![(&mut prover_data, vec![vec![Challenge::ZERO]]).into()],
                &mut challenger,
            );
        assert!(matches!(
            profile_error,
            Err(SecretHidingFriOpenError::RandomCodewordCountMismatch)
        ));
        assert!(!prover_data.opened.load(Ordering::Acquire));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_quotient_rejects_non_hiding_chunk_count_before_using_rng() {
        const DOMAIN_SIZE: usize = 64;

        let rejected_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let control_pcs = make_secret_hiding_pcs(NUM_RANDOM_CODEWORDS);
        let domain = <SecretHidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &rejected_pcs,
            DOMAIN_SIZE,
        );
        let rejected = <SecretHidingPcs as SecretUnivariateStarkPcsCommitter<
            Challenge,
            Challenger,
        >>::commit_quotient_secret(
            &rejected_pcs,
            domain,
            SecretProverData::new(RowMajorMatrix::new(vec![Val::ONE; DOMAIN_SIZE], 1)),
            1,
        );
        assert!(matches!(
            rejected,
            Err(SecretHidingFriCommitError::Inner(
                crate::SecretFriCommitError::InvalidQuotientChunkCount
            ))
        ));

        let valid_matrix = RowMajorMatrix::new(vec![Val::TWO; DOMAIN_SIZE], 1);
        let (after_rejection, _after_data) = <SecretHidingPcs as
            SecretUnivariateStarkPcsCommitter<Challenge, Challenger>>::commit_quotient_secret(
            &rejected_pcs,
            domain,
            SecretProverData::new(valid_matrix.clone()),
            2,
        )
        .expect("valid guarded quotient after rejection must succeed");
        let (control, _control_data) = <SecretHidingPcs as SecretUnivariateStarkPcsCommitter<
            Challenge,
            Challenger,
        >>::commit_quotient_secret(
            &control_pcs,
            domain,
            SecretProverData::new(valid_matrix),
            2,
        )
        .expect("valid guarded control quotient must succeed");

        assert_eq!(after_rejection, control);
    }

    /// Run a real prover roundtrip and return `(pcs, claims, proof, challenger)`
    /// ready to verify, with `challenger` advanced past the commitment.
    ///
    /// One round, one matrix, one point, so the random-opening tree nests as:
    ///
    /// ```text
    ///     proof.0          = [round_0]      // 1 round
    ///     proof.0[0]       = [matrix_0]     // 1 matrix
    ///     proof.0[0][0]    = [point_0]      // 1 point
    ///     proof.0[0][0][0] = [v_0, .., v_3] // NUM_RANDOM_CODEWORDS values
    /// ```
    ///
    /// Each test perturbs one level to trip the matching count check.
    fn make_fixture() -> (MyPcs, Vec<(Commitment, Claims)>, Proof, Challenger) {
        make_fixture_with_points(1)
    }

    fn make_fixture_with_points(
        num_points: usize,
    ) -> (MyPcs, Vec<(Commitment, Claims)>, Proof, Challenger) {
        // Fixed seeds keep the roundtrip deterministic.
        let mut rng = SmallRng::seed_from_u64(1);

        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());

        let val_mmcs = ValMmcs::new(hash, compress, 0, StdRng::seed_from_u64(3));
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        // Minimal sound parameters: blowup 2, binary folding, 2 queries.
        let fri_params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            batch_proof_of_work_bits: 0,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: challenge_mmcs,
        };

        // The wrapper owns an independently seeded RNG for its random codewords.
        let pcs = MyPcs::new(
            Dft::default(),
            val_mmcs,
            fri_params,
            NUM_RANDOM_CODEWORDS,
            StdRng::seed_from_u64(2),
        );

        // The wrapper interleaves the trace with random rows, doubling its
        // height, so (like the zk prover) we commit against a `2 * height` domain.
        let log_degree = if num_points == 1 { 4 } else { 5 };
        let width = 4;
        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 2 << log_degree);
        let trace = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_degree, width);
        let (commitment, prover_data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain, trace)]).unwrap();

        // Prover: observe, sample the point, prove.
        let mut p_challenger = Challenger::new(perm.clone());
        p_challenger.observe(&commitment);
        let zeta: Challenge = p_challenger.sample_algebra_element();
        let (opened_values, proof) = pcs
            .open(
                vec![OpeningRequest {
                    prover_data: &prover_data,
                    points: vec![vec![zeta; num_points]],
                }],
                &mut p_challenger,
            )
            .unwrap();

        // Verifier: replay up to the point sample so a valid proof must pass.
        let mut v_challenger = Challenger::new(perm);
        v_challenger.observe(&commitment);
        let v_zeta: Challenge = v_challenger.sample_algebra_element();
        assert_eq!(
            v_zeta, zeta,
            "prover and verifier must sample the same point"
        );

        // Public claims; the hidden values stay in `proof.0` until `verify`.
        let claims = vec![(
            commitment,
            vec![(
                domain,
                opened_values[0][0]
                    .iter()
                    .map(|values| (zeta, values.clone()))
                    .collect(),
            )],
        )];

        (pcs, claims, proof, v_challenger)
    }

    /// Verify with fully qualified syntax so the type parameters are unambiguous.
    fn run_verify(
        pcs: &MyPcs,
        claims: Vec<(Commitment, Claims)>,
        proof: &Proof,
        challenger: &mut Challenger,
    ) -> Result<(), TestError> {
        <MyPcs as Pcs<Challenge, Challenger>>::verify(
            pcs,
            claims.into_iter().map(Into::into).collect(),
            proof,
            challenger,
        )
    }

    #[test]
    fn preprocessing_request_can_precede_hidden_requests() {
        let (pcs, _, _, mut prover_challenger) = make_fixture();
        let mut verifier_challenger = prover_challenger.clone();
        let mut rng = SmallRng::seed_from_u64(17);
        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 32);
        let (pre_commit, pre_data) =
            <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::commit_preprocessing(
                &pcs,
                [(domain, RowMajorMatrix::<Val>::rand(&mut rng, 16, 3))],
            )
            .unwrap();
        let (trace_commit, trace_data) = <MyPcs as Pcs<Challenge, Challenger>>::commit(
            &pcs,
            [(domain, RowMajorMatrix::<Val>::rand(&mut rng, 16, 4))],
        )
        .unwrap();
        for challenger in [&mut prover_challenger, &mut verifier_challenger] {
            challenger.observe(&pre_commit);
            challenger.observe(&trace_commit);
        }
        let zeta: Challenge = prover_challenger.sample_algebra_element();
        assert_eq!(zeta, verifier_challenger.sample_algebra_element());
        let (values, proof) = pcs
            .open_with_preprocessing(
                vec![
                    OpeningRequest {
                        prover_data: &pre_data,
                        points: vec![vec![zeta]],
                    },
                    OpeningRequest {
                        prover_data: &trace_data,
                        points: vec![vec![zeta]],
                    },
                ],
                &mut prover_challenger,
                Some(0),
            )
            .unwrap();
        assert_eq!(values[0][0][0].len(), 3);
        assert_eq!(values[1][0][0].len(), 4);
        assert!(proof.0[0][0][0].is_empty());
        assert_eq!(proof.0[1][0][0].len(), NUM_RANDOM_CODEWORDS);
        let claims = vec![
            (
                pre_commit,
                vec![(domain, vec![(zeta, values[0][0][0].clone())])],
            )
                .into(),
            (
                trace_commit,
                vec![(domain, vec![(zeta, values[1][0][0].clone())])],
            )
                .into(),
        ];
        for index in [None, Some(1), Some(2)] {
            let mut rejected = verifier_challenger.clone();
            let error = pcs
                .verify_with_preprocessing(claims.clone(), &proof, &mut rejected, index)
                .unwrap_err();
            if index == Some(2) {
                assert!(matches!(
                    error,
                    FriError::HidingPreprocessedCommitmentOutOfBounds {
                        index: 2,
                        num_rounds: 2
                    }
                ));
            } else {
                assert!(matches!(
                    error,
                    FriError::HidingRandomOpeningValueCountMismatch {
                        round: 0,
                        matrix: 0,
                        point: 0,
                        expected: 4,
                        got: 0
                    }
                ));
            }
            let mut before = verifier_challenger.clone();
            assert_eq!(
                rejected.sample_algebra_element::<Challenge>(),
                before.sample_algebra_element::<Challenge>()
            );
        }
        pcs.verify_with_preprocessing(claims, &proof, &mut verifier_challenger, Some(0))
            .unwrap();
    }

    #[test]
    fn valid_proof_passes() {
        // Baseline: an unmodified proof verifies, so the mismatch tests below
        // start from a genuinely valid proof.
        let (pcs, claims, proof, mut challenger) = make_fixture();
        run_verify(&pcs, claims, &proof, &mut challenger)
            .expect("valid hiding proof should verify");
    }

    #[test]
    fn verifier_rejects_different_random_codeword_count() {
        for verifier_count in [5, 8] {
            let (mut pcs, claims, proof, mut challenger) = make_fixture();
            pcs.num_random_codewords = verifier_count;
            let mut before = challenger.clone();
            assert!(
                matches!(run_verify(&pcs, claims, &proof, &mut challenger), Err(FriError::HidingRandomOpeningValueCountMismatch {
                round: 0, matrix: 0, point: 0, expected, got: 4,
            }) if expected == verifier_count)
            );
            assert_eq!(
                challenger.sample_algebra_element::<Challenge>(),
                before.sample_algebra_element::<Challenge>()
            );
        }
    }

    #[test]
    fn random_opening_value_count_is_checked_at_later_points() {
        let (pcs, claims, mut proof, mut challenger) = make_fixture_with_points(2);
        let mut valid = challenger.clone();
        run_verify(&pcs, claims.clone(), &proof, &mut valid).unwrap();
        proof.0[0][0][1].pop();
        let mut before = challenger.clone();
        assert!(matches!(
            run_verify(&pcs, claims, &proof, &mut challenger),
            Err(FriError::HidingRandomOpeningValueCountMismatch {
                round: 0,
                matrix: 0,
                point: 1,
                expected: 4,
                got: 3,
            })
        ));
        assert_eq!(
            challenger.sample_algebra_element::<Challenge>(),
            before.sample_algebra_element::<Challenge>()
        );
    }

    #[test]
    fn terminal_height_rejection_preserves_hiding_opening_state() {
        let (mut pcs, _, _, mut challenger) = make_fixture();
        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 32);
        let (_, data) = <MyPcs as Pcs<Challenge, Challenger>>::commit(
            &pcs,
            [(domain, RowMajorMatrix::new(vec![Val::ONE; 16], 1))],
        )
        .unwrap();
        pcs.inner.fri.log_final_poly_len = 5;
        let mut before = challenger.clone();
        let result = pcs.open(
            vec![(&data, vec![vec![Challenge::TWO]]).into()],
            &mut challenger,
        );
        assert!(matches!(
            result,
            Err(HidingFriProverError::Fri(
                FriProverError::InputHeightTooSmall {
                    log_input_height: 6,
                    log_final_height: 6,
                }
            ))
        ));
        assert_eq!(
            challenger.sample_algebra_element::<Challenge>(),
            before.sample_algebra_element::<Challenge>()
        );
        pcs.inner.fri.log_final_poly_len = 0;
        pcs.open(
            vec![(&data, vec![vec![Challenge::TWO]]).into()],
            &mut challenger,
        )
        .unwrap();
    }

    #[test]
    fn failed_commit_validates_all_matrices_before_rng() {
        let (pcs, _, _, _) = make_fixture();
        let (control, _, _, _) = make_fixture();
        let before: u64 = control.rng.lock().random();
        let large = Domain::new(Val::ONE, 6).unwrap();
        let small = Domain::new(Val::ONE, 2).unwrap();
        let result = <MyPcs as Pcs<Challenge, Challenger>>::commit(
            &pcs,
            [
                (large, RowMajorMatrix::new(Val::zero_vec(64), 1)),
                (small, RowMajorMatrix::new(Val::zero_vec(2), 1)),
            ],
        );
        assert!(matches!(
            result,
            Err(HidingFriProverError::HidingBudgetExceeded { .. })
        ));
        assert_eq!(pcs.rng.lock().random::<u64>(), before);
    }

    #[test]
    fn commit_rejects_trace_shorter_than_hiding_budget() {
        let (pcs, _, _, _) = make_fixture();
        let domain = Domain::new(Val::ONE, 2).unwrap();
        let result = <MyPcs as Pcs<Challenge, Challenger>>::commit(
            &pcs,
            [(domain, RowMajorMatrix::new(Val::zero_vec(2), 1))],
        );
        assert!(matches!(
            result,
            Err(HidingFriProverError::HidingBudgetExceeded { mask_height: 2, .. })
        ));
    }

    #[test]
    fn preprocessing_rejects_short_domain() {
        let (pcs, _, _, _) = make_fixture();
        let domain = Domain::new(Val::ONE, 2).unwrap();
        let result = <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::commit_preprocessing(
            &pcs,
            [(domain, RowMajorMatrix::new(Val::zero_vec(2), 1))],
        );
        assert!(matches!(
            result,
            Err(HidingFriProverError::HidingBudgetExceeded { .. })
        ));
    }

    #[test]
    fn randomization_commitment_rejects_short_domain() {
        let (pcs, _, _, _) = make_fixture();
        let domain = Domain::new(Val::ONE, 2).unwrap();
        let result = <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::get_opt_randomization_poly_commitment(
            &pcs,
            [domain],
        );
        assert!(matches!(
            result,
            Err(HidingFriProverError::HidingBudgetExceeded { .. })
        ));
    }

    #[test]
    fn quotient_ldes_reject_chunks_shorter_than_hiding_budget() {
        let (pcs, _, _, _) = make_fixture();
        let domains = Domain::new(Val::GENERATOR, 2).unwrap().split_domains(2);
        let result = <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::get_quotient_ldes(
            &pcs,
            domains
                .into_iter()
                .map(|domain| (domain, RowMajorMatrix::new(Val::zero_vec(2), 1))),
            2,
        );
        assert!(matches!(
            result,
            Err(HidingFriProverError::HidingBudgetExceeded { .. })
        ));
    }

    #[test]
    fn quotient_ldes_reject_a_chunk_count_other_than_the_evaluation_count() {
        let (pcs, _, _, _) = make_fixture();
        // Fixture state: four chunks of two rows, split from a coset of eight points.
        let chunks = || {
            Domain::new(Val::GENERATOR, 3)
                .unwrap()
                .split_domains(4)
                .into_iter()
                .map(|domain| (domain, RowMajorMatrix::new(Val::zero_vec(2), 1)))
        };

        // Invariant: the masks cancel only when both counts agree.
        //
        // Mutation: two chunks declared for four evaluations.
        // Chunk 1 would balance chunk 0 alone, and chunk 2 would keep its draw.
        //
        //     draws:   r_0   r_1                  r_2   0
        //     masks:   r_0   r_1 - (c_0/c_1) r_0  r_2   0
        //     sum c_i masks = c_1 r_1 + c_2 r_2, not zero
        let result = <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::get_quotient_ldes(
            &pcs,
            chunks(),
            2,
        );
        assert_eq!(
            result.unwrap_err(),
            HidingFriProverError::QuotientChunkCountMismatch {
                expected: 2,
                got: 4
            }
        );

        // Mutation: eight chunks declared for four evaluations.
        // Chunk 7 has no evaluation to read.
        let result = <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::get_quotient_ldes(
            &pcs,
            chunks(),
            8,
        );
        assert_eq!(
            result.unwrap_err(),
            HidingFriProverError::QuotientChunkCountMismatch {
                expected: 8,
                got: 4
            }
        );
    }

    #[test]
    fn commit_ldes_rejects_short_mask() {
        let (pcs, _, _, _) = make_fixture();
        let result = <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::commit_ldes(
            &pcs,
            vec![RowMajorMatrix::new(Val::zero_vec(8), 1)],
        );
        assert!(matches!(
            result,
            Err(HidingFriProverError::HidingBudgetExceeded { .. })
        ));
    }

    fn commit_opening_fixture(pcs: &MyPcs) -> <MyPcs as Pcs<Challenge, Challenger>>::ProverData {
        let domain = Domain::new(Val::ONE, 6).unwrap();
        <MyPcs as Pcs<Challenge, Challenger>>::commit(
            pcs,
            [(domain, RowMajorMatrix::new(Val::zero_vec(32), 1))],
        )
        .unwrap()
        .1
    }

    #[cfg(panic = "unwind")]
    fn assert_mismatched_prover_data_rejected(change_config: impl FnOnce(&mut MyPcs)) {
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let (mut pcs, _, _, mut challenger) = make_fixture();
        pcs.inner.fri.log_blowup = 2;
        let data = commit_opening_fixture(&pcs);
        let mut receiver = pcs;
        change_config(&mut receiver);
        let domain = Domain::new(Val::GENERATOR, 6).unwrap();
        let evaluations = catch_unwind(AssertUnwindSafe(|| {
            <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::get_evaluations_on_domain(
                &receiver, &data, 0, domain,
            );
        }));
        let evaluations_no_random = catch_unwind(AssertUnwindSafe(|| {
            <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::get_evaluations_on_domain_no_random(
                &receiver, &data, 0, domain,
            );
        }));
        let zeta = challenger.sample_algebra_element::<Challenge>();
        let mut untouched_challenger = challenger.clone();
        let opening = catch_unwind(AssertUnwindSafe(|| {
            receiver
                .open(
                    vec![OpeningRequest {
                        prover_data: &data,
                        points: vec![vec![zeta]],
                    }],
                    &mut challenger,
                )
                .unwrap();
        }));
        assert!(
            evaluations.is_err() && evaluations_no_random.is_err() && opening.is_err(),
            "every prover-data entry point must reject a mismatched hiding configuration"
        );
        assert!(!data.opened.load(Ordering::Relaxed));
        assert_eq!(
            challenger.sample_algebra_element::<Challenge>(),
            untouched_challenger.sample_algebra_element::<Challenge>(),
        );
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn prover_data_rejects_changed_random_codeword_count() {
        assert_mismatched_prover_data_rejected(|pcs| pcs.num_random_codewords += 1);
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn prover_data_rejects_changed_blowup() {
        assert_mismatched_prover_data_rejected(|pcs| pcs.inner.fri.log_blowup -= 1);
    }

    #[test]
    fn open_rejects_excess_opening_points() {
        let (pcs, _, _, mut challenger) = make_fixture();
        let data = commit_opening_fixture(&pcs);
        let points = (0..5)
            .map(|_| challenger.sample_algebra_element::<Challenge>())
            .collect();
        let before: Challenge = challenger.clone().sample_algebra_element();
        let result = pcs.open(
            vec![OpeningRequest {
                prover_data: &data,
                points: vec![points],
            }],
            &mut challenger,
        );
        assert!(matches!(
            result,
            Err(HidingFriProverError::HidingBudgetExceeded {
                num_opening_points: 5,
                ..
            })
        ));
        assert_eq!(challenger.sample_algebra_element::<Challenge>(), before);
        // Rejection did not claim the single-use mask: a smaller opening can still succeed.
        pcs.open(
            vec![OpeningRequest {
                prover_data: &data,
                points: vec![vec![Challenge::from_u8(7)]],
            }],
            &mut challenger,
        )
        .unwrap();
    }

    #[cfg(not(feature = "secret-memory"))]
    #[test]
    fn open_rejects_a_matrix_without_opening_points() {
        let (pcs, _, _, mut challenger) = make_fixture();
        let data = commit_opening_fixture(&pcs);
        let before: Challenge = challenger.clone().sample_algebra_element();

        // Fixture state: one private round, one matrix, no opening point.
        //
        //     round 0, matrix 0:  points = []  ->  nothing pins its width
        let result = pcs.open(
            vec![OpeningRequest {
                prover_data: &data,
                points: vec![vec![]],
            }],
            &mut challenger,
        );
        assert!(matches!(
            result,
            Err(HidingFriProverError::Fri(
                FriProverError::MatrixWithoutOpeningPoints {
                    batch: 0,
                    matrix: 0
                }
            ))
        ));

        // The rejection observed nothing and left the single-use mask unclaimed.
        assert_eq!(challenger.sample_algebra_element::<Challenge>(), before);
        assert!(!data.opened.load(Ordering::Relaxed));

        // So a real opening of the same commitment still succeeds.
        pcs.open(
            vec![OpeningRequest {
                prover_data: &data,
                points: vec![vec![Challenge::from_u8(7)]],
            }],
            &mut challenger,
        )
        .unwrap();
    }

    #[cfg(not(feature = "secret-memory"))]
    #[test]
    #[should_panic(expected = "already opened")]
    fn private_commitment_cannot_be_opened_twice() {
        let (pcs, _, _, mut challenger) = make_fixture();
        let data = commit_opening_fixture(&pcs);
        let cloned_data = data.clone();
        for data in [&data, &cloned_data] {
            let zeta = challenger.sample_algebra_element::<Challenge>();
            pcs.open(
                vec![OpeningRequest {
                    prover_data: data,
                    points: vec![vec![zeta]],
                }],
                &mut challenger,
            )
            .unwrap();
        }
    }

    #[test]
    #[should_panic(expected = "already opened")]
    fn duplicate_private_requests_cannot_multiply_hiding_budget() {
        let (pcs, _, _, mut challenger) = make_fixture();
        let data = commit_opening_fixture(&pcs);
        let zeta = challenger.sample_algebra_element::<Challenge>();
        let request = OpeningRequest {
            prover_data: &data,
            points: vec![vec![zeta]],
        };
        pcs.open(vec![request.clone(), request], &mut challenger)
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "preprocessing index")]
    fn private_commitment_cannot_skip_checks_as_preprocessing() {
        let (pcs, _, _, mut challenger) = make_fixture();
        let data = commit_opening_fixture(&pcs);
        let zeta = challenger.sample_algebra_element::<Challenge>();
        pcs.open_with_preprocessing(
            vec![OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut challenger,
            Some(0),
        )
        .unwrap();
    }

    #[test]
    fn public_preprocessing_can_be_reused() {
        let (pcs, _, _, mut challenger) = make_fixture();
        let domain = Domain::new(Val::ONE, 5).unwrap();
        let (_, data) = <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::commit_preprocessing(
            &pcs,
            [(domain, RowMajorMatrix::new(Val::zero_vec(16), 1))],
        )
        .unwrap();
        for _ in 0..2 {
            let zeta = challenger.sample_algebra_element::<Challenge>();
            let (values, proof) = pcs
                .open_with_preprocessing(
                    vec![OpeningRequest {
                        prover_data: &data,
                        points: vec![vec![zeta]],
                    }],
                    &mut challenger,
                    Some(0),
                )
                .unwrap();
            assert_eq!(values[0][0][0], vec![Challenge::ZERO]);
            assert!(proof.0[0][0][0].is_empty());
        }
    }

    /// Produce the proof an older, unchecked hiding wrapper could emit. Going directly through
    /// the inner PCS ensures verifier regressions are independent of the new prover guards.
    fn unchecked_proof(
        pcs: &MyPcs,
        mask_height: usize,
        num_points: usize,
        mut challenger: Challenger,
    ) -> (Vec<(Commitment, Claims)>, Proof, Challenger) {
        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(pcs, 2 * mask_height);
        let mut rng = StdRng::seed_from_u64(37);
        let mut trace = RowMajorMatrix::new(Val::zero_vec(mask_height), 1)
            .with_random_cols(1 + 2 * NUM_RANDOM_CODEWORDS, &mut rng);
        trace.width = 1 + NUM_RANDOM_CODEWORDS;
        let (commitment, data) =
            Pcs::<Challenge, Challenger>::commit(&pcs.inner, [(domain, trace)]).unwrap();
        challenger.observe(&commitment);
        let points = (0..num_points)
            .map(|_| challenger.sample_algebra_element::<Challenge>())
            .collect_vec();
        let verifier_challenger = challenger.clone();
        let (mut values, proof) = pcs
            .inner
            .open(
                vec![OpeningRequest {
                    prover_data: &data,
                    points: vec![points.clone()],
                }],
                &mut challenger,
            )
            .unwrap();
        let random_values = values[0][0]
            .iter_mut()
            .map(|point_values| point_values.split_off(1))
            .collect_vec();
        let claims = vec![(
            commitment,
            vec![(
                domain,
                points.into_iter().zip(values.remove(0).remove(0)).collect(),
            )],
        )];
        (
            claims,
            (vec![vec![random_values]], proof),
            verifier_challenger,
        )
    }

    #[test]
    fn verifier_rejects_legacy_short_trace_proof() {
        let (pcs, _, _, challenger) = make_fixture();
        let (claims, proof, mut challenger) = unchecked_proof(&pcs, 2, 1, challenger);
        assert!(matches!(
            run_verify(&pcs, claims, &proof, &mut challenger),
            Err(FriError::HidingBudgetExceeded { mask_height: 2, .. })
        ));
    }

    #[test]
    fn verifier_counts_all_opening_points() {
        let (pcs, _, _, challenger) = make_fixture();
        let (claims, proof, mut challenger) = unchecked_proof(&pcs, 16, 2, challenger);
        assert!(matches!(
            run_verify(&pcs, claims, &proof, &mut challenger),
            Err(FriError::HidingBudgetExceeded {
                mask_height: 16,
                num_opening_points: 2,
                ..
            })
        ));
    }

    #[test]
    fn verifier_accepts_exact_hiding_budget() {
        let (mut pcs, _, _, challenger) = make_fixture();
        pcs.inner.fri.num_queries = 4;
        let (claims, proof, mut challenger) = unchecked_proof(&pcs, 16, 1, challenger);
        run_verify(&pcs, claims, &proof, &mut challenger).unwrap();
    }

    #[test]
    fn verifier_rejects_overflowing_hiding_budget() {
        let (mut pcs, claims, proof, mut challenger) = make_fixture();
        pcs.inner.fri.num_queries = usize::MAX;
        assert!(matches!(
            run_verify(&pcs, claims, &proof, &mut challenger),
            Err(FriError::HidingBudgetExceeded { .. })
        ));
        assert_eq!(pcs.hiding_budget(4, usize::MAX), None);
    }

    #[test]
    fn commit_rejects_insufficient_random_codewords() {
        let (mut pcs, _, _, _) = make_fixture();
        pcs.num_random_codewords = 3;
        let domain = Domain::new(Val::ONE, 6).unwrap();
        let result = <MyPcs as Pcs<Challenge, Challenger>>::commit(
            &pcs,
            [(domain, RowMajorMatrix::new(Val::zero_vec(64), 1))],
        );
        assert!(matches!(
            result,
            Err(HidingFriProverError::InsufficientRandomCodewords {
                required: 4,
                got: 3
            })
        ));
    }

    #[test]
    fn verifier_rejects_insufficient_random_codewords() {
        let (mut pcs, claims, proof, mut challenger) = make_fixture();
        pcs.num_random_codewords = 3;
        assert!(matches!(
            run_verify(&pcs, claims, &proof, &mut challenger),
            Err(FriError::InsufficientHidingRandomCodewords {
                required: 4,
                got: 3
            })
        ));
    }

    #[test]
    fn random_opening_round_count_mismatch() {
        let (pcs, claims, mut proof, mut challenger) = make_fixture();

        // One random-opening entry is required per public round.
        //     claims:  [round_0]          -> expected 1
        //     proof.0: [round_0, EXTRA]   -> got 2
        let expected_rounds = claims.len();
        proof.0.push(vec![]);

        let err = run_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("should reject an extra random-opening round");

        match err {
            FriError::HidingRandomOpeningRoundCountMismatch { expected, got } => {
                assert_eq!(expected, expected_rounds);
                assert_eq!(got, expected_rounds + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn random_opening_matrix_count_mismatch() {
        let (pcs, claims, mut proof, mut challenger) = make_fixture();

        // Round counts match, so the per-round matrix check fires next.
        //     claims[0]:  [matrix_0]          -> expected 1
        //     proof.0[0]: [matrix_0, EXTRA]   -> got 2
        let expected_mats = claims[0].1.len();
        proof.0[0].push(vec![]);

        let err = run_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("should reject an extra random-opening matrix");

        match err {
            FriError::HidingRandomOpeningMatrixCountMismatch {
                round,
                expected,
                got,
            } => {
                assert_eq!(round, 0);
                assert_eq!(expected, expected_mats);
                assert_eq!(got, expected_mats + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn random_opening_point_count_mismatch() {
        let (pcs, claims, mut proof, mut challenger) = make_fixture();

        // Round and matrix counts match, so the per-matrix point check fires.
        //     claims[0][0]:  [point_0]          -> expected 1
        //     proof.0[0][0]: [point_0, EXTRA]   -> got 2
        let expected_points = claims[0].1[0].1.len();
        proof.0[0][0].push(vec![]);

        let err = run_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("should reject an extra random-opening point");

        match err {
            FriError::HidingRandomOpeningPointCountMismatch {
                round,
                matrix,
                expected,
                got,
            } => {
                assert_eq!(round, 0);
                assert_eq!(matrix, 0);
                assert_eq!(expected, expected_points);
                assert_eq!(got, expected_points + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    /// Seed for the wrapper's mask RNG, replayed by [`expected_quotient_ldes`].
    const QUOTIENT_RNG_SEED: u64 = 5;

    /// Blowup used by the quotient-LDE tests below.
    const QUOTIENT_LOG_BLOWUP: usize = 1;

    /// One quotient chunk's LDE written as two independent forward transforms: a coset LDE of the
    /// chunk plus a full-size DFT of the `v_H * r` mask, summed pointwise.
    fn quotient_lde_two_transforms<D: TwoAdicSubgroupDft<Val>>(
        dft: &D,
        evals: RowMajorMatrix<Val>,
        random_values: &[Val],
        shift: Val,
    ) -> RowMajorMatrix<Val> {
        let h = evals.height();
        let w = evals.width();

        let mut lde_evals = dft
            .coset_lde_batch(evals, QUOTIENT_LOG_BLOWUP + 1, shift)
            .to_row_major_matrix();

        let mut vanishing_poly_coeffs = Val::zero_vec((h * w) << (QUOTIENT_LOG_BLOWUP + 1));
        let p = shift.exp_u64(h as u64);
        Val::GENERATOR
            .powers()
            .take(h)
            .enumerate()
            .for_each(|(i, p_i)| {
                for j in 0..w {
                    let mul_coeff = p_i * random_values[i * w + j];
                    vanishing_poly_coeffs[i * w + j] -= mul_coeff;
                    vanishing_poly_coeffs[(h + i) * w + j] = p * mul_coeff;
                }
            });
        let random_eval = dft
            .dft_batch(DenseMatrix::new(vanishing_poly_coeffs, w))
            .to_row_major_matrix();

        for i in 0..h * w * (1 << (QUOTIENT_LOG_BLOWUP + 1)) {
            lde_evals.values[i] += random_eval.values[i];
        }

        lde_evals.bit_reverse_rows().to_row_major_matrix()
    }

    /// Replay the mask draws `get_quotient_ldes` makes from `QUOTIENT_RNG_SEED`, then extend every
    /// chunk with [`quotient_lde_two_transforms`].
    fn expected_quotient_ldes<D: TwoAdicSubgroupDft<Val>>(
        dft: &D,
        domains: &[Domain],
        mats: &[RowMajorMatrix<Val>],
    ) -> Vec<RowMajorMatrix<Val>> {
        // The prover draws its masks from a private RNG forked off the seeded one.
        let mut rng = StdRng::from_rng(&mut StdRng::seed_from_u64(QUOTIENT_RNG_SEED));

        let cis = get_zp_cis(domains);
        let last_chunk = domains.len() - 1;
        let last_chunk_ci_inv = cis[last_chunk].inverse();
        let mul_coeffs = (0..last_chunk)
            .map(|i| cis[i] * last_chunk_ci_inv)
            .collect_vec();

        let randomized = mats
            .iter()
            .map(|mat| mat.with_random_cols(NUM_RANDOM_CODEWORDS, &mut rng))
            .collect_vec();
        let h = randomized[0].height();
        let w = randomized[0].width();
        let mut all_random_values = (0..last_chunk * h * w)
            .map(|_| rng.random())
            .chain(core::iter::repeat_n(Val::ZERO, h * w))
            .collect::<Vec<_>>();
        for j in 0..last_chunk {
            let mul_coeff = mul_coeffs[j];
            for k in 0..h * w {
                let t = all_random_values[j * h * w + k] * mul_coeff;
                all_random_values[last_chunk * h * w + k] -= t;
            }
        }

        domains
            .iter()
            .zip(randomized)
            .enumerate()
            .map(|(i, (domain, evals))| {
                let shift = Val::GENERATOR / domain.shift();
                quotient_lde_two_transforms(
                    dft,
                    evals,
                    &all_random_values[i * h * w..(i + 1) * h * w],
                    shift,
                )
            })
            .collect()
    }

    /// Check one backend's quotient LDE against the same result built from two transforms.
    ///
    /// Sharing a single forward transform between a chunk and its mask relies on two backend
    /// conventions, so every backend is pinned separately:
    ///
    /// - The inverse transform returns coefficients in natural order.
    /// - Undoing the row permutation recovers the order the commitment expects.
    fn check_fused_quotient_ldes<D: TwoAdicSubgroupDft<Val> + Clone + Sync>(dft: &D) {
        for (log_h, width, num_chunks) in [(4, 4, 2), (4, 3, 4)] {
            let mut rng = SmallRng::seed_from_u64(11);
            let perm = Perm::new_from_rng_128(&mut rng);
            let val_mmcs = ValMmcs::new(
                MyHash::new(perm.clone()),
                MyCompress::new(perm),
                0,
                StdRng::seed_from_u64(3),
            );
            let fri_params = FriParameters {
                log_blowup: QUOTIENT_LOG_BLOWUP,
                log_final_poly_len: 0,
                max_log_arity: 1,
                num_queries: 2,
                batch_proof_of_work_bits: 0,
                commit_proof_of_work_bits: 0,
                query_proof_of_work_bits: 0,
                mmcs: ChallengeMmcs::new(val_mmcs.clone()),
            };
            let pcs = HidingPcs::new(
                dft.clone(),
                val_mmcs,
                fri_params,
                NUM_RANDOM_CODEWORDS,
                StdRng::seed_from_u64(QUOTIENT_RNG_SEED),
            );

            // Chunk domains as the quotient prover builds them: a disjoint domain, split evenly.
            let trace_domain =
                Pcs::<Challenge, Challenger>::natural_domain_for_degree(&pcs, 1 << log_h);
            let domains = trace_domain
                .create_disjoint_domain(num_chunks << log_h)
                .split_domains(num_chunks);
            let mats = domains
                .iter()
                .map(|domain| RowMajorMatrix::<Val>::rand(&mut rng, domain.size(), width))
                .collect_vec();

            let expected = expected_quotient_ldes(dft, &domains, &mats);
            let fused = UnivariateStarkPcs::<Challenge, Challenger>::get_quotient_ldes(
                &pcs,
                domains.iter().copied().zip(mats).collect_vec(),
                num_chunks,
            )
            .unwrap();

            assert_eq!(fused.len(), expected.len());
            for (got, want) in fused.iter().zip(&expected) {
                assert_eq!(
                    got.values, want.values,
                    "log_h={log_h} width={width} num_chunks={num_chunks}"
                );
            }
        }
    }

    #[test]
    fn fused_quotient_lde_matches_two_transforms_radix2_dit() {
        check_fused_quotient_ldes(&Radix2Dit::<Val>::default());
    }

    #[test]
    fn fused_quotient_lde_matches_two_transforms_radix2_bowers() {
        check_fused_quotient_ldes(&Radix2Bowers);
    }

    #[test]
    fn fused_quotient_lde_matches_two_transforms_small_batch() {
        check_fused_quotient_ldes(&Radix2DFTSmallBatch::<Val>::default());
    }

    #[test]
    fn fused_quotient_lde_matches_two_transforms_dit_parallel() {
        // Why this backend earns its own case: it is the only one whose forward transform
        // hands back a permuted view rather than natural order.
        //
        //     other backends : forward transform -> rows already in natural order
        //     this backend   : forward transform -> rows in bit-reversed order, behind a view
        //
        // So it is the one place where the shared-transform rewrite could pick up the wrong
        // row order without any other test noticing.
        check_fused_quotient_ldes(&Radix2DitParallel::<Val>::default());
    }

    /// A lock check run from inside the parallel work of the object that owns the lock.
    ///
    /// The probed DFT lives inside the PCS it observes, so the check is armed after construction.
    #[derive(Clone, Default)]
    struct LockProbe {
        /// Returns whether the observed lock is free; unset until armed.
        is_free: Arc<Once<Box<dyn Fn() -> bool + Send + Sync>>>,
        /// Number of checks run, so a test can prove the probe actually fired.
        hits: Arc<AtomicUsize>,
    }

    impl LockProbe {
        /// Arm the probe on `target`, through a weak handle so the two do not keep each other alive.
        fn arm<T: Send + Sync + 'static>(&self, target: &Arc<T>, is_free: fn(&T) -> bool) {
            let target = Arc::downgrade(target);
            self.is_free.call_once(|| {
                Box::new(move || target.upgrade().is_none_or(|target| is_free(&target)))
            });
        }

        /// Fail if the observed lock is held; a no-op until armed.
        fn assert_lock_free(&self) {
            if let Some(is_free) = self.is_free.get() {
                assert!(
                    is_free(),
                    "parallel work must not run under the mask RNG lock"
                );
                self.hits.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// A DFT that runs its probe before every transform.
    ///
    /// Every default method of the trait funnels into `dft_batch`, so this one hook sees them all.
    #[derive(Clone, Default)]
    struct ProbedDft {
        inner: Dft,
        probe: LockProbe,
    }

    impl TwoAdicSubgroupDft<Val> for ProbedDft {
        type Evaluations = <Dft as TwoAdicSubgroupDft<Val>>::Evaluations;

        fn dft_batch(&self, mat: RowMajorMatrix<Val>) -> Self::Evaluations {
            self.probe.assert_lock_free();
            self.inner.dft_batch(mat)
        }
    }

    /// A shared generator that counts how many times it is drawn from.
    struct CountingRng {
        inner: StdRng,
        draws: Arc<AtomicUsize>,
    }

    impl CountingRng {
        /// Record one draw, then hand back the generator that serves it.
        fn draw(&mut self) -> &mut StdRng {
            self.draws.fetch_add(1, Ordering::Relaxed);
            &mut self.inner
        }
    }

    impl TryRng for CountingRng {
        type Error = Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Infallible> {
            self.draw().try_next_u32()
        }

        fn try_next_u64(&mut self) -> Result<u64, Infallible> {
            self.draw().try_next_u64()
        }

        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
            self.draw().try_fill_bytes(dst)
        }
    }

    impl TryCryptoRng for CountingRng {}

    type ProbedPcs = HidingFriPcs<Val, ProbedDft, ValMmcs, ChallengeMmcs, CountingRng>;

    /// A probed PCS, its armed lock probe, and the draw counter of its shared generator.
    fn probed_pcs() -> (Arc<ProbedPcs>, LockProbe, Arc<AtomicUsize>) {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let val_mmcs = ValMmcs::new(
            MyHash::new(perm.clone()),
            MyCompress::new(perm),
            0,
            StdRng::seed_from_u64(3),
        );
        let fri_params = FriParameters {
            log_blowup: QUOTIENT_LOG_BLOWUP,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            batch_proof_of_work_bits: 0,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: ChallengeMmcs::new(val_mmcs.clone()),
        };
        let probe = LockProbe::default();
        let draws = Arc::new(AtomicUsize::new(0));
        let pcs = Arc::new(ProbedPcs::new(
            ProbedDft {
                inner: Dft::default(),
                probe: probe.clone(),
            },
            val_mmcs,
            fri_params,
            NUM_RANDOM_CODEWORDS,
            CountingRng {
                inner: StdRng::seed_from_u64(QUOTIENT_RNG_SEED),
                draws: draws.clone(),
            },
        ));
        probe.arm(&pcs, |pcs| !pcs.rng.is_locked());
        (pcs, probe, draws)
    }

    /// Commit a random trace of the given shape through the probed PCS.
    fn probed_commit(pcs: &ProbedPcs, log_h: usize, width: usize) {
        // The wrapper doubles the trace height, so the domain is twice the trace.
        let domain = Pcs::<Challenge, Challenger>::natural_domain_for_degree(pcs, 2 << log_h);
        let trace = RowMajorMatrix::<Val>::rand(&mut SmallRng::seed_from_u64(3), 1 << log_h, width);
        Pcs::<Challenge, Challenger>::commit(pcs, [(domain, trace)]).unwrap();
    }

    /// Extend random quotient chunks of the given shape through the probed PCS.
    fn probed_quotient_ldes(pcs: &ProbedPcs, log_h: usize, width: usize) {
        let num_chunks = 2;
        let trace_domain = Pcs::<Challenge, Challenger>::natural_domain_for_degree(pcs, 1 << log_h);
        let mut rng = SmallRng::seed_from_u64(3);
        let chunks = trace_domain
            .create_disjoint_domain(num_chunks << log_h)
            .split_domains(num_chunks)
            .into_iter()
            .map(|domain| {
                let mat = RowMajorMatrix::<Val>::rand(&mut rng, domain.size(), width);
                (domain, mat)
            })
            .collect_vec();
        UnivariateStarkPcs::<Challenge, Challenger>::get_quotient_ldes(pcs, chunks, num_chunks)
            .unwrap();
    }

    #[test]
    fn commit_transforms_without_holding_the_rng_lock() {
        // A rayon worker that waits inside a transform may run another commit on this PCS.
        //
        // That commit would spin forever if the transform ran under the mask lock.
        let (pcs, probe, _) = probed_pcs();
        probed_commit(&pcs, 4, 4);

        // The check must have run, or the test would pass without probing anything.
        assert!(probe.hits.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn quotient_ldes_transform_without_holding_the_rng_lock() {
        // The batch prover calls this from a parallel loop over AIR instances.
        //
        // A worker waiting inside one instance's transform may run another instance's call.
        //
        // That call would spin forever if the transform ran under the mask lock.
        let (pcs, probe, _) = probed_pcs();
        probed_quotient_ldes(&pcs, 4, 4);

        // The check must have run, or the test would pass without probing anything.
        assert!(probe.hits.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn shared_rng_only_seeds_the_masks() {
        // The masks are filled after a parallel row copy that no transform probe can observe.
        //
        // Drawing them from the shared generator would put that copy under its lock.
        //
        // So each call must draw a fixed seed from it, however large the input.
        let (pcs, _, draws) = probed_pcs();
        let draws_for = |run: &dyn Fn(&ProbedPcs)| {
            let before = draws.load(Ordering::Relaxed);
            run(&pcs);
            draws.load(Ordering::Relaxed) - before
        };

        let small_commit = draws_for(&|pcs| probed_commit(pcs, 4, 2));
        let large_commit = draws_for(&|pcs| probed_commit(pcs, 6, 8));
        assert!(small_commit > 0);
        assert_eq!(small_commit, large_commit);

        let small_quotient = draws_for(&|pcs| probed_quotient_ldes(pcs, 4, 2));
        let large_quotient = draws_for(&|pcs| probed_quotient_ldes(pcs, 6, 8));
        assert!(small_quotient > 0);
        assert_eq!(small_quotient, large_quotient);
    }
}
