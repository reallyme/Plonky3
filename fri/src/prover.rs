// Modified by ReallyMe Technologies Limited, 2026.
// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, OpeningRequest};
#[cfg(feature = "secret-memory")]
use p3_commit::{SecretMmcs, SecretProverData};
#[cfg(feature = "secret-memory")]
use p3_dft::{Radix2Bowers, SecretDftError, SecretTwoAdicSubgroupDft};
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
#[cfg(feature = "secret-memory")]
use p3_matrix::secret::SecretMatrixError;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};
use tracing::{debug_span, info_span, instrument};

use crate::{
    BatchMultiOpening, CommitPhaseMultiStep, FriFoldingStrategy, FriParameters, FriProof,
    FriProverError, FriShape, ProverDataWithOpeningPoints, ProverTranscript,
};
#[cfg(feature = "secret-memory")]
use crate::{SecretFriFoldingStrategy, transcript::SecretProverTranscriptError};

/// Create a proof that an opening `f(zeta)` is correct by proving that the
/// function `(f(x) - f(zeta))/(x - zeta)` is low degree.
///
/// This further supports proving a batch of these claims for a collection of polynomials of shrinking degrees.
/// Polynomials of equal degree can be combined using randomness before calling this function.
///
/// The Soundness error from prove_fri comes from the paper:
/// Proximity Gaps for Reed-Solomon Codes (<https://eprint.iacr.org/2020/654>)
/// and is either `rate^{num_queries}` or `rate^{num_queries/2}` depending on if you rely on conjectured or
/// proven soundness. Particularly safety conscious users may want to set `num_queries` slightly higher than
/// this to account for the fact that most implementations batch inputs using a single random challenge
/// instead of one challenge for each polynomial and due to the birthday paradox,
/// there is a non trivial chance that two queried indices will be equal.
///
/// Arguments:
/// - `folding`: The FRI folding scheme to use.
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `inputs`: The evaluation vectors of all polynomials we are applying FRI to. The function assumes that
///   commitments to these vectors have been produced and observed by the challenger earlier in the protocol.
/// - `challenger`: The Fiat-Shamir challenger to use for sampling challenges.
/// - `log_global_max_height`: The log of the maximum height of the input matrices.
/// - `prover_data_with_opening_points`: A list of pairs of a batch commitment to a collection
///   of matrices and a list of points to open those matrices at.
/// - `batch_pow_witness`: The proof of work guarding the challenge that batched the inputs.
///   That challenge is consumed in building the inputs, so it is drawn before this call.
///   The witness still travels in the proof assembled here, which is the proof the caller ships.
///   A caller that batches nothing passes a zero witness and a zero difficulty.
///
/// Returns a terminal-height configuration error before changing the challenger if any
/// input is too short. A constant final polynomial permits the zero-fold boundary.
#[instrument(name = "FRI prover", skip_all)]
// The argument list is the protocol's own shape: the folding strategy, the parameters, the inputs,
// the transcript, the instance height, the committed data, its MMCS, and the caller's batch witness.
// Grouping any of them into a struct would only move the same fields behind another name.
#[allow(clippy::too_many_arguments)]
pub fn prove_fri<Folding, Val, Challenge, InputMmcs, FriMmcs, Challenger>(
    folding: &Folding,
    params: &FriParameters<FriMmcs>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
    log_global_max_height: usize,
    prover_data_with_opening_points: &[ProverDataWithOpeningPoints<
        '_,
        Challenge,
        InputMmcs::ProverData<RowMajorMatrix<Val>>,
    >],
    input_mmcs: &InputMmcs,
    batch_pow_witness: Val,
) -> Result<FriProof<Challenge, FriMmcs, Val, Folding::InputProof>, FriProverError>
where
    Val: TwoAdicField + PrimeField64,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenger:
        FieldChallenger<Val> + GrindingChallenger<Witness = Val> + CanObserve<FriMmcs::Commitment>,
    Folding:
        FriFoldingStrategy<Val, Challenge, InputProof = Vec<BatchMultiOpening<Val, InputMmcs>>>,
{
    prove_fri_with_input_opener(
        folding,
        params,
        inputs,
        challenger,
        log_global_max_height,
        batch_pow_witness,
        |indices| {
            Ok(open_inputs(
                log_global_max_height,
                indices,
                prover_data_with_opening_points,
                input_mmcs,
            ))
        },
    )
}

/// Prove with a caller-supplied folding schedule.
///
/// An honest prover passes `None` and the schedule is derived.
///
/// A test passes `Some` to forge a schedule the verifier will not derive.
/// Every commitment, opening and transcript value then agrees with the forgery.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn prove_fri_with_schedule<Folding, Val, Challenge, InputMmcs, FriMmcs, Challenger>(
    folding: &Folding,
    params: &FriParameters<FriMmcs>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
    log_global_max_height: usize,
    prover_data_with_opening_points: &[ProverDataWithOpeningPoints<
        '_,
        Challenge,
        InputMmcs::ProverData<RowMajorMatrix<Val>>,
    >],
    input_mmcs: &InputMmcs,
    batch_pow_witness: Challenger::Witness,
    schedule: Option<Vec<usize>>,
) -> Result<FriProof<Challenge, FriMmcs, Val, Folding::InputProof>, FriProverError>
where
    Val: TwoAdicField + PrimeField64,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenger:
        FieldChallenger<Val> + GrindingChallenger<Witness = Val> + CanObserve<FriMmcs::Commitment>,
    Folding:
        FriFoldingStrategy<Val, Challenge, InputProof = Vec<BatchMultiOpening<Val, InputMmcs>>>,
{
    prove_fri_with_schedule_and_input_opener(
        folding,
        params,
        inputs,
        challenger,
        log_global_max_height,
        batch_pow_witness,
        schedule,
        |indices| {
            Ok(open_inputs(
                log_global_max_height,
                indices,
                prover_data_with_opening_points,
                input_mmcs,
            ))
        },
    )
}

/// Prove FRI while delegating input commitment openings until query indices
/// have been derived from the Fiat-Shamir transcript.
///
/// This boundary lets a guarded PCS erase full committed matrices after their
/// final quotient-reduction read, retain only authentication state, and
/// reconstruct the selected rows once the transcript reveals the queries.
/// The callback cannot influence the transcript or query indices and produces
/// the same [`BatchMultiOpening`] wire type as the ordinary retained-data path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_fri_with_input_opener<
    Folding,
    Val,
    Challenge,
    InputMmcs,
    FriMmcs,
    Challenger,
    OpenError,
    OpenInputs,
>(
    folding: &Folding,
    params: &FriParameters<FriMmcs>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
    log_global_max_height: usize,
    batch_pow_witness: Challenger::Witness,
    open_inputs: OpenInputs,
) -> Result<FriProof<Challenge, FriMmcs, Val, Folding::InputProof>, OpenError>
where
    Val: TwoAdicField + PrimeField64,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenger:
        FieldChallenger<Val> + GrindingChallenger<Witness = Val> + CanObserve<FriMmcs::Commitment>,
    Folding:
        FriFoldingStrategy<Val, Challenge, InputProof = Vec<BatchMultiOpening<Val, InputMmcs>>>,
    OpenError: From<FriProverError>,
    OpenInputs: FnOnce(&[usize]) -> Result<Vec<BatchMultiOpening<Val, InputMmcs>>, OpenError>,
{
    prove_fri_with_schedule_and_input_opener(
        folding,
        params,
        inputs,
        challenger,
        log_global_max_height,
        batch_pow_witness,
        None,
        open_inputs,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_fri_with_schedule_and_input_opener<
    Folding,
    Val,
    Challenge,
    InputMmcs,
    FriMmcs,
    Challenger,
    OpenError,
    OpenInputs,
>(
    folding: &Folding,
    params: &FriParameters<FriMmcs>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
    log_global_max_height: usize,
    batch_pow_witness: Challenger::Witness,
    schedule: Option<Vec<usize>>,
    open_inputs: OpenInputs,
) -> Result<FriProof<Challenge, FriMmcs, Val, Folding::InputProof>, OpenError>
where
    Val: TwoAdicField + PrimeField64,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenger:
        FieldChallenger<Val> + GrindingChallenger<Witness = Val> + CanObserve<FriMmcs::Commitment>,
    Folding:
        FriFoldingStrategy<Val, Challenge, InputProof = Vec<BatchMultiOpening<Val, InputMmcs>>>,
    OpenError: From<FriProverError>,
    OpenInputs: FnOnce(&[usize]) -> Result<Vec<BatchMultiOpening<Val, InputMmcs>>, OpenError>,
{
    assert!(!inputs.is_empty());
    assert!(
        params.num_queries > 0,
        "num_queries must be at least 1 for FRI soundness"
    );
    assert!(
        params.log_blowup > 0,
        "log_blowup must be at least 1 for FRI soundness"
    );
    assert!(
        params.max_log_arity > 0,
        "max_log_arity must be at least 1 to guarantee folding progress"
    );
    assert!(
        inputs
            .iter()
            .tuple_windows()
            .all(|(l, r)| l.len() >= r.len()),
        "Inputs are not sorted in descending order of length."
    );

    // Index sampling and `open_input` must agree on the height; the caller's value is canonical.
    assert_eq!(
        log_global_max_height,
        log2_strict_usize(inputs[0].len()),
        "log_global_max_height must match the largest input length"
    );
    let log_min_height = log2_strict_usize(inputs.last().unwrap().len());
    params.validate_input_height(log_min_height)?;

    // Describe the transcript before running it.
    //
    // Every number comes from the parameters and the input heights.
    // The verifier builds the identical description from its own configuration.
    let input_log_heights: Vec<usize> = inputs
        .iter()
        .map(|input| log2_strict_usize(input.len()))
        .collect();
    let index_bits = log_global_max_height + folding.extra_query_index_bits();

    // A forged schedule stands in for the derived one, so no derivation runs.
    //
    // `fold_schedule` asserts strictly decreasing heights.
    // Inputs of equal length clear this function's own sort check but not that one.
    //
    // Folding, the described transcript and the proof all agree with whichever wins.
    let shape = schedule.map_or_else(
        || FriShape::new(params, &input_log_heights, index_bits),
        |log_arities| FriShape::with_schedule(params, log_arities, index_bits),
    );

    // Folding walks the described schedule, so the two cannot drift apart.
    let log_arities = shape.log_arities.clone();

    let mut transcript = ProverTranscript::<Challenger, Val, Challenge>::new(challenger, shape);

    // Fold the inputs down until the degree reaches the final polynomial's.
    // Every commitment and folding challenge passes through the transcript.
    let commit_phase_result = commit_phase(folding, params, inputs, &log_arities, &mut transcript);

    // Bind the final polynomial, grind, and draw every query index.
    //
    // Indices are drawn in one block, so the birthday bound applies.
    // At 100 queries over 2^20 positions, two collide about 0.5% of the time.
    let (indices, pow_witness) = transcript.query_phase(&commit_phase_result.final_poly);
    let pow_witness = pow_witness.unwrap_or(Val::ZERO);

    // Every described step has now been played.
    transcript.finish();

    let (input_openings, commit_phase_openings) = info_span!("query phase").in_scope(|| {
        // Openings of the inputs and of every commit-phase codeword at all
        // queried locations. Queries into one tree share a single proof, so
        // overlapping authentication paths are deduplicated instead of being
        // shipped once per query.
        let input_openings = open_inputs(&indices)?;
        let domain_indices: Vec<usize> = indices
            .iter()
            .map(|&index| index >> folding.extra_query_index_bits())
            .collect();
        let commit_phase_openings = answer_queries(
            params,
            &commit_phase_result.log_arities,
            &commit_phase_result.data,
            &domain_indices,
        );
        Ok::<_, OpenError>((input_openings, commit_phase_openings))
    })?;

    Ok(FriProof {
        batch_pow_witness,
        commit_phase_commits: commit_phase_result.commits,
        commit_pow_witnesses: commit_phase_result.pow_witnesses,
        input_openings,
        commit_phase_openings,
        final_poly: commit_phase_result.final_poly,
        query_pow_witness: pow_witness,
    })
}

/// Failure while driving the complete guarded FRI protocol.
///
/// The variants deliberately identify protocol phases without formatting their
/// nested errors. Some implementations attach internal commitment state to
/// those errors; keeping it out of `Display` prevents accidental disclosure in
/// logs while preserving the concrete type for programmatic handling.
#[cfg(feature = "secret-memory")]
#[derive(Debug, thiserror::Error)]
pub(crate) enum SecretFriDriverError<OpenError, MmcsError, FoldError> {
    /// No reduced-opening codeword was supplied.
    #[error("guarded FRI has no input codeword")]
    EmptyInputs,
    /// The configured query count is zero.
    #[error("guarded FRI query count must be nonzero")]
    ZeroQueries,
    /// The configured blowup is zero.
    #[error("guarded FRI blowup logarithm must be nonzero")]
    ZeroLogBlowup,
    /// The configured maximum folding arity cannot make progress.
    #[error("guarded FRI maximum folding arity must be nonzero")]
    ZeroMaxLogArity,
    /// A guarded input is empty, non-binary, or not a single column.
    #[error("guarded FRI input matrix shape is invalid")]
    InvalidInputShape,
    /// Input heights are not strictly decreasing.
    #[error("guarded FRI input heights must be strictly decreasing")]
    InputHeightOrder,
    /// The caller's global height does not match the tallest input.
    #[error("guarded FRI global height does not match its tallest input")]
    GlobalHeightMismatch,
    /// A configured height or query-index width cannot be represented.
    #[error("guarded FRI configuration overflows the platform index width")]
    IndexWidthOverflow,
    /// Driver metadata storage could not be reserved.
    #[error("guarded FRI driver allocation failed")]
    AllocationFailed,
    /// FRI parameter validation failed before transcript mutation.
    #[error("guarded FRI parameter validation failed")]
    Parameters(FriProverError),
    /// A guarded commit round failed.
    #[error("guarded FRI commit phase failed")]
    Commit(SecretFriCommitPhaseError<MmcsError, FoldError>),
    /// Terminal evaluations could not be converted into disclosed coefficients.
    #[error("guarded FRI terminal polynomial conversion failed")]
    FinalPolynomial(SecretFriFinalPolynomialError),
    /// The guarded transcript did not close cleanly.
    #[error("guarded FRI transcript finalization failed")]
    Transcript(SecretProverTranscriptError),
    /// The caller could not open the input commitments at the fixed queries.
    #[error("guarded FRI input opening failed")]
    InputOpening(OpenError),
    /// A guarded commit-phase layer could not be opened at the fixed queries.
    #[error("guarded FRI query answering failed")]
    Query(SecretFriQueryError<MmcsError>),
}

/// Complete guarded FRI result with its phase-specific failure type.
#[cfg(feature = "secret-memory")]
type SecretFriDriverOutput<Folding, Val, Challenge, FriMmcs, OpenError> = Result<
    FriProof<Challenge, FriMmcs, Val, <Folding as FriFoldingStrategy<Val, Challenge>>::InputProof>,
    SecretFriDriverError<
        OpenError,
        <FriMmcs as SecretMmcs<Challenge>>::SecretError,
        <Folding as SecretFriFoldingStrategy<Val, Challenge>>::SecretFoldError,
    >,
>;

/// Prove one complete FRI instance while retaining every secret-derived
/// codeword under a zeroizing owner.
///
/// All configuration and input-shape checks finish before the challenger is
/// borrowed by the transcript. Once Fiat-Shamir fixes the query indices, the
/// caller-supplied opener materializes only the public input rows and
/// authentication data needed by the proof. This ordering lets the surrounding
/// PCS erase full input matrices before opening without giving the callback any
/// influence over query selection.
#[cfg(feature = "secret-memory")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_fri_secret_with_input_opener<
    Folding,
    Val,
    Challenge,
    FriMmcs,
    Challenger,
    OpenError,
    OpenInputs,
>(
    folding: &Folding,
    params: &FriParameters<FriMmcs>,
    inputs: Vec<SecretProverData<RowMajorMatrix<Challenge>>>,
    challenger: &mut Challenger,
    log_global_max_height: usize,
    batch_pow_witness: Val,
    open_inputs: OpenInputs,
) -> SecretFriDriverOutput<Folding, Val, Challenge, FriMmcs, OpenError>
where
    Val: TwoAdicField + PrimeField64 + zeroize::Zeroize,
    Challenge: ExtensionField<Val> + zeroize::Zeroize,
    FriMmcs: Mmcs<Challenge> + SecretMmcs<Challenge>,
    Challenger:
        FieldChallenger<Val> + GrindingChallenger<Witness = Val> + CanObserve<FriMmcs::Commitment>,
    Folding: SecretFriFoldingStrategy<Val, Challenge>,
    OpenInputs: FnOnce(&[usize]) -> Result<Folding::InputProof, OpenError>,
{
    if params.num_queries == 0 {
        return Err(SecretFriDriverError::ZeroQueries);
    }
    if params.log_blowup == 0 {
        return Err(SecretFriDriverError::ZeroLogBlowup);
    }
    if params.max_log_arity == 0 {
        return Err(SecretFriDriverError::ZeroMaxLogArity);
    }

    let platform_index_bits =
        usize::try_from(usize::BITS).map_err(|_| SecretFriDriverError::IndexWidthOverflow)?;
    if params.log_final_poly_len >= platform_index_bits {
        return Err(SecretFriDriverError::IndexWidthOverflow);
    }
    params
        .log_blowup
        .checked_add(params.log_final_poly_len)
        .ok_or(SecretFriDriverError::IndexWidthOverflow)?;

    let first_input = inputs.first().ok_or(SecretFriDriverError::EmptyInputs)?;
    if inputs.iter().any(|input| {
        input.width != 1 || input.values.is_empty() || !input.values.len().is_power_of_two()
    }) {
        return Err(SecretFriDriverError::InvalidInputShape);
    }
    if inputs
        .windows(2)
        .any(|pair| pair[0].values.len() <= pair[1].values.len())
    {
        return Err(SecretFriDriverError::InputHeightOrder);
    }

    let first_log_height = log2_strict_usize(first_input.values.len());
    if first_log_height != log_global_max_height {
        return Err(SecretFriDriverError::GlobalHeightMismatch);
    }
    let last_log_height = inputs
        .last()
        .map(|input| log2_strict_usize(input.values.len()))
        .ok_or(SecretFriDriverError::EmptyInputs)?;
    params
        .validate_input_height(last_log_height)
        .map_err(SecretFriDriverError::Parameters)?;

    let extra_query_index_bits = folding.extra_query_index_bits();
    if extra_query_index_bits >= platform_index_bits {
        return Err(SecretFriDriverError::IndexWidthOverflow);
    }
    let index_bits = log_global_max_height
        .checked_add(extra_query_index_bits)
        .filter(|&bits| bits <= platform_index_bits)
        .ok_or(SecretFriDriverError::IndexWidthOverflow)?;

    let mut input_log_heights = Vec::new();
    input_log_heights
        .try_reserve_exact(inputs.len())
        .map_err(|_| SecretFriDriverError::AllocationFailed)?;
    input_log_heights.extend(
        inputs
            .iter()
            .map(|input| log2_strict_usize(input.values.len())),
    );

    // Preflight guarantees the invariants required by `FriShape::new`: the
    // heights are strictly decreasing, the target-height addition is bounded,
    // and every fold advances by at least one level.
    let shape = FriShape::new(params, &input_log_heights, index_bits);
    let mut log_arities = Vec::new();
    log_arities
        .try_reserve_exact(shape.log_arities.len())
        .map_err(|_| SecretFriDriverError::AllocationFailed)?;
    log_arities.extend_from_slice(&shape.log_arities);

    let mut transcript = ProverTranscript::<Challenger, Val, Challenge>::new(challenger, shape);
    let commit_result =
        match commit_phase_secret(folding, params, inputs, &log_arities, &mut transcript) {
            Ok(result) => result,
            Err(error) => {
                transcript.abort();
                return Err(SecretFriDriverError::Commit(error));
            }
        };

    let SecretCommitPhaseResult {
        commits,
        data,
        log_arities,
        pow_witnesses,
        final_evaluations,
    } = commit_result;
    let final_poly = match disclose_final_poly_secret::<Val, Challenge>(final_evaluations) {
        Ok(coefficients) => coefficients,
        Err(error) => {
            transcript.abort();
            return Err(SecretFriDriverError::FinalPolynomial(error));
        }
    };

    let (indices, query_pow_witness) = transcript.query_phase(&final_poly);
    let query_pow_witness = query_pow_witness.unwrap_or(Val::ZERO);
    transcript
        .finish_secret()
        .map_err(SecretFriDriverError::Transcript)?;

    let mut domain_indices = Vec::new();
    domain_indices
        .try_reserve_exact(indices.len())
        .map_err(|_| SecretFriDriverError::AllocationFailed)?;
    domain_indices.extend(indices.iter().map(|&index| index >> extra_query_index_bits));

    let input_openings = open_inputs(&indices).map_err(SecretFriDriverError::InputOpening)?;
    let commit_phase_openings = answer_queries_secret(params, &log_arities, data, &domain_indices)
        .map_err(SecretFriDriverError::Query)?;

    Ok(FriProof {
        batch_pow_witness,
        commit_phase_commits: commits,
        commit_pow_witnesses: pow_witnesses,
        input_openings,
        commit_phase_openings,
        final_poly,
        query_pow_witness,
    })
}

struct CommitPhaseResult<F: Field, M: Mmcs<F>, Witness> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    log_arities: Vec<usize>,
    pow_witnesses: Vec<Witness>,
    final_poly: Vec<F>,
}

/// Failure while building guarded FRI commit rounds.
#[cfg(feature = "secret-memory")]
#[derive(Debug, thiserror::Error)]
pub(crate) enum SecretFriCommitPhaseError<MmcsError, FoldError> {
    /// The FRI input collection is empty.
    #[error("guarded FRI commit phase has no input codeword")]
    EmptyInputs,
    /// An input or retained MMCS matrix has an inconsistent row-major shape.
    #[error("guarded FRI commit phase matrix shape is invalid")]
    InvalidMatrixShape,
    /// A folding arity cannot be represented by `usize`.
    #[error("guarded FRI commit phase arity overflows usize")]
    ArityOverflow,
    /// Commit-phase metadata storage could not be reserved.
    #[error("guarded FRI commit phase allocation failed")]
    AllocationFailed,
    /// The supplied input heights do not match the folding schedule.
    #[error("guarded FRI inputs do not match the folding schedule")]
    InputScheduleMismatch,
    /// The configured final polynomial length cannot be represented.
    #[error("guarded FRI final polynomial length overflows usize")]
    FinalPolynomialLengthOverflow,
    /// A guarded MMCS commitment failed.
    #[error("guarded FRI MMCS commitment failed")]
    Mmcs(MmcsError),
    /// A guarded folding operation failed.
    #[error("guarded FRI folding failed")]
    Fold(FoldError),
}

/// Guarded state produced by the FRI commit phase.
///
/// The final evaluations remain guarded because converting them into the
/// intentionally disclosed final polynomial is a separate ownership boundary.
#[cfg(feature = "secret-memory")]
pub(crate) struct SecretCommitPhaseResult<F, M, Witness>
where
    F: Field + zeroize::Zeroize,
    M: SecretMmcs<F>,
{
    pub(crate) commits: Vec<M::Commitment>,
    pub(crate) data: Vec<SecretProverData<M::SecretData<RowMajorMatrix<F>>>>,
    pub(crate) log_arities: Vec<usize>,
    pub(crate) pow_witnesses: Vec<Witness>,
    pub(crate) final_evaluations: SecretProverData<RowMajorMatrix<F>>,
}

/// Typed result of guarded FRI commit-phase construction.
#[cfg(feature = "secret-memory")]
type SecretCommitPhaseOutput<F, M, Witness, FoldError> = Result<
    SecretCommitPhaseResult<F, M, Witness>,
    SecretFriCommitPhaseError<<M as SecretMmcs<F>>::SecretError, FoldError>,
>;

/// Failure while converting guarded terminal evaluations into the public FRI polynomial.
#[cfg(feature = "secret-memory")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum SecretFriFinalPolynomialError {
    /// The terminal evaluation matrix is empty or not a single extension-field column.
    #[error("guarded FRI terminal evaluation shape is invalid")]
    InvalidShape,
    /// Guarded extension-field flattening failed.
    #[error("guarded FRI terminal representation conversion failed")]
    Matrix(SecretMatrixError),
    /// The guarded inverse transform failed.
    #[error("guarded FRI terminal inverse transform failed")]
    Dft(SecretDftError),
    /// The public final-polynomial allocation could not be reserved.
    #[error("guarded FRI final-polynomial allocation failed")]
    AllocationFailed,
}

/// Failure while opening guarded FRI commit-phase layers at transcript-derived queries.
#[cfg(feature = "secret-memory")]
#[derive(Debug, thiserror::Error)]
pub(crate) enum SecretFriQueryError<MmcsError> {
    /// A proof with no query positions cannot satisfy the FRI soundness contract.
    #[error("guarded FRI query set is empty")]
    EmptyQuerySet,
    /// The committed-layer count differs from the folding schedule.
    #[error("guarded FRI query layer count does not match the folding schedule")]
    RoundCountMismatch,
    /// A query round cannot fold by arity one.
    #[error("guarded FRI query log arity is zero")]
    ZeroLogArity,
    /// A query round's arity cannot be represented by `usize`.
    #[error("guarded FRI query arity overflows usize")]
    ArityOverflow,
    /// Query or proof metadata could not be reserved.
    #[error("guarded FRI query allocation failed")]
    AllocationFailed,
    /// A guarded MMCS opening did not have the single-row shape fixed by FRI.
    #[error("guarded FRI query opening shape is invalid")]
    InvalidOpeningShape,
    /// The guarded MMCS rejected a query opening.
    #[error("guarded FRI query MMCS opening failed")]
    Mmcs(MmcsError),
}

/// Perform the commit phase of the FRI protocol.
///
/// In each round we reduce our evaluations over `H` to evaluations over `H^k` (where k is the arity)
/// by folding with a random challenge. For instance, for arity 2, we have:
/// ```text
///     f_{i + 1}(x^2) = (f_i(x) + f_i(-x))/2 + beta_i (f_i(x) - f_i(-x))/2x
/// ```
/// We then commit to the evaluation vector over the smaller domain, i.e. `f_{i + 1}` over `H^2` for arity 2.
///
/// The arity for each round is dynamically computed to ensure we always commit at each input height level.
///
/// Once the degree of our polynomial falls below `final_poly_degree`, we compute the coefficients of our
/// polynomial and return them along with all intermediate evaluations and corresponding commitments.
///
/// Arguments:
/// - `folding`: The FRI folding scheme used by the prover.
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `inputs`: The evaluation vectors of the polynomials. These must be sorted in descending order of length and each
///   evaluation vector must be in bit reversed order. This function assumes that commitments to these vectors
///   have already been produced and observed by the challenger.
/// - `log_arities`: The log2 of the arity each round folds by, in round order.
/// - `transcript`: The described transcript every commitment and challenge passes through.
#[instrument(name = "commit phase", skip_all)]
fn commit_phase<Folding, Val, Challenge, M, Challenger>(
    folding: &Folding,
    params: &FriParameters<M>,
    inputs: Vec<Vec<Challenge>>,
    log_arities: &[usize],
    transcript: &mut ProverTranscript<'_, Challenger, Val, Challenge>,
) -> CommitPhaseResult<Challenge, M, Val>
where
    Val: TwoAdicField + PrimeField64,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger:
        FieldChallenger<Val> + GrindingChallenger<Witness = Val> + CanObserve<M::Commitment>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    assert!(
        params.max_log_arity > 0,
        "max_log_arity must be at least 1 to guarantee folding progress"
    );

    let mut inputs_iter = inputs.into_iter().peekable();
    let mut folded = inputs_iter.next().unwrap();
    let mut commits = vec![];
    let mut data = vec![];
    let mut pow_witnesses = vec![];

    for &log_arity in log_arities {
        let arity = 1 << log_arity;

        // As folded is in bit reversed order, the evaluations at conjugate points are adjacent.
        // We reinterpret the vector as a matrix of width `arity`.
        let leaves = RowMajorMatrix::new(folded, arity);

        // Commit to these evaluations.
        let (commit, prover_data) = params.mmcs.commit_matrix(leaves);

        // One call binds the commitment, grinds, and draws the folding challenge.
        let (beta, witness) = transcript.commit_round(commit.clone());
        commits.push(commit);
        // A zero difficulty still occupies a slot, keeping the proof shape fixed.
        pow_witnesses.push(witness.unwrap_or(Val::ZERO));

        // We passed ownership of `leaves` to the MMCS, so get a reference to it
        let leaves = params.mmcs.get_matrices(&prover_data).pop().unwrap();
        // Do the folding operation with the computed arity
        folded = folding.fold_matrix(beta, log_arity, leaves.as_view());

        data.push(prover_data);

        // If we have reached the size of the next input vector, we can add it to the current vector.
        if let Some(v) = inputs_iter.next_if(|v| v.len() == folded.len()) {
            // Each element of `inputs_iter` is a reduced opening polynomial, which is itself a
            // random linear combination `f_{i, 0} + alpha f_{i, 1} + ...`, when we add it
            // to the current folded polynomial, we need to multiply by a random factor.
            // We use beta^arity as the random factor to maintain independence.
            let beta_pow = beta.exp_power_of_2(log_arity);
            // One item reads the running coefficient and the incoming one.
            // It rewrites the running coefficient in place, so it moves three elements.
            //
            // Each round folds the vector down by the arity.
            // The last rounds are therefore short enough that a dispatch outweighs the pass.
            let item_bytes = 3 * size_of::<Challenge>();
            folded
                .par_iter_mut()
                .zip(v.par_iter())
                .for_each_min_task_bytes(item_bytes, |(c, &x)| *c += beta_pow * x);
        }
    }

    // Now we need to get the coefficients of the final polynomial. As we know that the degree
    // is `<= params.final_poly_len()` and the evaluations are stored in bit-reversed order,
    // we can just truncate the folded vector, bit-reverse again and run an IDFT.
    folded.truncate(params.final_poly_len());
    reverse_slice_index_bits(&mut folded);
    let final_poly = debug_span!("idft final poly")
        .in_scope(|| Radix2DFTSmallBatch::default().idft_algebra(folded));

    CommitPhaseResult {
        commits,
        data,
        log_arities: log_arities.to_vec(),
        pow_witnesses,
        final_poly,
    }
}

/// Build every FRI commit round without releasing a secret-derived codeword
/// allocation outside an erasing owner.
///
/// This remains crate-private until the guarded FRI driver composes it with
/// terminal-polynomial disclosure and query answering. Exposing the pieces as
/// a complete prover would overstate the memory-erasure contract.
#[cfg(feature = "secret-memory")]
pub(crate) fn commit_phase_secret<Folding, Val, Challenge, M, Challenger>(
    folding: &Folding,
    params: &FriParameters<M>,
    inputs: Vec<SecretProverData<RowMajorMatrix<Challenge>>>,
    log_arities: &[usize],
    transcript: &mut ProverTranscript<'_, Challenger, Val, Challenge>,
) -> SecretCommitPhaseOutput<Challenge, M, Val, Folding::SecretFoldError>
where
    Val: TwoAdicField + PrimeField64,
    Challenge: ExtensionField<Val> + zeroize::Zeroize,
    M: Mmcs<Challenge> + SecretMmcs<Challenge>,
    Challenger:
        FieldChallenger<Val> + GrindingChallenger<Witness = Val> + CanObserve<M::Commitment>,
    Folding: SecretFriFoldingStrategy<Val, Challenge>,
{
    let mut inputs_iter = inputs.into_iter().peekable();
    let mut folded = inputs_iter
        .next()
        .ok_or(SecretFriCommitPhaseError::EmptyInputs)?;
    if folded.width != 1 || folded.values.is_empty() {
        return Err(SecretFriCommitPhaseError::InvalidMatrixShape);
    }

    let mut commits = Vec::new();
    let mut data = Vec::new();
    let mut retained_log_arities = Vec::new();
    let mut pow_witnesses = Vec::new();
    commits
        .try_reserve_exact(log_arities.len())
        .map_err(|_| SecretFriCommitPhaseError::AllocationFailed)?;
    data.try_reserve_exact(log_arities.len())
        .map_err(|_| SecretFriCommitPhaseError::AllocationFailed)?;
    retained_log_arities
        .try_reserve_exact(log_arities.len())
        .map_err(|_| SecretFriCommitPhaseError::AllocationFailed)?;
    retained_log_arities.extend_from_slice(log_arities);
    pow_witnesses
        .try_reserve_exact(log_arities.len())
        .map_err(|_| SecretFriCommitPhaseError::AllocationFailed)?;

    for &log_arity in log_arities {
        if log_arity == 0 {
            return Err(SecretFriCommitPhaseError::InputScheduleMismatch);
        }
        let arity_shift =
            u32::try_from(log_arity).map_err(|_| SecretFriCommitPhaseError::ArityOverflow)?;
        let arity = 1_usize
            .checked_shl(arity_shift)
            .ok_or(SecretFriCommitPhaseError::ArityOverflow)?;
        if folded.width != 1
            || folded.values.is_empty()
            || !folded.values.len().is_multiple_of(arity)
        {
            return Err(SecretFriCommitPhaseError::InvalidMatrixShape);
        }

        // Reinterpret the same guarded allocation as arity-wide Merkle leaves.
        // No value moves into an unguarded temporary during this shape change.
        folded.width = arity;
        let (commitment, prover_data) = params
            .mmcs
            .commit_secret(vec![folded])
            .map_err(SecretFriCommitPhaseError::Mmcs)?;

        let (beta, witness) = transcript.commit_round(commitment.clone());
        commits.push(commitment);
        pow_witnesses.push(witness.unwrap_or(Val::ZERO));

        let matrices = params.mmcs.get_matrices_secret(&prover_data);
        let matrix = match matrices.as_slice() {
            [matrix] => matrix.as_view(),
            _ => return Err(SecretFriCommitPhaseError::InvalidMatrixShape),
        };
        folded = folding
            .fold_matrix_secret(beta, log_arity, matrix)
            .map_err(SecretFriCommitPhaseError::Fold)?;
        data.push(prover_data);

        if let Some(next_input) = inputs_iter
            .next_if(|input| input.width == 1 && input.values.len() == folded.values.len())
        {
            let beta_power = beta.exp_power_of_2(log_arity);
            folded
                .values
                .par_iter_mut()
                .zip(next_input.values.par_iter())
                .for_each(|(current, &incoming)| *current += beta_power * incoming);
        }
    }

    if inputs_iter.next().is_some() {
        return Err(SecretFriCommitPhaseError::InputScheduleMismatch);
    }

    let final_shift = u32::try_from(params.log_final_poly_len)
        .map_err(|_| SecretFriCommitPhaseError::FinalPolynomialLengthOverflow)?;
    let final_len = 1_usize
        .checked_shl(final_shift)
        .ok_or(SecretFriCommitPhaseError::FinalPolynomialLengthOverflow)?;
    if final_len > folded.values.len() {
        return Err(SecretFriCommitPhaseError::InputScheduleMismatch);
    }
    if let Some(discarded) = folded.values.get_mut(final_len..) {
        for value in discarded {
            zeroize::Zeroize::zeroize(value);
        }
    }
    folded.values.truncate(final_len);
    folded.width = 1;
    reverse_slice_index_bits(&mut folded.values);

    Ok(SecretCommitPhaseResult {
        commits,
        data,
        log_arities: retained_log_arities,
        pow_witnesses,
        final_evaluations: folded,
    })
}

/// Convert guarded terminal evaluations into the final polynomial disclosed by FRI.
///
/// The extension-field source and flattened base-field transform remain under
/// zeroizing owners. Only the coefficient vector that the protocol already
/// publishes in [`FriProof::final_poly`] crosses into ordinary ownership.
#[cfg(feature = "secret-memory")]
pub(crate) fn disclose_final_poly_secret<Val, Challenge>(
    final_evaluations: SecretProverData<RowMajorMatrix<Challenge>>,
) -> Result<Vec<Challenge>, SecretFriFinalPolynomialError>
where
    Val: TwoAdicField + zeroize::Zeroize,
    Challenge: ExtensionField<Val> + zeroize::Zeroize,
{
    if final_evaluations.width != 1 || final_evaluations.values.is_empty() {
        return Err(SecretFriFinalPolynomialError::InvalidShape);
    }
    let expected_coefficients = final_evaluations.values.len();
    let extension_dimension = Challenge::DIMENSION;
    if extension_dimension == 0 {
        return Err(SecretFriFinalPolynomialError::InvalidShape);
    }

    let mut flattened = final_evaluations
        .try_flatten_extension_to_base::<Val>()
        .map_err(SecretFriFinalPolynomialError::Matrix)?;
    Radix2Bowers
        .idft_batch_secret(&mut flattened)
        .map_err(SecretFriFinalPolynomialError::Dft)?;

    if flattened.width != extension_dimension
        || !flattened.values.len().is_multiple_of(extension_dimension)
    {
        return Err(SecretFriFinalPolynomialError::InvalidShape);
    }
    let coefficient_count = flattened.values.len() / extension_dimension;
    if coefficient_count != expected_coefficients {
        return Err(SecretFriFinalPolynomialError::InvalidShape);
    }

    let mut final_poly = Vec::new();
    final_poly
        .try_reserve_exact(coefficient_count)
        .map_err(|_| SecretFriFinalPolynomialError::AllocationFailed)?;
    for coefficients in flattened.values.chunks_exact(extension_dimension) {
        let coefficient = Challenge::from_basis_coefficients_slice(coefficients)
            .ok_or(SecretFriFinalPolynomialError::InvalidShape)?;
        final_poly.push(coefficient);
    }

    // The public coefficients have been copied out. Erase the transform before
    // returning so its backing allocation never reaches an ordinary owner.
    zeroize::Zeroize::zeroize(&mut flattened);
    Ok(final_poly)
}

/// Open every guarded FRI layer and consume its secret-derived prover data.
///
/// The returned siblings and authentication proof are protocol-public proof
/// material. Each retained MMCS layer is explicitly erased immediately after
/// its opening has been materialized. On an error, ownership of the current
/// and remaining layers stays in this function and their drop guards erase them.
#[cfg(feature = "secret-memory")]
pub(crate) fn answer_queries_secret<F, M>(
    config: &FriParameters<M>,
    log_arities: &[usize],
    folded_polynomial_commits: Vec<SecretProverData<M::SecretData<RowMajorMatrix<F>>>>,
    start_indices: &[usize],
) -> Result<Vec<CommitPhaseMultiStep<F, M>>, SecretFriQueryError<M::SecretError>>
where
    F: Field + zeroize::Zeroize,
    M: Mmcs<F> + SecretMmcs<F>,
{
    if start_indices.is_empty() {
        return Err(SecretFriQueryError::EmptyQuerySet);
    }
    if folded_polynomial_commits.len() != log_arities.len() {
        return Err(SecretFriQueryError::RoundCountMismatch);
    }

    let mut current_indices = Vec::new();
    current_indices
        .try_reserve_exact(start_indices.len())
        .map_err(|_| SecretFriQueryError::AllocationFailed)?;
    current_indices.extend_from_slice(start_indices);

    let mut openings = Vec::new();
    openings
        .try_reserve_exact(log_arities.len())
        .map_err(|_| SecretFriQueryError::AllocationFailed)?;

    for (&log_arity, mut commit) in log_arities.iter().zip(folded_polynomial_commits) {
        if log_arity == 0 {
            return Err(SecretFriQueryError::ZeroLogArity);
        }
        let arity_shift =
            u32::try_from(log_arity).map_err(|_| SecretFriQueryError::ArityOverflow)?;
        let arity = 1_usize
            .checked_shl(arity_shift)
            .ok_or(SecretFriQueryError::ArityOverflow)?;

        let mut group_indices = Vec::new();
        group_indices
            .try_reserve_exact(current_indices.len())
            .map_err(|_| SecretFriQueryError::AllocationFailed)?;
        group_indices.extend(current_indices.iter().map(|&index| index >> log_arity));

        let (opened_rows, opening_proof) = config
            .mmcs
            .open_multi_batch_secret(&group_indices, &commit)
            .map_err(SecretFriQueryError::Mmcs)?;
        zeroize::Zeroize::zeroize(&mut commit);
        if opened_rows.len() != current_indices.len() {
            return Err(SecretFriQueryError::InvalidOpeningShape);
        }

        let sibling_count = arity
            .checked_sub(1)
            .ok_or(SecretFriQueryError::ZeroLogArity)?;
        let mut sibling_values = Vec::new();
        sibling_values
            .try_reserve_exact(current_indices.len())
            .map_err(|_| SecretFriQueryError::AllocationFailed)?;
        for (&index, rows) in current_indices.iter().zip(opened_rows) {
            let mut rows = rows.into_iter();
            let opened_row = rows
                .next()
                .ok_or(SecretFriQueryError::InvalidOpeningShape)?;
            if rows.next().is_some() || opened_row.len() != arity {
                return Err(SecretFriQueryError::InvalidOpeningShape);
            }

            let index_in_group = index % arity;
            let mut siblings = Vec::new();
            siblings
                .try_reserve_exact(sibling_count)
                .map_err(|_| SecretFriQueryError::AllocationFailed)?;
            siblings.extend(
                opened_row
                    .into_iter()
                    .enumerate()
                    .filter_map(|(position, value)| (position != index_in_group).then_some(value)),
            );
            if siblings.len() != sibling_count {
                return Err(SecretFriQueryError::InvalidOpeningShape);
            }
            sibling_values.push(siblings);
        }

        current_indices = group_indices;
        openings.push(CommitPhaseMultiStep {
            sibling_values,
            opening_proof,
        });
    }

    Ok(openings)
}

/// Given the query indices, produce a proof that every chain of folds is correct.
/// This is the prover's complement to the verifier's fold-and-verify pass.
///
/// In addition to the output of this function, the prover must also supply the verifier with the input values
/// (with associated opening proofs). These are produced by the `open_inputs` function passed into `prove_fri`.
///
/// For each round, this returns the sibling values of every query (all values in the group
/// except the queried one) along with one shared opening proof covering all queries.
/// The verifier can then reconstruct each full group, verify the round's openings together,
/// and fold to get the value at the parent index.
///
/// With variable arity, the index shifts by `log_arities[i]` each round instead of always by 1
/// (i.e. when arity is fixed to 2).
///
/// Arguments:
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `log_arities`: The log2 of the arity used for each round.
/// - `folded_polynomial_commits`: A slice of commitments to the intermediate stage polynomials.
/// - `start_indices`: The opening indices for the unfolded polynomial, one per query.
#[inline]
fn answer_queries<F, M>(
    config: &FriParameters<M>,
    log_arities: &[usize],
    folded_polynomial_commits: &[M::ProverData<RowMajorMatrix<F>>],
    start_indices: &[usize],
) -> Vec<CommitPhaseMultiStep<F, M>>
where
    F: Field,
    M: Mmcs<F>,
{
    let mut current_indices = start_indices.to_vec();

    folded_polynomial_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            let log_arity = log_arities[i];
            let arity = 1 << log_arity;

            // Index of each query's group (row in the committed matrix).
            let group_indices: Vec<usize> = current_indices
                .iter()
                .map(|&index| index >> log_arity)
                .collect();

            // One shared proof that all queried groups are correct.
            let (opened_rows, opening_proof) = config.mmcs.open_multi_batch(&group_indices, commit);

            // For each query, keep all siblings (exclude the queried element).
            let sibling_values: Vec<Vec<F>> = izip!(&current_indices, opened_rows)
                .map(|(&index, mut rows)| {
                    // Each group opening should contain just the values in this group.
                    assert_eq!(rows.len(), 1);
                    let opened_row = rows.pop().unwrap();
                    assert_eq!(
                        opened_row.len(),
                        arity,
                        "Committed data should have arity {} elements",
                        arity
                    );
                    let index_in_group = index % arity;
                    opened_row
                        .into_iter()
                        .enumerate()
                        .filter(|(j, _)| *j != index_in_group)
                        .map(|(_, v)| v)
                        .collect()
                })
                .collect();

            current_indices = group_indices;

            // Add the siblings and the shared proof to the vector.
            CommitPhaseMultiStep {
                sibling_values,
                opening_proof,
            }
        })
        .collect()
}

/// Given the query indices, produce one batch multi-opening proof for each collection
/// of matrices combined into a single mmcs commitment.
///
/// In cases where the maximum height of a batch of matrices is smaller than the
/// global max height, shift the indices down to compensate.
///
/// Arguments:
/// - `log_global_max_height`: The log of the maximum height of the input matrices.
/// - `indices`: The indices to open the matrices at, one per query.
/// - `prover_data_with_opening_points`: A list of pairs of a batch commitment to a collection
///   of matrices and a list of points to open those matrices at.
/// - `mmcs`: The mixed matrix commitment scheme used to produce the batch commitments.
#[inline]
fn open_inputs<Val, Challenge, InputMmcs>(
    log_global_max_height: usize,
    indices: &[usize],
    prover_data_with_opening_points: &[ProverDataWithOpeningPoints<
        '_,
        Challenge,
        InputMmcs::ProverData<RowMajorMatrix<Val>>,
    >],
    mmcs: &InputMmcs,
) -> Vec<BatchMultiOpening<Val, InputMmcs>>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
{
    // This gives the verifier access to evaluations `f(x)` from which it can compute
    // `(f(zeta) - f(x))/(zeta - x)` and then combine them together and roll into FRI
    // as appropriate.
    prover_data_with_opening_points
        .iter()
        .map(
            |OpeningRequest {
                 prover_data: data, ..
             }| {
                let log_max_height = log2_strict_usize(mmcs.get_max_height(data));
                let bits_reduced = log_global_max_height - log_max_height;
                // If a matrix is smaller than global max height, we roll it into
                // fri in a later round.
                let reduced_indices: Vec<usize> =
                    indices.iter().map(|&index| index >> bits_reduced).collect();
                let (opened_values, opening_proof) = mmcs.open_multi_batch(&reduced_indices, data);
                BatchMultiOpening {
                    opened_values,
                    opening_proof,
                }
            },
        )
        .collect()
}

#[cfg(all(test, feature = "secret-memory"))]
mod secret_tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use core::cell::Cell;
    use core::marker::PhantomData;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{DuplexChallenger, FieldChallenger};
    use p3_commit::{ExtensionMmcs, ExtensionSecretMmcsError, Mmcs, SecretMmcs, SecretProverData};
    use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::{MerkleTreeError, MerkleTreeMmcs};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{
        SecretFriCommitPhaseError, SecretFriDriverError, SecretFriFinalPolynomialError,
        SecretFriQueryError, answer_queries, answer_queries_secret, commit_phase,
        commit_phase_secret, disclose_final_poly_secret, prove_fri_secret_with_input_opener,
        prove_fri_with_input_opener,
    };
    use crate::{
        BatchMultiOpening, FriParameters, FriProverError, FriShape, ProverTranscript,
        TwoAdicFriFolding, TwoAdicFriFoldingForMmcs,
    };

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type Permutation = Poseidon2BabyBear<16>;
    type Hash = PaddingFreeSponge<Permutation, 16, 8, 8>;
    type Compress = TruncatedPermutation<Permutation, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, Hash, Compress, 2, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Challenger = DuplexChallenger<Val, Permutation, 16, 8>;

    fn make_parameters(permutation: &Permutation) -> FriParameters<ChallengeMmcs> {
        make_parameters_with_final_log(permutation, 0)
    }

    fn make_parameters_with_final_log(
        permutation: &Permutation,
        log_final_poly_len: usize,
    ) -> FriParameters<ChallengeMmcs> {
        let hash = Hash::new(permutation.clone());
        let compress = Compress::new(permutation.clone());
        let value_mmcs = ValMmcs::new(hash, compress, 0);
        FriParameters {
            log_blowup: 1,
            log_final_poly_len,
            max_log_arity: 1,
            num_queries: 2,
            batch_proof_of_work_bits: 0,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: ChallengeMmcs::new(value_mmcs),
        }
    }

    fn test_inputs() -> (Vec<Challenge>, Vec<Challenge>) {
        let large = (0..16)
            .map(|value| Challenge::from_usize(value + 1))
            .collect::<Vec<_>>();
        let small = (0..8)
            .map(|value| Challenge::from_usize(value + 101))
            .collect::<Vec<_>>();
        (large, small)
    }

    #[test]
    fn guarded_driver_matches_ordinary_driver_output() {
        let mut rng = SmallRng::seed_from_u64(0x4652_4944_5249_5645);
        let permutation = Permutation::new_from_rng_128(&mut rng);
        let parameters = make_parameters_with_final_log(&permutation, 1);
        let folding: TwoAdicFriFoldingForMmcs<Val, ValMmcs> = TwoAdicFriFolding(PhantomData);
        let (large, small) = test_inputs();

        let mut ordinary_challenger = Challenger::new(permutation.clone());
        let ordinary = prove_fri_with_input_opener::<
            _,
            Val,
            Challenge,
            ValMmcs,
            ChallengeMmcs,
            _,
            FriProverError,
            _,
        >(
            &folding,
            &parameters,
            vec![large.clone(), small.clone()],
            &mut ordinary_challenger,
            4,
            Val::ZERO,
            |_| Ok(Vec::<BatchMultiOpening<Val, ValMmcs>>::new()),
        )
        .expect("valid ordinary FRI proof must succeed");

        let mut guarded_challenger = Challenger::new(permutation);
        let guarded = prove_fri_secret_with_input_opener(
            &folding,
            &parameters,
            vec![
                SecretProverData::new(RowMajorMatrix::new_col(large)),
                SecretProverData::new(RowMajorMatrix::new_col(small)),
            ],
            &mut guarded_challenger,
            4,
            Val::ZERO,
            |_| Ok::<_, FriProverError>(Vec::<BatchMultiOpening<Val, ValMmcs>>::new()),
        )
        .expect("valid guarded FRI proof must succeed");

        assert_eq!(guarded.batch_pow_witness, ordinary.batch_pow_witness);
        assert_eq!(guarded.commit_phase_commits, ordinary.commit_phase_commits);
        assert_eq!(guarded.commit_pow_witnesses, ordinary.commit_pow_witnesses);
        assert!(guarded.input_openings.is_empty());
        assert_eq!(guarded.final_poly, ordinary.final_poly);
        assert_eq!(guarded.query_pow_witness, ordinary.query_pow_witness);
        assert_eq!(
            guarded.commit_phase_openings.len(),
            ordinary.commit_phase_openings.len()
        );
        for (guarded_step, ordinary_step) in guarded
            .commit_phase_openings
            .iter()
            .zip(&ordinary.commit_phase_openings)
        {
            assert_eq!(guarded_step.sibling_values, ordinary_step.sibling_values);
            assert_eq!(
                guarded_step.opening_proof.sibling_hashes,
                ordinary_step.opening_proof.sibling_hashes
            );
        }
        assert_eq!(
            guarded_challenger.sample_algebra_element::<Challenge>(),
            ordinary_challenger.sample_algebra_element::<Challenge>()
        );
    }

    #[test]
    fn guarded_driver_rejects_invalid_shape_before_transcript_and_callback() {
        let mut rng = SmallRng::seed_from_u64(0x4652_4950_5245_464c);
        let permutation = Permutation::new_from_rng_128(&mut rng);
        let parameters = make_parameters_with_final_log(&permutation, 1);
        let folding: TwoAdicFriFoldingForMmcs<Val, ValMmcs> = TwoAdicFriFolding(PhantomData);
        let (large, small) = test_inputs();
        let mut challenger = Challenger::new(permutation);
        let mut untouched = challenger.clone();
        let callback_called = Cell::new(false);

        let result = prove_fri_secret_with_input_opener(
            &folding,
            &parameters,
            vec![
                SecretProverData::new(RowMajorMatrix::new_col(small)),
                SecretProverData::new(RowMajorMatrix::new_col(large)),
            ],
            &mut challenger,
            3,
            Val::ZERO,
            |_| {
                callback_called.set(true);
                Ok::<_, FriProverError>(Vec::<BatchMultiOpening<Val, ValMmcs>>::new())
            },
        );

        assert!(matches!(
            result,
            Err(SecretFriDriverError::InputHeightOrder)
        ));
        assert!(!callback_called.get());
        assert_eq!(
            challenger.sample_algebra_element::<Challenge>(),
            untouched.sample_algebra_element::<Challenge>()
        );
    }

    #[derive(Debug, PartialEq, Eq)]
    enum TestInputOpeningError {
        Rejected,
    }

    #[test]
    fn guarded_driver_defers_input_opening_until_queries_are_fixed() {
        let mut rng = SmallRng::seed_from_u64(0x4652_494f_5045_4e52);
        let permutation = Permutation::new_from_rng_128(&mut rng);
        let parameters = make_parameters_with_final_log(&permutation, 1);
        let folding: TwoAdicFriFoldingForMmcs<Val, ValMmcs> = TwoAdicFriFolding(PhantomData);
        let (large, small) = test_inputs();
        let mut successful_challenger = Challenger::new(permutation.clone());
        let successful = prove_fri_secret_with_input_opener(
            &folding,
            &parameters,
            vec![
                SecretProverData::new(RowMajorMatrix::new_col(large.clone())),
                SecretProverData::new(RowMajorMatrix::new_col(small.clone())),
            ],
            &mut successful_challenger,
            4,
            Val::ZERO,
            |indices| {
                assert_eq!(indices.len(), parameters.num_queries);
                Ok::<_, TestInputOpeningError>(Vec::<BatchMultiOpening<Val, ValMmcs>>::new())
            },
        );
        assert!(successful.is_ok());

        let callback_called = Cell::new(false);
        let mut rejected_challenger = Challenger::new(permutation);
        let rejected = prove_fri_secret_with_input_opener(
            &folding,
            &parameters,
            vec![
                SecretProverData::new(RowMajorMatrix::new_col(large)),
                SecretProverData::new(RowMajorMatrix::new_col(small)),
            ],
            &mut rejected_challenger,
            4,
            Val::ZERO,
            |indices| {
                callback_called.set(true);
                assert_eq!(indices.len(), parameters.num_queries);
                Err(TestInputOpeningError::Rejected)
            },
        );

        assert!(matches!(
            rejected,
            Err(SecretFriDriverError::InputOpening(
                TestInputOpeningError::Rejected
            ))
        ));
        assert!(callback_called.get());
        assert_eq!(
            rejected_challenger.sample_algebra_element::<Challenge>(),
            successful_challenger.sample_algebra_element::<Challenge>()
        );
    }

    #[test]
    fn guarded_commit_phase_matches_ordinary_transcript_and_erases_layers() {
        let mut rng = SmallRng::seed_from_u64(0x4652_4943_4f4d_4d49);
        let permutation = Permutation::new_from_rng_128(&mut rng);
        let parameters = make_parameters(&permutation);
        let schedule = vec![1, 1, 1, 1];
        let shape = FriShape::with_schedule(&parameters, schedule.clone(), 4);
        let folding: TwoAdicFriFoldingForMmcs<Val, ValMmcs> = TwoAdicFriFolding(PhantomData);

        let (large, small) = test_inputs();

        let mut ordinary_challenger = Challenger::new(permutation.clone());
        let mut ordinary_transcript =
            ProverTranscript::new(&mut ordinary_challenger, shape.clone());
        let ordinary = commit_phase(
            &folding,
            &parameters,
            vec![large.clone(), small.clone()],
            &schedule,
            &mut ordinary_transcript,
        );

        let mut guarded_challenger = Challenger::new(permutation);
        let mut guarded_transcript = ProverTranscript::new(&mut guarded_challenger, shape);
        let mut guarded = commit_phase_secret(
            &folding,
            &parameters,
            vec![
                SecretProverData::new(RowMajorMatrix::new_col(large)),
                SecretProverData::new(RowMajorMatrix::new_col(small)),
            ],
            &schedule,
            &mut guarded_transcript,
        )
        .expect("valid guarded FRI commit phase must succeed");

        assert_eq!(guarded.commits, ordinary.commits);
        assert_eq!(guarded.log_arities, ordinary.log_arities);
        assert_eq!(guarded.pow_witnesses, ordinary.pow_witnesses);
        assert_eq!(guarded.data.len(), ordinary.data.len());
        for ((commitment, ordinary_data), guarded_data) in ordinary
            .commits
            .iter()
            .zip(&ordinary.data)
            .zip(&guarded.data)
        {
            let ordinary_matrices = parameters.mmcs.get_matrices(ordinary_data);
            let guarded_matrices = parameters.mmcs.get_matrices_secret(guarded_data);
            assert_eq!(ordinary_matrices.len(), 1);
            assert_eq!(guarded_matrices.len(), 1);
            assert_eq!(guarded_matrices[0].width, ordinary_matrices[0].width);
            assert_eq!(guarded_matrices[0].values, ordinary_matrices[0].values);

            let indices = [0];
            let ordinary_opening = parameters.mmcs.open_multi_batch(&indices, ordinary_data);
            let guarded_opening = parameters
                .mmcs
                .open_multi_batch_secret(&indices, guarded_data)
                .expect("guarded extension-field opening must succeed");
            assert_eq!(guarded_opening.0, ordinary_opening.0);
            let dimensions = [p3_matrix::Dimensions {
                width: guarded_matrices[0].width,
                height: guarded_matrices[0].height(),
            }];
            parameters
                .mmcs
                .verify_multi_batch(
                    commitment,
                    &dimensions,
                    &indices,
                    &guarded_opening.0,
                    &guarded_opening.1,
                )
                .expect("guarded extension-field opening must verify");
        }

        let guarded_final_poly = Radix2DFTSmallBatch::<Val>::default()
            .idft_algebra(guarded.final_evaluations.values.clone());
        assert_eq!(guarded_final_poly, ordinary.final_poly);

        let ordinary_queries = ordinary_transcript.query_phase(&ordinary.final_poly).0;
        let guarded_queries = guarded_transcript.query_phase(&ordinary.final_poly).0;
        assert_eq!(guarded_queries, ordinary_queries);
        ordinary_transcript.finish();
        guarded_transcript.finish();

        for guarded_data in &mut guarded.data {
            zeroize::Zeroize::zeroize(guarded_data);
            let erased_matrices = parameters.mmcs.get_matrices_secret(guarded_data);
            assert_eq!(erased_matrices.len(), 1);
            assert_eq!(erased_matrices[0].width, 0);
            assert!(
                erased_matrices[0]
                    .values
                    .iter()
                    .all(|value| *value == Challenge::ZERO)
            );
        }
        zeroize::Zeroize::zeroize(&mut guarded.final_evaluations);
        assert_eq!(guarded.final_evaluations.width, 0);
        assert!(
            guarded
                .final_evaluations
                .values
                .iter()
                .all(|value| *value == Challenge::ZERO)
        );
    }

    #[test]
    fn guarded_terminal_and_query_components_match_ordinary_path() {
        let mut rng = SmallRng::seed_from_u64(0x4652_4954_4552_4d51);
        let permutation = Permutation::new_from_rng_128(&mut rng);
        let parameters = make_parameters_with_final_log(&permutation, 2);
        let schedule = vec![1, 1];
        let shape = FriShape::with_schedule(&parameters, schedule.clone(), 4);
        let folding: TwoAdicFriFoldingForMmcs<Val, ValMmcs> = TwoAdicFriFolding(PhantomData);
        let (large, small) = test_inputs();

        let mut ordinary_challenger = Challenger::new(permutation.clone());
        let mut ordinary_transcript =
            ProverTranscript::new(&mut ordinary_challenger, shape.clone());
        let ordinary = commit_phase(
            &folding,
            &parameters,
            vec![large.clone(), small.clone()],
            &schedule,
            &mut ordinary_transcript,
        );

        let mut guarded_challenger = Challenger::new(permutation);
        let mut guarded_transcript = ProverTranscript::new(&mut guarded_challenger, shape);
        let guarded = commit_phase_secret(
            &folding,
            &parameters,
            vec![
                SecretProverData::new(RowMajorMatrix::new_col(large)),
                SecretProverData::new(RowMajorMatrix::new_col(small)),
            ],
            &schedule,
            &mut guarded_transcript,
        )
        .expect("valid guarded FRI commit phase must succeed");

        let guarded_final_poly =
            disclose_final_poly_secret::<Val, Challenge>(guarded.final_evaluations)
                .expect("guarded final polynomial conversion must succeed");
        assert_eq!(guarded_final_poly, ordinary.final_poly);

        let ordinary_queries = ordinary_transcript.query_phase(&ordinary.final_poly).0;
        let guarded_queries = guarded_transcript.query_phase(&guarded_final_poly).0;
        assert_eq!(guarded_queries, ordinary_queries);
        ordinary_transcript.finish();
        guarded_transcript.finish();

        let ordinary_openings = answer_queries(
            &parameters,
            &ordinary.log_arities,
            &ordinary.data,
            &ordinary_queries,
        );
        let guarded_openings = answer_queries_secret(
            &parameters,
            &guarded.log_arities,
            guarded.data,
            &guarded_queries,
        )
        .expect("guarded FRI query answering must succeed");
        assert_eq!(guarded_openings.len(), ordinary_openings.len());
        for (guarded_step, ordinary_step) in guarded_openings.iter().zip(&ordinary_openings) {
            assert_eq!(guarded_step.sibling_values, ordinary_step.sibling_values);
            assert_eq!(
                guarded_step.opening_proof.sibling_hashes,
                ordinary_step.opening_proof.sibling_hashes
            );
        }
    }

    #[test]
    fn guarded_query_components_match_high_arity_ordinary_path() {
        let mut rng = SmallRng::seed_from_u64(0x4652_4948_4941_5249);
        let permutation = Permutation::new_from_rng_128(&mut rng);
        let mut parameters = make_parameters_with_final_log(&permutation, 2);
        parameters.max_log_arity = 2;
        let schedule = vec![2];
        let shape = FriShape::with_schedule(&parameters, schedule.clone(), 4);
        let folding: TwoAdicFriFoldingForMmcs<Val, ValMmcs> = TwoAdicFriFolding(PhantomData);
        let (large, _) = test_inputs();

        let mut ordinary_challenger = Challenger::new(permutation.clone());
        let mut ordinary_transcript =
            ProverTranscript::new(&mut ordinary_challenger, shape.clone());
        let ordinary = commit_phase(
            &folding,
            &parameters,
            vec![large.clone()],
            &schedule,
            &mut ordinary_transcript,
        );

        let mut guarded_challenger = Challenger::new(permutation);
        let mut guarded_transcript = ProverTranscript::new(&mut guarded_challenger, shape);
        let guarded = commit_phase_secret(
            &folding,
            &parameters,
            vec![SecretProverData::new(RowMajorMatrix::new_col(large))],
            &schedule,
            &mut guarded_transcript,
        )
        .expect("valid high-arity guarded FRI commit phase must succeed");
        let guarded_final_poly =
            disclose_final_poly_secret::<Val, Challenge>(guarded.final_evaluations)
                .expect("high-arity guarded final polynomial conversion must succeed");
        assert_eq!(guarded_final_poly, ordinary.final_poly);

        let ordinary_queries = ordinary_transcript.query_phase(&ordinary.final_poly).0;
        let guarded_queries = guarded_transcript.query_phase(&guarded_final_poly).0;
        assert_eq!(guarded_queries, ordinary_queries);
        ordinary_transcript.finish();
        guarded_transcript.finish();

        let ordinary_openings = answer_queries(
            &parameters,
            &ordinary.log_arities,
            &ordinary.data,
            &ordinary_queries,
        );
        let guarded_openings = answer_queries_secret(
            &parameters,
            &guarded.log_arities,
            guarded.data,
            &guarded_queries,
        )
        .expect("high-arity guarded FRI query answering must succeed");
        assert_eq!(guarded_openings.len(), 1);
        assert_eq!(guarded_openings[0].sibling_values[0].len(), 3);
        assert_eq!(
            guarded_openings[0].sibling_values,
            ordinary_openings[0].sibling_values
        );
        assert_eq!(
            guarded_openings[0].opening_proof.sibling_hashes,
            ordinary_openings[0].opening_proof.sibling_hashes
        );
    }

    #[test]
    fn guarded_terminal_conversion_rejects_invalid_shape_and_height() {
        let invalid_shape = SecretProverData::new(RowMajorMatrix::new(vec![Challenge::ONE; 4], 2));
        let shape_result = disclose_final_poly_secret::<Val, Challenge>(invalid_shape);
        assert_eq!(
            shape_result,
            Err(SecretFriFinalPolynomialError::InvalidShape)
        );

        let invalid_height =
            SecretProverData::new(RowMajorMatrix::new_col(vec![Challenge::ONE; 3]));
        let height_result = disclose_final_poly_secret::<Val, Challenge>(invalid_height);
        assert_eq!(
            height_result,
            Err(SecretFriFinalPolynomialError::Dft(
                p3_dft::SecretDftError::InvalidHeight
            ))
        );
    }

    #[test]
    fn guarded_query_answering_rejects_round_mismatch_and_out_of_bounds_index() {
        let mut rng = SmallRng::seed_from_u64(0x4652_4951_5545_5252);
        let permutation = Permutation::new_from_rng_128(&mut rng);
        let parameters = make_parameters_with_final_log(&permutation, 2);

        let empty =
            answer_queries_secret::<Challenge, ChallengeMmcs>(&parameters, &[], Vec::new(), &[]);
        assert!(matches!(empty, Err(SecretFriQueryError::EmptyQuerySet)));

        let mismatch =
            answer_queries_secret::<Challenge, ChallengeMmcs>(&parameters, &[1], Vec::new(), &[0]);
        assert!(matches!(
            mismatch,
            Err(SecretFriQueryError::RoundCountMismatch)
        ));

        let schedule = vec![1, 1];
        let shape = FriShape::with_schedule(&parameters, schedule.clone(), 4);
        let folding: TwoAdicFriFoldingForMmcs<Val, ValMmcs> = TwoAdicFriFolding(PhantomData);
        let (large, small) = test_inputs();
        let mut challenger = Challenger::new(permutation);
        let mut transcript = ProverTranscript::new(&mut challenger, shape);
        let guarded = commit_phase_secret(
            &folding,
            &parameters,
            vec![
                SecretProverData::new(RowMajorMatrix::new_col(large)),
                SecretProverData::new(RowMajorMatrix::new_col(small)),
            ],
            &schedule,
            &mut transcript,
        )
        .expect("valid guarded FRI commit phase must succeed");
        let final_poly = disclose_final_poly_secret::<Val, Challenge>(guarded.final_evaluations)
            .expect("guarded final polynomial conversion must succeed");
        let _ = transcript.query_phase(&final_poly);
        transcript.finish();

        let out_of_bounds =
            answer_queries_secret(&parameters, &guarded.log_arities, guarded.data, &[16]);
        assert!(matches!(
            out_of_bounds,
            Err(SecretFriQueryError::Mmcs(ExtensionSecretMmcsError::Inner(
                MerkleTreeError::IndexOutOfBounds {
                    max_height: 8,
                    index: 8
                }
            )))
        ));
    }

    #[test]
    fn guarded_commit_phase_rejects_empty_inputs_with_typed_error() {
        let mut rng = SmallRng::seed_from_u64(0x4652_4945_4d50_5459);
        let permutation = Permutation::new_from_rng_128(&mut rng);
        let parameters = make_parameters(&permutation);
        let schedule = Vec::new();
        let shape = FriShape::with_schedule(&parameters, schedule.clone(), 1);
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);
        let mut challenger = Challenger::new(permutation);
        let mut transcript = ProverTranscript::new(&mut challenger, shape);

        let result = commit_phase_secret(
            &folding,
            &parameters,
            Vec::new(),
            &schedule,
            &mut transcript,
        );
        let _ = transcript.query_phase(&[Challenge::ZERO]);
        transcript.finish();
        assert!(matches!(
            result,
            Err(SecretFriCommitPhaseError::EmptyInputs)
        ));
    }

    #[test]
    fn guarded_commit_phase_rejects_malformed_matrix_with_typed_error() {
        let mut rng = SmallRng::seed_from_u64(0x4652_494d_414c_464d);
        let permutation = Permutation::new_from_rng_128(&mut rng);
        let parameters = make_parameters(&permutation);
        let schedule = Vec::new();
        let shape = FriShape::with_schedule(&parameters, schedule.clone(), 1);
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);
        let mut challenger = Challenger::new(permutation);
        let mut transcript = ProverTranscript::new(&mut challenger, shape);
        let malformed = RowMajorMatrix::new(vec![Challenge::ONE; 4], 2);

        let result = commit_phase_secret(
            &folding,
            &parameters,
            vec![SecretProverData::new(malformed)],
            &schedule,
            &mut transcript,
        );
        let _ = transcript.query_phase(&[Challenge::ZERO]);
        transcript.finish();
        assert!(matches!(
            result,
            Err(SecretFriCommitPhaseError::InvalidMatrixShape)
        ));
    }
}
