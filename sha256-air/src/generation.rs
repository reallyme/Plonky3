// Modified by ReallyMe Technologies Limited, 2026.
// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Trace generation for the SHA-256 compression AIR.

use alloc::vec::Vec;
use core::array;
#[cfg(feature = "secret-memory")]
use core::borrow::BorrowMut;

use p3_air::utils::u32_to_bits_le;
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;
#[cfg(feature = "secret-memory")]
use p3_matrix::secret::ZeroizingOwner;
use p3_maybe_rayon::prelude::*;
#[cfg(feature = "secret-memory")]
use thiserror::Error;
use tracing::instrument;
#[cfg(feature = "secret-memory")]
use zeroize::Zeroize;
use zeroize::Zeroizing;

use crate::columns::{CHAIN_LEN, NUM_SHA256_COLS, Sha256Cols, Sha256RoundCols};
use crate::constants::{
    BLOCK_WORDS, NUM_COMPRESSION_ROUNDS, SCHEDULE_EXTENSIONS, SHA256_K, STATE_WORDS, u32_to_limbs,
};

/// Number of 32-bit words supplied per row.
///
/// Layout:
/// - Indices 0..16 hold the message block.
/// - Indices 16..24 hold the input chaining state.
pub const INPUT_WORDS: usize = BLOCK_WORDS + STATE_WORDS;

/// Stable failures from guarded SHA-256 trace construction.
#[cfg(feature = "secret-memory")]
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum SecretSha256TraceError {
    /// At least one compression input is required.
    #[error("guarded SHA-256 trace input is empty")]
    EmptyInput,
    /// The compression count must be a power of two.
    #[error("guarded SHA-256 trace input count is not a power of two")]
    NonPowerOfTwoInput,
    /// The trace or reserved-capacity length cannot be represented.
    #[error("guarded SHA-256 trace dimensions overflow")]
    DimensionOverflow,
    /// The trace allocation could not be reserved.
    #[error("guarded SHA-256 trace allocation failed")]
    AllocationFailed,
    /// The allocated row geometry does not match the SHA-256 column layout.
    #[error("guarded SHA-256 trace row geometry is invalid")]
    InvalidShape,
}

/// Build a SHA-256 compression trace without releasing secret-bearing input
/// or output allocations from zeroizing ownership.
///
/// The output allocation is placed under its erasure owner while it contains
/// only public zeroes. Witness-derived field values are written only after
/// that transition. The input owner is explicitly erased after its last use;
/// it is also erased on every typed failure and supported panic unwind.
#[cfg(feature = "secret-memory")]
#[instrument(name = "generate guarded SHA-256 trace", skip_all)]
pub fn generate_trace_rows_secret<F>(
    mut inputs: ZeroizingOwner<Vec<[u32; INPUT_WORDS]>>,
    extra_capacity_bits: usize,
) -> Result<ZeroizingOwner<RowMajorMatrix<F>>, SecretSha256TraceError>
where
    F: PrimeField64 + Zeroize,
{
    let num_rows = inputs.len();
    if num_rows == 0 {
        return Err(SecretSha256TraceError::EmptyInput);
    }
    if !num_rows.is_power_of_two() {
        return Err(SecretSha256TraceError::NonPowerOfTwoInput);
    }
    let trace_length = num_rows
        .checked_mul(NUM_SHA256_COLS)
        .ok_or(SecretSha256TraceError::DimensionOverflow)?;
    let shift = u32::try_from(extra_capacity_bits)
        .map_err(|_| SecretSha256TraceError::DimensionOverflow)?;
    let reserved_length = trace_length
        .checked_shl(shift)
        .ok_or(SecretSha256TraceError::DimensionOverflow)?;

    let mut values = Vec::new();
    values
        .try_reserve_exact(reserved_length)
        .map_err(|_| SecretSha256TraceError::AllocationFailed)?;
    values.resize(trace_length, F::ZERO);
    let mut trace = ZeroizingOwner::new(RowMajorMatrix::new(values, NUM_SHA256_COLS));

    for (row_index, input) in inputs.iter().enumerate() {
        let row_start = row_index
            .checked_mul(NUM_SHA256_COLS)
            .ok_or(SecretSha256TraceError::DimensionOverflow)?;
        let row_end = row_start
            .checked_add(NUM_SHA256_COLS)
            .ok_or(SecretSha256TraceError::DimensionOverflow)?;
        let row_values = trace
            .values
            .get_mut(row_start..row_end)
            .ok_or(SecretSha256TraceError::InvalidShape)?;
        let row: &mut Sha256Cols<F> = row_values.borrow_mut();
        generate_trace_row_for_compression(row, input);
    }
    inputs.zeroize();
    Ok(trace)
}

/// Populate a trace matrix where each row encodes one compression.
///
/// # Arguments
///
/// - `inputs`: one 24-word array per row. Length must be a power of two.
/// - `extra_capacity_bits`: log2 blowup factor reserved for the LDE.
///
/// # Returns
///
/// A row-major matrix of field elements ready to be handed to the prover.
///
/// # Panics
///
/// Panics if `inputs.len()` is not a power of two.
#[instrument(name = "generate SHA-256 trace", skip_all)]
pub fn generate_trace_rows<F: PrimeField64>(
    inputs: Vec<[u32; INPUT_WORDS]>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    let num_rows = inputs.len();
    // Power-of-two count is required for the LDE domain to align.
    assert!(
        num_rows.is_power_of_two(),
        "Callers must pad SHA-256 inputs to a power-of-two count"
    );

    // Reserve backing storage for the full blowup plus the logical trace.
    let trace_length = num_rows * NUM_SHA256_COLS;
    let mut long_trace = F::zero_vec(trace_length << extra_capacity_bits);
    // Shrink the logical view back down to the committed row count.
    long_trace.truncate(trace_length);

    // Wrap the flat buffer as a row-major matrix and re-interpret it as an
    // array of typed rows for easy field access.
    let mut trace = RowMajorMatrix::new(long_trace, NUM_SHA256_COLS);
    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<Sha256Cols<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), num_rows);

    // Fill each row in parallel - rows are fully independent.
    rows.par_iter_mut()
        .zip(inputs)
        .for_each(|(row, input)| generate_trace_row_for_compression(row, &input));

    trace
}

/// Populate a single row by executing the SHA-256 compression.
///
/// # Algorithm
///
/// 1. Split the 24-word input into a block and an input chaining state.
/// 2. Expand the message schedule: 48 new `W` words plus their per-step
///    sigma intermediates.
/// 3. Run 64 compression rounds, storing each round's packed intermediates
///    and extending the two working-variable chains.
/// 4. Write `H' = H + state` into the output row.
fn generate_trace_row_for_compression<F: PrimeField64>(
    row: &mut Sha256Cols<F>,
    input: &[u32; INPUT_WORDS],
) {
    // Step 1: unpack the input into a block and a chaining state.
    let block = Zeroizing::new(array::from_fn::<_, BLOCK_WORDS, _>(|i| input[i]));
    let h_in = Zeroizing::new(array::from_fn::<_, STATE_WORDS, _>(|i| {
        input[BLOCK_WORDS + i]
    }));

    // Commit the input chaining state in packed form - one entry per H word.
    row.h_in = array::from_fn(|index| u32_to_limbs(h_in[index]).map(F::from_u16));

    // Step 2: message-schedule expansion.
    //
    // Keep the running u32 values in a scratch array so sigma computations
    // stay simple; commit bits to the row as we go.
    let mut w_words = Zeroizing::new([0u32; NUM_COMPRESSION_ROUNDS]);
    for (dst, src) in w_words.iter_mut().zip(block.iter()) {
        *dst = *src;
    }
    // Commit the 16 block words as bits - pure witness, no auxiliary math.
    for (w_bits, &word) in row.w[..BLOCK_WORDS].iter_mut().zip(block.iter()) {
        *w_bits = u32_to_bits_le(word);
    }

    for t in BLOCK_WORDS..NUM_COMPRESSION_ROUNDS {
        // Auxiliary column index (0..48) for the packed columns.
        let i = t - BLOCK_WORDS;

        // Fetch the four schedule operands.
        let w_m15 = w_words[t - 15];
        let w_m2 = w_words[t - 2];
        let w_m7 = w_words[t - 7];
        let w_m16 = w_words[t - 16];

        // small_sigma0(x) = ROTR_7(x) XOR ROTR_18(x) XOR SHR_3(x).
        let s0 = w_m15.rotate_right(7) ^ w_m15.rotate_right(18) ^ (w_m15 >> 3);

        // small_sigma1(x) = ROTR_17(x) XOR ROTR_19(x) XOR SHR_10(x).
        let s1 = w_m2.rotate_right(17) ^ w_m2.rotate_right(19) ^ (w_m2 >> 10);

        // Partial sum - splits the four-term recurrence into two add3-sized
        // pieces.
        let tmp = s1.wrapping_add(w_m7);

        // Final schedule word for this index.
        let w_t = tmp.wrapping_add(s0).wrapping_add(w_m16);
        w_words[t] = w_t;

        // Commit every auxiliary value for step `t`.
        //
        // `W[t]`: bit form only — the AIR re-packs from bits when an add
        // constraint needs the packed view.
        row.w[t] = u32_to_bits_le(w_t);
        row.sched_sigma0[i] = u32_to_limbs(s0).map(F::from_u16);
        row.sched_sigma1[i] = u32_to_limbs(s1).map(F::from_u16);
        row.sched_tmp[i] = u32_to_limbs(tmp).map(F::from_u16);
    }
    // Sanity check for the extension count.
    debug_assert_eq!(SCHEDULE_EXTENSIONS, NUM_COMPRESSION_ROUNDS - BLOCK_WORDS);

    // Step 3: compression loop.
    //
    // Chain prefix: indices 0..4 hold (d, c, b, a) of the initial state -
    // that is H[3..=0] for the `a` chain and H[7..=4] for the `e` chain.
    row.a_chain[0] = u32_to_bits_le(h_in[3]);
    row.a_chain[1] = u32_to_bits_le(h_in[2]);
    row.a_chain[2] = u32_to_bits_le(h_in[1]);
    row.a_chain[3] = u32_to_bits_le(h_in[0]);
    row.e_chain[0] = u32_to_bits_le(h_in[7]);
    row.e_chain[1] = u32_to_bits_le(h_in[6]);
    row.e_chain[2] = u32_to_bits_le(h_in[5]);
    row.e_chain[3] = u32_to_bits_le(h_in[4]);

    // Working variables, initialised from the input state in canonical
    // SHA-256 order.
    let mut working = Zeroizing::new(*h_in);

    for t in 0..NUM_COMPRESSION_ROUNDS {
        let round: &mut Sha256RoundCols<F> = &mut row.rounds[t];

        // big_sigma1(e) = ROTR_6(e) XOR ROTR_11(e) XOR ROTR_25(e).
        let mut scratch = Zeroizing::new([0_u32; 8]);
        scratch[0] =
            working[4].rotate_right(6) ^ working[4].rotate_right(11) ^ working[4].rotate_right(25);

        // Ch(e, f, g) = (e AND f) XOR (NOT e AND g).
        scratch[1] = (working[4] & working[5]) ^ ((!working[4]) & working[6]);

        // Partial sum: tmp1 = h + big_sigma1 + Ch.
        scratch[2] = working[7].wrapping_add(scratch[0]).wrapping_add(scratch[1]);

        // T1 = tmp1 + K[t] + W[t].
        scratch[3] = scratch[2]
            .wrapping_add(SHA256_K[t])
            .wrapping_add(w_words[t]);

        // big_sigma0(a) = ROTR_2(a) XOR ROTR_13(a) XOR ROTR_22(a).
        scratch[4] =
            working[0].rotate_right(2) ^ working[0].rotate_right(13) ^ working[0].rotate_right(22);

        // Maj(a, b, c) = (a AND b) XOR (a AND c) XOR (b AND c).
        scratch[5] =
            (working[0] & working[1]) ^ (working[0] & working[2]) ^ (working[1] & working[2]);

        // Shifted slots for the next round.
        //
        // Mirrors the fused AIR add: `new_a = T_1 + big_sigma_0 + maj`
        // replaces the spec's two-step `T_2 = big_sigma_0 + maj; new_a = T_1 + T_2`.
        scratch[6] = scratch[3].wrapping_add(scratch[4]).wrapping_add(scratch[5]);
        scratch[7] = working[3].wrapping_add(scratch[3]);

        // Commit every packed intermediate for this round.
        //
        // `new_a` / `new_e`: bit form only — written into the next chain
        // slot below. The AIR re-packs from those bits on demand.
        round.sigma1_e = u32_to_limbs(scratch[0]).map(F::from_u16);
        round.ch = u32_to_limbs(scratch[1]).map(F::from_u16);
        round.tmp1 = u32_to_limbs(scratch[2]).map(F::from_u16);
        round.t1 = u32_to_limbs(scratch[3]).map(F::from_u16);
        round.sigma0_a = u32_to_limbs(scratch[4]).map(F::from_u16);
        round.maj = u32_to_limbs(scratch[5]).map(F::from_u16);

        // Extend the chains so the next round can read the shifted variables
        // directly.
        row.a_chain[t + 4] = u32_to_bits_le(scratch[6]);
        row.e_chain[t + 4] = u32_to_bits_le(scratch[7]);

        // Apply the SHA-256 working-variable shift.
        *working = [
            scratch[6], working[0], working[1], working[2], scratch[7], working[4], working[5],
            working[6],
        ];
    }
    // Sanity check: the chain length matches the expected extent.
    debug_assert_eq!(CHAIN_LEN, 4 + NUM_COMPRESSION_ROUNDS);

    // Step 4: finalization.
    //
    // After the loop the working variables line up with the tail of the two
    // chains:
    //     (a, b, c, d) = (a_chain[67], a_chain[66], a_chain[65], a_chain[64])
    //     (e, f, g, h) = (e_chain[67], e_chain[66], e_chain[65], e_chain[64])
    for i in 0..STATE_WORDS {
        // Output word i = H[i] + final_state[i] (mod 2^32).
        //
        // Bit form: the AIR repacks it for the addition check.
        // The bits are what bound the two 16-bit limbs.
        let h_out_word = h_in[i].wrapping_add(working[i]);
        row.h_out[i] = u32_to_bits_le(h_out_word);
    }
}

#[cfg(all(test, feature = "secret-memory"))]
mod secret_tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_matrix::secret::ZeroizingOwner;

    use super::{
        INPUT_WORDS, SecretSha256TraceError, generate_trace_rows, generate_trace_rows_secret,
    };
    use crate::SHA256_IV;

    fn compression_input(marker: u32) -> [u32; INPUT_WORDS] {
        let mut input = [0_u32; INPUT_WORDS];
        input[0] = marker;
        input[16..].copy_from_slice(&SHA256_IV);
        input
    }

    #[test]
    fn guarded_trace_matches_the_ordinary_generator() {
        let inputs = vec![
            compression_input(0x524d_454e),
            compression_input(0x5642_3100),
        ];
        let ordinary = generate_trace_rows::<BabyBear>(inputs.clone(), 2);
        let guarded = generate_trace_rows_secret::<BabyBear>(ZeroizingOwner::new(inputs), 2)
            .expect("valid guarded SHA-256 trace must generate");

        assert_eq!(guarded.width, ordinary.width);
        assert_eq!(guarded.values, ordinary.values);
    }

    #[test]
    fn guarded_trace_rejects_empty_and_non_power_of_two_inputs() {
        let empty = generate_trace_rows_secret::<BabyBear>(ZeroizingOwner::new(Vec::new()), 0);
        assert!(matches!(empty, Err(SecretSha256TraceError::EmptyInput)));

        let invalid = generate_trace_rows_secret::<BabyBear>(
            ZeroizingOwner::new(vec![
                compression_input(1),
                compression_input(2),
                compression_input(3),
            ]),
            0,
        );
        assert!(matches!(
            invalid,
            Err(SecretSha256TraceError::NonPowerOfTwoInput)
        ));
    }
}
