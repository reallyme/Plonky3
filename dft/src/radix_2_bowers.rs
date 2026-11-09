// Modified by ReallyMe Technologies Limited, 2026.
// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
#[cfg(feature = "secret-memory")]
use p3_matrix::secret::{ZeroizingMatrix, ZeroizingRowMajorMatrix};
use p3_matrix::util::reverse_matrix_index_bits;
use p3_maybe_rayon::prelude::*;
use p3_util::{flatten_to_base, log2_strict_usize, reverse_slice_index_bits};
use tracing::instrument;

use crate::butterflies::{Butterfly, DifButterfly, DitButterfly, TwiddleFreeButterfly};
#[cfg(feature = "secret-memory")]
use crate::util::coset_shift_cols;
use crate::util::divide_by_height;
use crate::{Layout, TwoAdicSubgroupDft};
#[cfg(feature = "secret-memory")]
use crate::{SecretDftError, SecretTwoAdicSubgroupDft};

/// The Bowers G FFT algorithm.
/// See: "Improved Twiddle Access for Fast Fourier Transforms"
#[derive(Default, Clone)]
pub struct Radix2Bowers;

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for Radix2Bowers {
    type Evaluations = RowMajorMatrix<F>;

    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        reverse_matrix_index_bits(&mut mat);
        bowers_g(&mut mat.as_view_mut());
        mat
    }

    /// Compute the inverse DFT of each column in `mat`.
    fn idft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        bowers_g_t(&mut mat.as_view_mut());
        divide_by_height(&mut mat);
        reverse_matrix_index_bits(&mut mat);
        mat
    }

    fn lde_batch(&self, mut mat: RowMajorMatrix<F>, added_bits: usize) -> RowMajorMatrix<F> {
        bowers_g_t(&mut mat.as_view_mut());
        divide_by_height(&mut mat);
        mat = mat.bit_reversed_zero_pad(added_bits);
        bowers_g(&mut mat.as_view_mut());
        mat
    }

    #[instrument(skip_all, level = "debug", fields(dims = %mat.dimensions(), added_bits))]
    fn coset_lde_batch(
        &self,
        mut mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> RowMajorMatrix<F> {
        let h = mat.height();
        let log_h = log2_strict_usize(h);
        // It's cheaper to use div_2exp_u64 as this usually avoids an inversion.
        // It's also cheaper to work in the PrimeSubfield whenever possible.
        let h_inv_subfield = F::PrimeSubfield::ONE.div_2exp_u64(log_h as u64);
        let h_inv = F::from_prime_subfield(h_inv_subfield);

        bowers_g_t(&mut mat.as_view_mut());

        // Rescale coefficients in two ways:
        // - divide by height (since we're doing an inverse DFT)
        // - multiply by powers of the coset shift (see default coset LDE impl for an explanation)
        let mut weights = shift.shifted_powers(h_inv).collect_n(h);
        // reverse_bits because mat is encoded in bit-reversed order
        reverse_slice_index_bits(&mut weights);

        mat.par_rows_mut()
            .zip(weights.into_par_iter())
            .for_each(|(row, weight)| row.iter_mut().for_each(|elem| *elem *= weight));

        mat = mat.bit_reversed_zero_pad(added_bits);

        bowers_g(&mut mat.as_view_mut());

        mat
    }

    #[instrument(skip_all, level = "debug", fields(dims = %mat.dimensions(), added_bits))]
    fn coset_lde_batch_with_transform<T>(
        &self,
        mut mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
        transform: T,
    ) -> RowMajorMatrix<F>
    where
        T: FnOnce(&mut RowMajorMatrixViewMut<'_, F>, Layout),
    {
        let h = mat.height();
        let log_h = log2_strict_usize(h);
        let h_inv_subfield = F::PrimeSubfield::ONE.div_2exp_u64(log_h as u64);
        let h_inv = F::from_prime_subfield(h_inv_subfield);

        // Inverse butterfly leaves coefficients in bit-reversed memory.
        bowers_g_t(&mut mat.as_view_mut());

        // Normalise to true polynomial coefficients before invoking the closure.
        // (Costs one extra O(h) pass relative to the no-op `coset_lde_batch` path,
        // which fuses normalisation with the coset-shift weights.)
        mat.values.par_iter_mut().for_each(|v| *v *= h_inv);

        transform(&mut mat.as_view_mut(), Layout::BitReversed);

        // Apply coset-shift weights in bit-reversed order.
        let mut shift_powers = shift.powers().collect_n(h);
        reverse_slice_index_bits(&mut shift_powers);
        mat.par_rows_mut()
            .zip(shift_powers.into_par_iter())
            .for_each(|(row, sp)| row.iter_mut().for_each(|e| *e *= sp));

        mat = mat.bit_reversed_zero_pad(added_bits);
        bowers_g(&mut mat.as_view_mut());
        mat
    }
}

#[cfg(feature = "secret-memory")]
impl<F> SecretTwoAdicSubgroupDft<F> for Radix2Bowers
where
    F: TwoAdicField + zeroize::Zeroize,
{
    fn idft_batch_secret(
        &self,
        matrix: &mut ZeroizingRowMajorMatrix<F>,
    ) -> Result<(), SecretDftError> {
        validate_secret_transform_matrix(matrix)?;
        bowers_g_t(&mut matrix.as_view_mut());
        divide_by_height(matrix);
        reverse_matrix_index_bits(matrix);
        Ok(())
    }

    fn dft_batch_secret(
        &self,
        matrix: &mut ZeroizingRowMajorMatrix<F>,
    ) -> Result<(), SecretDftError> {
        validate_secret_transform_matrix(matrix)?;
        reverse_matrix_index_bits(matrix);
        bowers_g(&mut matrix.as_view_mut());
        Ok(())
    }

    fn coset_idft_batch_secret(
        &self,
        matrix: &mut ZeroizingRowMajorMatrix<F>,
        shift: F,
    ) -> Result<(), SecretDftError> {
        self.idft_batch_secret(matrix)?;
        coset_shift_cols(matrix, shift.inverse());
        Ok(())
    }

    fn coset_lde_batch_secret(
        &self,
        matrix: &mut ZeroizingRowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Result<(), SecretDftError> {
        let height = validate_secret_transform_matrix(matrix)?;

        let log_height = log2_strict_usize(height);
        let log_height_u64 = u64::try_from(log_height).map_err(|_| SecretDftError::SizeOverflow)?;
        let height_inverse_subfield = F::PrimeSubfield::ONE.div_2exp_u64(log_height_u64);
        let height_inverse = F::from_prime_subfield(height_inverse_subfield);

        bowers_g_t(&mut matrix.as_view_mut());

        let mut weights = shift.shifted_powers(height_inverse).collect_n(height);
        reverse_slice_index_bits(&mut weights);
        matrix
            .par_rows_mut()
            .zip(weights.into_par_iter())
            .for_each(|(row, weight)| row.iter_mut().for_each(|element| *element *= weight));

        secret_bit_reversed_zero_pad(matrix, added_bits)?;
        bowers_g(&mut matrix.as_view_mut());
        Ok(())
    }

    fn coset_lde_batch_secret_with_coefficients(
        &self,
        matrix: &mut ZeroizingRowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Result<ZeroizingRowMajorMatrix<F>, SecretDftError> {
        self.idft_batch_secret(matrix)?;
        coset_shift_cols(matrix, shift);

        let added_bits_u32 = u32::try_from(added_bits).map_err(|_| SecretDftError::SizeOverflow)?;
        let lde_height = matrix
            .height()
            .checked_shl(added_bits_u32)
            .ok_or(SecretDftError::SizeOverflow)?;
        let output_len = lde_height
            .checked_mul(matrix.width)
            .ok_or(SecretDftError::SizeOverflow)?;
        let coefficient_len = matrix.values.len();
        let width = matrix.width;

        // Allocate the final LDE directly instead of cloning the coefficients
        // and then growing their allocation. Keeping the original allocation
        // as the returned coefficient owner avoids a third secret-bearing
        // allocation at the quotient-preparation high-water mark.
        let mut lde_values = Vec::new();
        lde_values
            .try_reserve_exact(output_len)
            .map_err(|_| SecretDftError::AllocationFailed)?;
        lde_values.resize(output_len, F::ZERO);
        let mut lde = ZeroizingMatrix::new(RowMajorMatrix::new(lde_values, width));
        lde.values
            .get_mut(..coefficient_len)
            .ok_or(SecretDftError::InvalidShape)?
            .copy_from_slice(&matrix.values);

        self.dft_batch_secret(&mut lde)?;
        core::mem::swap(matrix, &mut lde);

        // After the swap, `matrix` owns the full LDE and `lde` owns the
        // original compact coefficient allocation.
        let coefficients = lde;
        Ok(coefficients)
    }
}

#[cfg(feature = "secret-memory")]
fn validate_secret_transform_matrix<F>(
    matrix: &ZeroizingRowMajorMatrix<F>,
) -> Result<usize, SecretDftError>
where
    F: TwoAdicField + zeroize::Zeroize,
{
    if matrix.width == 0 || !matrix.values.len().is_multiple_of(matrix.width) {
        return Err(SecretDftError::InvalidShape);
    }
    let height = matrix.values.len() / matrix.width;
    if height == 0 || !height.is_power_of_two() {
        return Err(SecretDftError::InvalidHeight);
    }
    Ok(height)
}

#[cfg(feature = "secret-memory")]
fn secret_bit_reversed_zero_pad<F>(
    matrix: &mut ZeroizingRowMajorMatrix<F>,
    added_bits: usize,
) -> Result<(), SecretDftError>
where
    F: TwoAdicField + zeroize::Zeroize,
{
    if added_bits == 0 {
        return Ok(());
    }

    let row_scale = 1_usize
        .checked_shl(u32::try_from(added_bits).map_err(|_| SecretDftError::SizeOverflow)?)
        .ok_or(SecretDftError::SizeOverflow)?;
    let output_len = matrix
        .values
        .len()
        .checked_mul(row_scale)
        .ok_or(SecretDftError::SizeOverflow)?;
    let width = matrix.width;
    let mut output_values = Vec::new();
    output_values
        .try_reserve_exact(output_len)
        .map_err(|_| SecretDftError::AllocationFailed)?;
    output_values.resize(output_len, F::ZERO);
    let mut output = ZeroizingMatrix::new(RowMajorMatrix::new(output_values, width));

    output
        .par_row_chunks_exact_mut(row_scale)
        .zip(matrix.par_row_slices())
        .for_each(|(mut chunk, row)| chunk.row_mut(0).copy_from_slice(row));

    zeroize::Zeroize::zeroize(matrix);
    core::mem::swap(matrix, &mut output);
    Ok(())
}

#[cfg(all(test, feature = "secret-memory"))]
mod secret_memory_tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::secret::ZeroizingMatrix;
    use p3_matrix::util::reverse_matrix_index_bits;

    use super::Radix2Bowers;
    use crate::{SecretDftError, SecretTwoAdicSubgroupDft, TwoAdicSubgroupDft};

    fn fixture() -> RowMajorMatrix<BabyBear> {
        RowMajorMatrix::new((1_u8..=8).map(BabyBear::from_u8).collect::<Vec<_>>(), 2)
    }

    #[test]
    fn secret_coset_lde_matches_the_standard_transform() {
        let dft = Radix2Bowers;
        let input = fixture();
        let expected = dft.coset_lde_batch(input.clone(), 1, BabyBear::GENERATOR);
        let mut secret = ZeroizingMatrix::new(input);

        dft.coset_lde_batch_secret(&mut secret, 1, BabyBear::GENERATOR)
            .expect("valid fixture must transform");

        assert_eq!(secret.width, expected.width);
        assert_eq!(secret.values, expected.values);
    }

    #[test]
    fn compact_coefficients_recompute_every_bit_reversed_lde_row() {
        let dft = Radix2Bowers;
        let mut lde = ZeroizingMatrix::new(fixture());
        let compact_allocation = lde.values.as_ptr();
        let coefficients = dft
            .coset_lde_batch_secret_with_coefficients(&mut lde, 1, BabyBear::GENERATOR)
            .expect("valid fixture must retain compact coefficients");
        reverse_matrix_index_bits(&mut lde);

        let indices = (0..lde.height()).collect::<Vec<_>>();
        let recomputed = dft
            .recompute_coset_lde_rows_secret(&coefficients, 1, &indices)
            .expect("every committed row must be recomputable");

        assert_eq!(coefficients.height() * 2, lde.height());
        assert_eq!(coefficients.values.as_ptr(), compact_allocation);
        assert_eq!(recomputed.width, lde.width);
        assert_eq!(recomputed.values, lde.values);

        let recomputed_full = dft
            .recompute_coset_lde_secret(&coefficients, 1)
            .expect("complete committed LDE must be recomputable");
        assert_eq!(recomputed_full.width, lde.width);
        assert_eq!(recomputed_full.values, lde.values);
        assert!(matches!(
            dft.recompute_coset_lde_rows_secret(&coefficients, 1, &[]),
            Err(SecretDftError::EmptyQuerySet)
        ));
        let out_of_bounds_index = lde.height();
        assert!(matches!(
            dft.recompute_coset_lde_rows_secret(&coefficients, 1, &[out_of_bounds_index]),
            Err(SecretDftError::QueryIndexOutOfBounds { index, height })
                if index == out_of_bounds_index && height == out_of_bounds_index
        ));
    }

    #[test]
    fn complete_lde_reconstruction_rejects_malformed_coefficients() {
        let dft = Radix2Bowers;
        let zero_width = ZeroizingMatrix::new(RowMajorMatrix::new(Vec::<BabyBear>::new(), 0));
        assert!(matches!(
            dft.recompute_coset_lde_secret(&zero_width, 1),
            Err(SecretDftError::InvalidShape)
        ));

        let invalid_height = ZeroizingMatrix::new(RowMajorMatrix::new(
            (0..6).map(|_| BabyBear::ONE).collect::<Vec<_>>(),
            2,
        ));
        assert!(matches!(
            dft.recompute_coset_lde_secret(&invalid_height, 1),
            Err(SecretDftError::InvalidHeight)
        ));
    }

    #[test]
    fn secret_forward_and_inverse_transforms_match_standard_transforms() {
        let dft = Radix2Bowers;
        let input = fixture();
        let expected_evaluations = dft.dft_batch(input.clone());
        let mut secret = ZeroizingMatrix::new(input.clone());

        dft.dft_batch_secret(&mut secret)
            .expect("valid fixture must support guarded DFT");
        assert_eq!(secret.width, expected_evaluations.width);
        assert_eq!(secret.values, expected_evaluations.values);

        dft.idft_batch_secret(&mut secret)
            .expect("valid fixture must support guarded inverse DFT");
        assert_eq!(secret.width, input.width);
        assert_eq!(secret.values, input.values);
    }

    #[test]
    fn secret_coset_inverse_transform_matches_standard_transform() {
        let dft = Radix2Bowers;
        let input = fixture();
        let expected = dft.coset_idft_batch(input.clone(), BabyBear::GENERATOR);
        let mut secret = ZeroizingMatrix::new(input);

        dft.coset_idft_batch_secret(&mut secret, BabyBear::GENERATOR)
            .expect("valid fixture must support guarded coset inverse DFT");

        assert_eq!(secret.width, expected.width);
        assert_eq!(secret.values, expected.values);
    }

    #[test]
    fn secret_transforms_reject_zero_width_without_releasing_the_owner() {
        let dft = Radix2Bowers;
        let mut malformed = fixture();
        malformed.width = 0;
        let expected_values = malformed.values.clone();
        let mut secret = ZeroizingMatrix::new(malformed);

        let result = dft.dft_batch_secret(&mut secret);

        assert_eq!(result, Err(SecretDftError::InvalidShape));
        assert_eq!(secret.values, expected_values);
        assert_eq!(secret.width, 0);
    }

    #[test]
    fn secret_coset_lde_rejects_invalid_height_without_releasing_the_owner() {
        let dft = Radix2Bowers;
        let mut secret = ZeroizingMatrix::new(RowMajorMatrix::new(Vec::<BabyBear>::new(), 1));

        let result = dft.coset_lde_batch_secret(&mut secret, 1, BabyBear::GENERATOR);

        assert_eq!(result, Err(SecretDftError::InvalidHeight));
        assert!(secret.values.is_empty());
    }

    #[test]
    fn secret_coset_lde_rejects_output_size_overflow() {
        let dft = Radix2Bowers;
        let mut secret = ZeroizingMatrix::new(fixture());

        let oversized_shift =
            usize::try_from(usize::BITS).expect("usize bit width must fit in usize");
        let result = dft.coset_lde_batch_secret(&mut secret, oversized_shift, BabyBear::GENERATOR);

        assert_eq!(result, Err(SecretDftError::SizeOverflow));
    }

    #[test]
    fn coefficient_retaining_lde_overflow_preserves_the_guarded_allocation() {
        let dft = Radix2Bowers;
        let mut secret = ZeroizingMatrix::new(fixture());
        let guarded_allocation = secret.values.as_ptr();
        let original_len = secret.values.len();
        let original_width = secret.width;
        let oversized_shift =
            usize::try_from(usize::BITS).expect("usize bit width must fit in usize");

        let result = dft.coset_lde_batch_secret_with_coefficients(
            &mut secret,
            oversized_shift,
            BabyBear::GENERATOR,
        );

        assert!(matches!(result, Err(SecretDftError::SizeOverflow)));
        assert_eq!(secret.values.as_ptr(), guarded_allocation);
        assert_eq!(secret.values.len(), original_len);
        assert_eq!(secret.width, original_width);
    }
}

/// Executes the Bowers G network. This is like a DFT, except it assumes the input is in
/// bit-reversed order.
fn bowers_g<F: TwoAdicField>(mat: &mut RowMajorMatrixViewMut<'_, F>) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);

    let root = F::two_adic_generator(log_h);
    let mut twiddles = root.powers().collect_n(h / 2);
    reverse_slice_index_bits(&mut twiddles);
    // SAFETY: DifButterfly is `repr(transparent)`
    let twiddles: Vec<DifButterfly<F>> = unsafe { flatten_to_base(twiddles) };

    for log_half_block_size in 0..log_h {
        butterfly_layer(mat, 1 << log_half_block_size, &twiddles);
    }
}

/// Executes the Bowers G^T network. This is like an inverse DFT, except we skip rescaling by
/// 1/height, and the output is bit-reversed.
fn bowers_g_t<F: TwoAdicField>(mat: &mut RowMajorMatrixViewMut<'_, F>) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);

    let root_inv = F::two_adic_generator(log_h).inverse();
    let mut twiddles = root_inv.powers().collect_n(h / 2);
    reverse_slice_index_bits(&mut twiddles);
    // SAFETY: DitButterfly is `repr(transparent)`
    let twiddles: Vec<DitButterfly<F>> = unsafe { flatten_to_base(twiddles) };

    for log_half_block_size in (0..log_h).rev() {
        butterfly_layer(mat, 1 << log_half_block_size, &twiddles);
    }
}

fn butterfly_layer<F: Field, B: Butterfly<F>>(
    mat: &mut RowMajorMatrixViewMut<'_, F>,
    half_block_size: usize,
    twiddles: &[B],
) {
    mat.par_row_chunks_exact_mut(2 * half_block_size)
        .enumerate()
        .for_each(|(block, mut chunks)| {
            let (mut hi_chunks, mut lo_chunks) = chunks.split_rows_mut(half_block_size);
            hi_chunks
                .par_rows_mut()
                .zip(lo_chunks.par_rows_mut())
                .for_each(|(hi_chunk, lo_chunk)| {
                    if block == 0 {
                        TwiddleFreeButterfly.apply_to_rows(hi_chunk, lo_chunk);
                    } else {
                        twiddles[block].apply_to_rows(hi_chunk, lo_chunk);
                    }
                });
        });
}
