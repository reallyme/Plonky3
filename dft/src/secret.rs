// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in DFT operations that preserve zeroizing ownership.

use alloc::vec::Vec;

use p3_field::TwoAdicField;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::secret::ZeroizingRowMajorMatrix;
use p3_matrix::util::reverse_matrix_index_bits;
use p3_util::{log2_strict_usize, reverse_bits_len};
use thiserror::Error;

/// Stable failure classes for secret-memory DFT operations.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum SecretDftError {
    /// The matrix has zero width or a backing length inconsistent with its width.
    #[error("secret DFT matrix shape is invalid")]
    InvalidShape,
    /// The input height is zero or not a power of two.
    #[error("secret DFT input height must be a nonzero power of two")]
    InvalidHeight,
    /// Output dimensions cannot be represented by `usize`.
    #[error("secret DFT output dimensions overflow")]
    SizeOverflow,
    /// The output allocation could not be reserved.
    #[error("secret DFT output allocation failed")]
    AllocationFailed,
    /// Recomputing no rows cannot produce a usable MMCS opening.
    #[error("secret DFT row query set is empty")]
    EmptyQuerySet,
    /// A requested physical row does not exist in the extended domain.
    #[error("secret DFT row query index is out of bounds")]
    QueryIndexOutOfBounds { index: usize, height: usize },
}

/// DFT operations whose secret-derived matrix allocation remains guarded.
///
/// Implementations must not release any superseded backing allocation before
/// it has been erased. Errors leave `matrix` under the caller's existing
/// zeroizing owner.
pub trait SecretTwoAdicSubgroupDft<F>
where
    F: TwoAdicField + zeroize::Zeroize,
{
    /// Replace subgroup evaluations with polynomial coefficients in place.
    fn idft_batch_secret(
        &self,
        matrix: &mut ZeroizingRowMajorMatrix<F>,
    ) -> Result<(), SecretDftError>;

    /// Replace polynomial coefficients with subgroup evaluations in place.
    fn dft_batch_secret(
        &self,
        matrix: &mut ZeroizingRowMajorMatrix<F>,
    ) -> Result<(), SecretDftError>;

    /// Replace coset evaluations with polynomial coefficients in place.
    fn coset_idft_batch_secret(
        &self,
        matrix: &mut ZeroizingRowMajorMatrix<F>,
        shift: F,
    ) -> Result<(), SecretDftError>;

    /// Replace `matrix` with its low-degree extension over `shift * K`.
    fn coset_lde_batch_secret(
        &self,
        matrix: &mut ZeroizingRowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Result<(), SecretDftError>;

    /// Replace `matrix` with its LDE while retaining the compact,
    /// coset-adjusted coefficients needed to reconstruct committed rows.
    ///
    /// The returned coefficients are in natural degree order. Their guarded
    /// allocation is smaller than the LDE by the configured blowup factor and
    /// must remain secret until every requested opening row has been emitted.
    fn coset_lde_batch_secret_with_coefficients(
        &self,
        matrix: &mut ZeroizingRowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Result<ZeroizingRowMajorMatrix<F>, SecretDftError>;

    /// Rebuild the complete bit-reversed LDE committed by the PCS from its
    /// retained natural-order, coset-adjusted coefficients.
    ///
    /// This is the bounded-memory counterpart to retaining an LDE for the
    /// lifetime of a proof. Callers can erase committed value leaves after
    /// commitment, reconstruct one matrix for its final reduction, and erase
    /// that temporary matrix immediately afterward. The returned ordering is
    /// the physical ordering expected by the MMCS and therefore deliberately
    /// differs from [`Self::dft_batch_secret`]'s natural evaluation order.
    fn recompute_coset_lde_secret(
        &self,
        coefficients: &ZeroizingRowMajorMatrix<F>,
        added_bits: usize,
    ) -> Result<ZeroizingRowMajorMatrix<F>, SecretDftError> {
        if coefficients.width == 0 || !coefficients.values.len().is_multiple_of(coefficients.width)
        {
            return Err(SecretDftError::InvalidShape);
        }
        let coefficient_height = coefficients.height();
        if coefficient_height == 0 || !coefficient_height.is_power_of_two() {
            return Err(SecretDftError::InvalidHeight);
        }

        let added_bits_u32 = u32::try_from(added_bits).map_err(|_| SecretDftError::SizeOverflow)?;
        let lde_height = coefficient_height
            .checked_shl(added_bits_u32)
            .ok_or(SecretDftError::SizeOverflow)?;
        let coefficient_len = coefficient_height
            .checked_mul(coefficients.width)
            .ok_or(SecretDftError::SizeOverflow)?;
        let output_len = lde_height
            .checked_mul(coefficients.width)
            .ok_or(SecretDftError::SizeOverflow)?;

        let mut values = Vec::new();
        values
            .try_reserve_exact(output_len)
            .map_err(|_| SecretDftError::AllocationFailed)?;
        values.extend_from_slice(
            coefficients
                .values
                .get(..coefficient_len)
                .ok_or(SecretDftError::InvalidShape)?,
        );
        values.resize(output_len, F::ZERO);

        let mut lde = ZeroizingRowMajorMatrix::new(RowMajorMatrix::new(values, coefficients.width));
        self.dft_batch_secret(&mut lde)?;
        reverse_matrix_index_bits(&mut lde);
        Ok(lde)
    }

    /// Recompute selected physical rows of a bit-reversed LDE from retained
    /// natural-order, coset-adjusted coefficients.
    ///
    /// `indices` use the exact physical row numbering committed by the PCS.
    /// The returned matrix follows query order and remains zeroizing because
    /// the rows are witness-derived until they are deliberately disclosed.
    fn recompute_coset_lde_rows_secret(
        &self,
        coefficients: &ZeroizingRowMajorMatrix<F>,
        added_bits: usize,
        indices: &[usize],
    ) -> Result<ZeroizingRowMajorMatrix<F>, SecretDftError> {
        if coefficients.width == 0 || !coefficients.values.len().is_multiple_of(coefficients.width)
        {
            return Err(SecretDftError::InvalidShape);
        }
        let coefficient_height = coefficients.height();
        if coefficient_height == 0 || !coefficient_height.is_power_of_two() {
            return Err(SecretDftError::InvalidHeight);
        }
        if indices.is_empty() {
            return Err(SecretDftError::EmptyQuerySet);
        }

        let added_bits_u32 = u32::try_from(added_bits).map_err(|_| SecretDftError::SizeOverflow)?;
        let lde_height = coefficient_height
            .checked_shl(added_bits_u32)
            .ok_or(SecretDftError::SizeOverflow)?;
        for &index in indices {
            if index >= lde_height {
                return Err(SecretDftError::QueryIndexOutOfBounds {
                    index,
                    height: lde_height,
                });
            }
        }

        let output_len = indices
            .len()
            .checked_mul(coefficients.width)
            .ok_or(SecretDftError::SizeOverflow)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_len)
            .map_err(|_| SecretDftError::AllocationFailed)?;
        output.resize(output_len, F::ZERO);

        let log_lde_height = log2_strict_usize(lde_height);
        let generator = F::two_adic_generator(log_lde_height);
        for (&physical_index, output_row) in indices
            .iter()
            .zip(output.chunks_exact_mut(coefficients.width))
        {
            let natural_index = reverse_bits_len(physical_index, log_lde_height);
            let exponent =
                u64::try_from(natural_index).map_err(|_| SecretDftError::SizeOverflow)?;
            let point = generator.exp_u64(exponent);
            for coefficient_row in coefficients.row_slices().rev() {
                for (value, coefficient) in output_row.iter_mut().zip(coefficient_row) {
                    *value = *value * point + *coefficient;
                }
            }
        }

        Ok(ZeroizingRowMajorMatrix::new(RowMajorMatrix::new(
            output,
            coefficients.width,
        )))
    }
}
