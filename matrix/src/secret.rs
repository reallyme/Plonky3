// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Explicit owners for allocations whose storage contains secret-derived values.
//!
//! Ordinary matrices retain their existing allocation and drop behavior. A
//! caller selects this owner when the allocation is part of a secret prover
//! path. The backing storage is erased on normal return and during unwinding.
//! There is intentionally no operation that extracts the owned matrix: code
//! that moves the allocation into another prover stage must keep a zeroizing
//! owner around it at that stage.

use core::ops::{Deref, DerefMut};

use p3_field::{ExtensionField, Field};
use rand::distr::{Distribution, StandardUniform};
use rand::{CryptoRng, Rng, RngExt};
use thiserror::Error;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use crate::Matrix;
use crate::dense::{DenseMatrix, RowMajorMatrix};

/// Non-debuggable owner that erases a matrix before releasing its storage.
///
/// This type does not implement `Clone`, `Copy`, serialization, or a method
/// that releases the inner matrix without erasing it. Those omissions are
/// part of the secret-memory contract.
#[must_use = "dropping the owner is what erases the matrix storage"]
pub struct ZeroizingOwner<T: Zeroize> {
    inner: Zeroizing<T>,
}

/// Stable failure classes for guarded matrix allocation and shape changes.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum SecretMatrixError {
    /// A matrix dimension or backing-buffer length cannot be represented.
    #[error("secret matrix dimensions overflow")]
    DimensionOverflow,
    /// The replacement backing allocation could not be reserved.
    #[error("secret matrix allocation failed")]
    AllocationFailed,
    /// The source matrix does not have a valid, nonzero row width.
    #[error("secret matrix shape is invalid")]
    InvalidShape,
}

impl<T: Zeroize> ZeroizingOwner<T> {
    /// Takes responsibility for erasing `value` before its storage is freed.
    pub fn new(value: T) -> Self {
        Self {
            inner: Zeroizing::new(value),
        }
    }
}

impl<T> ZeroizingOwner<RowMajorMatrix<T>>
where
    T: Field + Zeroize,
{
    /// Copy this matrix directly into a distinct zeroizing owner.
    ///
    /// Some protocols must retain the source evaluations while a PCS consumes
    /// and transforms its own copy. Returning another guarded owner prevents
    /// that necessary duplication from passing through an ordinary `Vec`.
    pub fn try_clone_guarded(&self) -> Result<Self, SecretMatrixError> {
        let width = self.width;
        if width == 0 || !self.values.len().is_multiple_of(width) {
            return Err(SecretMatrixError::InvalidShape);
        }

        let mut values = alloc::vec::Vec::new();
        values
            .try_reserve_exact(self.values.len())
            .map_err(|_| SecretMatrixError::AllocationFailed)?;
        values.resize(self.values.len(), T::ZERO);
        values.copy_from_slice(&self.values);
        Ok(Self::new(RowMajorMatrix::new(values, width)))
    }
}

impl<T> ZeroizingOwner<RowMajorMatrix<T>>
where
    T: Field + Zeroize,
    StandardUniform: Distribution<T>,
{
    /// Allocate a row-major matrix under zeroizing ownership before filling it
    /// with cryptographically generated values.
    ///
    /// The allocation initially contains only public zeroes. Random values are
    /// written after the erasure owner exists, so no secret-bearing ordinary
    /// `Vec` is created even transiently.
    pub fn try_random<R>(
        height: usize,
        width: usize,
        rng: &mut R,
    ) -> Result<Self, SecretMatrixError>
    where
        R: Rng + CryptoRng,
    {
        if height == 0 || width == 0 {
            return Err(SecretMatrixError::InvalidShape);
        }
        let length = height
            .checked_mul(width)
            .ok_or(SecretMatrixError::DimensionOverflow)?;
        let mut values = alloc::vec::Vec::new();
        values
            .try_reserve_exact(length)
            .map_err(|_| SecretMatrixError::AllocationFailed)?;
        values.resize(length, T::ZERO);
        let mut matrix = Self::new(RowMajorMatrix::new(values, width));
        for value in &mut matrix.values {
            *value = rng.random();
        }
        Ok(matrix)
    }

    /// Replace this matrix with a widened copy whose trailing columns are random.
    ///
    /// Both the old and replacement allocations remain guarded throughout the
    /// operation. The old allocation is erased before ownership is swapped.
    pub fn try_widen_with_random_cols<R>(
        &mut self,
        random_columns: usize,
        rng: &mut R,
    ) -> Result<(), SecretMatrixError>
    where
        R: Rng + CryptoRng,
    {
        let old_width = self.width;
        if old_width == 0 || !self.values.len().is_multiple_of(old_width) {
            return Err(SecretMatrixError::InvalidShape);
        }
        let height = self.values.len() / old_width;
        let new_width = old_width
            .checked_add(random_columns)
            .ok_or(SecretMatrixError::DimensionOverflow)?;
        let output_len = height
            .checked_mul(new_width)
            .ok_or(SecretMatrixError::DimensionOverflow)?;

        let mut output_values = alloc::vec::Vec::new();
        output_values
            .try_reserve_exact(output_len)
            .map_err(|_| SecretMatrixError::AllocationFailed)?;
        output_values.resize(output_len, T::ZERO);
        let mut output = Self::new(RowMajorMatrix::new(output_values, new_width));

        for (source, destination) in self
            .values
            .chunks_exact(old_width)
            .zip(output.values.chunks_exact_mut(new_width))
        {
            destination[..old_width].copy_from_slice(source);
            for value in &mut destination[old_width..] {
                *value = rng.random();
            }
        }

        self.zeroize();
        core::mem::swap(self, &mut output);
        Ok(())
    }
}

impl<T> ZeroizingOwner<RowMajorMatrix<T>>
where
    T: Field + Zeroize,
{
    /// Flatten a guarded extension-field matrix into a guarded base-field matrix.
    ///
    /// The replacement allocation is placed under erasure ownership before
    /// coefficients are copied. The extension-field source is explicitly
    /// erased before this function returns, so no live secret allocation is
    /// released during the representation change.
    pub fn try_flatten_extension_to_base<F>(
        mut self,
    ) -> Result<ZeroizingRowMajorMatrix<F>, SecretMatrixError>
    where
        T: ExtensionField<F>,
        F: Field + Zeroize,
    {
        let width = T::DIMENSION;
        if self.width == 0 || width == 0 || !self.values.len().is_multiple_of(self.width) {
            return Err(SecretMatrixError::InvalidShape);
        }
        let output_width = self
            .width
            .checked_mul(width)
            .ok_or(SecretMatrixError::DimensionOverflow)?;
        let output_len = self
            .values
            .len()
            .checked_mul(width)
            .ok_or(SecretMatrixError::DimensionOverflow)?;
        let mut values = alloc::vec::Vec::new();
        values
            .try_reserve_exact(output_len)
            .map_err(|_| SecretMatrixError::AllocationFailed)?;
        values.resize(output_len, F::ZERO);
        let mut output = ZeroizingOwner::new(RowMajorMatrix::new(values, output_width));

        for (value, destination) in self
            .values
            .iter()
            .zip(output.values.chunks_exact_mut(width))
        {
            let coefficients = value.as_basis_coefficients_slice();
            if coefficients.len() != width {
                return Err(SecretMatrixError::InvalidShape);
            }
            destination.copy_from_slice(coefficients);
        }

        self.zeroize();
        Ok(output)
    }
}

impl<T> ZeroizingOwner<RowMajorMatrix<T>>
where
    T: Field + Zeroize,
{
    /// Replace this matrix with the leftmost `new_width` columns.
    ///
    /// The replacement is allocated and guarded before any values are copied.
    /// The superseded allocation, including columns that are intentionally
    /// discarded, is erased before ownership moves to the replacement.
    pub fn try_truncate_columns(&mut self, new_width: usize) -> Result<(), SecretMatrixError> {
        let old_width = self.width;
        if old_width == 0
            || new_width == 0
            || new_width > old_width
            || !self.values.len().is_multiple_of(old_width)
        {
            return Err(SecretMatrixError::InvalidShape);
        }
        let height = self.values.len() / old_width;
        let output_len = height
            .checked_mul(new_width)
            .ok_or(SecretMatrixError::DimensionOverflow)?;

        let mut output_values = alloc::vec::Vec::new();
        output_values
            .try_reserve_exact(output_len)
            .map_err(|_| SecretMatrixError::AllocationFailed)?;
        output_values.resize(output_len, T::ZERO);
        let mut output = Self::new(RowMajorMatrix::new(output_values, new_width));

        for (source, destination) in self
            .values
            .chunks_exact(old_width)
            .zip(output.values.chunks_exact_mut(new_width))
        {
            destination.copy_from_slice(&source[..new_width]);
        }

        self.zeroize();
        core::mem::swap(self, &mut output);
        Ok(())
    }

    /// Consume this matrix and retain a fixed-stride sequence of rows.
    ///
    /// The replacement is allocated under zeroizing ownership before rows are
    /// copied. The complete source, including every omitted row, is explicitly
    /// erased before its allocation is released. This is intended for protocol
    /// layouts that interleave witness rows with masking rows.
    pub fn try_select_strided_rows(
        mut self,
        first_row: usize,
        row_stride: usize,
        row_count: usize,
    ) -> Result<Self, SecretMatrixError> {
        let width = self.width;
        if width == 0
            || row_stride == 0
            || row_count == 0
            || !self.values.len().is_multiple_of(width)
        {
            return Err(SecretMatrixError::InvalidShape);
        }
        let source_height = self.values.len() / width;
        let final_offset = row_count
            .checked_sub(1)
            .and_then(|count| count.checked_mul(row_stride))
            .and_then(|offset| first_row.checked_add(offset))
            .ok_or(SecretMatrixError::DimensionOverflow)?;
        if final_offset >= source_height {
            return Err(SecretMatrixError::InvalidShape);
        }
        let output_len = row_count
            .checked_mul(width)
            .ok_or(SecretMatrixError::DimensionOverflow)?;

        let mut output_values = alloc::vec::Vec::new();
        output_values
            .try_reserve_exact(output_len)
            .map_err(|_| SecretMatrixError::AllocationFailed)?;
        output_values.resize(output_len, T::ZERO);
        let mut output = Self::new(RowMajorMatrix::new(output_values, width));

        for destination_row in 0..row_count {
            let source_row = destination_row
                .checked_mul(row_stride)
                .and_then(|offset| first_row.checked_add(offset))
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            let source_start = source_row
                .checked_mul(width)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            let source_end = source_start
                .checked_add(width)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            let destination_start = destination_row
                .checked_mul(width)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            let destination_end = destination_start
                .checked_add(width)
                .ok_or(SecretMatrixError::DimensionOverflow)?;
            let source = self
                .values
                .get(source_start..source_end)
                .ok_or(SecretMatrixError::InvalidShape)?;
            let destination = output
                .values
                .get_mut(destination_start..destination_end)
                .ok_or(SecretMatrixError::InvalidShape)?;
            destination.copy_from_slice(source);
        }

        self.zeroize();
        Ok(output)
    }

    /// Grow the matrix to `new_height`, filling added rows with zeroes.
    ///
    /// The replacement allocation is guarded before any existing value is
    /// copied. The superseded allocation is erased before ownership moves to
    /// the replacement, avoiding an ordinary `Vec` reallocation of live data.
    pub fn try_grow_zeroed_rows(&mut self, new_height: usize) -> Result<(), SecretMatrixError> {
        let width = self.width;
        if width == 0 || !self.values.len().is_multiple_of(width) {
            return Err(SecretMatrixError::InvalidShape);
        }
        let old_height = self.values.len() / width;
        if new_height < old_height {
            return Err(SecretMatrixError::InvalidShape);
        }
        let output_len = new_height
            .checked_mul(width)
            .ok_or(SecretMatrixError::DimensionOverflow)?;

        let mut output_values = alloc::vec::Vec::new();
        output_values
            .try_reserve_exact(output_len)
            .map_err(|_| SecretMatrixError::AllocationFailed)?;
        output_values.resize(output_len, T::ZERO);
        let mut output = Self::new(RowMajorMatrix::new(output_values, width));
        let destination = output
            .values
            .get_mut(..self.values.len())
            .ok_or(SecretMatrixError::InvalidShape)?;
        destination.copy_from_slice(&self.values);

        self.zeroize();
        core::mem::swap(self, &mut output);
        Ok(())
    }
}

impl<T: Zeroize> Deref for ZeroizingOwner<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: Zeroize> DerefMut for ZeroizingOwner<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T: Zeroize> Zeroize for ZeroizingOwner<T> {
    fn zeroize(&mut self) {
        self.inner.zeroize();
    }
}

impl<T: Zeroize> ZeroizeOnDrop for ZeroizingOwner<T> {}

impl<T, M> Matrix<T> for ZeroizingOwner<M>
where
    T: Send + Sync + Clone,
    M: Matrix<T> + Zeroize,
{
    fn width(&self) -> usize {
        self.inner.width()
    }

    fn height(&self) -> usize {
        self.inner.height()
    }

    unsafe fn row_unchecked(
        &self,
        row: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        // SAFETY: This adapter preserves the wrapped matrix's dimensions, so the
        // caller's `row < self.height()` obligation is exactly the obligation
        // required by the delegated operation.
        unsafe { self.inner.row_unchecked(row) }
    }
}

/// Zeroizing owner specialized to matrix-like values.
pub type ZeroizingMatrix<M> = ZeroizingOwner<M>;

/// Owned row-major matrix whose allocation is erased on drop.
pub type ZeroizingRowMajorMatrix<T> = ZeroizingMatrix<RowMajorMatrix<T>>;

/// Borrowed mutable matrix whose referenced storage is erased when the owner
/// leaves scope.
pub type ZeroizingDenseMatrix<T, V> = ZeroizingMatrix<DenseMatrix<T, V>>;

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use core::borrow::{Borrow, BorrowMut};
    use core::sync::atomic::{AtomicU8, Ordering};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::{Arc, Mutex};
    use std::thread;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::{SecretMatrixError, ZeroizingMatrix, ZeroizingOwner};
    use crate::Matrix;
    use crate::dense::{DenseMatrix, DenseStorage, RowMajorMatrix, RowMajorMatrixViewMut};

    const STAGE_SENSITIVE: u8 = 1;
    const STAGE_USED: u8 = 2;
    const STAGE_ERASE_STARTED: u8 = 3;
    const STAGE_ERASE_VERIFIED: u8 = 4;
    const STAGE_DEALLOCATED: u8 = 5;

    struct LifecycleState {
        stage: AtomicU8,
    }

    impl LifecycleState {
        fn sensitive() -> Arc<Self> {
            Arc::new(Self {
                stage: AtomicU8::new(STAGE_SENSITIVE),
            })
        }

        fn stage(&self) -> u8 {
            self.stage.load(Ordering::SeqCst)
        }
    }

    struct LifecycleCanary {
        values: [u32; 4],
        state: Arc<LifecycleState>,
    }

    impl LifecycleCanary {
        fn new(marker: u32, state: Arc<LifecycleState>) -> Self {
            assert_ne!(marker, 0, "the lifecycle canary must be nonzero");
            Self {
                values: [marker, 0xdead_beef, 0x5245_414c, 0x4c59_4d45],
                state,
            }
        }

        fn mark_used(&self) {
            self.state
                .stage
                .compare_exchange(
                    STAGE_SENSITIVE,
                    STAGE_USED,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .expect("a canary must be sensitive before it is used");
        }
    }

    impl zeroize::Zeroize for LifecycleCanary {
        fn zeroize(&mut self) {
            let stage = self.state.stage();
            if stage == STAGE_ERASE_VERIFIED {
                return;
            }
            assert!(
                stage == STAGE_SENSITIVE || stage == STAGE_USED,
                "erasure must begin from a live sensitive allocation"
            );
            self.state
                .stage
                .store(STAGE_ERASE_STARTED, Ordering::SeqCst);
            self.values.zeroize();
            assert_eq!(self.values, [0_u32; 4]);
            self.state
                .stage
                .store(STAGE_ERASE_VERIFIED, Ordering::SeqCst);
        }
    }

    impl Drop for LifecycleCanary {
        fn drop(&mut self) {
            assert_eq!(
                self.state.stage(),
                STAGE_ERASE_VERIFIED,
                "sensitive allocation reached deallocation before verified erasure"
            );
            self.state.stage.store(STAGE_DEALLOCATED, Ordering::SeqCst);
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
    enum InjectedFailure {
        #[error("injected lifecycle failure")]
        AfterSensitiveUse,
    }

    fn fail_after_sensitive_use(state: Arc<LifecycleState>) -> Result<(), InjectedFailure> {
        let canary = ZeroizingOwner::new(LifecycleCanary::new(0xa5a5_5a5a, state));
        canary.mark_used();
        Err(InjectedFailure::AfterSensitiveUse)
    }

    struct RecyclingPool {
        buffers: Mutex<Vec<Vec<u32>>>,
    }

    impl RecyclingPool {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                buffers: Mutex::new(Vec::new()),
            })
        }
    }

    struct RecyclingStorage {
        values: Vec<u32>,
        pool: Arc<RecyclingPool>,
        recycle_on_drop: bool,
    }

    impl RecyclingStorage {
        fn take(pool: Arc<RecyclingPool>, marker: u32, len: usize) -> Self {
            let mut values = pool
                .buffers
                .lock()
                .expect("recycling pool lock must not be poisoned")
                .pop()
                .unwrap_or_else(|| vec![0_u32; len]);
            assert_eq!(values.len(), len);
            assert!(values.iter().all(|value| *value == 0));
            values.fill(marker);
            Self {
                values,
                pool,
                recycle_on_drop: true,
            }
        }
    }

    impl Borrow<[u32]> for RecyclingStorage {
        fn borrow(&self) -> &[u32] {
            &self.values
        }
    }

    impl BorrowMut<[u32]> for RecyclingStorage {
        fn borrow_mut(&mut self) -> &mut [u32] {
            &mut self.values
        }
    }

    impl DenseStorage<u32> for RecyclingStorage {
        fn to_vec(mut self) -> Vec<u32> {
            self.recycle_on_drop = false;
            core::mem::take(&mut self.values)
        }
    }

    impl Drop for RecyclingStorage {
        fn drop(&mut self) {
            if !self.recycle_on_drop {
                return;
            }
            assert!(
                self.values.iter().all(|value| *value == 0),
                "recycled storage reached its allocator before erasure"
            );
            let values = core::mem::take(&mut self.values);
            self.pool
                .buffers
                .lock()
                .expect("recycling pool lock must not be poisoned")
                .push(values);
        }
    }

    #[test]
    fn explicit_zeroize_erases_values_and_shape() {
        let mut matrix = ZeroizingMatrix::new(RowMajorMatrix::new(
            vec![0xdead_beef_u32, 0x5245_414c, 0x4c59_4d45, 0x0102_0304],
            2,
        ));

        zeroize::Zeroize::zeroize(&mut matrix);

        assert_eq!(matrix.values, [0_u32; 4]);
        assert_eq!(matrix.width, 0);
    }

    #[test]
    fn guarded_clone_preserves_shape_without_aliasing_owners() {
        let original_values = [1_u8, 2, 3, 4].map(BabyBear::from_u8).to_vec();
        let source = ZeroizingMatrix::new(RowMajorMatrix::new(original_values.clone(), 2));
        let mut duplicate = source
            .try_clone_guarded()
            .expect("a valid guarded matrix must clone into a guarded owner");

        assert_eq!(duplicate.values, original_values);
        assert_eq!(duplicate.width, 2);

        zeroize::Zeroize::zeroize(&mut duplicate);
        assert_eq!(duplicate.values, [BabyBear::ZERO; 4]);
        assert_eq!(duplicate.width, 0);
        assert_eq!(source.values, original_values);
        assert_eq!(source.width, 2);
    }

    #[test]
    fn guarded_clone_rejects_invalid_shape_without_changing_source() {
        let mut malformed = RowMajorMatrix::new(
            [BabyBear::ONE, BabyBear::TWO, BabyBear::from_u8(3)].to_vec(),
            1,
        );
        malformed.width = 2;
        let source = ZeroizingMatrix::new(malformed);

        let result = source.try_clone_guarded();

        assert!(matches!(result, Err(SecretMatrixError::InvalidShape)));
        assert_eq!(
            source.values,
            [BabyBear::ONE, BabyBear::TWO, BabyBear::from_u8(3)]
        );
        assert_eq!(source.width, 2);
    }

    #[test]
    fn guarded_extension_column_flattening_preserves_coefficients() {
        type Extension = BinomialExtensionField<BabyBear, 4>;

        let first = Extension::from_basis_coefficients_fn(|index| match index {
            0 => BabyBear::from_u8(0x52),
            1 => BabyBear::from_u8(0x4d),
            2 => BabyBear::from_u8(0x45),
            _ => BabyBear::from_u8(0x4e),
        });
        let second = Extension::from_basis_coefficients_fn(|index| match index {
            0 => BabyBear::from_u8(0x56),
            1 => BabyBear::from_u8(0x42),
            2 => BabyBear::from_u8(0x31),
            _ => BabyBear::from_u8(0x21),
        });
        let guarded = ZeroizingOwner::new(RowMajorMatrix::new(vec![first, second], 1));

        let mut flattened = guarded
            .try_flatten_extension_to_base::<BabyBear>()
            .expect("a valid fixed-degree extension column must flatten");

        assert_eq!(flattened.width, 4);
        assert_eq!(flattened.height(), 2);
        assert_eq!(
            flattened.values,
            [
                BabyBear::from_u8(0x52),
                BabyBear::from_u8(0x4d),
                BabyBear::from_u8(0x45),
                BabyBear::from_u8(0x4e),
                BabyBear::from_u8(0x56),
                BabyBear::from_u8(0x42),
                BabyBear::from_u8(0x31),
                BabyBear::from_u8(0x21),
            ]
        );

        zeroize::Zeroize::zeroize(&mut flattened);
        assert_eq!(flattened.values, [BabyBear::ZERO; 8]);
        assert_eq!(flattened.width, 0);
    }

    #[test]
    fn drop_erases_borrowed_storage() {
        let mut values = [0xdead_beef_u32, 0x5245_414c, 0x4c59_4d45, 0x0102_0304];
        {
            let view = RowMajorMatrixViewMut::new(&mut values, 2);
            let _secret = ZeroizingMatrix::new(view);
        }

        assert_eq!(values, [0_u32; 4]);
    }

    #[test]
    fn unwind_erases_borrowed_storage() {
        let mut values = [0xdead_beef_u32, 0x5245_414c, 0x4c59_4d45, 0x0102_0304];
        let result = catch_unwind(AssertUnwindSafe(|| {
            let view = RowMajorMatrixViewMut::new(&mut values, 2);
            let _secret = ZeroizingMatrix::new(view);
            panic!("exercise unwind cleanup");
        }));

        assert!(result.is_err());
        assert_eq!(values, [0_u32; 4]);
    }

    #[test]
    fn lifecycle_reaches_verified_erasure_before_deallocation() {
        let state = LifecycleState::sensitive();
        {
            let canary = ZeroizingOwner::new(LifecycleCanary::new(0x0102_0304, state.clone()));
            canary.mark_used();
            assert_eq!(state.stage(), STAGE_USED);
        }
        assert_eq!(state.stage(), STAGE_DEALLOCATED);
    }

    #[test]
    fn lifecycle_rejects_deallocation_without_verified_erasure() {
        let state = LifecycleState::sensitive();
        let drop_state = state.clone();

        let result = catch_unwind(AssertUnwindSafe(|| {
            let canary = LifecycleCanary::new(0x5566_7788, drop_state);
            drop(canary);
        }));

        assert!(result.is_err());
        assert_eq!(state.stage(), STAGE_SENSITIVE);
    }

    #[test]
    fn early_result_error_erases_before_deallocation() {
        let state = LifecycleState::sensitive();

        let result = fail_after_sensitive_use(state.clone());

        assert_eq!(result, Err(InjectedFailure::AfterSensitiveUse));
        assert_eq!(state.stage(), STAGE_DEALLOCATED);
    }

    #[test]
    fn lifecycle_unwind_erases_before_deallocation() {
        let state = LifecycleState::sensitive();
        let unwind_state = state.clone();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let canary = ZeroizingOwner::new(LifecycleCanary::new(0x1122_3344, unwind_state));
            canary.mark_used();
            panic!("exercise lifecycle unwind cleanup");
        }));

        assert!(result.is_err());
        assert_eq!(state.stage(), STAGE_DEALLOCATED);
    }

    #[test]
    fn repeated_sensitive_owners_erase_independently() {
        let states: Vec<_> = [0x1111_1111_u32, 0x2222_2222, 0x3333_3333]
            .into_iter()
            .map(|marker| {
                let state = LifecycleState::sensitive();
                {
                    let canary = ZeroizingOwner::new(LifecycleCanary::new(marker, state.clone()));
                    canary.mark_used();
                }
                state
            })
            .collect();

        assert!(
            states
                .iter()
                .all(|state| state.stage() == STAGE_DEALLOCATED)
        );
    }

    #[test]
    fn parallel_sensitive_owners_erase_without_cross_run_state() {
        let handles: Vec<_> = (1_u32..=8)
            .map(|marker| {
                thread::spawn(move || {
                    let state = LifecycleState::sensitive();
                    {
                        let canary = ZeroizingOwner::new(LifecycleCanary::new(
                            marker.checked_mul(0x0101_0101).expect("bounded marker"),
                            state.clone(),
                        ));
                        canary.mark_used();
                        thread::yield_now();
                    }
                    state
                })
            })
            .collect();

        for handle in handles {
            let state = handle.join().expect("lifecycle worker must not panic");
            assert_eq!(state.stage(), STAGE_DEALLOCATED);
        }
    }

    #[test]
    fn allocator_reuse_only_receives_erased_matrix_storage() {
        let pool = RecyclingPool::new();
        let mut first_pointer: *const u32 = core::ptr::null();

        for marker in [0x1111_1111_u32, 0x2222_2222, 0x3333_3333] {
            let storage = RecyclingStorage::take(pool.clone(), marker, 16);
            let pointer = storage.values.as_ptr();
            if first_pointer.is_null() {
                first_pointer = pointer;
            } else {
                assert_eq!(
                    pointer, first_pointer,
                    "the controlled allocation must be reused"
                );
            }
            let matrix = DenseMatrix::new(storage, 4);
            let owner = ZeroizingMatrix::new(matrix);
            assert!(
                <RecyclingStorage as Borrow<[u32]>>::borrow(&owner.values)
                    .iter()
                    .all(|value| *value == marker)
            );
            drop(owner);

            let buffers = pool
                .buffers
                .lock()
                .expect("recycling pool lock must not be poisoned");
            assert_eq!(buffers.len(), 1);
            assert!(buffers[0].iter().all(|value| *value == 0));
            drop(buffers);
        }
    }

    #[test]
    fn guarded_widening_preserves_values_and_adds_random_columns() {
        let mut rng = StdRng::seed_from_u64(0x5245_414c_4c59_4d45);
        let mut matrix = ZeroizingMatrix::new(RowMajorMatrix::new(
            [1_u8, 2, 3, 4].map(BabyBear::from_u8).to_vec(),
            2,
        ));

        matrix
            .try_widen_with_random_cols(2, &mut rng)
            .expect("valid dimensions must widen");

        assert_eq!(matrix.width, 4);
        assert_eq!(matrix.height(), 2);
        assert_eq!(matrix.values[0], BabyBear::ONE);
        assert_eq!(matrix.values[1], BabyBear::TWO);
        assert_eq!(matrix.values[4], BabyBear::from_u8(3));
        assert_eq!(matrix.values[5], BabyBear::from_u8(4));
    }

    #[test]
    fn guarded_random_matrix_matches_standard_rng_order() {
        const HEIGHT: usize = 8;
        const WIDTH: usize = 3;
        const SEED: u64 = 0x5241_4e44_5f4d_4154;
        let mut ordinary_rng = StdRng::seed_from_u64(SEED);
        let mut guarded_rng = StdRng::seed_from_u64(SEED);
        let ordinary = RowMajorMatrix::<BabyBear>::rand(&mut ordinary_rng, HEIGHT, WIDTH);

        let guarded = ZeroizingMatrix::<RowMajorMatrix<BabyBear>>::try_random(
            HEIGHT,
            WIDTH,
            &mut guarded_rng,
        )
        .expect("valid guarded random matrix allocation must succeed");

        assert_eq!(guarded.width, ordinary.width);
        assert_eq!(guarded.values, ordinary.values);
    }

    #[test]
    fn guarded_random_matrix_rejects_empty_and_overflowing_shapes() {
        let mut rng = StdRng::seed_from_u64(0x5348_4150_455f_4552);

        assert!(matches!(
            ZeroizingMatrix::<RowMajorMatrix<BabyBear>>::try_random(0, 1, &mut rng),
            Err(SecretMatrixError::InvalidShape)
        ));
        assert!(matches!(
            ZeroizingMatrix::<RowMajorMatrix<BabyBear>>::try_random(1, 0, &mut rng),
            Err(SecretMatrixError::InvalidShape)
        ));
        assert!(matches!(
            ZeroizingMatrix::<RowMajorMatrix<BabyBear>>::try_random(usize::MAX, 2, &mut rng),
            Err(SecretMatrixError::DimensionOverflow)
        ));
    }

    #[test]
    fn guarded_widening_rejects_dimension_overflow_without_releasing_owner() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut matrix = ZeroizingMatrix::new(RowMajorMatrix::new(
            [BabyBear::ONE, BabyBear::TWO].to_vec(),
            1,
        ));

        let result = matrix.try_widen_with_random_cols(usize::MAX, &mut rng);

        assert_eq!(result, Err(SecretMatrixError::DimensionOverflow));
        assert_eq!(matrix.values, [BabyBear::ONE, BabyBear::TWO]);
        assert_eq!(matrix.width, 1);
    }

    #[test]
    fn guarded_column_truncation_preserves_left_columns() {
        let mut matrix = ZeroizingMatrix::new(RowMajorMatrix::new(
            [1_u8, 2, 3, 4, 5, 6].map(BabyBear::from_u8).to_vec(),
            3,
        ));

        matrix
            .try_truncate_columns(2)
            .expect("valid guarded column truncation must succeed");

        assert_eq!(matrix.width, 2);
        assert_eq!(matrix.height(), 2);
        assert_eq!(
            matrix.values,
            [
                BabyBear::ONE,
                BabyBear::TWO,
                BabyBear::from_u8(4),
                BabyBear::from_u8(5),
            ]
        );
    }

    #[test]
    fn guarded_column_truncation_rejects_invalid_width_without_releasing_owner() {
        let original_values = [1_u8, 2, 3, 4].map(BabyBear::from_u8).to_vec();
        let mut matrix = ZeroizingMatrix::new(RowMajorMatrix::new(original_values.clone(), 2));

        let result = matrix.try_truncate_columns(0);

        assert_eq!(result, Err(SecretMatrixError::InvalidShape));
        assert_eq!(matrix.values, original_values);
        assert_eq!(matrix.width, 2);
    }

    #[test]
    fn guarded_strided_row_selection_retains_only_selected_rows() {
        let matrix = ZeroizingMatrix::new(RowMajorMatrix::new(
            (1_u8..=12).map(BabyBear::from_u8).collect(),
            2,
        ));

        let selected = matrix
            .try_select_strided_rows(0, 2, 3)
            .expect("valid guarded row selection must succeed");

        assert_eq!(selected.width, 2);
        assert_eq!(selected.height(), 3);
        assert_eq!(
            selected.values,
            [
                BabyBear::ONE,
                BabyBear::TWO,
                BabyBear::from_u8(5),
                BabyBear::from_u8(6),
                BabyBear::from_u8(9),
                BabyBear::from_u8(10),
            ]
        );
    }

    #[test]
    fn guarded_strided_row_selection_rejects_invalid_geometry() {
        let matrix = ZeroizingMatrix::new(RowMajorMatrix::new(
            [1_u8, 2, 3, 4].map(BabyBear::from_u8).to_vec(),
            2,
        ));

        let result = matrix.try_select_strided_rows(0, 2, 2);

        assert!(matches!(result, Err(SecretMatrixError::InvalidShape)));
    }

    #[test]
    fn guarded_row_growth_preserves_values_and_zero_fills_new_rows() {
        let mut matrix = ZeroizingMatrix::new(RowMajorMatrix::new(
            [1_u8, 2, 3, 4].map(BabyBear::from_u8).to_vec(),
            2,
        ));

        matrix
            .try_grow_zeroed_rows(4)
            .expect("valid guarded row growth must succeed");

        assert_eq!(matrix.width, 2);
        assert_eq!(matrix.height(), 4);
        assert_eq!(
            matrix.values,
            [
                BabyBear::ONE,
                BabyBear::TWO,
                BabyBear::from_u8(3),
                BabyBear::from_u8(4),
                BabyBear::ZERO,
                BabyBear::ZERO,
                BabyBear::ZERO,
                BabyBear::ZERO,
            ]
        );
    }

    #[test]
    fn guarded_row_growth_rejects_shrinking_without_releasing_owner() {
        let original_values = [1_u8, 2, 3, 4].map(BabyBear::from_u8).to_vec();
        let mut matrix = ZeroizingMatrix::new(RowMajorMatrix::new(original_values.clone(), 2));

        let result = matrix.try_grow_zeroed_rows(1);

        assert_eq!(result, Err(SecretMatrixError::InvalidShape));
        assert_eq!(matrix.values, original_values);
        assert_eq!(matrix.width, 2);
    }
}
