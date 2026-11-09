// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::array;
use core::cmp::Reverse;
use core::marker::PhantomData;

use p3_commit::{
    SecretDetachedCommitment, SecretProverData, SecretStreamingMmcsCommitment,
    SecretStreamingRecomputableMmcs,
};
use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::{
    CryptographicHasher, IncrementalCryptographicHasher, PseudoCompressionFunction,
};
use serde::{Deserialize, Serialize};

use crate::mmcs::{derive_arity_schedule, validate_commit_reachable_heights};
use crate::{MerkleCap, MerkleTree, MerkleTreeAuthenticationData, MerkleTreeError, MerkleTreeMmcs};

struct LeafGroup<State, W, const DIGEST_ELEMS: usize> {
    height: usize,
    expected_matrices: usize,
    absorbed_matrices: usize,
    last_matrix_index: Option<usize>,
    states: Option<Vec<State>>,
    digests: Option<Vec<[W; DIGEST_ELEMS]>>,
}

/// Builds one canonical mixed-height Merkle tree without retaining input matrices.
///
/// Matrix dimensions are fixed before any secret is absorbed. Each matrix is
/// then supplied exactly once in commitment order. The accumulator retains one
/// zeroizing hash state per row until all matrices at that height have
/// contributed, after which it retains only the finalized row digests.
///
/// The caller remains responsible for erasing each matrix after
/// [`Self::try_absorb_matrix`] returns. The builder guarantees that its retained
/// hash states erase themselves on every drop path.
pub struct IncrementalMerkleTreeBuilder<'a, F, W, H, C, const N: usize, const DIGEST_ELEMS: usize>
where
    F: Clone + Send + Sync,
    W: Copy + Default,
    H: IncrementalCryptographicHasher<F, [W; DIGEST_ELEMS]>,
{
    hash: &'a H,
    compress: &'a C,
    dimensions: Vec<Dimensions>,
    groups: Vec<LeafGroup<H::State, W, DIGEST_ELEMS>>,
    arity_schedule: Vec<usize>,
    absorbed: Vec<bool>,
    absorbed_matrices: usize,
    next_matrix: usize,
    _field: PhantomData<F>,
}

/// Streaming detached commitment for a non-hiding Merkle MMCS.
#[cfg(feature = "secret-memory")]
pub struct IncrementalMerkleCommitment<'a, F, W, H, C, const N: usize, const DIGEST_ELEMS: usize>
where
    F: Clone + Send + Sync,
    W: Copy + Default,
    H: IncrementalCryptographicHasher<F, [W; DIGEST_ELEMS]>,
{
    tree: IncrementalMerkleTreeBuilder<'a, F, W, H, C, N, DIGEST_ELEMS>,
    cap_height: usize,
}

#[cfg(feature = "secret-memory")]
impl<'a, F, W, H, C, const N: usize, const DIGEST_ELEMS: usize>
    IncrementalMerkleCommitment<'a, F, W, H, C, N, DIGEST_ELEMS>
where
    F: Clone + Send + Sync + zeroize::Zeroize,
    W: Copy + Default + Clone + zeroize::Zeroize,
    H: IncrementalCryptographicHasher<F, [W; DIGEST_ELEMS]>,
    C: PseudoCompressionFunction<[W; DIGEST_ELEMS], N>,
{
    fn try_new(
        hash: &'a H,
        compress: &'a C,
        cap_height: usize,
        dimensions: Vec<Dimensions>,
    ) -> Result<Self, MerkleTreeError> {
        Ok(Self {
            tree: IncrementalMerkleTreeBuilder::try_new(hash, compress, dimensions)?,
            cap_height,
        })
    }

    fn try_absorb<M>(&mut self, mut matrix: SecretProverData<M>) -> Result<(), MerkleTreeError>
    where
        M: Matrix<F> + zeroize::Zeroize,
    {
        let result = self.tree.try_absorb_matrix(&*matrix);
        zeroize::Zeroize::zeroize(&mut matrix);
        result
    }

    pub(crate) fn try_absorb_at<M>(
        &mut self,
        matrix_index: usize,
        mut matrix: SecretProverData<M>,
    ) -> Result<(), MerkleTreeError>
    where
        M: Matrix<F> + zeroize::Zeroize,
    {
        let result = self.tree.try_absorb_matrix_at(matrix_index, &*matrix);
        zeroize::Zeroize::zeroize(&mut matrix);
        result
    }

    #[allow(
        clippy::type_complexity,
        reason = "the detached MMCS output keeps its matrix type linked to the empty data shell"
    )]
    fn try_finish<M>(
        self,
    ) -> Result<
        SecretDetachedCommitment<
            MerkleCap<F, [W; DIGEST_ELEMS]>,
            MerkleTree<F, W, SecretProverData<M>, N, DIGEST_ELEMS>,
            MerkleTreeAuthenticationData<F, W, N, DIGEST_ELEMS>,
        >,
        MerkleTreeError,
    >
    where
        M: Matrix<F> + zeroize::Zeroize,
    {
        let authentication_data = self.tree.try_finish()?;
        let commitment = authentication_data.try_cap(self.cap_height)?;
        Ok(SecretDetachedCommitment {
            commitment,
            prover_data: SecretProverData::new(MerkleTree::detached_secret_shell()),
            authentication_data,
        })
    }
}

#[cfg(feature = "secret-memory")]
impl<'a, F, W, H, C, M, const N: usize, const DIGEST_ELEMS: usize>
    SecretStreamingMmcsCommitment<F, M>
    for IncrementalMerkleCommitment<'a, F, W, H, C, N, DIGEST_ELEMS>
where
    F: Clone + Send + Sync + zeroize::Zeroize,
    W: Copy + Default + Clone + zeroize::Zeroize,
    H: IncrementalCryptographicHasher<F, [W; DIGEST_ELEMS]>,
    C: PseudoCompressionFunction<[W; DIGEST_ELEMS], N>,
    M: Matrix<F> + zeroize::Zeroize,
{
    type Commitment = MerkleCap<F, [W; DIGEST_ELEMS]>;
    type Data = MerkleTree<F, W, SecretProverData<M>, N, DIGEST_ELEMS>;
    type AuthenticationData = MerkleTreeAuthenticationData<F, W, N, DIGEST_ELEMS>;
    type Error = MerkleTreeError;

    fn absorb(&mut self, matrix: SecretProverData<M>) -> Result<(), Self::Error> {
        self.try_absorb(matrix)
    }

    fn finish(
        self,
    ) -> Result<
        SecretDetachedCommitment<Self::Commitment, Self::Data, Self::AuthenticationData>,
        Self::Error,
    > {
        self.try_finish()
    }
}

#[cfg(feature = "secret-memory")]
impl<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize>
    SecretStreamingRecomputableMmcs<P::Value> for MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>
where
    P: PackedValue,
    P::Value: zeroize::Zeroize,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + IncrementalCryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
        + Sync,
    PW::Value: Copy + Default + Eq + Clone + zeroize::Zeroize,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    type Stream<'a, M>
        = IncrementalMerkleCommitment<'a, P::Value, PW::Value, H, C, N, DIGEST_ELEMS>
    where
        Self: 'a,
        M: Matrix<P::Value> + zeroize::Zeroize;

    fn begin_streaming_commitment<M>(
        &self,
        dimensions: Vec<Dimensions>,
    ) -> Result<Self::Stream<'_, M>, Self::SecretError>
    where
        M: Matrix<P::Value> + zeroize::Zeroize,
    {
        let effective_cap_height = self.effective_cap_height_for_dimensions(&dimensions)?;
        IncrementalMerkleCommitment::try_new(
            self.hash(),
            self.compress(),
            effective_cap_height,
            dimensions,
        )
    }
}

impl<'a, F, W, H, C, const N: usize, const DIGEST_ELEMS: usize>
    IncrementalMerkleTreeBuilder<'a, F, W, H, C, N, DIGEST_ELEMS>
where
    F: Clone + Send + Sync,
    W: Copy + Default,
    H: IncrementalCryptographicHasher<F, [W; DIGEST_ELEMS]>,
    C: PseudoCompressionFunction<[W; DIGEST_ELEMS], N>,
{
    /// Predeclare every matrix shape and initialize the per-row hash states.
    pub fn try_new(
        hash: &'a H,
        compress: &'a C,
        dimensions: Vec<Dimensions>,
    ) -> Result<Self, MerkleTreeError> {
        if N < 2 || !N.is_power_of_two() {
            return Err(MerkleTreeError::InvalidArity { arity: N });
        }
        validate_commit_reachable_heights(dimensions.iter().map(|value| value.height))?;
        let arity_schedule = derive_arity_schedule(&dimensions, N)?;

        let mut heights = Vec::new();
        heights
            .try_reserve_exact(dimensions.len())
            .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
        for dimension in &dimensions {
            if !heights.contains(&dimension.height) {
                heights.push(dimension.height);
            }
        }
        heights.sort_unstable_by_key(|height| Reverse(*height));

        let mut groups = Vec::new();
        groups
            .try_reserve_exact(heights.len())
            .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
        for height in heights {
            let expected_matrices = dimensions
                .iter()
                .filter(|dimension| dimension.height == height)
                .count();
            let mut states = Vec::new();
            states
                .try_reserve_exact(height)
                .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
            for _ in 0..height {
                states.push(hash.begin_hash());
            }
            groups.push(LeafGroup {
                height,
                expected_matrices,
                absorbed_matrices: 0,
                last_matrix_index: None,
                states: Some(states),
                digests: None,
            });
        }

        let mut absorbed = Vec::new();
        absorbed
            .try_reserve_exact(dimensions.len())
            .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
        absorbed.resize(dimensions.len(), false);

        Ok(Self {
            hash,
            compress,
            dimensions,
            groups,
            arity_schedule,
            absorbed,
            absorbed_matrices: 0,
            next_matrix: 0,
            _field: PhantomData,
        })
    }

    /// Absorb one matrix in the exact order declared at construction.
    pub fn try_absorb_matrix<M>(&mut self, matrix: &M) -> Result<(), MerkleTreeError>
    where
        M: Matrix<F>,
    {
        let matrix_index = self.next_matrix;
        self.try_absorb_matrix_at(matrix_index, matrix)?;
        self.next_matrix = matrix_index
            .checked_add(1)
            .ok_or(MerkleTreeError::TreeGeometryOverflow)?;
        Ok(())
    }

    /// Absorb one matrix by its original commitment index.
    ///
    /// Detached commitments use this entry point to process complete height
    /// groups from tallest to shortest while preserving the original matrix
    /// order inside each group. That lets each group be erased before the next
    /// group is read without changing leaf-hash concatenation order.
    pub fn try_absorb_matrix_at<M>(
        &mut self,
        matrix_index: usize,
        matrix: &M,
    ) -> Result<(), MerkleTreeError>
    where
        M: Matrix<F>,
    {
        let expected = self.dimensions.get(matrix_index).ok_or(
            MerkleTreeError::StreamedMatrixCountMismatch {
                expected: self.dimensions.len(),
                got: matrix_index
                    .checked_add(1)
                    .ok_or(MerkleTreeError::TreeGeometryOverflow)?,
            },
        )?;
        let actual = matrix.dimensions();
        if actual != *expected {
            return Err(MerkleTreeError::StreamedMatrixShapeMismatch {
                matrix: matrix_index,
                expected_width: expected.width,
                expected_height: expected.height,
                got_width: actual.width,
                got_height: actual.height,
            });
        }

        let absorbed = self
            .absorbed
            .get_mut(matrix_index)
            .ok_or(MerkleTreeError::MalformedAuthenticationData)?;
        if *absorbed {
            return Err(MerkleTreeError::MalformedAuthenticationData);
        }

        let group = self
            .groups
            .iter_mut()
            .find(|group| group.height == expected.height)
            .ok_or(MerkleTreeError::MalformedAuthenticationData)?;
        if group
            .last_matrix_index
            .is_some_and(|last_index| matrix_index <= last_index)
        {
            return Err(MerkleTreeError::MalformedAuthenticationData);
        }
        let states = group
            .states
            .as_mut()
            .ok_or(MerkleTreeError::MalformedAuthenticationData)?;
        for row_index in 0..expected.height {
            let state = states
                .get_mut(row_index)
                .ok_or(MerkleTreeError::MalformedAuthenticationData)?;
            let row = matrix
                .row(row_index)
                .ok_or(MerkleTreeError::StreamedRowUnavailable {
                    matrix: matrix_index,
                    row: row_index,
                })?;
            self.hash.absorb(state, row);
        }

        group.absorbed_matrices = group
            .absorbed_matrices
            .checked_add(1)
            .ok_or(MerkleTreeError::TreeGeometryOverflow)?;
        if group.absorbed_matrices > group.expected_matrices {
            return Err(MerkleTreeError::MalformedAuthenticationData);
        }
        if group.absorbed_matrices == group.expected_matrices {
            let states = group
                .states
                .take()
                .ok_or(MerkleTreeError::MalformedAuthenticationData)?;
            let padded_height = try_padded_len(group.height, N)?;
            let mut digests = try_default_digest_layer::<W, DIGEST_ELEMS>(padded_height)?;
            for (digest, state) in digests.iter_mut().zip(states) {
                *digest = self.hash.finalize_hash(state);
            }
            group.digests = Some(digests);
        }

        group.last_matrix_index = Some(matrix_index);
        *absorbed = true;
        self.absorbed_matrices = self
            .absorbed_matrices
            .checked_add(1)
            .ok_or(MerkleTreeError::TreeGeometryOverflow)?;
        Ok(())
    }

    /// Finish the tree after every predeclared matrix has been absorbed.
    pub fn try_finish(
        mut self,
    ) -> Result<MerkleTreeAuthenticationData<F, W, N, DIGEST_ELEMS>, MerkleTreeError> {
        if self.absorbed_matrices != self.dimensions.len()
            || self.absorbed.iter().any(|absorbed| !absorbed)
        {
            return Err(MerkleTreeError::StreamedMatrixCountMismatch {
                expected: self.dimensions.len(),
                got: self.absorbed_matrices,
            });
        }

        let layer_capacity = self
            .arity_schedule
            .len()
            .checked_add(1)
            .ok_or(MerkleTreeError::TreeGeometryOverflow)?;
        let mut digest_layers = Vec::new();
        digest_layers
            .try_reserve_exact(layer_capacity)
            .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;

        let first_group = self
            .groups
            .first_mut()
            .ok_or(MerkleTreeError::MalformedAuthenticationData)?;
        let first_layer = first_group
            .digests
            .take()
            .ok_or(MerkleTreeError::MalformedAuthenticationData)?;
        digest_layers.push(first_layer);

        for step in &self.arity_schedule {
            let previous = digest_layers
                .last()
                .ok_or(MerkleTreeError::MalformedAuthenticationData)?;
            if *step == 0 || *step > N || !previous.len().is_multiple_of(*step) {
                return Err(MerkleTreeError::MalformedAuthenticationData);
            }
            let raw_next = previous
                .len()
                .checked_div(*step)
                .ok_or(MerkleTreeError::TreeGeometryOverflow)?;
            let injection_height = raw_next
                .checked_next_power_of_two()
                .ok_or(MerkleTreeError::TreeGeometryOverflow)?;
            let injection = self
                .groups
                .iter_mut()
                .skip(1)
                .find(|group| {
                    group.digests.is_some()
                        && group.height.checked_next_power_of_two() == Some(injection_height)
                })
                .and_then(|group| group.digests.take());
            let next = try_compress_and_inject::<W, C, N, DIGEST_ELEMS>(
                previous,
                *step,
                injection.as_deref(),
                self.compress,
            )?;
            digest_layers.push(next);
        }

        if self.groups.iter().any(|group| group.digests.is_some())
            || digest_layers
                .last()
                .is_none_or(|root_layer| root_layer.len() != 1)
        {
            return Err(MerkleTreeError::MalformedAuthenticationData);
        }

        Ok(MerkleTreeAuthenticationData {
            dimensions: self.dimensions,
            digest_layers,
            arity_schedule: self.arity_schedule,
            _phantom: PhantomData,
        })
    }
}

fn try_padded_len(raw_len: usize, arity: usize) -> Result<usize, MerkleTreeError> {
    if raw_len <= 1 {
        return Ok(raw_len);
    }
    if raw_len < arity {
        return Ok(arity);
    }
    raw_len
        .checked_add(
            arity
                .checked_sub(1)
                .ok_or(MerkleTreeError::InvalidArity { arity })?,
        )
        .and_then(|value| value.checked_div(arity))
        .and_then(|value| value.checked_mul(arity))
        .ok_or(MerkleTreeError::TreeGeometryOverflow)
}

fn try_default_digest_layer<W, const DIGEST_ELEMS: usize>(
    len: usize,
) -> Result<Vec<[W; DIGEST_ELEMS]>, MerkleTreeError>
where
    W: Copy + Default,
{
    let mut layer = Vec::new();
    layer
        .try_reserve_exact(len)
        .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
    layer.resize(len, [W::default(); DIGEST_ELEMS]);
    Ok(layer)
}

fn try_compress_and_inject<W, C, const N: usize, const DIGEST_ELEMS: usize>(
    previous: &[[W; DIGEST_ELEMS]],
    step: usize,
    injection: Option<&[[W; DIGEST_ELEMS]]>,
    compress: &C,
) -> Result<Vec<[W; DIGEST_ELEMS]>, MerkleTreeError>
where
    W: Copy + Default,
    C: PseudoCompressionFunction<[W; DIGEST_ELEMS], N>,
{
    if step == 0 || step > N || !previous.len().is_multiple_of(step) {
        return Err(MerkleTreeError::MalformedAuthenticationData);
    }
    let raw_next = previous
        .len()
        .checked_div(step)
        .ok_or(MerkleTreeError::TreeGeometryOverflow)?;
    let padded_next = try_padded_len(raw_next, N)?;
    let default_digest = [W::default(); DIGEST_ELEMS];
    let mut next = try_default_digest_layer::<W, DIGEST_ELEMS>(padded_next)?;

    for (node_index, output) in next.iter_mut().take(raw_next).enumerate() {
        let first_child = node_index
            .checked_mul(step)
            .ok_or(MerkleTreeError::TreeGeometryOverflow)?;
        let mut children = [default_digest; N];
        for (child_index, child) in children.iter_mut().take(step).enumerate() {
            let source_index = first_child
                .checked_add(child_index)
                .ok_or(MerkleTreeError::TreeGeometryOverflow)?;
            *child = *previous
                .get(source_index)
                .ok_or(MerkleTreeError::MalformedAuthenticationData)?;
        }
        let folded = compress.compress(children);
        *output = injection.map_or(folded, |injection| {
            let row_digest = injection.get(node_index).copied().unwrap_or(default_digest);
            let inputs = array::from_fn(|index| match index {
                0 => folded,
                1 => row_digest,
                _ => default_digest,
            });
            compress.compress(inputs)
        });
    }
    Ok(next)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{
        CryptographicHasher, IncrementalCryptographicHasher, PseudoCompressionFunction,
    };
    use zeroize::Zeroizing;

    use super::IncrementalMerkleTreeBuilder;
    use crate::MerkleTree;

    const DIGEST_ELEMS: usize = 2;

    #[derive(Clone, Copy)]
    struct TestHasher;

    impl CryptographicHasher<BabyBear, [BabyBear; DIGEST_ELEMS]> for TestHasher {
        fn hash_iter<I>(&self, input: I) -> [BabyBear; DIGEST_ELEMS]
        where
            I: IntoIterator<Item = BabyBear>,
        {
            let mut state = self.begin_hash();
            self.absorb(&mut state, input);
            self.finalize_hash(state)
        }
    }

    impl IncrementalCryptographicHasher<BabyBear, [BabyBear; DIGEST_ELEMS]> for TestHasher {
        type State = Zeroizing<[BabyBear; 3]>;

        fn begin_hash(&self) -> Self::State {
            Zeroizing::new([BabyBear::ZERO; 3])
        }

        fn absorb<I>(&self, state: &mut Self::State, input: I)
        where
            I: IntoIterator<Item = BabyBear>,
        {
            for value in input {
                state[0] += value;
                state[1] = state[1] * BabyBear::from_u8(7) + value;
                state[2] += BabyBear::ONE;
            }
        }

        fn finalize_hash(&self, state: Self::State) -> [BabyBear; DIGEST_ELEMS] {
            [state[0] + state[2], state[1]]
        }
    }

    #[derive(Clone, Copy)]
    struct TestCompress;

    impl PseudoCompressionFunction<[BabyBear; DIGEST_ELEMS], 2> for TestCompress {
        fn compress(&self, input: [[BabyBear; DIGEST_ELEMS]; 2]) -> [BabyBear; DIGEST_ELEMS] {
            [
                input[0][0] + input[1][0] * BabyBear::from_u8(11),
                input[0][1] * BabyBear::from_u8(13) + input[1][1],
            ]
        }
    }

    fn matrix(height: usize, width: usize, seed: u8) -> RowMajorMatrix<BabyBear> {
        let values = (0..height * width)
            .map(|index| BabyBear::from_u8(seed.wrapping_add(index as u8)))
            .collect();
        RowMajorMatrix::new(values, width)
    }

    #[test]
    fn incremental_builder_matches_canonical_mixed_height_root() {
        let matrices = vec![
            matrix(3, 2, 1),
            matrix(9, 1, 11),
            matrix(5, 3, 21),
            matrix(9, 2, 31),
            matrix(1, 4, 41),
            matrix(3, 1, 51),
        ];
        let dimensions = matrices.iter().map(Matrix::dimensions).collect();
        let canonical = MerkleTree::<BabyBear, BabyBear, _, 2, DIGEST_ELEMS>::new::<
            BabyBear,
            BabyBear,
            TestHasher,
            TestCompress,
        >(&TestHasher, &TestCompress, matrices.clone());

        let mut builder = IncrementalMerkleTreeBuilder::<
            BabyBear,
            BabyBear,
            TestHasher,
            TestCompress,
            2,
            DIGEST_ELEMS,
        >::try_new(&TestHasher, &TestCompress, dimensions)
        .expect("valid dimensions");
        for matrix in &matrices {
            builder.try_absorb_matrix(matrix).expect("valid matrix");
        }
        let streamed = builder.try_finish().expect("complete stream");

        assert_eq!(streamed.digest_layers, canonical.digest_layers);
        assert_eq!(streamed.arity_schedule, canonical.arity_schedule);
    }

    #[test]
    fn incremental_builder_rejects_shape_drift_before_absorption() {
        let declared = matrix(5, 2, 1);
        let wrong = matrix(5, 3, 1);
        let mut builder =
            IncrementalMerkleTreeBuilder::<
                BabyBear,
                BabyBear,
                TestHasher,
                TestCompress,
                2,
                DIGEST_ELEMS,
            >::try_new(&TestHasher, &TestCompress, vec![declared.dimensions()])
            .expect("valid dimensions");

        assert!(builder.try_absorb_matrix(&wrong).is_err());
    }

    #[test]
    fn incremental_builder_rejects_incomplete_stream() {
        let first = matrix(5, 2, 1);
        let second = matrix(3, 2, 2);
        let dimensions = vec![first.dimensions(), second.dimensions()];
        let mut builder = IncrementalMerkleTreeBuilder::<
            BabyBear,
            BabyBear,
            TestHasher,
            TestCompress,
            2,
            DIGEST_ELEMS,
        >::try_new(&TestHasher, &TestCompress, dimensions)
        .expect("valid dimensions");
        builder.try_absorb_matrix(&first).expect("valid matrix");

        assert!(builder.try_finish().is_err());
    }
}
