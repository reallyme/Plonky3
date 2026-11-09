// Modified by ReallyMe Technologies Limited, 2026.
// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
#[cfg(feature = "secret-memory")]
use core::cmp::Reverse;

use itertools::Itertools;
use p3_commit::{BatchOpening, BatchOpeningRef, HidingMmcs, Mmcs};
#[cfg(feature = "secret-memory")]
use p3_commit::{SecretDetachedCommitment, SecretMmcs, SecretProverData, SecretRecomputableMmcs};
#[cfg(feature = "secret-memory")]
use p3_commit::{SecretStreamingMmcsCommitment, SecretStreamingRecomputableMmcs};
use p3_field::PackedValue;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::stack::HorizontalPair;
use p3_matrix::{Dimensions, Matrix};
#[cfg(feature = "secret-memory")]
use p3_symmetric::IncrementalCryptographicHasher;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
#[cfg(feature = "secret-memory")]
use p3_util::log2_ceil_usize;
use p3_util::zip_eq::zip_eq;
#[cfg(feature = "secret-memory")]
use rand::RngExt;
use rand::distr::{Distribution, StandardUniform};
use rand::{CryptoRng, SeedableRng};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use spin::Mutex;

use crate::mmcs::check_widths;
#[cfg(feature = "secret-memory")]
use crate::mmcs::validate_commit_reachable_heights;
use crate::pruning::PrunedMerklePaths;
#[cfg(feature = "secret-memory")]
use crate::{IncrementalMerkleTreeBuilder, MerkleTreeAuthenticationData};
use crate::{MerkleCap, MerkleTree, MerkleTreeError, MerkleTreeMmcs};

#[cfg(feature = "secret-memory")]
type GuardedHidingLeaf<F, M> =
    HorizontalPair<SecretProverData<M>, SecretProverData<RowMajorMatrix<F>>>;

#[cfg(feature = "secret-memory")]
type GuardedHidingTree<F, W, M, const N: usize, const DIGEST_ELEMS: usize> =
    MerkleTree<F, W, GuardedHidingLeaf<F, M>, N, DIGEST_ELEMS>;

#[cfg(feature = "secret-memory")]
type SplitRows<F> = (Vec<Vec<F>>, Vec<Vec<F>>);

#[cfg(feature = "secret-memory")]
type HidingMultiOpening<F, W, const DIGEST_ELEMS: usize> = (
    Vec<Vec<Vec<F>>>,
    (Vec<Vec<Vec<F>>>, PrunedMerklePaths<W, DIGEST_ELEMS>),
);

#[cfg(feature = "secret-memory")]
struct BorrowedMatrix<'a, M>(&'a M);

#[cfg(feature = "secret-memory")]
impl<T, M> Matrix<T> for BorrowedMatrix<'_, M>
where
    T: Send + Sync + Clone,
    M: Matrix<T>,
{
    fn width(&self) -> usize {
        self.0.width()
    }

    fn height(&self) -> usize {
        self.0.height()
    }

    unsafe fn row_unchecked(
        &self,
        row: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        // SAFETY: This adapter preserves the wrapped matrix's dimensions, so
        // the caller's row bound is exactly the delegated method's bound.
        unsafe { self.0.row_unchecked(row) }
    }
}

/// In-progress hiding commitment that releases each value matrix immediately
/// after its rows have entered the leaf hash states.
///
/// The complete unsalted geometry and all hiding salts are established before
/// the first secret matrix is accepted. Invalid public geometry therefore does
/// not advance the RNG, and salt generation cannot stop after only a prefix of
/// the matrices. Only salts and zeroizing incremental hash states remain live
/// between calls to [`Self::try_absorb_matrix`].
#[cfg(feature = "secret-memory")]
pub struct IncrementalHidingMerkleCommitment<
    'a,
    F,
    W,
    H,
    C,
    const N: usize,
    const DIGEST_ELEMS: usize,
> where
    F: Clone + Send + Sync + zeroize::Zeroize,
    W: Copy + Default,
    H: IncrementalCryptographicHasher<F, [W; DIGEST_ELEMS]>,
{
    tree: IncrementalMerkleTreeBuilder<'a, F, W, H, C, N, DIGEST_ELEMS>,
    salts: SecretProverData<Vec<SecretProverData<RowMajorMatrix<F>>>>,
    dimensions: Vec<Dimensions>,
    cap_height: usize,
    next_matrix: usize,
}

#[cfg(feature = "secret-memory")]
impl<'a, F, W, H, C, const N: usize, const DIGEST_ELEMS: usize>
    IncrementalHidingMerkleCommitment<'a, F, W, H, C, N, DIGEST_ELEMS>
where
    F: Clone + Send + Sync + zeroize::Zeroize,
    W: Copy + Default + Clone + zeroize::Zeroize,
    H: IncrementalCryptographicHasher<F, [W; DIGEST_ELEMS]>,
    C: PseudoCompressionFunction<[W; DIGEST_ELEMS], N>,
{
    /// Absorb and then erase the next value matrix in commitment order.
    pub fn try_absorb_matrix<M>(
        &mut self,
        mut matrix: SecretProverData<M>,
    ) -> Result<(), MerkleTreeError>
    where
        M: Matrix<F> + zeroize::Zeroize,
    {
        let matrix_index = self.next_matrix;
        let result = self.try_absorb_matrix_inner(matrix_index, &matrix);
        zeroize::Zeroize::zeroize(&mut matrix);
        if result.is_ok() {
            self.next_matrix = matrix_index
                .checked_add(1)
                .ok_or(MerkleTreeError::TreeGeometryOverflow)?;
        }
        result
    }

    pub(crate) fn try_absorb_matrix_at<M>(
        &mut self,
        matrix_index: usize,
        mut matrix: SecretProverData<M>,
    ) -> Result<(), MerkleTreeError>
    where
        M: Matrix<F> + zeroize::Zeroize,
    {
        let result = self.try_absorb_matrix_inner(matrix_index, &matrix);
        zeroize::Zeroize::zeroize(&mut matrix);
        result
    }

    fn try_absorb_matrix_inner<M>(
        &mut self,
        matrix_index: usize,
        matrix: &M,
    ) -> Result<(), MerkleTreeError>
    where
        M: Matrix<F> + zeroize::Zeroize,
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

        let salt = self
            .salts
            .get(matrix_index)
            .ok_or(MerkleTreeError::MalformedAuthenticationData)?;
        let salted = HorizontalPair::new(BorrowedMatrix(matrix), BorrowedMatrix(&**salt));
        self.tree.try_absorb_matrix_at(matrix_index, &salted)
    }

    /// Finish the canonical tree and return authentication-only prover state.
    #[allow(
        clippy::type_complexity,
        reason = "the detached MMCS output keeps its matrix type linked to the empty data shell"
    )]
    pub fn try_finish<M>(
        self,
    ) -> Result<
        SecretDetachedCommitment<
            MerkleCap<F, [W; DIGEST_ELEMS]>,
            GuardedHidingTree<F, W, M, N, DIGEST_ELEMS>,
            HidingMerkleAuthenticationData<F, W, N, DIGEST_ELEMS>,
        >,
        MerkleTreeError,
    >
    where
        M: Matrix<F> + zeroize::Zeroize,
    {
        let inner = self.tree.try_finish()?;
        let commitment = inner.try_cap(self.cap_height)?;
        Ok(SecretDetachedCommitment {
            commitment,
            prover_data: SecretProverData::new(MerkleTree::detached_secret_shell()),
            authentication_data: HidingMerkleAuthenticationData {
                inner,
                salts: self.salts,
                dimensions: self.dimensions,
            },
        })
    }
}

#[cfg(feature = "secret-memory")]
impl<'a, F, W, H, C, M, const N: usize, const DIGEST_ELEMS: usize>
    SecretStreamingMmcsCommitment<F, M>
    for IncrementalHidingMerkleCommitment<'a, F, W, H, C, N, DIGEST_ELEMS>
where
    F: Clone + Send + Sync + zeroize::Zeroize,
    W: Copy + Default + Clone + zeroize::Zeroize,
    H: IncrementalCryptographicHasher<F, [W; DIGEST_ELEMS]>,
    C: PseudoCompressionFunction<[W; DIGEST_ELEMS], N>,
    M: Matrix<F> + zeroize::Zeroize,
{
    type Commitment = MerkleCap<F, [W; DIGEST_ELEMS]>;
    type Data = GuardedHidingTree<F, W, M, N, DIGEST_ELEMS>;
    type AuthenticationData = HidingMerkleAuthenticationData<F, W, N, DIGEST_ELEMS>;
    type Error = MerkleTreeError;

    fn absorb(&mut self, matrix: SecretProverData<M>) -> Result<(), Self::Error> {
        self.try_absorb_matrix(matrix)
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

/// A vector commitment scheme backed by a `MerkleTree`.
///
/// This is similar to `MerkleTreeMmcs`, but each leaf is "salted" with random elements. This is
/// done to turn the Merkle tree into a hiding commitment. See e.g. Section 3 of
/// [Interactive Oracle Proofs](https://eprint.iacr.org/2016/116).
///
/// `SALT_ELEMS` should be set such that the product of `SALT_ELEMS` with the size of the value
/// (`P::Value`) is at least the target security parameter.
///
/// `R` is bounded by [`CryptoRng`], which rules out generators known to be unsuitable for
/// cryptographic use. The bound does not replace proper seeding: the salts only hide the leaves
/// if the generator is also seeded from unpredictable entropy, so production callers should seed
/// from the operating system.
///
/// Generics:
/// - `P`: a leaf value
/// - `PW`: an element of a digest
/// - `H`: the leaf hasher
/// - `C`: the digest compression function
/// - `R`: a random number generator for blinding leaves
#[derive(Debug)]
pub struct MerkleTreeHidingMmcs<
    P,
    PW,
    H,
    C,
    R,
    const N: usize,
    const DIGEST_ELEMS: usize,
    const SALT_ELEMS: usize,
> {
    inner: MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>,
    rng: Mutex<R>,
}

/// Merkle authentication state plus the still-secret salt matrices needed to
/// reproduce hiding openings after committed value matrices are erased.
///
/// The salt owner is neither cloneable nor serializable and zeroizes every salt
/// matrix on drop. Only salts at verifier-selected query rows are released into
/// a proof.
#[cfg(feature = "secret-memory")]
pub struct HidingMerkleAuthenticationData<
    F: zeroize::Zeroize,
    W,
    const N: usize,
    const DIGEST_ELEMS: usize,
> {
    inner: MerkleTreeAuthenticationData<F, W, N, DIGEST_ELEMS>,
    salts: SecretProverData<Vec<SecretProverData<RowMajorMatrix<F>>>>,
    dimensions: Vec<Dimensions>,
}

#[cfg(feature = "secret-memory")]
impl<F: zeroize::Zeroize, W, const N: usize, const DIGEST_ELEMS: usize>
    HidingMerkleAuthenticationData<F, W, N, DIGEST_ELEMS>
{
    /// Unsalted matrix shapes authenticated by this commitment.
    #[must_use]
    pub fn dimensions(&self) -> &[Dimensions] {
        &self.dimensions
    }
}

#[cfg(feature = "secret-memory")]
impl<F, W, const N: usize, const DIGEST_ELEMS: usize> zeroize::Zeroize
    for HidingMerkleAuthenticationData<F, W, N, DIGEST_ELEMS>
where
    F: zeroize::Zeroize,
    W: zeroize::Zeroize,
{
    fn zeroize(&mut self) {
        self.inner.zeroize();
        self.salts.zeroize();
        self.dimensions.clear();
    }
}

impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    MerkleTreeHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
{
    /// Create a new `MerkleTreeHidingMmcs` with the given hash and compression functions.
    ///
    /// # Arguments
    /// * `hash` - The hash function used to hash individual matrix rows (leaf level).
    /// * `compress` - The compression function used to hash internal tree nodes.
    /// * `cap_height` - The maximum number of layers below the root for the cap.
    ///   A height of 0 uses only the root. Mixed-height batches may use a smaller
    ///   height so the cap includes every matrix injection.
    /// * `rng` - A random number generator for generating salts.
    ///
    /// An empty salt hides nothing, so it fails to compile.
    ///
    /// Hiding is only as strong as the salt entropy.
    pub const fn new(hash: H, compress: C, cap_height: usize, rng: R) -> Self {
        const {
            assert!(
                SALT_ELEMS > 0,
                "a hiding MMCS needs at least one salt element per leaf"
            );
        }
        let inner = MerkleTreeMmcs::new(hash, compress, cap_height);
        Self {
            inner,
            rng: Mutex::new(rng),
        }
    }

    pub const fn cap_height(&self) -> usize {
        self.inner.cap_height()
    }
}

#[cfg(feature = "secret-memory")]
impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    MerkleTreeHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
where
    P: PackedValue,
    P::Value: Default + zeroize::Zeroize,
    PW: PackedValue,
    PW::Value: Copy + Default + Clone + zeroize::Zeroize,
    H: IncrementalCryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>,
    R: CryptoRng + SeedableRng + Send,
    StandardUniform: Distribution<P::Value>,
{
    /// Start a hiding commitment whose value matrices can be generated,
    /// absorbed, and erased one at a time.
    ///
    /// Every shape is validated and every required allocation succeeds before
    /// the RNG is advanced. Salts are then generated in the same matrix, row,
    /// and column order as [`SecretRecomputableMmcs::commit_detached_secret`].
    pub fn try_begin_incremental_secret_commitment(
        &self,
        dimensions: Vec<Dimensions>,
    ) -> Result<
        IncrementalHidingMerkleCommitment<'_, P::Value, PW::Value, H, C, N, DIGEST_ELEMS>,
        MerkleTreeError,
    > {
        if SALT_ELEMS == 0 {
            return Err(MerkleTreeError::SecretInvalidSaltWidth);
        }
        validate_commit_reachable_heights(dimensions.iter().map(|value| value.height))?;

        let mut salted_dimensions = Vec::new();
        salted_dimensions
            .try_reserve_exact(dimensions.len())
            .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
        let mut salt_lengths = Vec::new();
        salt_lengths
            .try_reserve_exact(dimensions.len())
            .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
        for dimension in &dimensions {
            let width = dimension
                .width
                .checked_add(SALT_ELEMS)
                .ok_or(MerkleTreeError::SecretDimensionOverflow)?;
            let salt_len = dimension
                .height
                .checked_mul(SALT_ELEMS)
                .ok_or(MerkleTreeError::SecretDimensionOverflow)?;
            salted_dimensions.push(Dimensions {
                width,
                height: dimension.height,
            });
            salt_lengths.push(salt_len);
        }

        // Complete geometry validation plus the state and salt allocations
        // needed before absorption prior to consuming randomness. A failed
        // preflight therefore leaves the RNG unchanged.
        let effective_cap_height = self
            .inner
            .effective_cap_height_for_dimensions(&salted_dimensions)?;
        let tree = IncrementalMerkleTreeBuilder::try_new(
            self.inner.hash(),
            self.inner.compress(),
            salted_dimensions,
        )?;
        let mut salts = SecretProverData::new(Vec::new());
        salts
            .try_reserve_exact(dimensions.len())
            .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
        for salt_len in salt_lengths {
            let mut values = Vec::new();
            values
                .try_reserve_exact(salt_len)
                .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
            values.resize(salt_len, P::Value::default());
            salts.push(SecretProverData::new(RowMajorMatrix::new(
                values, SALT_ELEMS,
            )));
        }

        {
            let mut rng = self.rng.lock();
            for salt_matrix in &mut *salts {
                for salt in &mut salt_matrix.values {
                    *salt = rng.random();
                }
            }
        }

        Ok(IncrementalHidingMerkleCommitment {
            tree,
            salts,
            dimensions,
            cap_height: effective_cap_height,
            next_matrix: 0,
        })
    }
}

/// Cloning forks the RNG stream by drawing a fresh seed from the source RNG, so the clone and the
/// original never salt two commitments with the same sequence. This mirrors `HidingFriPcs`, and it
/// is also what lets `R` be a generator that deliberately withholds `Clone`.
impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize> Clone
    for MerkleTreeHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
where
    MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>: Clone,
    R: CryptoRng + SeedableRng,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            rng: Mutex::new(R::from_rng(&mut *self.rng.lock())),
        }
    }
}

impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    HidingMmcs<P::Value> for MerkleTreeHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
where
    P: PackedValue,
    P::Value: Serialize + DeserializeOwned,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
        + Sync,
    R: CryptoRng + SeedableRng + Send,
    PW::Value: Eq + Clone,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<P::Value>,
{
}

impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    Mmcs<P::Value> for MerkleTreeHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
where
    P: PackedValue,
    P::Value: Serialize + DeserializeOwned,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
        + Sync,
    R: CryptoRng + SeedableRng + Send,
    PW::Value: Eq + Clone,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<P::Value>,
{
    type ProverData<M> = MerkleTree<
        P::Value,
        PW::Value,
        HorizontalPair<M, RowMajorMatrix<P::Value>>,
        N,
        DIGEST_ELEMS,
    >;
    type Commitment = MerkleCap<P::Value, [PW::Value; DIGEST_ELEMS]>;
    /// The first item is salts; the second is the usual Merkle proof (sibling digests).
    type Proof = (Vec<Vec<P::Value>>, Vec<[PW::Value; DIGEST_ELEMS]>);
    /// The first item is the per-query salts (`salts[q][m]`).
    /// The second is the shared pruned multiproof.
    type MultiProof = (
        Vec<Vec<Vec<P::Value>>>,
        PrunedMerklePaths<PW::Value, DIGEST_ELEMS>,
    );
    type Error = MerkleTreeError;

    fn commit<M: Matrix<P::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        // Draw every salt under the lock, then release it before hashing.
        //
        // Why: tree building is parallel.
        // A waiting rayon worker may run a queued commit on this same MMCS.
        // That commit would spin forever on a lock its own thread still holds.
        let salted_inputs = {
            let mut rng = self.rng.lock();
            inputs
                .into_iter()
                .map(|mat| {
                    let salts = RowMajorMatrix::rand(&mut *rng, mat.height(), SALT_ELEMS);
                    HorizontalPair::new(mat, salts)
                })
                .collect()
        };
        self.inner.commit(salted_inputs)
    }

    fn open_batch<M: Matrix<P::Value>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<P::Value, Self> {
        let (salted_openings, siblings) = self.inner.open_batch(index, prover_data).unpack();
        let (openings, salts) = salted_openings
            .into_iter()
            .map(|row| {
                let (a, b) = row.split_at(row.len() - SALT_ELEMS);
                (a.to_vec(), b.to_vec())
            })
            .unzip();
        BatchOpening::new(openings, (salts, siblings))
    }

    fn get_matrices<'a, M: Matrix<P::Value>>(
        &self,
        prover_data: &'a Self::ProverData<M>,
    ) -> Vec<&'a M> {
        prover_data.leaves.iter().map(|mat| &mat.left).collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, P::Value, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, (salts, siblings)) = batch_opening.unpack();
        self.verify_batch_borrowed(commit, dimensions, index, opened_values, salts, siblings)
    }

    fn open_multi_batch<M: Matrix<P::Value>>(
        &self,
        indices: &[usize],
        prover_data: &Self::ProverData<M>,
    ) -> (Vec<Vec<Vec<P::Value>>>, Self::MultiProof) {
        // The inner tree opens salted rows: each row is `real_row ++ salt`.
        let (salted_values, pruned) = self.inner.open_multi_batch(indices, prover_data);

        // Split every salted row back into its unsalted prefix and its salt suffix.
        let (opened_values, salts): (Vec<_>, Vec<_>) = salted_values
            .into_iter()
            .map(|rows_at_index| {
                rows_at_index
                    .into_iter()
                    .map(|row| {
                        let (a, b) = row.split_at(row.len() - SALT_ELEMS);
                        (a.to_vec(), b.to_vec())
                    })
                    .unzip()
            })
            .unzip();

        (opened_values, (salts, pruned))
    }

    fn verify_multi_batch<Row: AsRef<[P::Value]> + PartialEq>(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
        opened_values: &[Vec<Row>],
        proof: &Self::MultiProof,
    ) -> Result<(), Self::Error> {
        let (salts, pruned) = proof;

        // Re-attach salts query by query, mirroring the single-opening path.
        let mut salted_values = Vec::with_capacity(opened_values.len());
        let expected_salt_batches = opened_values.len();
        let got_salt_batches = salts.len();
        for (rows, salts_at_index) in zip_eq(
            opened_values,
            salts,
            MerkleTreeError::WrongBatchSize {
                expected: expected_salt_batches,
                got: got_salt_batches,
            },
        )? {
            // Pin each unsalted row to its matrix width before salting.
            // The inner tree only sees salted widths.
            // An over-long row could otherwise hide behind an under-long salt.
            check_widths(dimensions, rows)?;

            let expected_salts = rows.len();
            let got_salts = salts_at_index.len();
            let salted_rows = zip_eq(
                rows,
                salts_at_index,
                MerkleTreeError::WrongBatchSize {
                    expected: expected_salts,
                    got: got_salts,
                },
            )?
            .map(|(opened, salt)| {
                opened
                    .as_ref()
                    .iter()
                    .chain(salt.iter())
                    .copied()
                    .collect_vec()
            })
            .collect_vec();
            salted_values.push(salted_rows);
        }

        // The inner tree commits to salt-widened rows, so widen the dimensions to match.
        let salted_dimensions = dimensions
            .iter()
            .map(|dims| Dimensions {
                width: dims.width + SALT_ELEMS,
                height: dims.height,
            })
            .collect_vec();

        self.inner
            .verify_multi_batch(commit, &salted_dimensions, indices, &salted_values, pruned)
    }
}

impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    MerkleTreeHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
where
    P: PackedValue,
    P::Value: Serialize + DeserializeOwned,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
        + Sync,
    R: CryptoRng + SeedableRng + Send,
    PW::Value: Eq + Clone,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<P::Value>,
{
    /// Verify rows, salts and an authentication path borrowed separately.
    ///
    /// Performs the same checks as the trait verification, without an owned `(salts, siblings)` proof tuple.
    pub fn verify_batch_borrowed<S: AsRef<[P::Value]>>(
        &self,
        commit: &MerkleCap<P::Value, [PW::Value; DIGEST_ELEMS]>,
        dimensions: &[Dimensions],
        index: usize,
        opened_values: &[Vec<P::Value>],
        salts: &[S],
        siblings: &[[PW::Value; DIGEST_ELEMS]],
    ) -> Result<(), MerkleTreeError> {
        // Pin each opened row to its matrix width before salting.
        // The inner tree only sees salted widths.
        // Without this, an over-long row could be masked by an under-long salt.
        check_widths(dimensions, opened_values)?;

        let expected_salts = opened_values.len();
        let got_salts = salts.len();
        let opened_salted_values = zip_eq(
            opened_values,
            salts,
            MerkleTreeError::WrongBatchSize {
                expected: expected_salts,
                got: got_salts,
            },
        )?
        .map(|(opened, salt)| {
            opened
                .iter()
                .chain(salt.as_ref().iter())
                .copied()
                .collect_vec()
        })
        .collect_vec();

        // The inner tree commits to rows widened by the salt columns,
        // so the widths must be widened the same way.
        let salted_dimensions = dimensions
            .iter()
            .map(|dims| Dimensions {
                width: dims.width + SALT_ELEMS,
                height: dims.height,
            })
            .collect_vec();

        self.inner.verify_batch_borrowed(
            commit,
            &salted_dimensions,
            index,
            &opened_salted_values,
            siblings,
        )
    }
}

#[cfg(feature = "secret-memory")]
impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    MerkleTreeHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
where
    P: PackedValue,
    P::Value: Serialize + DeserializeOwned + zeroize::Zeroize,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
        + Sync,
    R: CryptoRng + SeedableRng + Send,
    PW::Value: Eq + Clone + zeroize::Zeroize,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<P::Value>,
{
    /// Move salt ownership and public authentication nodes out of a guarded
    /// hiding tree, then erase every committed value matrix in place.
    pub fn erase_leaves_into_authentication_data<M>(
        &self,
        prover_data: &mut SecretProverData<
            GuardedHidingTree<P::Value, PW::Value, M, N, DIGEST_ELEMS>,
        >,
    ) -> Result<HidingMerkleAuthenticationData<P::Value, PW::Value, N, DIGEST_ELEMS>, MerkleTreeError>
    where
        M: Matrix<P::Value> + zeroize::Zeroize,
    {
        if SALT_ELEMS == 0 {
            return Err(MerkleTreeError::SecretInvalidSaltWidth);
        }
        if prover_data.leaves.is_empty() {
            return Err(MerkleTreeError::EmptyBatch);
        }

        let leaf_count = prover_data.leaves.len();
        let mut dimensions = Vec::new();
        dimensions
            .try_reserve_exact(leaf_count)
            .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
        let mut salted_dimensions = Vec::new();
        salted_dimensions
            .try_reserve_exact(leaf_count)
            .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
        for leaf in &prover_data.leaves {
            dimensions.push(leaf.left.dimensions());
            salted_dimensions.push(leaf.dimensions());
        }

        let mut salts = Vec::new();
        salts
            .try_reserve_exact(leaf_count)
            .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
        for leaf in &mut prover_data.leaves {
            let empty_salts = SecretProverData::new(RowMajorMatrix::new(Vec::new(), SALT_ELEMS));
            salts.push(core::mem::replace(&mut leaf.right, empty_salts));
        }

        let inner =
            prover_data.erase_leaves_into_authentication_data_with_dimensions(salted_dimensions)?;
        Ok(HidingMerkleAuthenticationData {
            inner,
            salts: SecretProverData::new(salts),
            dimensions,
        })
    }

    /// Open one verifier-selected index from rows recomputed after value-leaf erasure.
    pub fn open_batch_with_recomputed_rows<Row: AsRef<[P::Value]>>(
        &self,
        index: usize,
        opened_rows: &[Row],
        authentication_data: &HidingMerkleAuthenticationData<P::Value, PW::Value, N, DIGEST_ELEMS>,
    ) -> Result<BatchOpening<P::Value, Self>, MerkleTreeError> {
        let salted_rows = Self::collect_salted_rows(index, opened_rows, authentication_data)?;
        let inner_opening = self.inner.open_batch_with_recomputed_rows(
            index,
            &salted_rows,
            &authentication_data.inner,
        )?;
        let (opened_values, salts) =
            Self::split_salted_rows(inner_opening.opened_values, &authentication_data.dimensions)?;
        Ok(BatchOpening::new(
            opened_values,
            (salts, inner_opening.opening_proof),
        ))
    }

    /// Open several verifier-selected indices with one pruned proof after value-leaf erasure.
    pub fn open_multi_batch_with_recomputed_rows<Row: AsRef<[P::Value]>>(
        &self,
        indices: &[usize],
        opened_rows: &[Vec<Row>],
        authentication_data: &HidingMerkleAuthenticationData<P::Value, PW::Value, N, DIGEST_ELEMS>,
    ) -> Result<HidingMultiOpening<P::Value, PW::Value, DIGEST_ELEMS>, MerkleTreeError> {
        if opened_rows.len() != indices.len() {
            return Err(MerkleTreeError::WrongBatchSize {
                expected: indices.len(),
                got: opened_rows.len(),
            });
        }

        let mut salted_rows = Vec::new();
        salted_rows
            .try_reserve_exact(indices.len())
            .map_err(|_| MerkleTreeError::RecomputedOpeningAllocationFailed)?;
        for (&index, rows) in indices.iter().zip(opened_rows) {
            salted_rows.push(Self::collect_salted_rows(index, rows, authentication_data)?);
        }

        let inner_opening = self.inner.open_batch_pruned_with_recomputed_rows(
            indices,
            &salted_rows,
            &authentication_data.inner,
        )?;
        let mut opened_values = Vec::new();
        opened_values
            .try_reserve_exact(inner_opening.opened_values.len())
            .map_err(|_| MerkleTreeError::RecomputedOpeningAllocationFailed)?;
        let mut salts = Vec::new();
        salts
            .try_reserve_exact(inner_opening.opened_values.len())
            .map_err(|_| MerkleTreeError::RecomputedOpeningAllocationFailed)?;
        for rows in inner_opening.opened_values {
            let (values, salts_at_index) =
                Self::split_salted_rows(rows, &authentication_data.dimensions)?;
            opened_values.push(values);
            salts.push(salts_at_index);
        }

        Ok((opened_values, (salts, inner_opening.pruned_proof)))
    }

    fn collect_salted_rows<Row: AsRef<[P::Value]>>(
        index: usize,
        opened_rows: &[Row],
        authentication_data: &HidingMerkleAuthenticationData<P::Value, PW::Value, N, DIGEST_ELEMS>,
    ) -> Result<Vec<Vec<P::Value>>, MerkleTreeError> {
        check_widths(&authentication_data.dimensions, opened_rows)?;
        if authentication_data.salts.len() != authentication_data.dimensions.len() {
            return Err(MerkleTreeError::MalformedAuthenticationData);
        }
        let max_height = validate_commit_reachable_heights(
            authentication_data
                .dimensions
                .iter()
                .map(|dimensions| dimensions.height),
        )?;
        if index >= max_height {
            return Err(MerkleTreeError::IndexOutOfBounds { max_height, index });
        }
        let log_max_height = log2_ceil_usize(max_height);

        let mut salted_rows = Vec::new();
        salted_rows
            .try_reserve_exact(opened_rows.len())
            .map_err(|_| MerkleTreeError::RecomputedOpeningAllocationFailed)?;
        for (matrix, ((row, dimensions), salt_matrix)) in opened_rows
            .iter()
            .zip(&authentication_data.dimensions)
            .zip(authentication_data.salts.iter())
            .enumerate()
        {
            if salt_matrix.width() != SALT_ELEMS || salt_matrix.height() != dimensions.height {
                return Err(MerkleTreeError::MalformedAuthenticationData);
            }
            let bits_reduced = log_max_height
                .checked_sub(log2_ceil_usize(dimensions.height))
                .ok_or(MerkleTreeError::MalformedAuthenticationData)?;
            let row_index = index >> bits_reduced;
            let salt = salt_matrix.row_slice(row_index).ok_or(
                MerkleTreeError::RecomputedRowUnavailable {
                    matrix,
                    row: row_index,
                },
            )?;
            if salt.len() != SALT_ELEMS {
                return Err(MerkleTreeError::MalformedAuthenticationData);
            }
            let salted_width = dimensions
                .width
                .checked_add(SALT_ELEMS)
                .ok_or(MerkleTreeError::SecretDimensionOverflow)?;
            let mut salted_row = Vec::new();
            salted_row
                .try_reserve_exact(salted_width)
                .map_err(|_| MerkleTreeError::RecomputedOpeningAllocationFailed)?;
            salted_row.extend_from_slice(row.as_ref());
            salted_row.extend_from_slice(&salt);
            salted_rows.push(salted_row);
        }
        Ok(salted_rows)
    }

    fn split_salted_rows(
        salted_rows: Vec<Vec<P::Value>>,
        dimensions: &[Dimensions],
    ) -> Result<SplitRows<P::Value>, MerkleTreeError> {
        if salted_rows.len() != dimensions.len() {
            return Err(MerkleTreeError::WrongBatchSize {
                expected: dimensions.len(),
                got: salted_rows.len(),
            });
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(salted_rows.len())
            .map_err(|_| MerkleTreeError::RecomputedOpeningAllocationFailed)?;
        let mut salts = Vec::new();
        salts
            .try_reserve_exact(salted_rows.len())
            .map_err(|_| MerkleTreeError::RecomputedOpeningAllocationFailed)?;
        for (matrix, (row, matrix_dimensions)) in
            salted_rows.into_iter().zip(dimensions).enumerate()
        {
            let expected_width = matrix_dimensions
                .width
                .checked_add(SALT_ELEMS)
                .ok_or(MerkleTreeError::SecretDimensionOverflow)?;
            if row.len() != expected_width {
                return Err(MerkleTreeError::WrongWidth {
                    matrix,
                    expected: expected_width,
                    got: row.len(),
                });
            }
            let (value_row, salt_row) = row.split_at(matrix_dimensions.width);
            let mut value_copy = Vec::new();
            value_copy
                .try_reserve_exact(value_row.len())
                .map_err(|_| MerkleTreeError::RecomputedOpeningAllocationFailed)?;
            value_copy.extend_from_slice(value_row);
            let mut salt_copy = Vec::new();
            salt_copy
                .try_reserve_exact(salt_row.len())
                .map_err(|_| MerkleTreeError::RecomputedOpeningAllocationFailed)?;
            salt_copy.extend_from_slice(salt_row);
            values.push(value_copy);
            salts.push(salt_copy);
        }
        Ok((values, salts))
    }
}

#[cfg(feature = "secret-memory")]
impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    SecretMmcs<P::Value> for MerkleTreeHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
where
    P: PackedValue,
    P::Value: Serialize + DeserializeOwned + zeroize::Zeroize,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
        + Sync,
    R: CryptoRng + SeedableRng + Send,
    PW::Value: Eq + Clone + zeroize::Zeroize,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<P::Value>,
{
    type SecretData<M>
        = MerkleTree<
        P::Value,
        PW::Value,
        HorizontalPair<SecretProverData<M>, SecretProverData<RowMajorMatrix<P::Value>>>,
        N,
        DIGEST_ELEMS,
    >
    where
        M: Matrix<P::Value> + zeroize::Zeroize;
    type SecretError = MerkleTreeError;

    fn commit_secret<M>(
        &self,
        inputs: Vec<SecretProverData<M>>,
    ) -> Result<(Self::Commitment, SecretProverData<Self::SecretData<M>>), Self::SecretError>
    where
        M: Matrix<P::Value> + zeroize::Zeroize,
    {
        if SALT_ELEMS == 0 {
            return Err(MerkleTreeError::SecretInvalidSaltWidth);
        }
        validate_commit_reachable_heights(inputs.iter().map(Matrix::height))?;

        let mut salted_inputs = Vec::new();
        salted_inputs
            .try_reserve_exact(inputs.len())
            .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
        let mut rng = self.rng.lock();
        for matrix in inputs {
            let salt_len = matrix
                .height()
                .checked_mul(SALT_ELEMS)
                .ok_or(MerkleTreeError::SecretDimensionOverflow)?;
            let mut salt_values = Vec::new();
            salt_values
                .try_reserve_exact(salt_len)
                .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
            salt_values.resize(salt_len, P::Value::default());
            let mut salts = SecretProverData::new(RowMajorMatrix::new(salt_values, SALT_ELEMS));
            for salt in &mut salts.values {
                *salt = rng.random();
            }
            salted_inputs.push(HorizontalPair::new(matrix, salts));
        }

        let (commitment, tree) = self.inner.commit(salted_inputs);
        Ok((commitment, SecretProverData::new(tree)))
    }

    fn open_batch_secret<M>(
        &self,
        index: usize,
        prover_data: &SecretProverData<Self::SecretData<M>>,
    ) -> Result<BatchOpening<P::Value, Self>, Self::SecretError>
    where
        M: Matrix<P::Value> + zeroize::Zeroize,
    {
        let max_height = prover_data
            .leaves
            .iter()
            .map(Matrix::height)
            .max()
            .ok_or(MerkleTreeError::EmptyBatch)?;
        if index >= max_height {
            return Err(MerkleTreeError::IndexOutOfBounds { max_height, index });
        }

        let (salted_openings, siblings) = self.inner.open_batch(index, prover_data).unpack();
        let (openings, salts) = salted_openings
            .into_iter()
            .map(|row| {
                let split = row.len().saturating_sub(SALT_ELEMS);
                let (values, salt) = row.split_at(split);
                (values.to_vec(), salt.to_vec())
            })
            .unzip();
        Ok(BatchOpening::new(openings, (salts, siblings)))
    }

    fn get_matrices_secret<'a, M>(
        &self,
        prover_data: &'a SecretProverData<Self::SecretData<M>>,
    ) -> Vec<&'a M>
    where
        M: Matrix<P::Value> + zeroize::Zeroize,
    {
        prover_data
            .leaves
            .iter()
            .map(|matrix| &*matrix.left)
            .collect()
    }

    fn open_multi_batch_secret<M>(
        &self,
        indices: &[usize],
        prover_data: &SecretProverData<Self::SecretData<M>>,
    ) -> Result<(Vec<Vec<Vec<P::Value>>>, Self::MultiProof), Self::SecretError>
    where
        M: Matrix<P::Value> + zeroize::Zeroize,
    {
        let max_height = prover_data
            .leaves
            .iter()
            .map(Matrix::height)
            .max()
            .ok_or(MerkleTreeError::EmptyBatch)?;
        if let Some(&index) = indices.iter().find(|&&index| index >= max_height) {
            return Err(MerkleTreeError::IndexOutOfBounds { max_height, index });
        }

        let (salted_values, pruned) = self.inner.open_multi_batch(indices, prover_data);
        let (opened_values, salts): (Vec<_>, Vec<_>) = salted_values
            .into_iter()
            .map(|rows_at_index| {
                rows_at_index
                    .into_iter()
                    .map(|row| {
                        let split = row.len().saturating_sub(SALT_ELEMS);
                        let (values, salt) = row.split_at(split);
                        (values.to_vec(), salt.to_vec())
                    })
                    .unzip()
            })
            .unzip();

        Ok((opened_values, (salts, pruned)))
    }
}

#[cfg(feature = "secret-memory")]
impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    SecretRecomputableMmcs<P::Value>
    for MerkleTreeHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
where
    P: PackedValue,
    P::Value: Default + Serialize + DeserializeOwned + zeroize::Zeroize,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + IncrementalCryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
        + Sync,
    R: CryptoRng + SeedableRng + Send,
    PW::Value: Copy + Default + Eq + Clone + zeroize::Zeroize,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<P::Value>,
{
    type AuthenticationData<M>
        = HidingMerkleAuthenticationData<P::Value, PW::Value, N, DIGEST_ELEMS>
    where
        M: Matrix<P::Value> + zeroize::Zeroize;

    fn commit_detached_secret<M>(
        &self,
        inputs: Vec<SecretProverData<M>>,
    ) -> Result<
        SecretDetachedCommitment<
            Self::Commitment,
            Self::SecretData<M>,
            Self::AuthenticationData<M>,
        >,
        Self::SecretError,
    >
    where
        M: Matrix<P::Value> + zeroize::Zeroize,
    {
        let mut dimensions = Vec::new();
        dimensions
            .try_reserve_exact(inputs.len())
            .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
        dimensions.extend(inputs.iter().map(|matrix| matrix.dimensions()));

        let mut stream =
            <Self as SecretStreamingRecomputableMmcs<P::Value>>::begin_streaming_commitment::<M>(
                self, dimensions,
            )?;
        let mut indexed_inputs = Vec::new();
        indexed_inputs
            .try_reserve_exact(inputs.len())
            .map_err(|_| MerkleTreeError::SecretAllocationFailed)?;
        indexed_inputs.extend(inputs.into_iter().enumerate());
        indexed_inputs.sort_by_key(|(_, matrix)| Reverse(matrix.height()));
        for (matrix_index, input) in indexed_inputs {
            stream.try_absorb_matrix_at(matrix_index, input)?;
        }
        stream.finish()
    }

    fn detach_authentication_data<M>(
        &self,
        prover_data: &mut SecretProverData<Self::SecretData<M>>,
    ) -> Result<Self::AuthenticationData<M>, Self::SecretError>
    where
        M: Matrix<P::Value> + zeroize::Zeroize,
    {
        self.erase_leaves_into_authentication_data(prover_data)
    }

    fn open_multi_batch_from_recomputed_rows<M>(
        &self,
        indices: &[usize],
        opened_rows: &[Vec<Vec<P::Value>>],
        authentication_data: &Self::AuthenticationData<M>,
    ) -> Result<(Vec<Vec<Vec<P::Value>>>, Self::MultiProof), Self::SecretError>
    where
        M: Matrix<P::Value> + zeroize::Zeroize,
    {
        self.open_multi_batch_with_recomputed_rows(indices, opened_rows, authentication_data)
    }
}

#[cfg(feature = "secret-memory")]
impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    SecretStreamingRecomputableMmcs<P::Value>
    for MerkleTreeHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
where
    P: PackedValue,
    P::Value: Serialize + DeserializeOwned + zeroize::Zeroize,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + IncrementalCryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
        + Sync,
    R: CryptoRng + SeedableRng + Send,
    PW::Value: Copy + Default + Eq + Clone + zeroize::Zeroize,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<P::Value>,
{
    type Stream<'a, M>
        = IncrementalHidingMerkleCommitment<'a, P::Value, PW::Value, H, C, N, DIGEST_ELEMS>
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
        self.try_begin_incremental_secret_commitment(dimensions)
    }
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::string::ToString;
    use alloc::sync::Arc;
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(feature = "secret-memory")]
    use std::thread;

    use itertools::Itertools;
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_commit::Mmcs;
    #[cfg(feature = "secret-memory")]
    use p3_commit::{SecretDetachedCommitment, SecretMmcs, SecretRecomputableMmcs};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    #[cfg(feature = "secret-memory")]
    use p3_matrix::dense::RowMajorMatrixViewMut;
    #[cfg(feature = "secret-memory")]
    use p3_matrix::secret::ZeroizingMatrix;
    use p3_symmetric::{CryptographicHasher, PaddingFreeSponge, TruncatedPermutation};
    #[cfg(feature = "secret-memory")]
    use p3_symmetric::{IncrementalCryptographicHasher, PseudoCompressionFunction};
    use p3_util::assert_sync;
    use rand::SeedableRng;
    use rand::rngs::{SmallRng, StdRng};
    use spin::Once;
    #[cfg(feature = "secret-memory")]
    use zeroize::Zeroizing;

    use super::MerkleTreeHidingMmcs;
    use crate::MerkleTreeError;

    type F = BabyBear;
    const SALT_ELEMS: usize = 4;

    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyMmcs = MerkleTreeHidingMmcs<
        <F as Field>::Packing,
        <F as Field>::Packing,
        MyHash,
        MyCompress,
        StdRng,
        2,
        8,
        SALT_ELEMS,
    >;

    /// The same MMCS with a leaf hasher that checks the salt RNG lock on every call.
    type ProbedMmcs = MerkleTreeHidingMmcs<
        <F as Field>::Packing,
        <F as Field>::Packing,
        ProbedHasher<MyHash>,
        MyCompress,
        StdRng,
        2,
        8,
        SALT_ELEMS,
    >;

    /// A lock check run from inside the parallel work of the object that owns the lock.
    ///
    /// The probed hasher lives inside the MMCS it observes, so the check is armed after construction.
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
                    "parallel work must not run under the salt RNG lock"
                );
                self.hits.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// A leaf hasher that runs its probe before forwarding every call to `inner`.
    #[derive(Clone)]
    struct ProbedHasher<H> {
        inner: H,
        probe: LockProbe,
    }

    impl<H, Item, Out> CryptographicHasher<Item, Out> for ProbedHasher<H>
    where
        H: CryptographicHasher<Item, Out>,
        Item: Clone,
    {
        const LANES: usize = H::LANES;

        fn hash_iter<I>(&self, input: I) -> Out
        where
            I: IntoIterator<Item = Item>,
        {
            self.probe.assert_lock_free();
            self.inner.hash_iter(input)
        }

        fn hash_iter_slices<'a, I>(&self, input: I) -> Out
        where
            I: IntoIterator<Item = &'a [Item]>,
            Item: 'a,
        {
            self.probe.assert_lock_free();
            self.inner.hash_iter_slices(input)
        }

        fn hash_slice(&self, input: &[Item]) -> Out {
            self.probe.assert_lock_free();
            self.inner.hash_slice(input)
        }

        fn hash_item(&self, input: Item) -> Out {
            self.probe.assert_lock_free();
            self.inner.hash_item(input)
        }

        fn hash_many(&self, input: &[Item], out: &mut [Out]) {
            self.probe.assert_lock_free();
            self.inner.hash_many(input, out);
        }
    }

    #[test]
    fn commit_hashes_without_holding_the_rng_lock() {
        // A rayon worker that waits inside the tree build may run another commit on this MMCS.
        //
        // That commit would spin forever if the build ran under the salt lock.
        //
        // The leaf hasher runs inside the build, so it checks the lock directly.
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let probe = LockProbe::default();
        let hash = ProbedHasher {
            inner: MyHash::new(perm.clone()),
            probe: probe.clone(),
        };
        let mmcs = Arc::new(ProbedMmcs::new(
            hash,
            MyCompress::new(perm),
            0,
            StdRng::seed_from_u64(0),
        ));
        probe.arm(&mmcs, |mmcs| !mmcs.rng.is_locked());

        // Tall enough that the build hashes rows across several rayon tasks.
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 1 << 10, 8);
        let _ = mmcs.commit(vec![mat]);

        // The check must have run, or the test would pass without probing anything.
        assert!(probe.hits.load(Ordering::Relaxed) > 0);
    }

    #[cfg(feature = "secret-memory")]
    fn secret_mmcs() -> MyMmcs {
        let mut rng = SmallRng::seed_from_u64(0x5345_4352_4554_4d4d);
        let permutation = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(permutation.clone());
        let compress = MyCompress::new(permutation);
        MyMmcs::new(hash, compress, 0, StdRng::seed_from_u64(7))
    }

    #[cfg(feature = "secret-memory")]
    const STREAM_DIGEST_ELEMS: usize = 2;

    #[cfg(feature = "secret-memory")]
    #[derive(Clone, Copy)]
    struct TestIncrementalHasher;

    #[cfg(feature = "secret-memory")]
    impl CryptographicHasher<F, [F; STREAM_DIGEST_ELEMS]> for TestIncrementalHasher {
        fn hash_iter<I>(&self, input: I) -> [F; STREAM_DIGEST_ELEMS]
        where
            I: IntoIterator<Item = F>,
        {
            let mut state = self.begin_hash();
            self.absorb(&mut state, input);
            self.finalize_hash(state)
        }
    }

    #[cfg(feature = "secret-memory")]
    impl IncrementalCryptographicHasher<F, [F; STREAM_DIGEST_ELEMS]> for TestIncrementalHasher {
        type State = Zeroizing<[F; 3]>;

        fn begin_hash(&self) -> Self::State {
            Zeroizing::new([F::ZERO; 3])
        }

        fn absorb<I>(&self, state: &mut Self::State, input: I)
        where
            I: IntoIterator<Item = F>,
        {
            for value in input {
                state[0] += value;
                state[1] = state[1] * F::from_u8(7) + value;
                state[2] += F::ONE;
            }
        }

        fn finalize_hash(&self, state: Self::State) -> [F; STREAM_DIGEST_ELEMS] {
            [state[0] + state[2], state[1]]
        }
    }

    #[cfg(feature = "secret-memory")]
    #[derive(Clone, Copy)]
    struct TestIncrementalCompress;

    #[cfg(feature = "secret-memory")]
    impl PseudoCompressionFunction<[F; STREAM_DIGEST_ELEMS], 2> for TestIncrementalCompress {
        fn compress(&self, input: [[F; STREAM_DIGEST_ELEMS]; 2]) -> [F; STREAM_DIGEST_ELEMS] {
            [
                input[0][0] + input[1][0] * F::from_u8(11),
                input[0][1] * F::from_u8(13) + input[1][1],
            ]
        }
    }

    #[cfg(feature = "secret-memory")]
    type TestIncrementalMmcs = MerkleTreeHidingMmcs<
        F,
        F,
        TestIncrementalHasher,
        TestIncrementalCompress,
        StdRng,
        2,
        STREAM_DIGEST_ELEMS,
        2,
    >;

    #[cfg(feature = "secret-memory")]
    fn incremental_mmcs() -> TestIncrementalMmcs {
        TestIncrementalMmcs::new(
            TestIncrementalHasher,
            TestIncrementalCompress,
            0,
            StdRng::seed_from_u64(0x5354_5245_414d_4d4d),
        )
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn incremental_hiding_commit_matches_detached_commit_and_erases_each_matrix() {
        let batch_mmcs = incremental_mmcs();
        let streamed_mmcs = incremental_mmcs();
        let middle_values = (101_u32..=104).map(F::from_u32).collect_vec();
        let tall_values = (1_u32..=16).map(F::from_u32).collect_vec();
        let short_values = (201_u32..=206).map(F::from_u32).collect_vec();
        let dimensions = vec![
            p3_matrix::Dimensions {
                width: 1,
                height: 4,
            },
            p3_matrix::Dimensions {
                width: 2,
                height: 8,
            },
            p3_matrix::Dimensions {
                width: 3,
                height: 2,
            },
        ];
        let batch = batch_mmcs
            .commit_detached_secret(vec![
                p3_commit::SecretProverData::new(RowMajorMatrix::new(middle_values.clone(), 1)),
                p3_commit::SecretProverData::new(RowMajorMatrix::new(tall_values.clone(), 2)),
                p3_commit::SecretProverData::new(RowMajorMatrix::new(short_values.clone(), 3)),
            ])
            .expect("valid detached hiding commitment");

        let mut streamed = streamed_mmcs
            .try_begin_incremental_secret_commitment(dimensions.clone())
            .expect("valid incremental hiding plan");
        let mut middle_canary = middle_values.clone();
        streamed
            .try_absorb_matrix(ZeroizingMatrix::new(RowMajorMatrixViewMut::new(
                &mut middle_canary,
                1,
            )))
            .expect("valid middle matrix");
        assert_eq!(middle_canary, vec![F::ZERO; middle_values.len()]);

        let mut tall_canary = tall_values.clone();
        streamed
            .try_absorb_matrix(ZeroizingMatrix::new(RowMajorMatrixViewMut::new(
                &mut tall_canary,
                2,
            )))
            .expect("valid tall matrix");
        assert_eq!(tall_canary, vec![F::ZERO; tall_values.len()]);

        let mut short_canary = short_values.clone();
        streamed
            .try_absorb_matrix(ZeroizingMatrix::new(RowMajorMatrixViewMut::new(
                &mut short_canary,
                3,
            )))
            .expect("valid short matrix");
        assert_eq!(short_canary, vec![F::ZERO; short_values.len()]);

        let streamed = streamed
            .try_finish::<RowMajorMatrix<F>>()
            .expect("complete incremental hiding commitment");
        assert_eq!(streamed.commitment, batch.commitment);
        assert_eq!(streamed.authentication_data.dimensions(), dimensions);
        assert_eq!(
            streamed.authentication_data.salts.len(),
            batch.authentication_data.salts.len()
        );
        for (streamed_salt, batch_salt) in streamed
            .authentication_data
            .salts
            .iter()
            .zip(batch.authentication_data.salts.iter())
        {
            assert_eq!(streamed_salt.values, batch_salt.values);
        }
        assert!(
            streamed_mmcs
                .get_matrices_secret(&streamed.prover_data)
                .is_empty()
        );

        let indices = [0_usize, 3, 7];
        let recomputed_rows = indices
            .iter()
            .map(|index| {
                let middle_index = index >> 1;
                let tall_start = index * 2;
                let short_start = (index >> 2) * 3;
                vec![
                    vec![middle_values[middle_index]],
                    tall_values[tall_start..tall_start + 2].to_vec(),
                    short_values[short_start..short_start + 3].to_vec(),
                ]
            })
            .collect_vec();
        let batch_opening = batch_mmcs
            .open_multi_batch_with_recomputed_rows(
                &indices,
                &recomputed_rows,
                &batch.authentication_data,
            )
            .expect("batch opening");
        let streamed_opening = streamed_mmcs
            .open_multi_batch_with_recomputed_rows(
                &indices,
                &recomputed_rows,
                &streamed.authentication_data,
            )
            .expect("streamed opening");
        assert_eq!(streamed_opening.0, batch_opening.0);
        assert_eq!(streamed_opening.1.0, batch_opening.1.0);
        assert_eq!(
            streamed_opening.1.1.sibling_hashes,
            batch_opening.1.1.sibling_hashes
        );
        streamed_mmcs
            .verify_multi_batch(
                &streamed.commitment,
                streamed.authentication_data.dimensions(),
                &indices,
                &streamed_opening.0,
                &streamed_opening.1,
            )
            .expect("streamed opening must verify");

        let next_matrix = RowMajorMatrix::new(tall_values, 2);
        let batch_next = batch_mmcs
            .commit_detached_secret(vec![p3_commit::SecretProverData::new(next_matrix.clone())])
            .expect("next batch commitment");
        let streamed_next = streamed_mmcs
            .commit_detached_secret(vec![p3_commit::SecretProverData::new(next_matrix)])
            .expect("next streamed commitment");
        assert_eq!(streamed_next.commitment, batch_next.commitment);
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn incremental_hiding_rejects_shape_drift_and_erases_rejected_matrix() {
        let mmcs = incremental_mmcs();
        let mut streamed = mmcs
            .try_begin_incremental_secret_commitment(vec![p3_matrix::Dimensions {
                width: 2,
                height: 4,
            }])
            .expect("valid incremental hiding plan");
        let mut canary = [F::NEG_ONE; 12];
        let error = streamed
            .try_absorb_matrix(ZeroizingMatrix::new(RowMajorMatrixViewMut::new(
                &mut canary,
                3,
            )))
            .expect_err("shape drift must fail closed");

        assert!(matches!(
            error,
            MerkleTreeError::StreamedMatrixShapeMismatch {
                matrix: 0,
                expected_width: 2,
                expected_height: 4,
                got_width: 3,
                got_height: 4,
            }
        ));
        assert_eq!(canary, [F::ZERO; 12]);
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn incremental_hiding_invalid_preflight_does_not_advance_rng() {
        let control_mmcs = incremental_mmcs();
        let tested_mmcs = incremental_mmcs();
        let invalid = vec![
            p3_matrix::Dimensions {
                width: 2,
                height: 8,
            },
            p3_matrix::Dimensions {
                width: 2,
                height: 3,
            },
        ];
        assert!(
            tested_mmcs
                .try_begin_incremental_secret_commitment(invalid)
                .is_err()
        );

        let values = (1_u32..=16).map(F::from_u32).collect_vec();
        let control = control_mmcs
            .commit_detached_secret(vec![p3_commit::SecretProverData::new(RowMajorMatrix::new(
                values.clone(),
                2,
            ))])
            .expect("control commitment");
        let tested = tested_mmcs
            .commit_detached_secret(vec![p3_commit::SecretProverData::new(RowMajorMatrix::new(
                values, 2,
            ))])
            .expect("commitment after rejected preflight");
        assert_eq!(tested.commitment, control.commitment);
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn incremental_hiding_rejects_incomplete_stream() {
        let mmcs = incremental_mmcs();
        let dimensions = vec![
            p3_matrix::Dimensions {
                width: 2,
                height: 8,
            },
            p3_matrix::Dimensions {
                width: 1,
                height: 4,
            },
        ];
        let mut streamed = mmcs
            .try_begin_incremental_secret_commitment(dimensions)
            .expect("valid incremental hiding plan");
        streamed
            .try_absorb_matrix(p3_commit::SecretProverData::new(RowMajorMatrix::new(
                vec![F::ONE; 16],
                2,
            )))
            .expect("first matrix");
        let Err(error) = streamed.try_finish::<RowMajorMatrix<F>>() else {
            panic!("incomplete stream must fail closed");
        };
        assert!(matches!(
            error,
            MerkleTreeError::StreamedMatrixCountMismatch {
                expected: 2,
                got: 1,
            }
        ));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn guarded_hiding_commitment_opens_and_erases_borrowed_storage() {
        let mmcs = secret_mmcs();
        let dimensions = [p3_matrix::Dimensions {
            width: 2,
            height: 4,
        }];
        let mut values = [
            F::ONE,
            F::TWO,
            F::NEG_ONE,
            F::GENERATOR,
            F::ONE,
            F::TWO,
            F::NEG_ONE,
            F::GENERATOR,
        ];
        let (commitment, opening) = {
            let matrix = ZeroizingMatrix::new(RowMajorMatrixViewMut::new(&mut values, 2));
            let (commitment, prover_data) = mmcs
                .commit_secret(vec![matrix])
                .expect("valid guarded hiding commitment must succeed");
            let opening = mmcs
                .open_batch_secret(3, &prover_data)
                .expect("guarded hiding commitment must open");
            (commitment, opening)
        };

        assert_eq!(values, [F::ZERO; 8]);
        mmcs.verify_batch(&commitment, &dimensions, 3, (&opening).into())
            .expect("public opening material must survive private prover-data erasure");
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn detached_hiding_commit_matches_explicit_transition_and_preserves_rng_order() {
        let expected_mmcs = secret_mmcs();
        let detached_mmcs = secret_mmcs();
        let original_values = [
            F::ONE,
            F::TWO,
            F::NEG_ONE,
            F::GENERATOR,
            F::ONE,
            F::TWO,
            F::NEG_ONE,
            F::GENERATOR,
        ];
        let expected_matrix = RowMajorMatrix::new(original_values.to_vec(), 2);
        let (expected_commitment, mut expected_data) = expected_mmcs
            .commit_secret(vec![p3_commit::SecretProverData::new(expected_matrix)])
            .expect("valid guarded hiding commitment must succeed");
        let expected_authentication = expected_mmcs
            .detach_authentication_data(&mut expected_data)
            .expect("explicit hiding detachment must succeed");

        let mut guarded_values = original_values;
        let detached_matrix =
            ZeroizingMatrix::new(RowMajorMatrixViewMut::new(&mut guarded_values, 2));
        let SecretDetachedCommitment {
            commitment,
            prover_data,
            authentication_data,
        } = detached_mmcs
            .commit_detached_secret(vec![detached_matrix])
            .expect("direct hiding detachment must succeed");

        assert_eq!(commitment, expected_commitment);
        assert!(detached_mmcs.get_matrices_secret(&prover_data).is_empty());
        drop(prover_data);
        assert_eq!(
            authentication_data.dimensions(),
            expected_authentication.dimensions()
        );

        let query_index = 2;
        let recomputed_rows = [original_values[4..6].to_vec()];
        let expected_opening = expected_mmcs
            .open_batch_with_recomputed_rows(
                query_index,
                &recomputed_rows,
                &expected_authentication,
            )
            .expect("explicitly detached hiding commitment must open");
        let opening = detached_mmcs
            .open_batch_with_recomputed_rows(query_index, &recomputed_rows, &authentication_data)
            .expect("directly detached hiding commitment must open");
        assert_eq!(opening.opened_values, expected_opening.opened_values);
        assert_eq!(opening.opening_proof, expected_opening.opening_proof);

        let next_matrix = RowMajorMatrix::new(original_values.to_vec(), 2);
        let (expected_next, _expected_next_data) = expected_mmcs
            .commit_secret(vec![p3_commit::SecretProverData::new(next_matrix.clone())])
            .expect("next explicit commitment must succeed");
        let (detached_next, _detached_next_data) = detached_mmcs
            .commit_secret(vec![p3_commit::SecretProverData::new(next_matrix)])
            .expect("next direct commitment must succeed");
        assert_eq!(detached_next, expected_next);
        drop(authentication_data);
        drop(commitment);
        assert_eq!(guarded_values, [F::ZERO; 8]);
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn detached_hiding_commit_preserves_multi_height_salts_and_openings() {
        let expected_mmcs = secret_mmcs();
        let detached_mmcs = secret_mmcs();
        let tall_values = (1_u32..=16).map(F::from_u32).collect_vec();
        let middle_values = (101_u32..=104).map(F::from_u32).collect_vec();
        let short_values = (201_u32..=206).map(F::from_u32).collect_vec();
        let expected_inputs = vec![
            p3_commit::SecretProverData::new(RowMajorMatrix::new(middle_values.clone(), 1)),
            p3_commit::SecretProverData::new(RowMajorMatrix::new(tall_values.clone(), 2)),
            p3_commit::SecretProverData::new(RowMajorMatrix::new(short_values.clone(), 3)),
        ];
        let (expected_commitment, mut expected_data) = expected_mmcs
            .commit_secret(expected_inputs)
            .expect("valid multi-height hiding commitment must succeed");
        let expected_authentication = expected_mmcs
            .detach_authentication_data(&mut expected_data)
            .expect("explicit multi-height hiding detachment must succeed");

        let detached_inputs = vec![
            p3_commit::SecretProverData::new(RowMajorMatrix::new(middle_values.clone(), 1)),
            p3_commit::SecretProverData::new(RowMajorMatrix::new(tall_values.clone(), 2)),
            p3_commit::SecretProverData::new(RowMajorMatrix::new(short_values.clone(), 3)),
        ];
        let detached = detached_mmcs
            .commit_detached_secret(detached_inputs)
            .expect("direct multi-height hiding detachment must succeed");

        assert_eq!(detached.commitment, expected_commitment);
        assert_eq!(
            detached.authentication_data.dimensions(),
            expected_authentication.dimensions()
        );
        assert!(
            detached_mmcs
                .get_matrices_secret(&detached.prover_data)
                .is_empty()
        );

        let indices = [0_usize, 3, 7];
        let recomputed_rows = indices
            .iter()
            .map(|index| {
                let middle_index = index >> 1;
                let tall_start = index * 2;
                let short_start = (index >> 2) * 3;
                vec![
                    vec![middle_values[middle_index]],
                    tall_values[tall_start..tall_start + 2].to_vec(),
                    short_values[short_start..short_start + 3].to_vec(),
                ]
            })
            .collect_vec();
        let expected_opening = expected_mmcs
            .open_multi_batch_with_recomputed_rows(
                &indices,
                &recomputed_rows,
                &expected_authentication,
            )
            .expect("explicitly detached multi-height rows must open");
        let detached_opening = detached_mmcs
            .open_multi_batch_with_recomputed_rows(
                &indices,
                &recomputed_rows,
                &detached.authentication_data,
            )
            .expect("directly detached multi-height rows must open");

        assert_eq!(detached_opening.0, expected_opening.0);
        assert_eq!(detached_opening.1.0, expected_opening.1.0);
        assert_eq!(
            detached_opening.1.1.sibling_hashes,
            expected_opening.1.1.sibling_hashes
        );
        detached_mmcs
            .verify_multi_batch(
                &detached.commitment,
                detached.authentication_data.dimensions(),
                &indices,
                &detached_opening.0,
                &detached_opening.1,
            )
            .expect("direct multi-height hiding opening must verify");
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn detached_hiding_authentication_reproduces_single_and_pruned_openings() {
        let mmcs = secret_mmcs();
        let original_values = [
            F::from_u32(0x00ad_beef),
            F::from_u32(0x1245_414c),
            F::from_u32(0x0c59_4d45),
            F::from_u32(0x0102_0304),
            F::from_u32(0x0005_0607),
            F::from_u32(0x0008_090a),
            F::from_u32(0x000b_0c0d),
            F::from_u32(0x000e_0f10),
            F::from_u32(0x0011_1213),
            F::from_u32(0x0014_1516),
            F::from_u32(0x0017_1819),
            F::from_u32(0x001a_1b1c),
            F::from_u32(0x001d_1e1f),
            F::from_u32(0x0020_2122),
            F::from_u32(0x0023_2425),
            F::from_u32(0x0026_2728),
        ];
        let mut guarded_values = original_values;
        let dimensions = [p3_matrix::Dimensions {
            width: 2,
            height: 8,
        }];
        let single_index = 5;
        let indices = [0, 3, 5, 7];

        let matrix = ZeroizingMatrix::new(RowMajorMatrixViewMut::new(
            &mut guarded_values,
            dimensions[0].width,
        ));
        let (commitment, mut prover_data) = mmcs
            .commit_secret(vec![matrix])
            .expect("valid guarded hiding commitment must succeed");
        let ordinary_single = mmcs
            .open_batch_secret(single_index, &prover_data)
            .expect("retained hiding leaves must open");
        let (ordinary_multi_values, ordinary_multi_proof) = mmcs
            .open_multi_batch_secret(&indices, &prover_data)
            .expect("retained hiding leaves must produce a pruned opening");

        let authentication_data = mmcs
            .erase_leaves_into_authentication_data(&mut prover_data)
            .expect("hiding salts and public authentication data must detach");
        assert_eq!(authentication_data.dimensions(), dimensions);
        assert!(matches!(
            mmcs.erase_leaves_into_authentication_data(&mut prover_data),
            Err(MerkleTreeError::EmptyBatch)
        ));
        drop(prover_data);
        assert_eq!(guarded_values, [F::ZERO; 16]);

        let single_start = single_index * dimensions[0].width;
        let single_rows =
            [original_values[single_start..single_start + dimensions[0].width].to_vec()];
        let recomputed_single = mmcs
            .open_batch_with_recomputed_rows(single_index, &single_rows, &authentication_data)
            .expect("recomputed hiding row must open");
        assert_eq!(
            recomputed_single.opened_values,
            ordinary_single.opened_values
        );
        assert_eq!(
            recomputed_single.opening_proof,
            ordinary_single.opening_proof
        );
        mmcs.verify_batch(
            &commitment,
            &dimensions,
            single_index,
            (&recomputed_single).into(),
        )
        .expect("recomputed hiding opening must verify against the original commitment");

        let multi_rows = indices
            .iter()
            .map(|&index| {
                let start = index * dimensions[0].width;
                vec![original_values[start..start + dimensions[0].width].to_vec()]
            })
            .collect_vec();
        let (recomputed_multi_values, recomputed_multi_proof) = mmcs
            .open_multi_batch_with_recomputed_rows(&indices, &multi_rows, &authentication_data)
            .expect("recomputed hiding rows must produce a pruned opening");
        assert_eq!(recomputed_multi_values, ordinary_multi_values);
        assert_eq!(recomputed_multi_proof.0, ordinary_multi_proof.0);
        assert_eq!(
            recomputed_multi_proof.1.sibling_hashes,
            ordinary_multi_proof.1.sibling_hashes
        );
        mmcs.verify_multi_batch(
            &commitment,
            &dimensions,
            &indices,
            &recomputed_multi_values,
            &recomputed_multi_proof,
        )
        .expect("recomputed hiding multiproof must verify against the original commitment");

        let mut substituted_rows = single_rows.clone();
        substituted_rows[0][0] += F::ONE;
        let substituted = mmcs
            .open_batch_with_recomputed_rows(single_index, &substituted_rows, &authentication_data)
            .expect("a substituted row still has a well-formed authentication path");
        assert!(matches!(
            mmcs.verify_batch(
                &commitment,
                &dimensions,
                single_index,
                (&substituted).into(),
            ),
            Err(MerkleTreeError::CapMismatch)
        ));

        let wrong_width = [vec![F::ONE]];
        assert!(matches!(
            mmcs.open_batch_with_recomputed_rows(single_index, &wrong_width, &authentication_data,),
            Err(MerkleTreeError::WrongWidth {
                matrix: 0,
                expected: 2,
                got: 1,
            })
        ));
        assert!(matches!(
            mmcs.open_batch_with_recomputed_rows(8, &single_rows, &authentication_data),
            Err(MerkleTreeError::IndexOutOfBounds {
                max_height: 8,
                index: 8,
            })
        ));
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn repeated_guarded_hiding_commitments_erase_independently() {
        let mmcs = secret_mmcs();
        for marker in [F::ONE, F::TWO, F::NEG_ONE] {
            let mut values = [marker; 8];
            {
                let matrix = ZeroizingMatrix::new(RowMajorMatrixViewMut::new(&mut values, 2));
                let (_commitment, _prover_data) = mmcs
                    .commit_secret(vec![matrix])
                    .expect("valid repeated guarded commitment must succeed");
            }
            assert_eq!(values, [F::ZERO; 8]);
        }
    }

    #[cfg(feature = "secret-memory")]
    #[test]
    fn parallel_guarded_hiding_commitments_erase_distinct_canaries() {
        let mut canaries = [[F::ONE; 8], [F::TWO; 8], [F::NEG_ONE; 8], [F::GENERATOR; 8]];

        thread::scope(|scope| {
            for values in &mut canaries {
                scope.spawn(move || {
                    let mmcs = secret_mmcs();
                    let matrix =
                        ZeroizingMatrix::new(RowMajorMatrixViewMut::new(values.as_mut_slice(), 2));
                    let (_commitment, _prover_data) = mmcs
                        .commit_secret(vec![matrix])
                        .expect("valid parallel guarded commitment must succeed");
                });
            }
        });

        assert!(
            canaries
                .iter()
                .all(|values| values.iter().all(|value| *value == F::ZERO))
        );
    }

    #[test]
    #[should_panic]
    fn mismatched_heights() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0, StdRng::seed_from_u64(0));

        // attempt to commit to a mat with 8 rows and a mat with 7 rows. this should panic.
        let large_mat = RowMajorMatrix::new([1, 2, 3, 4, 5, 6, 7, 8].map(F::from_u8).to_vec(), 1);
        let small_mat = RowMajorMatrix::new([1, 2, 3, 4, 5, 6, 7].map(F::from_u8).to_vec(), 1);
        let _ = mmcs.commit(vec![large_mat, small_mat]);
    }

    /// Invariant: cloning forks the salt stream.
    ///
    /// The usual wiring hands a clone of the value MMCS to the extension MMCS, so a clone that
    /// copied the generator would salt two commitments with the same sequence and undo the
    /// hiding. Committing the same matrix through the original and through its clone must
    /// therefore land on different commitments.
    #[test]
    fn clone_forks_the_salt_stream() {
        let mut rng = SmallRng::seed_from_u64(1);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 16, 3);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        let mmcs = MyMmcs::new(hash, compress, 0, StdRng::seed_from_u64(0));
        let forked = mmcs.clone();

        let (commit, _) = mmcs.commit(vec![mat.clone()]);
        let (forked_commit, _) = forked.commit(vec![mat]);

        assert_ne!(
            commit, forked_commit,
            "a clone must not reuse the original's salt stream",
        );
    }

    #[test]
    fn different_widths() -> Result<(), MerkleTreeError> {
        let mut rng = SmallRng::seed_from_u64(1);
        // 10 mats with 32 rows where the ith mat has i + 1 cols
        let mats = (0..10)
            .map(|i| RowMajorMatrix::<F>::rand(&mut rng, 32, i + 1))
            .collect_vec();
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0, StdRng::seed_from_u64(0));

        let dims = mats.iter().map(|m| m.dimensions()).collect_vec();

        let (commit, prover_data) = mmcs.commit(mats);
        let batch_proof = mmcs.open_batch(17, &prover_data);
        mmcs.verify_batch(&commit, &dims, 17, (&batch_proof).into())
    }

    #[test]
    fn verify_rejects_wrong_row_width() {
        let mut rng = SmallRng::seed_from_u64(2);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        // Commit to one matrix of width 4 through the hiding wrapper.
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 8, 4);
        let dims = vec![mat.dimensions()];
        let mmcs = MyMmcs::new(hash, compress, 0, StdRng::seed_from_u64(0));
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        // Mutation: append one extra element to the opened row.
        //
        //     opened row:  [a, b, c, d, EXTRA]   (5 values, width is 4)
        let mut opening = mmcs.open_batch(3, &prover_data);
        opening.opened_values[0].push(F::ONE);

        // The width is checked on the unsalted row, before salt columns are appended.
        //
        //     expected: 4 (matrix width)
        //     got:      5 (opened row length)
        let err = mmcs
            .verify_batch(&commit, &dims, 3, (&opening).into())
            .expect_err("row longer than the matrix width must be rejected");
        assert!(matches!(
            err,
            MerkleTreeError::WrongWidth {
                matrix: 0,
                expected: 4,
                got: 5,
            }
        ));
    }

    #[test]
    fn verify_reports_the_opening_and_salt_counts() {
        let mut rng = SmallRng::seed_from_u64(5);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 8, 4);
        let dims = vec![mat.dimensions()];
        let mmcs = MyMmcs::new(hash, compress, 0, StdRng::seed_from_u64(0));
        let (commit, prover_data) = mmcs.commit(vec![mat]);
        let mut opening = mmcs.open_batch(3, &prover_data);
        opening.opening_proof.0.clear();

        let error = mmcs
            .verify_batch(&commit, &dims, 3, (&opening).into())
            .expect_err("an opening without its salt row must be rejected");
        assert_eq!(
            error.to_string(),
            "batch size mismatch: expected 1 entries, got 0"
        );
    }

    #[test]
    fn hiding_mmcs_is_sync() {
        assert_sync::<MyMmcs>();
    }

    #[test]
    fn multi_opening_round_trip() {
        let mut rng = SmallRng::seed_from_u64(3);
        // Three matrices of equal height but widths 2, 3, 4.
        let mats = (0..3)
            .map(|i| RowMajorMatrix::<F>::rand(&mut rng, 16, i + 2))
            .collect_vec();
        let dims = mats.iter().map(|m| m.dimensions()).collect_vec();
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0, StdRng::seed_from_u64(0));
        let (commit, prover_data) = mmcs.commit(mats);

        // Open four leaves at once.
        // Indices must be ascending and distinct.
        let indices = vec![1usize, 4, 9, 15];
        let (opened, proof) = mmcs.open_multi_batch(&indices, &prover_data);

        // One opened row set per query, one row per committed matrix.
        assert_eq!(opened.len(), indices.len());
        assert!(opened.iter().all(|rows| rows.len() == dims.len()));

        // The salt suffix is stripped: each returned row matches its matrix width.
        for rows in &opened {
            for (row, dim) in rows.iter().zip(&dims) {
                assert_eq!(row.len(), dim.width);
            }
        }

        // The single shared multiproof authenticates every opened row.
        mmcs.verify_multi_batch(&commit, &dims, &indices, &opened, &proof)
            .expect("round-trip multi-opening must verify");
    }

    #[test]
    fn verify_multi_rejects_wrong_row_width() {
        let mut rng = SmallRng::seed_from_u64(4);
        // Commit to one matrix of width 4 through the hiding wrapper.
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 8, 4);
        let dims = vec![mat.dimensions()];
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0, StdRng::seed_from_u64(0));
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        let indices = vec![2usize, 5];
        let (mut opened, proof) = mmcs.open_multi_batch(&indices, &prover_data);

        // Mutation: append one extra element to the first query's only row.
        //
        //     query 0 row:  [a, b, c, d, EXTRA]   (5 values, width is 4)
        opened[0][0].push(F::ONE);

        // The unsalted width is checked before salt columns are re-appended.
        // An over-long row cannot hide behind an under-long salt.
        //
        //     expected: 4 (matrix width)
        //     got:      5 (opened row length)
        let err = mmcs
            .verify_multi_batch(&commit, &dims, &indices, &opened, &proof)
            .expect_err("row longer than the matrix width must be rejected");
        assert!(matches!(
            err,
            MerkleTreeError::WrongWidth {
                matrix: 0,
                expected: 4,
                got: 5,
            }
        ));
    }

    #[test]
    fn verify_multi_reports_the_query_and_salt_counts() {
        let mut rng = SmallRng::seed_from_u64(6);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 8, 4);
        let dims = vec![mat.dimensions()];
        let mmcs = MyMmcs::new(hash, compress, 0, StdRng::seed_from_u64(0));
        let (commit, prover_data) = mmcs.commit(vec![mat]);
        let indices = vec![2usize, 5];
        let (opened, mut proof) = mmcs.open_multi_batch(&indices, &prover_data);

        proof.0.pop();
        let error = mmcs
            .verify_multi_batch(&commit, &dims, &indices, &opened, &proof)
            .expect_err("each query must carry one salt batch");
        assert_eq!(
            error.to_string(),
            "batch size mismatch: expected 2 entries, got 1"
        );

        let (_, mut proof) = mmcs.open_multi_batch(&indices, &prover_data);
        proof.0[0].clear();
        let error = mmcs
            .verify_multi_batch(&commit, &dims, &indices, &opened, &proof)
            .expect_err("each opened matrix row must carry one salt row");
        assert_eq!(
            error.to_string(),
            "batch size mismatch: expected 1 entries, got 0"
        );
    }
}
