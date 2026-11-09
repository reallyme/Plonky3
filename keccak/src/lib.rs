// Modified by ReallyMe Technologies Limited, 2026.
// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Keccak-f permutation, and hash functions built from it.

#![no_std]

#[cfg(test)]
extern crate alloc;

use p3_symmetric::{CryptographicHasher, CryptographicPermutation, Permutation};
use sha3::digest::Output;
use sha3::digest::zeroize::ZeroizeOnDrop;
use sha3::{Digest, Sha3_384 as ZeroizingSha3_384, Sha3_512 as ZeroizingSha3_512};
use tiny_keccak::{Hasher, Keccak, Sha3, keccakf};

mod batch;

// On x86-64 every wide backend is compiled, for the batched hash to pick at run time.
//
// Only the one the build enables is public, as the permutation of packed states.
//
// A soft-float target has no SSE2 and must not enable vector features, so it skips the wide ones.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub mod avx512;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx512f")
))]
mod avx512;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use avx512::*;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub mod avx2;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx2")
))]
mod avx2;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub use avx2::*;

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
pub mod sse2;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
pub use sse2::*;

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "sha3"
))]
pub mod neon_sha3;
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "sha3"
))]
pub use neon_sha3::*;

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    not(target_feature = "sha3")
))]
pub mod neon;
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    not(target_feature = "sha3")
))]
pub use neon::*;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm32_simd128;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use wasm32_simd128::*;

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
mod fallback;
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
pub use fallback::*;

/// The Keccak-f permutation.
#[derive(Copy, Clone, Debug)]
pub struct KeccakF;

impl Permutation<[u64; 25]> for KeccakF {
    fn permute_mut(&self, input: &mut [u64; 25]) {
        keccakf(input);
    }
}

impl CryptographicPermutation<[u64; 25]> for KeccakF {}

impl Permutation<[u8; 200]> for KeccakF {
    fn permute(&self, input_u8s: [u8; 200]) -> [u8; 200] {
        let mut state_u64s: [u64; 25] = core::array::from_fn(|i| {
            u64::from_le_bytes(input_u8s[i * 8..][..8].try_into().unwrap())
        });

        keccakf(&mut state_u64s);

        core::array::from_fn(|i| {
            let u64_limb = state_u64s[i / 8];
            u64_limb.to_le_bytes()[i % 8]
        })
    }
}

impl CryptographicPermutation<[u8; 200]> for KeccakF {}

/// Feed a byte stream to a one-message sponge and return its fixed-size digest.
#[inline]
fn finalize_iter<const N: usize, S: Hasher>(
    mut sponge: S,
    input: impl IntoIterator<Item = u8>,
) -> [u8; N] {
    const BUFLEN: usize = 512; // Tweakable parameter; determined by experiment
    p3_util::apply_to_chunks::<BUFLEN, _, _>(input, |buf| sponge.update(buf));

    let mut output = [0u8; N];
    sponge.finalize(&mut output);
    output
}

/// Feed a run of byte slices to a one-message sponge and return its fixed-size digest.
#[inline]
fn finalize_slices<'a, const N: usize, S: Hasher>(
    mut sponge: S,
    input: impl IntoIterator<Item = &'a [u8]>,
) -> [u8; N] {
    for chunk in input {
        sponge.update(chunk);
    }

    let mut output = [0u8; N];
    sponge.finalize(&mut output);
    output
}

/// Feed a byte stream to a RustCrypto sponge whose state is erased on drop.
#[inline]
fn finalize_zeroizing_iter<const N: usize, D>(
    mut sponge: D,
    input: impl IntoIterator<Item = u8>,
) -> [u8; N]
where
    D: Digest + ZeroizeOnDrop,
    Output<D>: Into<[u8; N]>,
{
    const BUFLEN: usize = 512;
    p3_util::apply_to_chunks::<BUFLEN, _, _>(input, |buf| Digest::update(&mut sponge, buf));
    sponge.finalize().into()
}

/// Feed byte slices to a RustCrypto sponge whose state is erased on drop.
#[inline]
fn finalize_zeroizing_slices<'a, const N: usize, D>(
    mut sponge: D,
    input: impl IntoIterator<Item = &'a [u8]>,
) -> [u8; N]
where
    D: Digest + ZeroizeOnDrop,
    Output<D>: Into<[u8; N]>,
{
    for chunk in input {
        Digest::update(&mut sponge, chunk);
    }
    sponge.finalize().into()
}

/// The `Keccak` hash functions defined in
/// [Keccak SHA3 submission](https://keccak.team/files/Keccak-submission-3.pdf).
#[derive(Copy, Clone, Debug)]
pub struct Keccak256Hash;

impl CryptographicHasher<u8, [u8; 32]> for Keccak256Hash {
    const LANES: usize = batch::LANES;

    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = u8>,
    {
        finalize_iter(Keccak::v256(), input)
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        finalize_slices(Keccak::v256(), input)
    }

    fn hash_many(&self, input: &[u8], out: &mut [[u8; 32]]) {
        batch::hash_many(batch::KECCAK_DOMAIN, input, out);
    }
}

/// The SHA3-256 hash function of [FIPS 202](https://doi.org/10.6028/NIST.FIPS.202).
///
/// It runs the same sponge as Keccak-256, at the same cost.
///
/// Only the padding differs, so the two digests differ for every input:
///
/// ```text
///     Keccak-256("abc") = 4e03657a ea45a94f ...
///     SHA3-256("abc")   = 3a985da7 4fe225b2 ...
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Sha3_256Hash;

impl CryptographicHasher<u8, [u8; 32]> for Sha3_256Hash {
    const LANES: usize = batch::LANES;

    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = u8>,
    {
        finalize_iter(Sha3::v256(), input)
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        finalize_slices(Sha3::v256(), input)
    }

    fn hash_many(&self, input: &[u8], out: &mut [[u8; 32]]) {
        batch::hash_many(batch::SHA3_DOMAIN, input, out);
    }
}

/// The SHA3-384 hash function of [FIPS 202](https://doi.org/10.6028/NIST.FIPS.202).
///
/// Its sponge state is zeroized when hashing completes or unwinds.
#[derive(Copy, Clone, Debug)]
pub struct Sha3_384Hash;

impl CryptographicHasher<u8, [u8; 48]> for Sha3_384Hash {
    fn hash_iter<I>(&self, input: I) -> [u8; 48]
    where
        I: IntoIterator<Item = u8>,
    {
        finalize_zeroizing_iter(ZeroizingSha3_384::new(), input)
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 48]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        finalize_zeroizing_slices(ZeroizingSha3_384::new(), input)
    }
}

impl CryptographicHasher<u8, [u64; 6]> for Sha3_384Hash {
    fn hash_iter<I>(&self, input: I) -> [u64; 6]
    where
        I: IntoIterator<Item = u8>,
    {
        sha3_384_digest_to_words(finalize_zeroizing_iter(ZeroizingSha3_384::new(), input))
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u64; 6]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        sha3_384_digest_to_words(finalize_zeroizing_slices(ZeroizingSha3_384::new(), input))
    }
}

/// Hash canonical word streams by absorbing `LE64(word[0]) || LE64(word[1]) || ...`.
///
/// This representation is used when SHA3-384 digests feed a Merkle compression layer.
/// It is independent of native endianness and in-memory layout.
impl CryptographicHasher<u64, [u64; 6]> for Sha3_384Hash {
    fn hash_iter<I>(&self, input: I) -> [u64; 6]
    where
        I: IntoIterator<Item = u64>,
    {
        let bytes = input.into_iter().flat_map(u64::to_le_bytes);
        sha3_384_digest_to_words(finalize_zeroizing_iter(ZeroizingSha3_384::new(), bytes))
    }
}

/// The SHA3-512 hash function of [FIPS 202](https://doi.org/10.6028/NIST.FIPS.202).
///
/// Its sponge state is zeroized when hashing completes or unwinds.
#[derive(Copy, Clone, Debug)]
pub struct Sha3_512Hash;

impl CryptographicHasher<u8, [u8; 64]> for Sha3_512Hash {
    fn hash_iter<I>(&self, input: I) -> [u8; 64]
    where
        I: IntoIterator<Item = u8>,
    {
        finalize_zeroizing_iter(ZeroizingSha3_512::new(), input)
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 64]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        finalize_zeroizing_slices(ZeroizingSha3_512::new(), input)
    }
}

impl CryptographicHasher<u8, [u64; 8]> for Sha3_512Hash {
    fn hash_iter<I>(&self, input: I) -> [u64; 8]
    where
        I: IntoIterator<Item = u8>,
    {
        sha3_512_digest_to_words(finalize_zeroizing_iter(ZeroizingSha3_512::new(), input))
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u64; 8]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        sha3_512_digest_to_words(finalize_zeroizing_slices(ZeroizingSha3_512::new(), input))
    }
}

/// Hash canonical word streams by absorbing `LE64(word[0]) || LE64(word[1]) || ...`.
///
/// This representation is used when SHA3-512 digests feed a Merkle compression layer.
/// It is independent of native endianness and in-memory layout.
impl CryptographicHasher<u64, [u64; 8]> for Sha3_512Hash {
    fn hash_iter<I>(&self, input: I) -> [u64; 8]
    where
        I: IntoIterator<Item = u64>,
    {
        let bytes = input.into_iter().flat_map(u64::to_le_bytes);
        sha3_512_digest_to_words(finalize_zeroizing_iter(ZeroizingSha3_512::new(), bytes))
    }
}

/// Interpret a SHA3-384 digest as six canonical little-endian words.
#[must_use]
pub fn sha3_384_digest_to_words(bytes: [u8; 48]) -> [u64; 6] {
    let (chunks, _) = bytes.as_chunks::<8>();
    let mut words = [0; 6];
    for (word, chunk) in words.iter_mut().zip(chunks) {
        *word = u64::from_le_bytes(*chunk);
    }
    words
}

/// Interpret a SHA3-512 digest as eight canonical little-endian words.
#[must_use]
pub fn sha3_512_digest_to_words(bytes: [u8; 64]) -> [u64; 8] {
    let (chunks, _) = bytes.as_chunks::<8>();
    let mut words = [0; 8];
    for (word, chunk) in words.iter_mut().zip(chunks) {
        *word = u64::from_le_bytes(*chunk);
    }
    words
}

/// Serialize six canonical SHA3-384 words to the original digest bytes.
///
/// This is the inverse of [`sha3_384_digest_to_words`].
#[must_use]
pub fn sha3_384_words_to_digest(words: [u64; 6]) -> [u8; 48] {
    let mut bytes = [0; 48];
    let (chunks, _) = bytes.as_chunks_mut::<8>();
    for (chunk, word) in chunks.iter_mut().zip(words) {
        *chunk = word.to_le_bytes();
    }
    bytes
}

/// Serialize eight canonical SHA3-512 words to the original digest bytes.
///
/// This is the inverse of [`sha3_512_digest_to_words`].
#[must_use]
pub fn sha3_512_words_to_digest(words: [u64; 8]) -> [u8; 64] {
    let mut bytes = [0; 64];
    let (chunks, _) = bytes.as_chunks_mut::<8>();
    for (chunk, word) in chunks.iter_mut().zip(words) {
        *chunk = word.to_le_bytes();
    }
    bytes
}

#[cfg(test)]
mod nist_tests;

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use hex_literal::hex;
    use p3_symmetric::{CompressionFunctionFromHasher, PseudoCompressionFunction};
    use proptest::prelude::*;
    use sha3::{Digest, Keccak256, Sha3_256};

    use super::*;
    use crate::batch::{KECCAK_DOMAIN, LANES, LAST_RATE_WORD, RATE, SHA3_DOMAIN, supported};

    /// Every message length that changes the shape of the absorb loop.
    ///
    /// The block boundaries are the interesting ones:
    ///
    /// - A length one short of the rate folds both padding marks into a single byte.
    /// - A length that is an exact multiple of the rate pads a block carrying no message bytes.
    /// - A length whose leftover bytes already sit in the closing word of the rate puts both
    ///   marks in that word, in different bytes.
    const SHAPE_LENGTHS: [usize; 16] = [
        0,
        1,
        7,
        8,
        9,
        RATE - 3,
        RATE - 1,
        RATE,
        RATE + 1,
        2 * RATE - 1,
        2 * RATE,
        2 * RATE + 1,
        3 * RATE,
        8 * LAST_RATE_WORD,
        400,
        1000,
    ];

    /// Hash each message on its own, which is the behaviour the batched path must reproduce.
    fn reference<H>(hasher: &H, messages: &[u8], len: usize, count: usize) -> Vec<[u8; 32]>
    where
        H: CryptographicHasher<u8, [u8; 32]>,
    {
        (0..count)
            .map(|k| {
                // A zero-length message still has a digest, so slice defensively.
                let message = if len == 0 {
                    &messages[..0]
                } else {
                    &messages[k * len..(k + 1) * len]
                };
                hasher.hash_slice(message)
            })
            .collect()
    }

    /// Batch every shape length at every count up to two lane groups plus one.
    ///
    /// The public batched path runs first, then every backend the CPU supports.
    fn check_block_shapes<H>(hasher: &H, domain: u8)
    where
        H: CryptographicHasher<u8, [u8; 32]>,
    {
        // Batch sizes below, at, and above one full group of the widest backend.
        // The final short group is then exercised at every lane count.
        let counts: Vec<usize> = (1..=2 * LANES + 1).collect();

        for len in SHAPE_LENGTHS {
            for &count in &counts {
                // Fixture: a deterministic byte ramp, distinct per position.
                let messages: Vec<u8> = (0..len * count).map(|i| (i * 31 + 7) as u8).collect();
                let expected = reference(hasher, &messages, len, count);

                let mut batched = vec![[0u8; 32]; count];
                hasher.hash_many(&messages, &mut batched);
                assert_eq!(batched, expected, "len {len}, count {count}");

                for kernel in supported() {
                    let mut batched = vec![[0u8; 32]; count];
                    kernel.hash_many(domain, &messages, len, &mut batched);
                    assert_eq!(batched, expected, "{kernel:?}, len {len}, count {count}");
                }
            }
        }
    }

    #[test]
    fn keccak256_hash_many_matches_scalar_across_block_shapes() {
        check_block_shapes(&Keccak256Hash, KECCAK_DOMAIN);
    }

    #[test]
    fn sha3_256_hash_many_matches_scalar_across_block_shapes() {
        check_block_shapes(&Sha3_256Hash, SHA3_DOMAIN);
    }

    #[test]
    fn hash_many_splits_input_by_digest_count() {
        // The message length is the input length divided by the digest count.
        // 64 bytes and 4 digests therefore read as four adjacent 16-byte messages.
        let messages: Vec<u8> = (0..64).map(|i| i as u8).collect();

        // 4 messages of 16 bytes each.
        let mut digests = [[0u8; 32]; 4];
        Keccak256Hash.hash_many(&messages, &mut digests);

        for (k, digest) in digests.iter().enumerate() {
            assert_eq!(*digest, Keccak256Hash.hash_slice(&messages[k * 16..][..16]));
        }
    }

    #[test]
    #[should_panic(expected = "must be a whole multiple")]
    fn hash_many_rejects_ragged_input() {
        // 5 bytes cannot split into 2 equal messages.
        // The contract fails up front rather than misaligning message boundaries.
        let mut digests = [[0u8; 32]; 2];
        Sha3_256Hash.hash_many(&[1, 2, 3, 4, 5], &mut digests);
    }

    #[test]
    fn the_two_hashes_differ_only_by_padding() {
        // Both sponges run the same permutation at the same rate.
        // The first padding byte alone separates them:
        //
        //     Keccak-256:  "abc" | 0x01 | 0x00 ... 0x00 | 0x80
        //     SHA3-256:    "abc" | 0x06 | 0x00 ... 0x00 | 0x80
        //
        // Published digests of "abc" for each.
        let keccak = hex!("4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45");
        let sha3 = hex!("3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532");

        // Scalar and batched paths agree with the published values.
        let mut batched = [[0u8; 32]; 1];
        Keccak256Hash.hash_many(b"abc", &mut batched);
        assert_eq!(batched[0], keccak);
        assert_eq!(Keccak256Hash.hash_slice(b"abc"), keccak);

        Sha3_256Hash.hash_many(b"abc", &mut batched);
        assert_eq!(batched[0], sha3);
        assert_eq!(Sha3_256Hash.hash_slice(b"abc"), sha3);
    }

    #[test]
    fn empty_messages_hash_to_the_published_digests() {
        // Zero-length messages take the padding-only branch of the batched path.
        // Three of them in one call must all get the digest of the empty string.
        let keccak = hex!("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");
        let sha3 = hex!("a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a");

        let mut digests = [[0u8; 32]; 3];
        Keccak256Hash.hash_many(&[], &mut digests);
        assert_eq!(digests, [keccak; 3]);

        Sha3_256Hash.hash_many(&[], &mut digests);
        assert_eq!(digests, [sha3; 3]);
    }

    #[test]
    fn sha3_384_and_512_match_fips_vectors() {
        let sha3_384_empty = hex!(
            "0c63a75b845e4f7d01107d852e4c2485c51a50aaaa94fc61995e71bbee983a2a"
            "c3713831264adb47fb6bd1e058d5f004"
        );
        let sha3_384_abc = hex!(
            "ec01498288516fc926459f58e2c6ad8df9b473cb0fc08c2596da7cf0e49be4b2"
            "98d88cea927ac7f539f1edf228376d25"
        );
        let sha3_512_empty = hex!(
            "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a6"
            "15b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26"
        );
        let sha3_512_abc = hex!(
            "b751850b1a57168a5693cd924b6b096e08f621827444f70d884f5d0240d2712e"
            "10e116e9192af3c91a7ec57647e3934057340b4cf408d5a56592f8274eec53f0"
        );

        let digest_384_empty: [u8; 48] = Sha3_384Hash.hash_slice(b"");
        let digest_384_abc: [u8; 48] = Sha3_384Hash.hash_slice(b"abc");
        let digest_512_empty: [u8; 64] = Sha3_512Hash.hash_slice(b"");
        let digest_512_abc: [u8; 64] = Sha3_512Hash.hash_slice(b"abc");

        assert_eq!(digest_384_empty, sha3_384_empty);
        assert_eq!(digest_384_abc, sha3_384_abc);
        assert_eq!(digest_512_empty, sha3_512_empty);
        assert_eq!(digest_512_abc, sha3_512_abc);
    }

    #[test]
    fn sha3_word_digests_round_trip_to_normative_bytes() {
        let bytes_384: [u8; 48] = Sha3_384Hash.hash_slice(b"canonical SHA3-384 encoding");
        let words_384: [u64; 6] = Sha3_384Hash.hash_slice(b"canonical SHA3-384 encoding");
        let bytes_512: [u8; 64] = Sha3_512Hash.hash_slice(b"canonical SHA3-512 encoding");
        let words_512: [u64; 8] = Sha3_512Hash.hash_slice(b"canonical SHA3-512 encoding");

        assert_eq!(sha3_384_words_to_digest(words_384), bytes_384);
        assert_eq!(sha3_512_words_to_digest(words_512), bytes_512);
        assert_eq!(sha3_384_digest_to_words(bytes_384), words_384);
        assert_eq!(sha3_512_digest_to_words(bytes_512), words_512);
    }

    #[test]
    fn sha3_word_inputs_use_canonical_little_endian_bytes() {
        let words = [
            0x0001_0203_0405_0607_u64,
            0x8899_aabb_ccdd_eeff,
            0x1020_3040_5060_7080,
        ];
        let bytes: Vec<u8> = words.into_iter().flat_map(u64::to_le_bytes).collect();

        let expected_384: [u8; 48] = Sha3_384Hash.hash_slice(&bytes);
        let actual_384: [u64; 6] = Sha3_384Hash.hash_slice(&words);
        let expected_512: [u8; 64] = Sha3_512Hash.hash_slice(&bytes);
        let actual_512: [u64; 8] = Sha3_512Hash.hash_slice(&words);

        assert_eq!(sha3_384_words_to_digest(actual_384), expected_384);
        assert_eq!(sha3_512_words_to_digest(actual_512), expected_512);
    }

    #[test]
    fn sha3_segmented_byte_inputs_match_contiguous_inputs() {
        let segments: [&[u8]; 4] = [b"segmented", b" SHA3", b" input", b""];
        let contiguous = b"segmented SHA3 input";

        let segmented_384: [u8; 48] = Sha3_384Hash.hash_iter_slices(segments);
        let contiguous_384: [u8; 48] = Sha3_384Hash.hash_slice(contiguous);
        let segmented_512: [u8; 64] = Sha3_512Hash.hash_iter_slices(segments);
        let contiguous_512: [u8; 64] = Sha3_512Hash.hash_slice(contiguous);

        assert_eq!(segmented_384, contiguous_384);
        assert_eq!(segmented_512, contiguous_512);
    }

    #[test]
    fn sha3_word_hashers_compose_with_binary_merkle_compression() {
        let children_384 = [[0x11_u64; 6], [0x22_u64; 6]];
        let compression_384 = CompressionFunctionFromHasher::<_, 2, 6>::new(Sha3_384Hash);
        let expected_384: [u64; 6] = Sha3_384Hash.hash_slice(children_384.as_flattened());
        assert_eq!(compression_384.compress(children_384), expected_384);

        let children_512 = [[0x33_u64; 8], [0x44_u64; 8]];
        let compression_512 = CompressionFunctionFromHasher::<_, 2, 8>::new(Sha3_512Hash);
        let expected_512: [u64; 8] = Sha3_512Hash.hash_slice(children_512.as_flattened());
        assert_eq!(compression_512.compress(children_512), expected_512);
    }

    /// A cheap deterministic byte stream, so a failing case shrinks reproducibly.
    fn stream(seed: u64, len: usize) -> Vec<u8> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    proptest! {
        #[test]
        fn keccak256_hash_many_matches_scalar_on_random_batches(
            len in 0usize..=400,
            count in 1usize..=17,
            seed in any::<u64>(),
        ) {
            let messages = stream(seed, len * count);

            let mut batched = vec![[0u8; 32]; count];
            Keccak256Hash.hash_many(&messages, &mut batched);

            prop_assert_eq!(batched, reference(&Keccak256Hash, &messages, len, count));
        }

        #[test]
        fn sha3_256_matches_an_independent_implementation(
            len in 0usize..=600,
            count in 1usize..=2 * LANES + 1,
            seed in any::<u64>(),
        ) {
            // Fixture: `count` random messages of `len` bytes, back to back.
            let messages = stream(seed, len * count);

            // Batched and one-message digests of this crate.
            let mut batched = vec![[0u8; 32]; count];
            Sha3_256Hash.hash_many(&messages, &mut batched);
            let scalar = reference(&Sha3_256Hash, &messages, len, count);

            // Digests of the RustCrypto implementation, which shares no code with this one.
            let expected: Vec<[u8; 32]> = (0..count)
                .map(|k| Sha3_256::digest(&messages[k * len..(k + 1) * len]).into())
                .collect();

            prop_assert_eq!(&batched, &expected);
            prop_assert_eq!(&scalar, &expected);

            // Every backend the CPU supports, not only the one picked at run time.
            for kernel in supported() {
                let mut batched = vec![[0u8; 32]; count];
                kernel.hash_many(SHA3_DOMAIN, &messages, len, &mut batched);
                prop_assert_eq!(&batched, &expected, "{:?}", kernel);
            }
        }

        #[test]
        fn sha3_384_and_512_match_independent_implementations(
            len in 0usize..=600,
            seed in any::<u64>(),
        ) {
            let message = stream(seed, len);
            let digest_384: [u8; 48] = Sha3_384Hash.hash_slice(&message);
            let digest_512: [u8; 64] = Sha3_512Hash.hash_slice(&message);
            let expected_384 = finalize_iter(Sha3::v384(), message.iter().copied());
            let expected_512 = finalize_iter(Sha3::v512(), message.iter().copied());

            prop_assert_eq!(digest_384, expected_384);
            prop_assert_eq!(digest_512, expected_512);
        }

        #[test]
        fn keccak256_matches_an_independent_implementation(
            len in 0usize..=600,
            count in 1usize..=2 * LANES + 1,
            seed in any::<u64>(),
        ) {
            // Fixture: `count` random messages of `len` bytes, back to back.
            let messages = stream(seed, len * count);

            // Digests of the RustCrypto implementation, which shares no code with this one.
            // Its Keccak-256 pads with the first byte 0x01 of the SHA-3 submission.
            let expected: Vec<[u8; 32]> = (0..count)
                .map(|k| Keccak256::digest(&messages[k * len..(k + 1) * len]).into())
                .collect();

            // One-message digests, through tiny-keccak.
            prop_assert_eq!(&reference(&Keccak256Hash, &messages, len, count), &expected);

            // The batched path, on the backend picked at run time.
            let mut batched = vec![[0u8; 32]; count];
            Keccak256Hash.hash_many(&messages, &mut batched);
            prop_assert_eq!(&batched, &expected);

            // Every backend the CPU supports, including the narrower ones.
            for kernel in supported() {
                let mut batched = vec![[0u8; 32]; count];
                kernel.hash_many(KECCAK_DOMAIN, &messages, len, &mut batched);
                prop_assert_eq!(&batched, &expected, "{:?}", kernel);
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    mod wasm_simd_tests {
        use tiny_keccak::keccakf;

        use super::*;

        fn permute_packed(states: [[u64; 25]; 2]) -> [[u64; 25]; 2] {
            let mut packed = core::array::from_fn(|word| [states[0][word], states[1][word]]);
            KeccakF.permute_mut(&mut packed);
            core::array::from_fn(|lane| core::array::from_fn(|word| packed[word][lane]))
        }

        fn permute_scalar(mut states: [[u64; 25]; 2]) -> [[u64; 25]; 2] {
            keccakf(&mut states[0]);
            keccakf(&mut states[1]);
            states
        }

        #[test]
        fn simd_permutation_matches_scalar_for_zero_and_ones() {
            let states = [[0; 25], [u64::MAX; 25]];
            assert_eq!(permute_packed(states), permute_scalar(states));
        }

        #[test]
        fn simd_lanes_do_not_influence_each_other() {
            let lane0 =
                core::array::from_fn(|i| 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(i as u64 + 1));
            let lane1_a = core::array::from_fn(|i| (i as u64).rotate_left(i as u32));
            let lane1_b = core::array::from_fn(|i| !(i as u64).wrapping_mul(0x0101_0101_0101_0101));

            let output_a = permute_packed([lane0, lane1_a]);
            let output_b = permute_packed([lane0, lane1_b]);

            assert_eq!(output_a[0], output_b[0]);
            assert_eq!(output_a, permute_scalar([lane0, lane1_a]));
            assert_eq!(output_b, permute_scalar([lane0, lane1_b]));
        }

        proptest! {
            #[test]
            fn simd_permutation_matches_scalar_for_arbitrary_words(
                states in prop::array::uniform2(prop::array::uniform25(any::<u64>()))
            ) {
                let expected = permute_scalar(states);
                let computed = permute_packed(states);
                prop_assert_eq!(computed, expected);
            }
        }
    }
}
