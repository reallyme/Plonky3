// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

use zeroize::ZeroizeOnDrop;

use crate::CryptographicHasher;

/// A cryptographic hasher whose input can be supplied in ordered chunks.
///
/// Incremental hashing is required when a caller cannot retain every input
/// fragment until the complete message is available. Implementations must
/// produce exactly the same digest as [`CryptographicHasher::hash_iter`] over
/// the concatenated chunks, including empty chunks and an empty message.
///
/// The live state can contain the entire absorbed message through the sponge
/// state or a partial block. Requiring [`ZeroizeOnDrop`] makes erasure part of
/// the type contract instead of relying on each caller to remember a cleanup
/// step on success, error, and unwind paths.
pub trait IncrementalCryptographicHasher<Item: Clone, Out>: CryptographicHasher<Item, Out> {
    /// Secret-bearing state retained between chunks.
    type State: ZeroizeOnDrop;

    /// Begin a new message.
    fn begin_hash(&self) -> Self::State;

    /// Append one ordered chunk to the message.
    fn absorb<I>(&self, state: &mut Self::State, input: I)
    where
        I: IntoIterator<Item = Item>;

    /// Finalize the digest and consume the secret-bearing state.
    fn finalize_hash(&self, state: Self::State) -> Out;
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use zeroize::Zeroizing;

    use super::IncrementalCryptographicHasher;
    use crate::CryptographicHasher;

    #[derive(Clone, Copy)]
    struct Fold;

    impl CryptographicHasher<u8, [u8; 2]> for Fold {
        fn hash_iter<I>(&self, input: I) -> [u8; 2]
        where
            I: IntoIterator<Item = u8>,
        {
            fold(input)
        }
    }

    impl IncrementalCryptographicHasher<u8, [u8; 2]> for Fold {
        type State = Zeroizing<Vec<u8>>;

        fn begin_hash(&self) -> Self::State {
            Zeroizing::new(Vec::new())
        }

        fn absorb<I>(&self, state: &mut Self::State, input: I)
        where
            I: IntoIterator<Item = u8>,
        {
            state.extend(input);
        }

        fn finalize_hash(&self, state: Self::State) -> [u8; 2] {
            fold(state.iter().copied())
        }
    }

    fn fold<I>(input: I) -> [u8; 2]
    where
        I: IntoIterator<Item = u8>,
    {
        let mut value = 0_u8;
        let mut length = 0_u8;
        for byte in input {
            value = value.wrapping_mul(31).wrapping_add(byte);
            length = length.wrapping_add(1);
        }
        [value, length]
    }

    #[test]
    fn chunked_hash_matches_one_shot_hash() {
        let message = [1_u8, 2, 3, 4, 5, 6, 7];
        let mut state = Fold.begin_hash();
        Fold.absorb(&mut state, message[..2].iter().copied());
        Fold.absorb(&mut state, core::iter::empty());
        Fold.absorb(&mut state, message[2..5].iter().copied());
        Fold.absorb(&mut state, message[5..].iter().copied());

        assert_eq!(Fold.finalize_hash(state), Fold.hash_slice(&message));
    }

    #[test]
    fn empty_chunked_hash_matches_empty_one_shot_hash() {
        let state = Fold.begin_hash();
        assert_eq!(Fold.finalize_hash(state), Fold.hash_slice(&[]));
    }
}
