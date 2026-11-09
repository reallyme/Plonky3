// Modified by ReallyMe Technologies Limited, 2026.
// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

mod builder;
mod hiding_mmcs;
#[cfg(feature = "secret-memory")]
mod incremental_merkle_tree;
mod merkle_tree;
mod mmcs;
mod pruning;
mod storage;

#[cfg(test)]
mod packed_row_tests;

pub use hiding_mmcs::*;
#[cfg(feature = "secret-memory")]
pub use incremental_merkle_tree::{IncrementalMerkleCommitment, IncrementalMerkleTreeBuilder};
pub use merkle_tree::{MerkleTree, MerkleTreeAuthenticationData};
pub use mmcs::{MerkleTreeError, MerkleTreeMmcs, PrunedBatchOpening, PrunedProofError};
pub use p3_symmetric::MerkleCap;
pub use pruning::{MerkleAuthPath, PrunedMerklePaths};
pub use storage::MerkleDigestStorage;
