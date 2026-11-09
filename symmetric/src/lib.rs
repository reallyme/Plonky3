// Modified by ReallyMe Technologies Limited, 2026.
// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A framework for symmetric cryptography primitives.

#![no_std]

extern crate alloc;

mod compression;
mod hash;
mod hasher;
#[cfg(feature = "secret-memory")]
mod incremental_hasher;
mod permutation;
mod rtl_sponge;
mod serializing_hasher;
mod sponge;

pub use compression::*;
pub use hash::*;
pub use hasher::*;
#[cfg(feature = "secret-memory")]
pub use incremental_hasher::*;
pub use permutation::*;
pub use rtl_sponge::*;
pub use serializing_hasher::*;
pub use sponge::*;
