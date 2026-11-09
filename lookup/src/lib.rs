// Modified by ReallyMe Technologies Limited, 2026.
// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod builder;
mod bus;
mod challenges;
mod count;
pub mod debug_util;
pub mod indexed;
pub mod logup;
pub mod protocol;
#[cfg(feature = "secret-memory")]
mod secret;
pub mod symbolic;
#[cfg(test)]
mod tests;
pub mod traits;
mod types;

pub use builder::{
    IndexedLookupBuilder, InteractionBuilder, LocalTuple, SymbolicInteraction,
    SymbolicLocalInteraction,
};
pub use bus::{LookupBus, PermutationCheckBus};
pub use challenges::Challenges;
pub use count::Count;
pub use indexed::{IndexedLookupError, IndexedLookups, IndexedRead, IndexedTable, TraceWindow};
pub use logup::LogUpGadget;
pub use protocol::LookupProtocol;
#[cfg(feature = "secret-memory")]
pub use secret::{SecretLookupGenerationError, SecretLookupTraceResult};
pub use symbolic::InteractionSymbolicBuilder;
pub use types::*;
