// Modified by ReallyMe Technologies Limited, 2026.
// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod adapters;
mod domain;
mod encoder;
mod mmcs;
mod pcs;
mod periodic;
#[cfg(feature = "secret-memory")]
mod secret;

#[cfg(any(test, feature = "test-utils"))]
pub mod testing;

pub use adapters::*;
pub use domain::*;
pub use encoder::*;
pub use mmcs::*;
pub use pcs::*;
pub use periodic::*;
#[cfg(feature = "secret-memory")]
pub use secret::*;
