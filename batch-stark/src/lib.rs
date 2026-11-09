// Modified by ReallyMe Technologies Limited, 2026.
// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

#[cfg(debug_assertions)]
mod check_constraints;
pub mod common;
pub mod config;
pub mod error;
pub mod folder;
pub mod proof;
pub mod prover;
pub mod security;
pub mod symbolic;
pub(crate) mod transcript;
pub mod verifier;

// Re-export main types and functions for convenience
pub use common::{CommonData, ProverData, ProverOnlyData};
pub use config::{
    Challenge, Commitment, Domain, PackedChallenge, PackedVal, PcsError, PcsProof,
    StarkGenericConfig, Val,
};
pub use error::BatchVerificationError;
pub use p3_uni_stark::{OpenedValues, ProvingError, VerificationError};
pub use proof::{BatchCommitments, BatchOpenedValues, BatchProof};
#[cfg(feature = "secret-memory")]
pub use prover::{
    GuardedStarkInstance, SecretBatchProvingError, SecretBatchProvingObserver,
    SecretBatchProvingPhase, SecretBatchProvingResult, SecretQuotientChunkAuditStatus,
    prove_batch_guarded_inputs, prove_batch_secret, prove_batch_secret_with_observer,
};
pub use prover::{StarkInstance, prove_batch};
pub use security::num_batched_openings;
pub use transcript::{
    BatchProverTranscript, BatchShape, BatchTranscriptFailure, BatchVerifierTranscript,
};
pub use verifier::{VerifierData, verify_batch};
