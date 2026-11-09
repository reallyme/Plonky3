// Modified by ReallyMe Technologies Limited, 2026.
// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error types raised while committing, opening, or verifying a Merkle MMCS.

use thiserror::Error;

/// Errors that may arise during Merkle tree commitment, opening, or verification.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MerkleTreeError {
    /// A Merkle arity cannot define a valid compression tree.
    #[error("Merkle arity {arity} must be a power of two greater than one")]
    InvalidArity {
        /// Invalid compression arity.
        arity: usize,
    },

    /// Merkle layer dimensions cannot be represented by `usize`.
    #[error("Merkle tree geometry overflows the platform index width")]
    TreeGeometryOverflow,

    /// The fallible Merkle scheduling allocation could not be reserved.
    #[error("Merkle tree schedule allocation failed")]
    TreePlanAllocationFailed,

    /// Packed leaf values and packed digest words cannot share a SIMD schedule.
    #[error(
        "Merkle packing width mismatch: leaf packing is {leaf_width}, digest packing is {digest_width}"
    )]
    PackingWidthMismatch {
        /// Number of scalar leaf values in one packed leaf value.
        leaf_width: usize,
        /// Number of scalar digest words in one packed digest word.
        digest_width: usize,
    },

    /// A guarded matrix or salt buffer length cannot be represented.
    #[error("secret commitment dimensions overflow")]
    SecretDimensionOverflow,

    /// A guarded commitment buffer could not be reserved.
    #[error("secret commitment allocation failed")]
    SecretAllocationFailed,

    /// Recomputed opening buffers could not be reserved.
    #[error("recomputed Merkle opening allocation failed")]
    RecomputedOpeningAllocationFailed,

    /// A hiding commitment requested no salt elements per row.
    #[error("secret hiding commitment requires at least one salt element")]
    SecretInvalidSaltWidth,

    /// A streaming commitment received too few or too many matrices.
    #[error("streaming commitment expected {expected} matrices, got {got}")]
    StreamedMatrixCountMismatch {
        /// Matrix count declared before secret absorption began.
        expected: usize,
        /// Matrix count supplied so far.
        got: usize,
    },

    /// A streamed matrix does not match its predeclared shape.
    #[error(
        "streamed matrix {matrix} has shape {got_width}x{got_height}; expected {expected_width}x{expected_height}"
    )]
    StreamedMatrixShapeMismatch {
        /// Matrix position in commitment order.
        matrix: usize,
        /// Width declared before secret absorption began.
        expected_width: usize,
        /// Height declared before secret absorption began.
        expected_height: usize,
        /// Width supplied by the streamed matrix.
        got_width: usize,
        /// Height supplied by the streamed matrix.
        got_height: usize,
    },

    /// A shape-checked streamed matrix failed to return one of its rows.
    #[error("streamed matrix {matrix} does not contain required row {row}")]
    StreamedRowUnavailable {
        /// Matrix position in commitment order.
        matrix: usize,
        /// Row required by the predeclared matrix height.
        row: usize,
    },

    /// The number of openings provided does not match the expected number.
    #[error("batch size mismatch: expected {expected} entries, got {got}")]
    WrongBatchSize {
        /// Number of openings fixed by the commitment or query set.
        expected: usize,
        /// Number of openings supplied by the proof.
        got: usize,
    },

    /// An opened row's length does not match the width of its matrix.
    #[error("wrong width: matrix {matrix} expected {expected} values, got {got}")]
    WrongWidth {
        /// Index of the offending matrix in the batch.
        matrix: usize,
        /// Width of the matrix according to the caller-supplied dimensions.
        expected: usize,
        /// Length of the opened row provided in the proof.
        got: usize,
    },

    /// The number of proof nodes does not match the expected tree height.
    #[error("wrong height: expected {expected_proof_len} siblings, got {num_siblings}")]
    WrongHeight {
        /// Expected number of sibling hashes in the proof.
        expected_proof_len: usize,

        /// Actual number of sibling hashes provided in the proof.
        num_siblings: usize,
    },

    /// Matrix heights are incompatible; they cannot share a common binary Merkle tree.
    #[error(
        "matrix height {height} incompatible with tallest height {max_height}; \
         expected ceil_div({max_height}, 2^{bits_reduced}) = {expected_height} \
         so every global index maps to a row at depth {bits_reduced}"
    )]
    IncompatibleHeights {
        /// Height that was provided.
        height: usize,
        /// Height of the tallest matrix in the batch.
        max_height: usize,
        /// The only height that covers every global index at this depth.
        expected_height: usize,
        /// Power-of-two reduction depth from the tallest matrix.
        bits_reduced: usize,
    },

    /// Padding prevents a binary bridge from shrinking or reaching the next injection.
    #[error(
        "cannot shrink padded layer of height {padded_height} with arity {arity} \
         to inject matrix of height {height}"
    )]
    NonShrinkingLayer {
        /// Digest count in the layer being folded.
        padded_height: usize,
        /// Height of the next matrix awaiting injection.
        height: usize,
        /// Full compression arity that fixes the minimum padded width.
        arity: usize,
    },

    /// Digest-layer geometry, total digest count, or payload bytes exceed `usize`.
    #[error("Merkle digest storage size overflows usize")]
    DigestStorageOverflow,

    /// The queried row index exceeds the maximum height.
    #[error("index out of bounds: index {index} exceeds max height {max_height}")]
    IndexOutOfBounds {
        /// Maximum admissible height.
        max_height: usize,
        /// Row index that was provided.
        index: usize,
    },

    /// Attempted to open an empty batch (no committed matrices).
    #[error("empty batch: attempted to open an empty batch with no committed matrices")]
    EmptyBatch,

    /// The tallest height rounds up past the largest power of two in `usize`.
    ///
    /// No tree has that leaf layer.
    #[error("tallest height {max_height} exceeds the largest power of two in usize")]
    HeightOverflow {
        /// Height of the tallest matrix in the batch.
        max_height: usize,
    },

    /// Recomputed matrices do not have the same shape as the committed matrices.
    #[error(
        "recomputed matrix {matrix} has shape {got_width}x{got_height}; expected {expected_width}x{expected_height}"
    )]
    RecomputedMatrixShapeMismatch {
        /// Index of the offending matrix in commitment order.
        matrix: usize,
        /// Committed matrix width.
        expected_width: usize,
        /// Committed matrix height.
        expected_height: usize,
        /// Recomputed matrix width.
        got_width: usize,
        /// Recomputed matrix height.
        got_height: usize,
    },

    /// Retained authentication data is internally inconsistent.
    #[error("Merkle authentication data is malformed")]
    MalformedAuthenticationData,

    /// A row expected from a shape-checked recomputed matrix was unavailable.
    #[error("recomputed matrix {matrix} does not contain required row {row}")]
    RecomputedRowUnavailable {
        /// Index of the matrix in commitment order.
        matrix: usize,
        /// Row required by the global query index.
        row: usize,
    },

    /// The computed Merkle digest does not match any entry in the cap.
    #[error("cap mismatch: computed digest does not match any entry in the Merkle cap")]
    CapMismatch,

    /// The cap does not hold as many roots as the tree's top layers produce.
    #[error("wrong cap size: expected {expected} roots, got {got}")]
    WrongCapSize {
        /// Roots the committed dimensions and cap height produce.
        expected: usize,
        /// Roots in the commitment.
        got: usize,
    },

    /// A pruned batch opening could not be restored or validated.
    #[error("malformed pruned proof: {0}")]
    MalformedPrunedProof(#[from] PrunedProofError),
}

/// Why a pruned batch opening was rejected.
///
/// Each variant pins one failure mode with its diagnostic fields.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PrunedProofError {
    /// The flat sibling list does not match the frontier the query set implies.
    ///
    /// The verifier's query set fixes an exact boundary-digest count.
    /// A proof carrying too few or too many digests lands here.
    #[error("wrong pruned sibling count: expected {expected}, got {got}")]
    SiblingCountMismatch {
        /// Digest count the verifier-derived frontier requires.
        expected: usize,
        /// Digest count the proof actually carried.
        got: usize,
    },

    /// Two queries map to one leaf but disagree on opened values.
    #[error("duplicate queries for path slot {slot} disagree on opened values")]
    InconsistentDuplicateOpenings {
        /// Unique-leaf slot the conflicting queries map to.
        slot: usize,
    },

    /// A non-lead member of a merged path group disagrees with the group's
    /// lead on the opened row of a matrix injected at their shared layer.
    #[error(
        "unique path slot {slot} disagrees with its group's lead on matrix {matrix}'s opened row"
    )]
    InconsistentGroupOpening {
        /// Unique-path slot of the disagreeing group member.
        slot: usize,
        /// Index (into the caller-supplied `dimensions`) of the matrix whose opened row diverges.
        matrix: usize,
    },
}
