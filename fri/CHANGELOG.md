<!-- Modified by ReallyMe Technologies Limited, 2026. -->
<!-- SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited -->
<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Add guarded two-adic, quotient, and hiding-FRI commitment construction with erasable retained
  LDEs.
- Retain guarded compact coefficients in the two-adic PCS so ordinary trace and masked quotient
  commitments can reconstruct only verifier-selected LDE rows after full Merkle leaves are erased.
- Add a typed PCS bridge that preserves mixed-height MMCS index semantics while reconstructing,
  detaching, and authenticating selected rows, plus a FRI driver boundary that defers input
  openings until Fiat-Shamir query indices are fixed.
- Add guarded FRI folding and commit-phase construction. Fold outputs and multi-round scratch
  storage are guarded before their first secret-derived write, while every committed extension
  codeword remains owned by a zeroizing MMCS data wrapper.
- Add guarded terminal-polynomial conversion and commit-layer query answering. The inverse DFT
  stays behind zeroizing extension- and base-field owners until the protocol-public coefficients
  are copied out, and query answering consumes and erases every retained MMCS layer.
- Add a complete guarded FRI driver with typed preflight, explicit transcript abort on fallible
  phase errors, non-panicking transcript finalization, and deferred input openings after
  Fiat-Shamir fixes the query indices.
- Add a complete guarded two-adic PCS opening path. Interpolation, quotient reduction, FRI
  folding, deferred input reconstruction, and Merkle authentication preserve zeroizing ownership;
  generated proofs are accepted by the ordinary verifier.
- Expose that guarded two-adic opening path through the protocol-level `SecretPcsOpener`
  capability with stable public failure categories.
- Implement guarded one-shot opening for hiding FRI commitments, including profile, shape, and
  disclosure-budget validation before consumption and ordinary-verifier proof parity.
- Add guarded two-adic domain evaluation from retained coefficients, including hiding-FRI profile
  validation and zeroizing removal of random codeword columns.
- Recover exact pre-randomization hiding-FRI source traces by reconstructing the committed domain,
  removing random codeword columns, and consuming the interleaved masking rows under zeroizing
  ownership. Domain-shape mismatches fail with typed errors.
- Stream hiding quotient masks one chunk at a time while retaining only the final constrained mask
  accumulator. Quotient sources are erased immediately after guarded chunk distribution, before
  randomization, coefficient retention, and LDE expansion. Commitment and post-commitment RNG
  parity remain identical to the ordinary path.
- Detach transparent and hiding quotient value leaves immediately after their global commitment is
  fixed. Compact coefficients provide checked opening geometry, while digest layers and hiding
  salts remain in authentication-only state for transcript-selected queries. Guarded opening now
  accepts this pre-detached state without changing commitments, proof bytes, or verifier behavior.
- Add an explicit guarded MMCS operation that returns authentication-only state as part of
  commitment construction. Transparent and hiding quotient commitments now use this single
  transition, with tests pinning source erasure, recomputed-row verification, hiding-salt parity,
  and unchanged hiding-RNG consumption. The default implementation composes the existing commit
  and detach operations so optimized MMCS implementations can later release matrices during tree
  construction without changing the PCS boundary.
- Add guarded hiding-FRI randomization-polynomial commitments. Generated evaluations enter a
  zeroizing owner before their first random write, and all domain budgets are validated before the
  prover RNG advances.
- Add mixed guarded PCS opening for transparent and hiding FRI. Public preprocessing remains
  reusable under ordinary immutable ownership while private value leaves are detached before
  quotient reduction; authentication data remains available for the later transcript-selected
  queries.
- Add guarded multi-instance quotient commitment for transparent and hiding FRI, retaining one
  global commitment root while validating every instance before transformation or hiding-RNG use.
- Add planned quotient generation for transparent and hiding FRI. Public geometry is preflighted
  before generation or RNG use, each contiguous quotient is released after chunk transformation,
  and the resulting LDEs retain the original single-root order.
- Add matrix-scoped reconstruction of complete committed LDEs from guarded retained coefficients,
  including exact parity before and after Merkle value-leaf detachment.
- Stream guarded trace masking, LDE construction, and recomputable MMCS absorption one matrix at a
  time, returning already-detached authentication state. Complete batch geometry is rejected
  before masking RNG use. Tests pin commitment, opening, verifier, and hiding-RNG parity with the
  retained commitment path.

### Changed
- Reduce the two-adic PCS opening live set by fusing row compression with opening-quotient
  accumulation and releasing interpolation and denominator tables at their final consumers.
- Factor interpolation and quotient reduction into an owned prepared-opening state. The ordinary
  PCS path now ends every committed-matrix borrow before FRI query selection, establishing the
  lifetime boundary required for guarded detachment without changing proof or verifier types.
- Route ordinary FRI input openings through the deferred opener used by the guarded path, keeping
  transcript order and proof wire types identical.
- Preserve ordinary commitments, folding challenges, final-polynomial values, and opening proofs
  across guarded FRI commit construction, with typed shape and allocation failures.
- Preserve ordinary final-polynomial bytes, query indices, sibling values, and pruned Merkle
  multiproofs across the guarded terminal and query boundaries.
- Route ordinary and guarded PCS openings through one reduction implementation with explicit
  storage and interpolation policies, rejecting unsupported opening points before transcript
  mutation and aborting incomplete transcripts on typed failures.
- Split claimed-opening interpolation from quotient reduction. Guarded openings now release every
  committed-matrix borrow, validate geometry, detach full private value leaves, and evaluate claims
  directly from retained coefficients before reconstructing, reducing, and erasing one private LDE
  matrix at a time. Independent interpolation, proof-component, and challenger-state tests preserve
  the transcript and proof format exactly.
- Distinguish typed input-detachment failures from failures reconstructing an already-detached
  matrix during guarded opening reduction.

## [0.8.0] - 2026-09-23
### Merged PRs
- Perf(fri): closed-form fold_row, in-place fold_matrix, hoisted verifier alpha ladder, fused hiding LDE (#2015)
- Test(fri): pin the fused quotient LDE for the bit-reversed DFT backend (#2025)
- Fix(fri)!: derive the folding schedule instead of accepting the prover's (#2033)
- Fix(security,fri): accounting and grinding hygiene fixes (#2048)
- Chore: expose the parallel feature on every crate that uses rayon (#2049)
- Feat(fri)!: drive the FRI transcript through the typed Fiat-Shamir layer (#2035)
- Refactor(commit): separate univariate STARK capabilities from PCS (#2059)
- Test: share PCS opening contracts and batch fixtures (#2061)
- Feat(fri)!: drive the FRI PCS transcript through the typed Fiat-Shamir layer (#2086)
- Fix(security)!: enforce PCS budgets and compose multi-STARK soundness (#2100)
- Fix(security): reject non-canonical grinding witnesses at zero difficulty (#2106)
- Test(fri): pin every shape field to the transcript seed (#2107)
- Test(challenger): assert no two protocols share a transcript seed (#2108)
- Feat(security): check recorded grinding difficulties against the security model (#2109)
- Feat(pcs)!: add batching grinding to STIR and Circle (#2112)
- Fix(transcript): address the review follow-ups from the typed-transcript stack (#2117)
- Review: couple tweaks here and there (#2120)
- Fix: address the minor follow-ups from the #2120 review (#2127)
- Refactor(fri)!: derive the commit-round shape from configuration rather than the proof (#2125)
- Perf(fri): only compute the adjusted-weight prefix interpolation reads (#2189)
- Perf(fri): drop the no-op twiddle rewrite in the multi-round fold (#2190)
- Perf(fri): shrink the extrapolation fallback's inverse transform to the polynomial's native height (#2191)
- Perf(fri): pack the multi-round fold and drop a multiplication from the opening quotient (#2204)
- Fix(uni-stark,circle): reject degree_bits below the PCS minimum trace height (#2257)
- Refactor(commit): own the periodic column shape rule where every consumer can reach it (#2276)
- Refactor!(uni-stark,batch-stark,commit): group the preprocessed openings and name the trace-height bounds as a pair (#2281)
- Perf(maybe-rayon): close the review comments on the cost model, and apply it to three more loops (#2283)

## [0.7.0] - 2026-09-04
### Merged PRs
- Feat: expose pruned-multiproof restoration and FRI/STARK opening internals (#2010)

## [0.6.0] - 2026-06-11
### Merged PRs
- Feat: add support for Periodic Columns at runtime (#1462)
- Refactor(verifier): remove unused MissingInput variant (#1484)
- Fix(fri): split InvalidProofShape into typed verifier errors (#1519)
- Refactor(fri): move FRI parameter constructors to associated methods (#1530)
- Refactor(matrix): move column-appending helpers to `DenseMatrix` methods (#1533)
- Refactor(interpolation): replace free functions with Interpolate extension trait (#1540)
- Fri: full test coverage for verifier errors (#1529)
- Perf(interpolation): optimize barycentric weights via algebraic identity (#1553)
- Chore: use T::zero_vec(n) instead of vec![T::ZERO; n] (#1633)
- Chore: remove needless_range_loop allows across the workspace (#1632)
- Enforce Positive max_log_arity (#1652)
- Fix(fri): enforce num_chunks > 1 in HidingFriPcs::get_quotient_ldes (#1595)
- Verifier: add a couple strengthening checks (#1666)
- Harden log_arity validation in FRI and Circle (#1676)
- Ci: tighten doc/release/TOML checks (#1689)
- Feat: add a `HornerIter` supertrait on `DoubleEndedIterator` (#1692)
- Chore: HornerIter follow-ups from #1692 review (#1693)
- Fri: reject global max height mismatches (#1698)
- Add fail-fast checks to periodic LDE builders (#1614)
- Fix(fri): type hiding PCS random opening errors (#1704)
- Couple miscellaneous tweaks (#1731)
- Fix(params): error instead of panic on infeasible WHIR/FRI params (#1737)
- Fix(fri): reject zero-query instances on both sides (#1735)
- Refactor(fri): hoist height-1 check and unify max-height in prover (#1756)
- Fix(merkle-tree): enforce opened row widths in batch verification (#1757)
- Fix(fri): reject opening point equal to the query point instead of panicking (#1762)
- Chore: update CHANGELOGs (#1785)
- Doc: add basic READMEs in main crates (#1786)
- Fix(fri): reject input matrices opened at zero points (#1793)

## [0.5.3] - 2026-05-15
## [0.5.2] - 2026-03-27
## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- Chore: minor fixes (#1246)
- Fix: reduce logging noise in batch-stark with multiple AIRs (#1258)
- [BREAKING] feat: Implement high-arity folding (#1277)
- Deps: update rand and rand_xoshiro (#1314)
- Feat: add Merkle Caps (#1321)
- Perf: reduce number of batch inversions in general FRI arity case (#1325)
- Feat: add Clone to StarkConfig and StarkGenericConfig (#1328)
- Fix: remove unnecessary clone of opening points (#1344)
- Feat(fri): handle extrapolation for LDEs (#1352)
- Feat: make HidingFriPcs Sync (#1395)
- Feat: add high-arity support in `MerkleTree` and `MMCS`  (#1373)

## [0.4.2] - 2026-01-05
### Merged PRs
- Enable ZK for preprocessing and in batch-stark (#1178) (Linda Guiga)
- Avoid change of Pcs's `open` method signature (#1230) (Linda Guiga)

### Authors
- Linda Guiga

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Field.rs: `Powers::packed_collect_n` (#888) (Adrian Hamelink)
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- More Clippy Complaints (#931) (AngusG)
- Shrink some test sizes (#524) (Daniel Lubarov)
- Update doc comment and some other comment fixes. (#959) (AngusG)
- Minor FRI refactor - make open input its own function (#961) (AngusG)
- Remove duplicated definition (#1031) (AngusG)
- Refactor: remove redundant clones in crypto modules (#1086) (Skylar Ray)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add nursery (#1103) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Clippy: add `needless_pass_by_value` (#1112) (Thomas Coratger)
- Core: add error messages to error enums via thiserror (#1168) (Thomas Coratger)
- Challenger: use `observe_algebra_slice` when possible (#1187) (Thomas Coratger)
- Feat: add PoW phase for batching in FRI commit phase (#1164) (Zach Langley)

### Authors
- Adrian Hamelink
- AngusG
- Daniel Lubarov
- Himess
- Skylar Ray
- Thomas Coratger
- Zach Langley
