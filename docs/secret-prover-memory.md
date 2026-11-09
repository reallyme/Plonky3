<!-- SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited -->
<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Secret prover memory

## Scope

This document defines an opt-in memory-erasure contract for applications that
prove statements over high-value secret witnesses. It does not claim that
erasure alone makes a prover secure, and it does not change proof-system
parameters, proof formats, or verifier behavior.

The ordinary prover remains the default. A secret-memory prover is a distinct
capability whose types preserve erasure ownership across every allocation that
can contain a witness or a value derived from one.

## Data classification

| Class | Examples | Erasure requirement |
| --- | --- | --- |
| Secret | witness columns, private trace rows, salts, masking polynomials | Erase before allocation release |
| Secret-derived | LDEs, quotient evaluations, FRI fold buffers, retained prover matrices | Erase before allocation release |
| Intentionally disclosed | queried openings and proof values | Erase temporary construction copies; the returned proof is public by protocol |
| Public | public inputs, commitments, Merkle roots, verifier configuration | No secret-memory requirement |

Classification is conservative. A buffer remains secret-derived until a
reviewed protocol boundary proves that its complete contents are intentionally
disclosed.

## Ownership invariant

For each secret or secret-derived allocation there must be exactly one owner
whose destruction erases the allocation before it is released. Transferring an
allocation between prover stages must transfer that obligation without an
unprotected interval.

The base ownership primitive is `ZeroizingOwner`; `ZeroizingMatrix` is its
matrix-specific alias. It deliberately omits `Clone`, serialization, `Debug`,
and extraction of the inner value. Ordinary `DenseMatrix` values do not acquire
implicit erasure costs. Enabling the `secret-memory` feature provides
`Zeroize` implementations that explicit owners use. The commitment layer names
the same contract as `SecretProverData` and exposes `ErasableProverData` for
generic bounds.

Guarded matrix widening allocates the replacement under a zeroizing owner,
uses checked dimension arithmetic and fallible reservation, fills random mask
columns, erases the old allocation, and only then swaps owners. This is the
required allocation pattern for a hiding PCS; the ordinary `with_random_cols`
API remains unchanged for standard-memory callers.

Guarded column truncation follows the same replacement pattern. The hiding PCS
uses it to remove random codeword columns from a quotient-domain evaluation
without dropping the discarded columns through an ordinary `Vec` path.

`Radix2Bowers` provides the first guarded DFT operation. Its secure coset-LDE
API borrows a zeroizing owner rather than consuming a raw matrix. When padding
requires a larger allocation, it builds the replacement under a second guard,
copies the live rows, explicitly erases the superseded allocation, and only
then swaps owners. This avoids relying on `Vec` reallocation, which may release
the former allocation without wiping it.

Merkle prover data is erasable transitively when its leaves and digest words
are erasable. Tests exercise normal drop, unwind after construction, and unwind
while invalid input geometry is being rejected. Hiding FRI prover data is also
erasable and marks itself consumed when explicitly wiped. Under the
`secret-memory` capability it does not implement `Clone` or `Debug`.

`SecretMmcs` is the guarded commitment and opening capability for retained
Merkle data. Implementations return `SecretProverData` rather than an ordinary
MMCS owner, retain the original guarded matrices without extracting their
allocations, and use typed errors for invalid shapes and opening indexes.

`SecretPcsCommitter` deliberately covers commitment construction only. The
two-adic implementation preserves the guard while creating and bit-reversing
each LDE. The hiding wrapper performs checked, fallible random-column widening
under the same guard, then nests the guarded inner PCS data inside guarded
hiding prover data. `SecretPcsOpener` is a separate capability whose mutable
requests permit irreversible leaf and coefficient erasure only after
Fiat-Shamir fixes query indices. `SecretPcsMixedOpener` adds an explicit owner
model for reusable public preprocessing without converting it into guarded
private state. Tests compare guarded and ordinary commitments and openings for
deterministic inputs, verify the guarded proof with the ordinary verifier, and
exercise retained-data erasure.

`SecretUnivariateStarkPcsCommitter` extends this boundary through two-adic
quotient splitting and commitment for both ordinary and hiding FRI. It
validates chunk geometry before copying, allocates zero-filled chunk buffers
under erasure owners before placing any secret-derived value in them, and
preserves those owners through LDE construction and MMCS commitment. The
hiding implementation additionally guards its random columns, quotient masks,
inverse-transform coefficients, zero-filled coefficient extension, forward
transform, and retained masked LDEs. Invalid hiding chunk counts are rejected
before consuming the mask RNG. Multi-instance quotient inputs are validated as
one batch, then split in instance and domain order under a single global PCS
root. This preserves the existing batch-STARK proof and verifier layout rather
than introducing one quotient commitment per instance. Tests pin deterministic
commitment parity with the corresponding ordinary APIs, mixed domain and chunk
counts, explicit retained-LDE erasure, invalid chunk rejection, fail-before-RNG
behavior, and malformed zero-width matrix rejection without a panic.

The matrix owner also provides guarded extension-column flattening for callers
of the compatibility quotient API. It allocates the base-field matrix under
erasure ownership, copies each extension element's canonical basis
coefficients, and erases the extension-field source before returning. The
complete batch prover avoids that replacement allocation: quotient evaluation
writes the same canonical coefficients directly into the base-field matrix
consumed by the PCS.

Batch-STARK quotient evaluation now has a guarded capability. It validates
domain and matrix geometry before allocating its secret output and keeps
packed main, preprocessed, permutation, scalar-packing, and constraint buffers
behind zeroizing owners. Reused row buffers are erased before their lengths are
reset, so old elements do not survive only in `Vec` spare capacity. The batch
path writes each quotient's extension-basis coefficients directly into its
guarded base-field output and transfers that owner into
`SecretUnivariateStarkPcsCommitter`. The public compatibility entry point
recomposes the conventional extension-field column under a replacement guard.
Tests pin the quotient values, complete proof bytes, and resulting commitment
to the ordinary path and cover typed shape rejection. The guarded hiding path
preserves the existing masking protocol rather than substituting the
non-hiding quotient construction.

Multi-instance quotient commitment accepts a complete public plan before it
requests any guarded matrix. The batch prover therefore evaluates one instance
quotient, transfers it into the PCS, and releases that contiguous source as
soon as its values have moved into guarded chunk owners. The hiding path also
avoids retaining a mask matrix for every chunk: it generates and consumes one
earlier-chunk mask at a time while accumulating the protocol-constrained final
mask under a separate zeroizing owner. This preserves the exact random draw
order, global commitment root, and post-commitment RNG state while removing
source-plus-mask overlap before coefficient retention and LDE expansion.

This transition shortens secret lifetime but does not, by itself, materially
reduce peak resident memory. Every transformed LDE and retained coefficient
matrix must still remain live until the one global quotient Merkle root is
fixed. Reducing that live set requires an incremental MMCS construction that
preserves the canonical mixed-height leaf ordering and identical root.

Guarded transparent and hiding quotient commitments now transition to
authentication-only MMCS state immediately after the global Merkle root is
fixed. No protocol step consumes their full value leaves between commitment
and opening, so retaining them across randomization and transcript preparation
only extended the secret lifetime. Checked LDE dimensions come from the compact
coefficient owners and remain available after detachment; hiding-FRI preflight
uses that metadata to preserve all shape and disclosure-budget validation.
Opening accepts both live commitments and commitments already detached at this
earlier boundary. Tests require exact commitment and hiding-RNG parity and
inspect that returned quotient data contains no live value matrices.

That lifecycle is expressed by `SecretRecomputableMmcs::commit_detached_secret`
rather than by PCS code manually sequencing commitment and detachment. The
default method intentionally delegates to the independently tested operations;
an MMCS may specialize the method to consume value matrices while constructing
the same global tree, but must preserve the commitment, authentication paths,
hiding salts, RNG draw order, and typed failure behavior exactly.

Merkle arity selection has a pure, fallible planning boundary derived only
from public matrix dimensions. The verifier and guarded consuming constructor
use that planner as their canonical schedule source. The constructor hashes
matrices in original commitment order within each height group, then transfers
ownership of that group to an erasure callback before reading the next group.
The ordinary adapter erases the complete matrix; the hiding adapter erases the
value owner while retaining its independently owned salt at the original
matrix index. Checked errors cover invalid arity, packing mismatch,
platform-width geometry overflow, and allocation failure.

Randomized binary and 4-ary tests require the consuming constructor's cap to
equal the ordinary constructor's cap for every generated geometry. Dedicated
multi-height tests enforce the lifecycle ordering, capped-root parity, hiding
salt and RNG order, authentication-path parity, and successful verification.
The retained prover-data shell contains no matrices after construction. This
reduces live secret ownership during tree construction without changing proof
or commitment encodings; allocators may retain released pages, so process RSS
is not itself an erasure guarantee.

The guarded two-adic PCS exposes this lifecycle separately as
`SecretDetachedPcsCommitter`. It validates the complete batch geometry before
starting commitment construction, then creates, bit-reverses, and absorbs one
LDE at a time. Only compact coefficient owners and authentication-only MMCS
state survive the handoff. The hiding wrapper preserves its canonical masking
layout and random draw order while masking, transforming, and absorbing one
source matrix before advancing to the next. It preflights every source/domain
pair before the masking RNG advances. The end-to-end batch prover uses this
operation for main and lookup-permutation commitments; standalone compatibility
helpers retain their established live-matrix behavior. Parity tests require
identical commitments, verifier-accepted openings, and identical
post-commitment hiding RNG state.

`SecretUnivariateStarkPcsEvaluator` closes the domain-evaluation boundary for
guarded two-adic prover data. It derives quotient-domain matrices from retained
coset-adjusted coefficients, uses checked allocation and an in-place guarded
DFT, and returns an owned zeroizing matrix. The hiding wrapper validates that
the retained profile matches the active PCS and erases discarded random
codeword columns. Tests pin generator-coset, larger disjoint-coset, hiding, and
untruncated preprocessing-layout outputs to the ordinary PCS behavior.

The batch-STARK composition helper now acquires guarded main and optional
permutation matrices through that capability and consumes them directly in
guarded quotient evaluation. Preprocessed columns remain an ordinary input by
design because they are public; callers must not route witness-bearing data
through that argument. The helper has typed failures that distinguish main,
permutation, and quotient phases.

Batch main-trace commitment also has explicit guarded handoffs. The public
compatibility helper preserves its callers' source evaluations by copying them
directly into a second zeroizing owner before transferring that owner to the
guarded PCS; it never materializes the duplicate as an ordinary matrix. The
complete batch prover uses a consuming handoff instead. After Fiat-Shamir fixes
the lookup challenges, it reconstructs only the current lookup-bearing trace
from compact committed coefficients. A hiding PCS evaluates the full committed
domain, removes its random codeword columns, selects the original even rows
from the interleaved masking layout, and erases each superseded owner. The batch
prover then validates the recovered geometry, generates its LogUp permutation,
and erases the temporary owner before advancing. Non-lookup traces are never
reconstructed.
This removes simultaneous ownership of the complete source batch and its PCS
copy without changing commitments, transcript order, or verifier behavior.

LogUp permutation generation now has a checked guarded path as well. Its
witness-derived denominator, multiplicity, inverse, row-total, prefix-offset,
and final auxiliary-trace allocations all remain under zeroizing owners.
Malformed geometry, challenge counts, exclusive selectors, and zero
denominators return stable typed failures. Batch-STARK can consume the guarded
extension-field trace, flatten it under guarded ownership, and commit the
base-field representation without an ordinary intermediate matrix.

These lower-layer capabilities are composed by `prove_batch_secret`, the
top-level guarded batch-STARK entry point. It validates batch, trace,
preprocessing, lookup, and quotient geometry before committing the transcript,
and maps each fallible prover phase into a domain-specific typed error.
`prove_batch_secret_with_observer` exposes the same protocol with typed,
infallible start and completion callbacks around preparation, commitment,
lookup generation, each instance quotient, opening, and proof assembly. The
observer receives only the phase identifier and instance index; no witness,
public-input, commitment, challenge, or proof value crosses the boundary. The
ordinary entry point installs a no-op observer, so diagnostics remain opt-in
and cannot alter transcript bytes or error behavior.

`p3-batch-stark` exposes a narrower `prove_batch_guarded_inputs` staging API.
It consumes source traces held by `SecretProverData`, then explicitly erases and
deallocates those source matrices after main-trace commitment and lookup
permutation construction. This prevents the original traces from surviving
through quotient evaluation and FRI opening. The API documentation states that
the ordinary PCS prover data produced by the same call is not guarded, so this
staging API must not be treated as an end-to-end erasure boundary. Existing
users of it must migrate to `prove_batch_secret` for that property.

The ordinary two-adic opening path now has a smaller temporary live set. It
releases its base-field coset after denominator construction, releases adjusted
interpolation weights after claimed openings are computed, fuses row
compression directly into opening-quotient accumulation, and releases the
denominator table before FRI folding. This removes avoidable overlap between
full-height temporary vectors without changing transcript order, commitments,
proof bytes, or verifier behavior. It is a memory-allocation optimization, not
an erasure guarantee: the remaining buffers still need guarded ownership before
the secret prover entry point can be exposed.

The Merkle layer also exposes an initial recomputation boundary. A guarded tree
can move its public digest layers, arity schedule, and committed matrix shapes
into `MerkleTreeAuthenticationData` while explicitly erasing and releasing all
retained leaf matrices in place. Later single-query and pruned multi-query
openings accept independently recomputed matrices, require exact shape parity,
and produce the same opening proof bytes as the retained-leaf path. The
transition does not extract secret matrices from `SecretProverData`, and the
authentication object is deliberately neither cloneable nor serializable.

The hiding-Merkle adapter retains its salt matrices separately under a
zeroizing, non-cloneable owner while the committed value matrices are erased.
It combines only verifier-selected recomputed rows with their original salt
rows and reproduces the ordinary single and pruned opening formats exactly.
Thus detachment neither regenerates salts nor changes the commitment, while
unqueried salts remain secret until their owner is erased.

The guarded two-adic PCS now retains natural-order, coset-adjusted coefficients
at the committed polynomial's original height. It can reconstruct only the
verifier-selected rows of the bit-reversed LDE, with typed rejection of an
empty query set, out-of-range indexes, and a retained blowup that differs from
the active PCS configuration. The hiding quotient path retains the compact
masked polynomial coefficients rather than an unmasked substitute. An
integration test commits through the PCS, erases all hiding-Merkle value
leaves, rebuilds selected rows from the compact owner, and verifies the rebuilt
opening against the original commitment and salt disclosures.

Coefficient-retaining coset LDE construction keeps the transformed compact
input allocation as the returned coefficient owner and allocates the final LDE
at its checked size from the outset. This removes the prior compact clone plus
grow/reallocation overlap at quotient preparation. Both allocations are under
zeroizing ownership before secret-derived bytes are copied, and an allocation-
identity test pins that the returned coefficient matrix owns the original
input allocation.

The same compact owner can also rebuild one complete committed LDE at a time
in the exact physical, bit-reversed order consumed by the Merkle commitment.
The reconstructed matrix is born under a zeroizing owner and is deliberately
matrix-scoped: a later opening reducer can reconstruct, consume, and erase one
matrix without reviving the entire commitment batch. Tests compare every field
element with the originally committed LDE, repeat reconstruction after the
original value leaves have been detached, and reject malformed coefficient
geometry and out-of-range matrix indexes with typed errors.

`SecretRecomputableMmcs` now makes the irreversible transition a generic MMCS
contract implemented by both ordinary and hiding Merkle commitments. The
two-adic PCS bridge preserves mixed-height index reduction, reconstructs only
the selected rows, erases every full value leaf, and authenticates the rebuilt
rows through that contract. Invalid queries are rejected before detachment, so
caller error cannot destroy otherwise usable prover data.

The FRI prover now accepts a typed input-opening callback that is invoked only
after its Fiat-Shamir query indices are fixed. The ordinary retained-data path
uses the same callback boundary, preserving transcript order and proof wire
types. This establishes the timing point at which the PCS bridge must run.

The two-adic PCS now separates opening geometry from matrix storage and shares
the same quotient reducer between ordinary and guarded paths. The guarded path
validates every dimension, point, denominator table, and retained blowup before
changing prover state. It then moves each private Merkle commitment to
authentication-only state and evaluates disclosed claims directly from the
compact natural-order coefficients with guarded Horner accumulators. Retained
coefficients represent `f(GENERATOR * X)`, so evaluation at
`point / GENERATOR` is exactly `f(point)` and requires no full-LDE
reconstruction. Independent coset interpolation and complete ordinary-proof
parity tests pin that convention. Opening-quotient reduction subsequently
rebuilds, reduces, and erases one private LDE matrix at a time; selected input
rows remain deferred until Fiat-Shamir fixes the query indices. The ordinary
opening contract suites for BabyBear and Mersenne-31 exercise the shared logic
across single, multiple, equal-height, mixed-height, reversed-order, and
high-arity commitments.

FRI folding now has a guarded strategy contract. The two-adic implementation
allocates every output and high-arity ping-pong buffer behind a zeroizing owner
before the first secret-derived write, explicitly erases discarded tails, and
returns typed shape, domain, arity, and allocation errors. The extension-field
MMCS adapter preserves that owner while exposing a flattened base-field view
to its inner commitment scheme.

The guarded FRI commit phase uses those capabilities for every committed
codeword. Tests pin its commitments, challenges, retained matrices, final
polynomial, and guarded openings to the ordinary implementation, then erase
and inspect every retained layer.

Terminal-polynomial conversion now consumes those guarded evaluations,
flattens them into a guarded base-field matrix, and runs an in-place guarded
inverse DFT. Only the coefficients already disclosed by the FRI proof cross
the explicit boundary into ordinary storage; both transform allocations are
erased. Guarded query answering similarly consumes each commit-phase MMCS
owner, emits the transcript-selected public sibling values and multiproof, and
erases the retained codeword immediately after opening. Typed failures cover
empty queries, schedule drift, arity overflow, malformed openings, allocation,
and inner MMCS rejection.

The guarded FRI driver composes commit construction, terminal-polynomial
disclosure, Fiat-Shamir query selection, deferred input opening, and
commit-layer query answering. It validates input geometry and platform index
widths before transcript mutation. Once a transcript exists, typed failures
explicitly abort its completeness check before guarded buffers unwind. Tests
pin the complete driver output and final challenger state to the ordinary FRI
driver, verify that invalid preflight leaves the challenger unchanged, and
verify that the input-opening callback cannot run before query selection is
complete.

The guarded two-adic PCS opening path now composes that driver with direct
coefficient opening, matrix-scoped opening-quotient reduction, and the
detachable input-opening bridge. Horner accumulators and every reduced quotient
codeword are placed under erasure ownership before their first secret-derived
write. Full private value leaves are detached before disclosed claims or
reduced codewords are allocated. Each temporary complete LDE is reconstructed
under a zeroizing owner, reduced, and dropped before the next private matrix is
rebuilt. The deferred callback later reconstructs and authenticates only
transcript-selected rows, then erases compact retained coefficients. Typed
failures distinguish detachment from post-detachment reconstruction and abort
the outer PCS transcript before guarded storage unwinds. Tests prove parity
with every ordinary proof component and challenger state, and an ordinary
verifier accepts the resulting guarded proof.

This is now an end-to-end guarded PCS opening capability exposed through
`SecretPcsOpener` for both two-adic and hiding FRI private commitments.
Guarded quotient evaluation can acquire private main and permutation
quotient-domain matrices from guarded PCS data, while public preprocessing
remains outside the secret owner model. Guarded hiding-FRI randomization
polynomials are also allocated under erasure ownership before the first random
value is written, committed through the guarded PCS, and covered by the same
one-shot opening contract.

Mixed PCS opening now preserves ordered public and private rounds without
converting their owners. Public preprocessing stays immutably borrowed and
reusable. Private rounds remain mutably borrowed; their full committed value
leaves are detached before quotient reduction, while authentication-only state
survives until Fiat-Shamir fixes and authenticates the query indices.
Hiding FRI validates the unique public-preprocessing marker, every profile and
shape, and all disclosure budgets before consuming any private commitment.
Batch-STARK can now transfer every guarded instance quotient into the one
global commitment required by the existing proof format. The
`prove_batch_secret` entry point composes guarded main-trace commitment,
lookup permutation generation, quotient evaluation and commitment, optional
hiding randomization, mixed public/private PCS opening, and transcript
finalization. It returns the existing `BatchProof` format, and proofs are
accepted by the existing verifier.

This capability is protocol-agnostic infrastructure. Application-specific
statement construction and witness ownership must still call the guarded entry
point explicitly; adding this API does not migrate an application's prover by
itself.

The guarded entry point preserves the erasure-owner invariant through:

```text
witness trace
    -> DFT and LDE workspaces
    -> commitment prover data and retained Merkle leaves
    -> quotient evaluations and quotient commitment data
    -> FRI batching and folding buffers
    -> opening construction temporaries
```

Every arrow above retains an erasure owner on success, typed failure, and
supported panic unwinding. Future prover phases must meet the same requirement
before they are added to the guarded entry point.

## Failure behavior

Erasure must run after:

- successful proof construction;
- commitment, quotient, or opening failure;
- an early return from any prover phase;
- panic unwinding on targets built with unwinding support; and
- repeated and parallel proving.

Abort-on-panic targets cannot run destructors. The API and documentation must
state this limitation rather than imply a guarantee the process cannot keep.
WASM validation must cover the actual target and panic strategy shipped by an
application.

## Verification strategy

Tests use recognizable canaries and borrowed controlled storage so the bytes
remain inspectable after an owner drops. Coverage must include explicit
erasure, normal drop, unwind cleanup, typed errors at each prover phase,
allocation reuse, repeated proofs, parallel proofs, and supported WASM builds.

The ownership primitive has an instrumented test lifecycle:

```text
SENSITIVE
    -> USED
    -> ERASE_STARTED
    -> ERASE_VERIFIED
    -> DEALLOCATED
```

The test allocation's destructor rejects any transition from live sensitive
state directly to deallocation. A separate recycling storage returns an erased
buffer to a controlled pool and forces the next synthetic secret to reuse the
same allocation. This observes the allocator boundary without reading memory
after it has been freed.

Higher-layer tests retain borrowed canaries through ordinary and hiding Merkle
prover data. They cover successful commitment and opening construction, typed
commitment and opening errors, normal drop, panic unwinding, repeated runs, and
concurrent runs with distinct canaries. Public commitments and opening proofs
remain verifiable after private prover data has been erased.

Optimized test builds are required because ordinary assignments can be removed
as dead stores. The implementation uses the `zeroize` crate rather than plain
buffer fills for production erasure. WASM builds exercise the same ownership
and erasure implementations, but destructor-based cleanup still depends on
normal return or an unwinding panic strategy.

Canary tests demonstrate the behavior of allocations controlled by the
library. They cannot prove that a compiler, allocator, operating system, crash
dump, swap subsystem, or hardware device never made another copy. The final
security statement must preserve that boundary.

Successful full-prover tests cover exact ordinary-proof parity for batches
without lookups and verifier acceptance for batches combining public
preprocessing with lookup permutations. In the lookup case, commitments and
claimed openings are exactly equal to the ordinary path; the Merkle
authentication proof may use a different valid encoding because the guarded
path reconstructs only transcript-selected rows rather than retaining the
ordinary tree. The batch staging API tests source-owner release, and the guarded
FRI tests cover folding buffers, retained extension-MMCS data, terminal
conversion, query construction, full low-level driver parity, typed transcript
completion, deferred input-opening order, and ordinary-verifier acceptance of
both transparent and hiding guarded PCS openings. Mixed public preprocessing
is covered by ordinary-proof parity, ordinary-verifier acceptance, public-data
reuse, and fail-before-private-consumption tests. Guarded randomization-
polynomial construction is covered by exact ordinary-path parity,
verifier-acceptance, checked-overflow, and preflight RNG tests. Multi-instance
quotient commitment is covered by single-root ordinary parity, mixed-size and
mixed-chunk layouts, explicit erasure, and fail-before-RNG tests. Top-level
batch composition is covered by typed rejection of empty batches and public
input count mismatches before proving begins.

## Upstreaming

The implementation is protocol-agnostic. Application-specific statement
types, policy, and cryptographic profiles do not belong in these crates. Each
layer should remain independently reviewable so the capability can be proposed
upstream without adopting a permanent application-specific fork.
