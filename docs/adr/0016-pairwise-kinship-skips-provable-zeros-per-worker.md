# ADR 0016: the pairwise kinship walk skips provable zeros and runs one walker per worker

**Status:** accepted
**Date:** 2026-10-01
**Context:** issue #34 (the serial value fill of `relationship_kinship_matrix`
sets wall time and peak memory at simACE scale). Amends ADR 0005's "one memo
per call" and its scaling consequence, and ADR 0007's list of serial kernels.
Keeps the ADR 0009 recurrence and its bit contract unchanged.

## Context

`support_values` (the value fill behind `relationship_kinship_matrix`) and
`pair_kinship` walked every requested pair with one memo on one thread. On
the PA-FGRS analysis pedigree of simACE `cure_rA50` (3,576,377 rows, six
generations, degree 2) the fill took 65 s of a 75 s build and the process
peaked at 7.6 GB, at any thread budget.

A census of the memo on `cure_rA50_200k` (357k rows, degree 2) found 35.5M
keys, 32.4M of them exactly zero, almost all pairs of ancestors with no
common ancestor: 15.7M founder pairs, 8.1M founder-by-generation-1 pairs,
and so on up. The walk descended each one to the founders to learn it was
0, and kept it until the call returned.

## Decision

**A key whose value needs no walk is neither pushed nor stored.** The walker
evaluates each key's rule first (`Rule` in `kinship/pairwise.rs`):

* a self-like key with a missing parent is `0.5`;
* a key whose peeled endpoint is a founder is `0`;
* a key whose endpoints share no ancestor is `0`, proved by **ancestor
  signatures** (`kinship/ancestry.rs`): a 256-bit set per row, holding one
  hashed bit for the row, one for its MZ co-twin, and both parents' sets.
  Disjoint sets prove no shared ancestor-or-self genome, so the recurrence
  gives exactly `+0.0`. A collision only makes unrelated rows look related,
  which costs a walk, never a value.

Every stored key is computed by the same float32 half-sum from the same
dependency bits, so every value is bit-identical to the recurrence ADR 0009
pins.

**Each call runs one walker per pool worker.** `pair_kinship` splits its pairs
into chunks of 4,096 and `support_values` its columns into chunks of 256;
each worker takes chunks from a shared counter and keeps one memo across
them. A key has one value whoever computes it, so the output is the same
bits at every budget; `test_both_entries_return_the_same_bits_under_every_thread_budget`
holds that, mapped in `test_architecture_guardrails.py`. Workers write the
output at disjoint positions through `AtomicU32` (the crate forbids
`unsafe`), converted to `f32` in the same allocation.

The signatures and the stack they are built with reserve through a new
allocation family, `kinship_signatures`, so the seam tests of
`kinship_memo` and `kinship_stack` still reach the workers' memo and stack.

## Measurements

Whole `relationship_kinship_matrix` build and the fill inside it, medians of
three interleaved fresh processes, i7-9750H capped at 2.6 GHz. `base` is
`main` at 56deaf1 and `new` this change, both built the same way and run
in one environment. Peak is `ru_maxrss`. Every matrix hashed identically
across variants and reps.

| input | degree | threads | base build / fill / peak | new build / fill / peak |
|---|---|---|---|---|
| cure_rA50_200k | 2 | 1 | 6.66 s / 5.86 s / 858 MB | 2.47 s / 1.67 s / 412 MB |
| cure_rA50_200k | 2 | 8 | 6.28 s / 5.85 s / 870 MB | 0.90 s / 0.45 s / 545 MB |
| cure_rA50_200k | 3 | 1 | 19.68 s / 17.34 s / 1,998 MB | 6.99 s / 4.73 s / 719 MB |
| cure_rA50_200k | 3 | 8 | 18.30 s / 17.23 s / 2,011 MB | 2.23 s / 1.23 s / 918 MB |
| cure_rA50 | 2 | 1 | 74.89 s / 65.29 s / 7,608 MB | 28.42 s / 18.94 s / 2,666 MB |
| cure_rA50 | 2 | 8 | 69.40 s / 64.64 s / 7,729 MB | 10.86 s / 5.14 s / 3,530 MB |
| cure_rA50 | 3 | 8 | not run | 24.57 s / 13.92 s / 6,827 MB |

On the 200k degree-2 support the memo falls from 35.5M keys to 4.6M with
256-bit signatures; 3.1M of the original keys are nonzero, so the remaining
overhead over the nonzero floor is small. 128 bits kept 5.4M keys and was
about 14% slower at full scale.

## Rejected

* **A shared concurrent memo.** No duplicated keys, but every lookup takes a
  lock or an atomic on rows every worker reaches (the founders and early
  generations). Once the zeros are gone the per-worker duplication is 1.4x
  the keys at eight workers, which does not justify it.
* **A depth-layered memo that frees a level once nothing above it can read
  it.** Bounds the memo by the widest live levels instead of the closure,
  but dependencies can skip levels on general pedigrees, and the zero
  pruning already removes 87% of the keys the layering was meant to bound.

## Consequences

* The walk's cost follows the *nonzero* ancestor pairs the requested pairs
  reach, plus the related-looking pairs the signatures cannot rule out,
  not the whole ancestral closure. ADR 0005's scaling note is amended
  accordingly.
* Every call pays 32 bytes a row for the signatures and, per worker that
  takes a chunk, 32 bytes a row for its memo's row vector (115 MB each at
  3.58M rows). A one-chunk query uses one worker.
* Peak memory grows with the budget, since workers duplicate shared ancestor
  keys: on `cure_rA50` degree 2, 2.7 GB at one thread and 3.5 GB at eight.
