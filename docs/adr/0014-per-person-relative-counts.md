# ADR 0014: per-person relative counts credit the first member and compare a threshold on the credited member

**Status:** accepted
**Date:** 2026-09-30
**Context:** issue #33 (pedsum#3, fitACE_epimight `create_input`); the
per-person reducer ADR 0013 reserved on the same engine pass. Plan: simACE
`plans/pedigree-graph-issue-33-relatives-per-person.md` (session-local draft).

## Context

Two consumers reduce the degree-3 pair list to a few integers per person
per relationship kind. pedsum's `epimight-input` counts each person's
relatives per EPIMIGHT kind. fitACE_epimight's `create_input` counts, per
trait and kind, each person's relatives and affected relatives. Its
aligned variant credits an affected relative only if
`onset[relative] <= cutoff[person]`. Both build `relationship_pairs` and
fold it with `np.add.at`, so peak memory follows the pair count: 439,217,825
pairs at 20M rows (`benchmarks/bench_pair_emitters.md`).

ADR 0013 sketched this reducer as "counts and sums of the other member's
values per person, category and cell". That sketch cannot express the
aligned count, because its threshold belongs to the credited person, not
to the relative, and the cutoff is continuous (it includes death age), so
no finite label on either member reproduces it.

## Decision

### One reducer on the ADR 0013 pass, crediting the first member

`crates/core/src/relationships/relatives.rs` adds a `Reducer` whose lane is
`()`. Every lane adds into one shared `[rows][categories][1 + K]` array of
relaxed `AtomicU32` counters, and the merge is empty. Integer adds commute,
so the counts are the same for every lane count and thread budget.

The reducer credits only the `first` member of each pair it receives, and
the entry point always runs `reduce_pairs` with `Symmetric::Both`. A
symmetric pair therefore reaches it in both orientations and credits both
members, and an asymmetric pair arrives once, junior first, and credits
only the junior: `MO`/`FO` count a person's parents, `GP` their
grandparents, `Av` their aunts and uncles. This is the rule both consumers
applied to pair blocks (`unidirectional` for PO, Av and 1G), so the
crediting rule is structural rather than a keyword. Crediting the senior
member of a directional pair as well would be a new keyword, and an
additive change.

A person's relatives in one category are distinct rows, fewer than the
receiver's row count and so below `2^32`, so a `u32` counter cannot wrap
and there is no post-pass check.

### A threshold column compares the relative's value with the credited member's threshold

A call names up to 32 columns, each a pair `(relative, threshold)`.
Column `k` counts a relative when
`relative_k[relative row] <= threshold_k[credited row]`, compared as IEEE
`f64`. NaN on either side never counts, and ±inf compare as IEEE says.
Column 0, `"relatives"`, is the pair count and is always present; the name
is reserved.

One comparison expresses both epimight counts exactly, for every float
input:

| count | `relative` | `threshold` |
|---|---|---|
| aligned affected | `where(affected, onset, nan)` | `cutoff` |
| unaligned affected | `where(affected, 0.0, nan)` | `0.0` |

The unaligned encoding does not use the onset: an affected relative with a
NaN onset counts today and must keep counting.

No value is quantized. `float32` and integers up to `2^53` convert to
`f64` exactly; larger integers and floats wider than `f64` are refused
because they would round, and bool is refused as a type error.

### Inputs are borrowed, and the counts leave the core without a copy

Each column's `relative` is a borrowed slice, and its `threshold` is a
borrowed slice or a scalar that is never broadcast. The host converts an
input to contiguous `float64` only when it is not already, and converts an
object shared by two columns once, so the epimight columns share one
`cutoff`. simACE writes onsets as float32, and `np.where(affected, onset,
np.nan)` keeps float32, which would cost a float64 copy per column; a
caller that passes `np.float64(np.nan)` gets float64 from the allocation
`np.where` makes anyway, and the widening is exact. Borrowed columns cost K reads per credit from separate arrays
rather than one cache line per member; the benchmark records the cost.

Core forbids unsafe code, so the settled counters become `Vec<u32>`
through `into_iter().map(AtomicU32::into_inner).collect()`, as
`relationship_burden` already did. That reuses the allocation through
std's in-place iteration specialization, which is not a language
guarantee; a unit test asserts the pointer is unchanged, so a toolchain
that stops reusing it fails the test rather than doubling the peak. The
binding hands the vector to NumPy with `into_pyarray`, and the host
reshapes it to `[rows, categories, 1 + K]` and freezes it.

### Grouping is a fold, and there is no memory budget

The result has one slot per requested registry code.
`RelativesPerPerson.sum(codes, column)` folds codes into one `int64`
array, adding each code's strided view in place, so the fold allocates
its result alone; a repeated or unrequested code is an error. EPIMIGHT's
groups overlap (HS, mHS and pHS share MHS and PHS), so taking groups in
the call would not partition the codes. This is ADR 0013's rule for
moments.

The call takes no memory budget. Its output is O(N) and the caller asked
for all of it: `4 · rows · categories · (1 + K)` bytes, 720 MB for pedsum's
nine codes at 20M rows and 2.16 GB with two threshold columns. A counter
array that cannot be allocated raises `ResourceError("allocation_failed")`
under the new family `relative_counts`, as `relationship_burden` does.

### Views and the compact path

A view credits view rows, and every column has one entry per view row.
Selection, closest-category precedence and the compact view path are
those of `relationship_moments`. A receiver of fewer than two rows, or an
empty selection, returns zeros without running the engine.

## Rejected

* **A label axis on the other member, with the aligned count left in
  Python over `relationship_pairs`.** It serves totals and unaligned
  counts, but the aligned path would keep the pair-list peak, which is the
  cost this issue removes.
* **Fixed-point sums of the other member's values** (the ADR 0013 sketch).
  It cannot place the threshold on the credited member, and bucketing a
  continuous cutoff would not be exact.
* **Category groups as a call argument.** The EPIMIGHT groups overlap, and
  the fold costs nine codes rather than eight kinds.
* **Row-major `[rows, K]` input tables.** They would copy every column,
  640 MB for epimight's two columns at 20M rows, where borrowing copies
  nothing.
* **A memory budget.** It would refuse a result the caller needs in full;
  moments' budget exists because its lanes multiply, and these do not.

## Consequences

* pedsum `epimight-input` and fitACE_epimight `create_input` can count
  relatives with no pair list; with `--pairs`, pedsum still builds pairs,
  but only for its pair export. Both leave #32's consumer list.
* One allocation family (`relative_counts`) joins the tables, and
  `relationships/relatives.rs` joins the parallel-module map with its
  thread-budget test.
* The R package exposes `relatives_per_person()` for graphs (R has no
  views). Its result is an integer array `[row, category, column]`, copied
  out of the core's row-major counts, so the call briefly holds the result
  twice.
* Wall time and peak RSS on `pedsum_2M` and `pedsum_20M` go in
  `benchmarks/bench_relatives_per_person.md`.
