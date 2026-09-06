# Performance opportunities from the tskit ecosystem

This note records performance ideas from
[tskit](https://tskit.dev/tskit/docs/stable/),
[msprime](https://tskit.dev/msprime/docs/stable/), and
[tstrait](https://tskit.dev/tstrait/docs/stable/) that may improve relatedness
and kinship operations in pedigree-graph. The common project hub is
[tskit.dev](https://tskit.dev/).

This is an exploration note, not an accepted design. Any public-semantic or
architectural decision still requires an ADR. In particular, optimisations must
preserve the relationship contracts in [ADR 0006](adr/0006-public-api-and-coordinate-semantics.md)
and the pinned `float32` kinship recurrence and deterministic peel rule in
[ADR 0009](adr/0009-kinship-is-a-pinned-float32-recurrence.md).

## Existing strengths

Several important lessons from the tskit ecosystem are already present or
accepted in pedigree-graph:

- Compact integer graph-space coordinates and structure-of-arrays native
  storage are part of the accepted Rust-core design in
  [ADR 0007](adr/0007-rust-core-host-boundary-and-release.md).
- The Rust relationship engine streams rows with O(N) global state and bounded
  per-thread scratch rather than materialising global sparse products
  ([ADR 0010](adr/0010-row-streaming-relationship-engine.md)).
- Dense approximate-matrix support is exactified by capturing requested values
  during one complete retiring-DP pass, avoiding repeated dependency discovery
  across pair chunks
  ([matrix exactification profile](../benchmarks/matrix_exactification.md)).
- Pairwise queries use exact on-demand recurrence rather than constructing a
  complete relationship matrix
  ([ADR 0005](adr/0005-exact-on-demand-pairwise-kinship.md)).
- The inbreeding walk already distinguishes genome nodes from individuals to
  represent monozygotic coidentity
  ([ADR 0008](adr/0008-inbreeding-via-genome-node-meuwissen-luo-walk.md)).

The opportunities below therefore extend existing modules rather than propose
reimplementing these ideas from scratch.

## Ranked opportunities

| Rank | Opportunity | Expected value | Implementation status | Principal risk |
|---:|---|---|---|---|
| 1 | Select the kinship algorithm from the query shape | Very high for dense or overlapping pair support | Mechanisms exist separately | Violating ADR 0009 value semantics |
| 2 | Add matrix-free kinship products and reductions | Very high for iterative fitting and summaries | One specialised reduction exists | Floating-point equivalence |
| 3 | Simplify small views to relevant ancestry | Potentially very high for small subsets | New | Closure construction can cost more than it saves |
| 4 | Generalise the Rust row engine with result sinks | High when pairs are only an intermediate | Count sink exists; pair sink is planned | Duplicating classification semantics in sinks |
| 5 | Finish the columnar, zero-copy native core | Moderate directly; foundational for the others | Accepted, partially implemented | Porting without algorithmic improvement |

### 1. Select the kinship algorithm from the query shape

#### Upstream pattern

tskit exposes pairwise, matrix, weighted, and matrix-vector relatedness
operations rather than implementing all query shapes as repeated scalar calls.
Its `genetic_relatedness_matrix` documentation explicitly notes that one matrix
calculation is usually more efficient than many pairwise requests. See the
[statistics interface](https://tskit.dev/tskit/docs/stable/stats.html) and the
[`genetic_relatedness_matrix` source](https://github.com/tskit-dev/tskit/blob/5aeadfa419b8ae03a6c43d1d2faf0acc8f88fd93/python/tskit/trees.py#L8993-L9102).

#### Current seam

pedigree-graph already has two exact value engines with different reuse and
working-set properties:

- `_kinship_pairwise.py` evaluates sparse arbitrary pairs with a recurrence
  memo.
- `_kinship_dp.py` evaluates complete rows and retires them after their last
  direct child.
- `_kinship_matrix.py::_exactify_support` uses deterministic pair chunks for
  sparse relationship-selected support.
- `_kinship_matrix.py::_exactify_approximate_support` captures dense candidate
  values during one complete retiring-DP pass.

The existing profile shows why the distinction matters. On the 20,400-row
fitACE pedigree with 4,991,524 upper-triangle candidates, exact-value evaluation
with one shared pairwise memo took 117.5 seconds and 960 MiB of evaluation RSS.
The final fused-DP operation, including support construction and output
assembly, took 6.48 seconds with 848 MiB peak process RSS. Fixed pair chunks
bounded individual memo sizes but repeated dependency discovery, reaching
207.3 to 404.1 seconds. These are measurements of the existing implementations,
not a general speedup guarantee; see
[`benchmarks/matrix_exactification.md`](../benchmarks/matrix_exactification.md).

#### Proposed module boundary

Add an internal bulk-kinship planner, separate from the public scalar
`pair_kinship` contract. Given stable-topology pair support, it would select:

1. pair recurrence for sparse, weakly overlapping requests; or
2. complete retiring DP with candidate capture for dense or strongly
   overlapping requests.

The first target should be `relationship_kinship_matrix`, whose complete set of
support coordinates is known before exactification. Do not silently change the
public `pair_kinship` value path: ADR 0009 specifically rejected an adapter that
could return DP values for some pair queries.

The planner should initially be benchmark-driven rather than use a guessed
support-density threshold. Useful predictors may include pair count, endpoint
coverage, pedigree depth, and estimated ancestor overlap, but each predictor is
a hypothesis until measured.

#### Benchmark gate

Extend `benchmarks/matrix_exactification.py` across:

- relationship support through each degree and selected category groups;
- shallow, deep, inbred, missing-parent, and MZ pedigrees;
- sparse random pairs, concentrated endpoint blocks, and dense blocks;
- cold and warm execution;
- wall time, peak RSS, dependency/memo count, and output checksum.

Every retained value must match the ADR 0009 pair-recurrence bits. The result of
the benchmark should determine whether a stable crossover rule exists.

### 2. Add matrix-free kinship products and reductions

#### Upstream pattern

tskit provides `genetic_relatedness_vector`, which directly computes a
relatedness-matrix/vector product and may restrict the output to focal nodes,
rather than first materialising the complete matrix. Its `general_stat`
interface propagates multidimensional sample weights and applies summary
functions during traversal. See:

- [`genetic_relatedness_vector`](https://github.com/tskit-dev/tskit/blob/5aeadfa419b8ae03a6c43d1d2faf0acc8f88fd93/python/tskit/trees.py#L9172-L9233)
- [`general_stat`](https://github.com/tskit-dev/tskit/blob/5aeadfa419b8ae03a6c43d1d2faf0acc8f88fd93/python/tskit/trees.py#L7932-L8000)

The transferable principle is to make the requested reduction the traversal's
output, rather than make a sparse or dense matrix an obligatory intermediate.

#### Current seam

`PedigreeGraph.per_gen_mean_kinship` already demonstrates a specialised form of
this interface: `_kinship_dp.py` computes per-generation summary values while
streaming DP storage, and `_ne_rates.py` consumes those summaries. General
matrix-vector products and grouped reductions still require callers to obtain a
kinship matrix or issue pairwise queries.

#### Proposed interface

Prototype focused operations such as:

```python
graph.kinship_matvec(weights, *, rows=None)
graph.kinship_reduce(groups, *, statistic=...)
```

The names and exact public surface are deliberately unlocked. The first
implementation should support a small fixed set of native reductions rather
than an arbitrary Python callback inside a compiled traversal. Likely consumers
should be identified in simACE and fitACE before the interface is finalised.

The core module should own traversal and accumulation; host facades should only
validate coordinates and shape returned arrays. This gives Python and future R
bindings the same execution semantics.

#### Expected benefit

When the caller needs `K @ W`, cohort means, quadratic forms, or another small
summary, memory can scale with traversal state, the weight columns, and the
requested output rather than with all retained entries of `K`. Wall-time gains
are plausible when matrix assembly is avoided, but have not been measured.

#### Semantic risk and benchmark gate

The reference is the public matrix operation, not a mathematically equivalent
higher-precision recurrence. Tests must compare proposed results with operations
on `kinship_matrix()` for small pedigrees containing inbreeding, MZ twins,
missing parents, and permuted input rows. Before accepting tolerances, determine
whether fixed traversal order can reproduce the reference `float32` operation;
if not, document and decide the numerical contract explicitly.

Benchmark at least 1, 8, and 32 weight columns, both all rows and small focal-row
sets, recording wall time, peak RSS, and allocation volume.

### 3. Simplify small views to relevant ancestry

#### Upstream pattern

tskit's `simplify` operation retains the history relevant to selected samples
and optionally returns an old-to-new node map. This makes later traversals pay
for the retained ancestral history rather than the original data set. See the
[succinct data model](https://tskit.dev/tskit/docs/stable/data-model.html) and
[`TreeSequence.simplify`](https://github.com/tskit-dev/tskit/blob/5aeadfa419b8ae03a6c43d1d2faf0acc8f88fd93/python/tskit/trees.py#L6973-L7050).

#### Current seam

`PedigreeView` owns graph-row to view-row coordinate maps, but
`_pair_extractor.py::view_relationship_pairs` classifies relationships in the
full graph and then `_pair_utils.py::project_pairs` discards pairs whose
endpoints are not selected. A small view can therefore pay full-graph
classification cost.

#### Proposed module

Add an internal query simplifier that constructs a compact, immutable working
pedigree containing:

- selected endpoint rows;
- the ancestor closure needed by the requested operation;
- MZ-linked rows or an equivalent canonical genome-node representation needed
  to preserve coidentity;
- explicit compact-row to graph-row and graph-row to compact-row maps.

The result is an execution detail, not a new caller-visible coordinate space.
Public results must still use view rows and the view's coordinate token.

The closure needed for relationship classification must be derived and tested
per category. It should not be assumed that a plain parent closure is sufficient
for all MZ and category-exclusion rules.

#### Expected benefit

The opportunity is largest when an ascertainment sample or fitACE subset is a
small fraction of a very large pedigree and its ancestry remains compact. It
disappears when many selected rows have a near-global ancestor closure.
Construction, remapping, and cache-locality costs may make simplification slower
for large views.

#### Benchmark gate

Measure view fractions of approximately 0.1%, 1%, 10%, and 50% across shallow
and deep pedigrees. Record selected size, closure size, construction time,
relationship/kinship time, peak RSS, and result parity. Use these measurements
to decide whether simplification should be explicit or automatically selected;
do not set a view-size threshold alone without also measuring closure size.

### 4. Generalise the Rust row engine with result sinks

#### Upstream pattern

tskit's statistics implementation accumulates requested summaries during tree
traversal instead of materialising every intermediate node/sample relation.
tstrait follows a related two-level pattern: it computes values on genome nodes
and then accumulates those values directly to individuals in a linear compiled
kernel. See:

- [tskit statistics](https://tskit.dev/tskit/docs/stable/stats.html)
- [tstrait genetic-value calculation](https://tskit.dev/tstrait/docs/stable/genetic.html)
- [`_compute_nodes_genetic_value`](https://github.com/tskit-dev/tstrait/blob/31fec9a7d996360d7afc2f0526544c897e61ab7a/tstrait/jit.py#L24-L60)
- [`_accumulate_individual_values`](https://github.com/tskit-dev/tstrait/blob/31fec9a7d996360d7afc2f0526544c897e61ab7a/tstrait/jit.py#L64-L76)

The transferable principle is to separate one traversal/classification module
from the representation of its result.

#### Current seam

The Rust engine in `crates/core/src/relationships/` already classifies one row
at a time with O(N) global state and bounded per-thread scratch. Its current
output is a count sink. ADR 0010 states that the production pair engine should
reuse this traversal with a pair sink rather than create a separate
implementation.

Recorded release-build measurements for all 23 categories were:

| Rows | Threads | Wall | Peak RSS |
|---:|---:|---:|---:|
| 2 million | 12 | 6.7 s | 302 MiB |
| 20 million | 12 | 83 s | 2.86 GiB |

These measurements establish the row traversal as a strong base; they do not
measure the proposed sinks.

#### Proposed module boundary

Keep classification and precedence in one engine, with internal adapters for:

- exact counts;
- bounded chunks of classified pairs;
- category/group summaries;
- kinship-support capture for a downstream bulk evaluator.

Prefer closed native sink types over arbitrary host callbacks. A Python callback
on every row or pair would reintroduce boundary overhead and make parallel
execution difficult. Any sink needing all pairs must make output-proportional
memory explicit; streaming cannot make the final materialised output smaller.

#### Benchmark gate

Benchmark the same traversal with count, pair-chunk, and one representative
summary sink. Separate classification time from host conversion and final-output
allocation. Verify identical category assignment across sinks, thread counts,
and parity fixtures. Include a downstream consumer that currently materialises
pairs only to aggregate them, since that is where a fused sink should provide
the clearest gain.

### 5. Finish the columnar, zero-copy native core

#### Upstream pattern

tskit stores fixed-type table columns contiguously and exposes immutable data
through array-oriented APIs. msprime's fixed-pedigree builder bulk-appends typed
individual and node columns rather than adding every row through the scalar
path. See:

- [tskit data model](https://tskit.dev/tskit/docs/stable/data-model.html)
- [tskit Python API](https://tskit.dev/tskit/docs/stable/python-api.html)
- [msprime fixed-pedigree model](https://tskit.dev/msprime/docs/stable/ancestry.html)
- [`PedigreeBuilder.add_individuals`](https://github.com/tskit-dev/msprime/blob/996f12d83a5231533fbb2f94c4684817709b79b9/msprime/pedigrees.py#L357-L377)

#### Current seam

ADR 0007 already accepts the corresponding pedigree-graph architecture:
structure-of-arrays core storage; typed individual and row IDs; host-neutral
ownership; one shared execution context; and PyO3 and future R bindings around
the same Rust core. The row-streaming relationship engine exists in
`crates/core`, but the complete native graph, Python binding, kinship kernels,
and host representations have not all landed.

#### Recommendation

Complete the accepted migration rather than add more long-lived
Python/Numba/SciPy seams:

1. move validated topology and coordinate maps into the host-neutral core;
2. expose bulk constructors from contiguous typed arrays;
3. define ownership so large outputs are allocated once where practical;
4. wire the row engine through PyO3;
5. port kernels only when their existing algorithm and benchmark gate are
   understood.

This is an enabling investment, not evidence that Rust alone will make every
operation faster. ADR 0007 correctly warns that the complete kinship DP is
output-dominated and that its storage representation must be benchmarked before
selection.

#### Benchmark gate

Measure construction, topology building, one relationship query, pairwise
kinship, and host result conversion separately. Include cold Python/Numba and
warm Python/Numba baselines so removal of compilation latency is not confused
with steady-state kernel speed. Record copied bytes or allocation counts where
possible, in addition to wall time and peak RSS.

## Kinship-kernel follow-up opportunities

The ranked opportunities above treat kinship mostly at the operation and module
level. Looking inside the recurrence and DP kernels exposes another set of
experiments. These are not additional public APIs by default; most are possible
internal implementations of opportunities 1 and 2.

### Three calculation shapes

pedigree-graph currently computes kinship in three materially different ways:

1. **Requested pairs:** `_kinship_pairwise.py::_pairwise_kinship_core` performs
   an iterative, post-order Karigl recurrence over only the requested pairs and
   their pair-state dependencies. One open-addressing memo is shared across the
   request (`pedigree_graph/_kinship_pairwise.py:214-352`).
2. **Matrices and selected support:** `_kinship_dp.py::_dp_kinship` processes
   the pedigree depth by depth, while `_kinship_dp_depth.py::_process_depth`
   merge-walks the two parent rows and writes symmetric child entries
   (`pedigree_graph/_kinship_dp_depth.py:195-268`).
3. **Inbreeding only:** `_inbreeding_kernel.py::_compute_F_meuwissen_luo`
   performs the genome-node Meuwissen-Luo walk using the decomposition
   `A = T D T'` without constructing pairwise kinship
   (`pedigree_graph/_inbreeding_kernel.py:53-210`).

These shapes have different outputs and working sets. A single universal
kinship kernel is unlikely to be optimal; the useful design target is a shared
value contract with query-specific execution modules.

### K1. Compile large pair requests into a row-bucketed recurrence DAG

This is the strongest additional pairwise-kinship experiment.

In stable topological order, a canonical pair state is `(lo, hi)`, and the
production kernel decodes `lo` as `other` and `hi` as `peeled`
(`pedigree_graph/_kinship_pairwise.py:275-279`). Every dependency has a
strictly smaller maximum endpoint:

- an ordinary state emits `(mother_hi, lo)` and `(father_hi, lo)`
  (`pedigree_graph/_kinship_pairwise.py:311-316`); and
- a self or MZ-like state emits `(mother_hi, father_hi)`
  (`pedigree_graph/_kinship_pairwise.py:296`).

Both parents precede the peeled child, and `lo < hi` for a non-self state, so
the recurrence DAG is naturally layered by `hi`. A bulk engine could exploit
that structure:

1. canonicalise requested roots and bucket each `lo` by its `hi`;
2. sweep `hi` downward, sorting/deduplicating each completed bucket and
   emitting its dependencies into lower buckets;
3. sweep `hi` upward, evaluating each state once after all dependencies have
   been evaluated; and
4. scatter requested root values back to caller order.

A compact representation could use row offsets plus one `int32 lo` and one
`float32` value per distinct state. The current global memo uses an `int64` key
and `float32` value—12 bytes per slot—and grows at a 70% load threshold
(`pedigree_graph/_kinship_pairwise.py:42`, `72-75`, `337-343`). At the maximum
load that is about 17 bytes of slot storage per live entry before stack/output
storage, and geometric growth temporarily retains both tables. ADR 0009
measured the rehash copy at about 30% of peak RSS on the 536k-row case
([ADR 0009](adr/0009-kinship-is-a-pinned-float32-recurrence.md)). Row buckets
could remove occupancy slack, global random probing, and whole-table rehashing.
Temporary closure arrays and bucket metadata must be included in any comparison;
the eight-byte steady-state pair is not a complete peak-memory estimate.

The arithmetic can remain bit-identical to ADR 0009: each state still performs
the same correctly rounded `float32` half-sum after its two named dependencies.
Evaluation order between independent states does not enter that formula.

The main risks are sorting cost, duplicate dependency generation, and poor
performance for small requests. Because every unique state emits at most two
dependencies, dependency generation should be measured rather than assumed to
explode. This should initially be a large-batch competitor to the current memo,
not a scalar-path replacement.

#### Benchmark gate

Compare the row-bucket prototype with the current memo on:

- isolated pairs, relationship blocks, and multi-million-pair requests;
- low- and high-overlap requests with the same pair count;
- shallow, deeply inbred, and MZ pedigrees;
- distinct states, generated dependencies before deduplication, sort time,
  evaluation time, peak RSS, and output checksum.

The gate is bit equality with the current recurrence plus a measured region in
which either wall time or peak memory improves materially.

### K2. Instrument and tune the existing hash memo first

Before replacing the memo, measure whether its simplest implementation choices
are a bottleneck. `_memo_slot` maps a canonical key directly with
`idx = key & mask`, without mixing the key bits
(`pedigree_graph/_kinship_pairwise.py:176-184`). Since the key is
`lo * n + hi` and relatedness workloads often concentrate endpoints, poor
low-bit distribution is plausible but unverified.

Add benchmark-only telemetry for:

- total probes and lookups;
- mean, percentile, and maximum probe-chain length;
- probe counts before each growth;
- entries created per requested root; and
- time and peak allocation attributable to rehashing.

Then compare direct masking with a cheap deterministic 64-bit mixer, several
load factors, and optional pre-sizing from an observed or sampled
states-per-root ratio. Hash placement cannot change recurrence values because
the original canonical key is still stored and compared.

The current `_memo_grow` allocates a complete doubled table and rehashes every
entry (`pedigree_graph/_kinship_pairwise.py:196-211`). Independently of key
mixing, the Rust-core implementation should benchmark segmented growth or
another no-predecessor-copy layout against the requirement already recorded in
[ADR 0007](adr/0007-rust-core-host-boundary-and-release.md). A standard Rust
hash map is not automatically better: its entry alignment, control bytes, load
factor, and growth peak all need measurement against the current structure of
arrays.

### K3. Use `A = T D T'` for an implicit kinship operator

Opportunity 2 proposed a matrix-free API. The existing inbreeding calculation
provides a concrete implementation path. `_compute_F_meuwissen_luo` already
constructs the Mendelian-sampling diagonal `D` and traverses path coefficients
from `T` on the genome-node pedigree
(`pedigree_graph/_inbreeding_kernel.py:68-80`, `136-202`), but currently returns
only `F` (`pedigree_graph/_inbreeding_kernel.py:210`).

In exact arithmetic, with numerator relationship matrix `A = T D T'` and
kinship `K = A / 2`, a product `K @ W` can be calculated without storing `K`:

1. initialise `u = W` and make a reverse topological pass, adding half of each
   child's `u` to each represented parent's `u`; this applies `T'`;
2. multiply each genome-node row by `D`;
3. make a forward topological pass, adding half of each represented parent's
   accumulated value to each child; this applies `T`;
4. divide by two; and
5. expand genome-node values back to individuals.

For MZ pairs, individual weights must first be summed onto their shared genome
node; the resulting node value is then copied back to both individual rows.
Children of either twin must reference that same canonical genome node. Once
`D` is available, the operator requires two passes over at most two represented
parent edges per genome node and per weight column, rather than matrix entry
construction.

#### Numerical contract

This cannot silently implement an existing matrix method. The Meuwissen-Luo
walk and `D` are float64, while ADR 0009 defines returned pair and matrix values
by a particular per-step `float32` recurrence. The existing contract only holds
`F = 2 * phi(i, i) - 1` within tolerance on deep pedigrees. Factorised
multiplication also changes summation order. Therefore an implicit operator
must either:

- declare a factorised, tolerance-based numerical contract; or
- demonstrate that a more constrained implementation reproduces
  `kinship_matrix() @ W` under a separately specified operator contract.

The first option is likely more useful scientifically, but it is a public
semantic decision requiring an ADR and consumer evidence.

#### Benchmark gate

Use a real simACE or fitACE matrix-vector workload. Compare against
`kinship_matrix() @ W` where the complete matrix fits, at 1, 8, and 32 weight
columns, measuring `D` construction separately from repeated products. Include
MZ, one-known-parent, inbred, and row-permuted fixtures, and report both absolute
and ULP differences rather than asserting bit parity.

### K4. Iterate contiguous depth spans instead of scanning all rows

The private topology is depth-major, but each depth operation currently scans
all `n` rows and rejects rows outside the active depth:

- candidate capture: `pedigree_graph/_kinship_dp_depth.py:41`;
- MZ pass: `pedigree_graph/_kinship_dp_depth.py:96`; and
- parent-row processing: `pedigree_graph/_kinship_dp_depth.py:212`.

Precompute `depth_start[d]` and `depth_end[d]` once, then pass the active range
to each kernel. This removes repeated `range(n)` scans and branches without
changing row order, recurrence arithmetic, retirement, or candidate capture.

The likely gain is modest on the usual approximately eight-generation simACE
pedigree, but it may matter on deep pedigrees because several control passes
currently scale with `n * number_of_depths`. Benchmark shallow and 50–60-depth
fixtures before implementation. This is the lowest-risk kernel-level
optimisation in this section.

### K5. Reuse parental merge templates within sibships

`_process_depth` merge-walks the same two parent rows independently for every
child (`pedigree_graph/_kinship_dp_depth.py:229-268`). Full siblings at the same
depth share the part of this calculation involving rows from earlier depths.
A family-batched implementation could:

1. group same-depth children by canonical `(mother, father)`;
2. merge the portions of the parent rows from earlier depths once;
3. stream or copy that recurrence-computed base into each child row; and
4. handle current-depth sibling entries, symmetric writes, and MZ cases
   separately.

The current parent rows gain entries for previously processed children, so
blindly copying the entire first child's result would be wrong. The reusable
template must end at the current depth boundary, and sibling-to-sibling values
must follow the pinned recurrence explicitly. One-known-parent and half-sib
families need separate treatment or can stay on the ordinary path.

This is most promising for livestock or other pedigrees with large sibships;
it may add overhead to ordinary two-child families. Before designing storage,
profile parent-row merge entries per child and the distribution of children per
canonical parent pair.

### K6. Fuse propagated-support discovery with exact-value capture

`approximate_kinship_matrix` currently runs a threshold-propagating DP to build
candidate support (`pedigree_graph/_kinship_matrix.py:517-537`) and then a
complete retiring DP to capture exact values on that support
(`pedigree_graph/_kinship_matrix.py:542`). A dual-state DP could carry:

- complete exact values used by the unpruned recurrence; and
- propagated values or presence flags used solely to decide candidate support.

When the propagated state survives, the operation could emit the corresponding
exact value directly. This might avoid a separate candidate CSC assembly,
graph-to-topology support remap, and second orchestration boundary.

The semantics are delicate: `min_propagated_kinship` defines propagation-pruned
support, not a final threshold on exact values. The propagated state therefore
cannot be replaced by `exact_value > threshold`.

The measured ceiling on the likely gain is limited. On the 20,400-row profile,
candidate construction took about 2.7–2.8 seconds and the complete retiring DP
about 2.9 seconds; final end-to-end time was 6.48 seconds
([matrix exactification profile](../benchmarks/matrix_exactification.md)). A
fused traversal may also increase peak live memory by carrying exact and
propagated state together, so it should rank below the pairwise memo and
factorised-operator experiments unless a larger profile shows support
construction dominating.

### Lower-priority kinship experiments

#### Exact zero-rejection fingerprints

Assign represented founders bits in a fixed 64- or 128-bit fingerprint and
propagate fingerprints to descendants with bitwise OR. If two endpoint
fingerprints have an empty intersection, they cannot share a represented
founder and their kinship is exactly zero. Founder collisions only produce
false positives that fall back to the recurrence; they cannot produce a false
zero. MZ founders must share one genome-node fingerprint, and missing parents
remain unrepresented unique ancestry under the current recurrence.

This adds fixed storage and an AND test per checked state. It may help arbitrary
mostly-unrelated pair requests, but relationship-selected pairs already share
ancestry by construction. Benchmark its rejection rate before adding it to the
core graph.

#### Memory-budgeted parallel pair shards

Independent pair batches can run in parallel with separate memos and produce
deterministic per-pair values. The tradeoff is repeated dependency discovery
and aggregate memo memory. If pursued in Rust, cluster pairs by endpoint or
ancestral locality, expose an explicit total memory budget, and compare total
state evaluations with the shared-memo baseline. Dense support should continue
to use the DP rather than parallelise an unsuitable pairwise plan.

#### Universal genome-node compression

Canonical genome nodes could remove duplicate MZ states from more kernels, but
ADR 0009's tie-break is defined in stable individual-row order. Replacing a
twin parent with another representative can change a same-depth peel path and
therefore deep-inbreeding rounding bits. Do not adopt this as a pairwise
optimisation without proving bit parity across the deep and row-permuted corpus.
It remains more natural for a separately specified factorised operator.

## Ideas considered but not ranked

### Incremental edge-difference traversal

tskit's `edge_diffs` updates state efficiently because adjacent genomic trees
share most edges. A pedigree graph has no corresponding sequence coordinate or
series of adjacent topologies: one pedigree topology is reused for all current
pedigree-expected calculations. The broader lesson—retain state across small
structural changes—is useful, but direct adoption would require a workload such
as repeated closely related views or incrementally appended generations. No such
workload has yet been shown to dominate, so this is not a top-five opportunity.

### msprime fixed-pedigree gene dropping

msprime can simulate inheritance through a supplied diploid pedigree, including
multiple independent genomic regions. Combined with tskit statistics, this
could estimate realised genomic relatedness or provide a Monte Carlo validation
backend. It is not a transparent acceleration of pedigree-expected kinship:
finite simulation introduces sampling error, founder assumptions affect the
result, and MZ coidentity needs explicit handling. It therefore changes the
estimand and should be a separate feature, not an optimisation of the current
exact recurrence.

## Recommended evaluation order

1. Add benchmark-only probe and rehash telemetry to the existing pairwise memo;
   test key mixing and confirm the actual lookup bottleneck before redesigning
   storage.
2. Prototype the row-bucketed recurrence DAG against the current memo on sparse,
   overlapping, and multi-million-pair requests.
3. Extend the exactification benchmark to relationship-selected support and
   determine whether a reliable pairwise/DP crossover exists.
4. Identify one real simACE or fitACE matrix-vector workload and prototype the
   `A = T D T'` operator against it, with numerical differences reported
   explicitly.
5. Benchmark depth-span iteration and measure parental-row merge duplication;
   pursue family templates only if sibship reuse is substantial.
6. Profile view queries by ancestor-closure size before designing a public
   simplification option.
7. Implement pair and summary sinks as extensions of the accepted Rust row
   engine, not as independent classifiers.
8. Evaluate dual-state support/value fusion only after a larger profile shows
   that support construction is worth its additional live state.
9. Use the native-core migration to consolidate successful prototypes after
   their algorithms and ownership requirements are measured.

## Source snapshots reviewed

The observations above were checked against these upstream source snapshots:

- tskit: [`5aeadfa419b8ae03a6c43d1d2faf0acc8f88fd93`](https://github.com/tskit-dev/tskit/tree/5aeadfa419b8ae03a6c43d1d2faf0acc8f88fd93)
- msprime: [`996f12d83a5231533fbb2f94c4684817709b79b9`](https://github.com/tskit-dev/msprime/tree/996f12d83a5231533fbb2f94c4684817709b79b9)
- tstrait: [`31fec9a7d996360d7afc2f0526544c897e61ab7a`](https://github.com/tskit-dev/tstrait/tree/31fec9a7d996360d7afc2f0526544c897e61ab7a)

Performance claims for pedigree-graph are limited to measurements already
recorded in this repository. All projected benefits remain hypotheses until the
benchmark gates above are run.
