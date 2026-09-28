# Architecture

Experiments W/O Stress separates a paper's scientific choices from the machinery
that runs and records them. Algorithms, data generators, and any custom
protocols or metrics belong to the study, conventionally in `experiment_code/`.
The library plans runs, manages state and artifacts, and analyzes saved
observations on one machine. For the public authoring interfaces, see the
[configuration guide](CONFIGURATION.md).

For an ordered walk through the implementation and its tests, see
[Reading the code](CODE_GUIDE.md).

## Data flow and responsibilities

```text
YAML → validated study → run plan → saved instance
     → protocol + algorithm + generator → observations and checkpoints
     → metrics → aggregation → figures
```

A **group** defines algorithms, parameters, and repetitions. A planner turns it
into **runs** with stable identities. Each run gets fresh component objects and
separate random streams. The **protocol** owns interaction order and step
progress; the **algorithm** owns learned state; the **generator** owns evolving
environment state. The executor records selected observations and persists
checkpoints. Metrics read those observations and the saved instance after
execution. Analysis never needs to construct the simulation components.

An `Instance` contains immutable scientific input, such as arm means, a graph,
or a dataset. It is persisted separately from changing component state. The
generator may define `create_instance`; the executor supplies the saved result
to its constructor. If evaluation needs a realized hidden quantity, the
generator must record it as a measurement rather than mutate the instance or
accumulate a metric internally.

## Source organization

The source tree follows six responsibilities:

| Package | Responsibility |
| --- | --- |
| `study/` | Validated configuration, serializable run specifications, planning, and RNG derivation. `specs.py` keeps saved descriptions independent of component loading. |
| `components/` | Extension contracts and dynamic loading. External study code implements these interfaces. |
| `builtins/` | Library-supplied protocols, generators, bandit environments, bandit metrics, and the optional Gymnasium adapter. |
| `execution/` | Request coordination, GPU allocation and worker lifecycle, compute observation/reporting, provenance, logging, and notifications. |
| `storage/` | Immutable artifact models, atomic files, checkpoint representations and transactions, experiment state, and cleanup. |
| `analysis/` | General metric contracts and field metrics, repeated-run aggregation, validated caches, and figure export. |

`ExecutionCoordinator` plans one invocation, publishes its prepared request, and
schedules bounded work while owning signals and notifications. One compact durable
request describes the plan; per-variant metadata is published just before first
use. Repeated source files are fingerprinted once per preparation. No per-run
filesystem materialization is required merely to plan a large grid. Each worker creates
a `RunSession`, which owns its components, RNGs, recorder, and checkpoint lifecycle.
The process-pool entry point remains a module-level function.

GPU execution adds a focused resource owner in `execution/resources.py`. With
`execution.gpu_ids` present, it launches one persistent spawned process per
configured GPU, including the one-worker case. The coordinator submits at most
one simulation or derivation task at a time to each worker. GPU visibility is inherited at process launch,
before spawn reimports the main script; a pool initializer would run too late
to protect against a framework imported there. Each process retains its mask
for its lifetime and constructs fresh components for each run. Execution's
planning, preflight checks, and source inspection take place in the first GPU
worker so they cannot initialize a scientific framework in the coordinator.
Metric and plotter preparation also stays in an assigned worker; only plain
dependency descriptors return to the coordinator.
Scientific algorithms use their worker-local device, normally `cuda:0`.
This execution boundary does not control study code the caller imported before
`run_experiment`; `ews count-runs` can import a custom planner in its own
process.

Configuration validates a nonempty list of distinct nonnegative GPU IDs and
requires one worker per ID. GPU execution requires `CUDA_VISIBLE_DEVICES` to be
unset in the launching environment; IDs belong to the host or container runtime's
unmasked GPU namespace. The resource owner restores the parent's environment
after launching each process and closes workers after completion or failure.
This allocation is exclusive within one invocation, not a machine-wide GPU
reservation. Without `gpu_ids`, CPU execution retains its local single-worker
or spawned process-pool behavior. See the
[GPU configuration](CONFIGURATION.md#gpu-execution) for a YAML example.

Worker launches serialize through an EWS lock and temporarily change the
parent's `CUDA_VISIBLE_DEVICES` during `Process.start()`. This lock does not
cover unrelated application threads that inspect or change the environment,
initialize a GPU framework, or launch subprocesses during that window. Avoid
those concurrent operations in an embedding application while EWS launches GPU
workers.

`ExperimentStore` owns the output root, lock, retained variants, and request
publication. Execution computes scientific identities before passing them to the
store. `RunStore` owns one variant's durable boundaries; `Recorder` buffers its
observations. `AnalysisCache` handles immutable cache generations and exports,
while metric computation and aggregation remain in the analysis pipeline.

`execution/progress.py` observes simulation and derived worker tasks. Workers send
small snapshots through a bounded multiprocessing queue at most four times per
second, with forced terminal updates. Saturated monitoring queues may discard
snapshots; coordinator results remain authoritative for counts and final 100%
completion. Both CPU and GPU workers use the same transport. Progress uses the
existing protocol step and budget; atomic offline/trial work has one unit and
custom protocols may supply `total_steps` when no budget is known.

The CLI owns `TerminalProgress`, whose single parent-side display thread drains
snapshots and updates Rich or periodic plain stderr lines. This also refreshes
elapsed time while a sequential fit blocks inside one operation. Rows show task kind, subject, elapsed time, and defensible progress/ETA. They are
keyed by process ID and replaced on worker reuse. Wall-clock finish estimates
always include the selected IANA timezone; UTC remains the timestamp standard
for persisted records. The context closes the display and
queue on success, failure, or interruption; workers never render. Notifications
continue to use their independent sparse sender and durable checkpoint readings.

`CheckpointBackend` in `storage/checkpoints.py` separates checkpoint representation
from `RunStore`'s transaction. The default `NumPyCheckpointBackend` uses explicit
JSON and NPZ encoding; a study backend can preserve native framework state or
write multiple files. The backend never owns execution order, checkpoint timing,
scientific identities, or the result boundary.

Storage has no execution dependency. Configuration validates notification options
without importing the sender. Figures share cache operations with the pipeline
and use its public analysis result. Scientific components are loaded only when
planning or execution requires them; saved results remain usable without the
study's simulation modules.

Root exports and documented historical submodules remain available through thin
compatibility facades. Internal code imports the owning modules. Existing built-in
aliases and qualified component strings retain their meaning.

## Reproducibility

The experiment seed feeds separate NumPy streams for instance generation, the
generator, algorithm, and protocol. Stable run identities keep those streams
independent of worker count, GPU assignment, scheduling, grid traversal order,
recording frequency, and requested budget. Data and instance identities exclude
algorithm choice so competing algorithms can share a problem instance. Action-dependent
environments can still produce different trajectories.

Metadata records the resolved configuration, seeds, component and library code,
tracked inputs, and numerical environment. Source fingerprints cover complete
component modules and base-class modules, plus declared dependency files. This
conservative boundary prevents code changes from silently reusing old simulation
results. It also means an edit to a metric in the same module as an algorithm
can invalidate the algorithm's run; separate modules make independent reuse
possible. A long-running Python process retains imported modules, so restart or
reload after editing already imported study code.

GPU IDs and actual worker assignments are execution diagnostics, retained in
resolved configuration, provenance, and run logs. They do not enter scientific
run IDs, RNG derivation, or stored variant selection. EWS manages resource
visibility without importing a GPU framework or requiring hardware detection. The
study remains responsible for framework RNGs, supported checkpoint state, and
numerical reproducibility across hardware; identical EWS streams do not imply
bitwise-identical GPU computations.

Checkpoint backend selection and its separately tracked source are operational
provenance, excluded from run IDs, RNG derivation, and stored variant selection.
Every generation retains the backend type, parameters, source fingerprints, and
representation metadata needed to identify its loader. Put backend code in its
own module: editing a file also used by a scientific component still changes that
component's fingerprint. These rules do not exempt library implementation edits
from the existing conservative compatibility checks.

Library fingerprints cover the implementation files in the new packages, rather
than their compatibility facades. The dependency-scheduler redesign therefore selects new
simulation and analysis variants conservatively. Existing artifact schemas and
readers are unchanged; retained results remain inspectable and analyzable with
their matching saved configuration.

Compute metadata never enters these identities. Reporting modules are excluded
from simulation fingerprints, and reporting does not change analysis cache
inputs. `execution/compatibility.py` maps exact, reviewed reporting-only versions
of the coordinator, worker, GPU resource, provenance, and figure-export files to
their preceding digests. This narrow compatibility bridge preserves existing
scientific variants and figure caches when compute or terminal instrumentation
is introduced. An unrecognized source
digest still invalidates reuse conservatively; it is not a general exemption for
implementation edits.

Compatibility includes Python and numerical-library versions, platform identity,
and resolved input paths. External dependencies that a study does not declare
still require management by that study. A container image alone does not make
checkpoints portable across hosts; the
[cloud guide](CLOUD.md#resources-and-compatibility) gives the practical
migration limits.

## Budget and reusable work

`budget.steps` requests execution length, separate from scientific parameters
such as a learning rule's design horizon. A larger budget can continue from a
saved endpoint only when the protocol, algorithm, and generator all declare
`supports_extension = True` and preserve the same scientific meaning. The
executor restores their state and RNGs, sets the new protocol budget, and
executes only the added steps. Otherwise, the changed budget selects another
variant.

Scientific settings, source code, tracked inputs, and recording selection can
also select new variants. Earlier artifacts are retained, while unaffected runs
remain reusable. The active request identifies the variants and result
boundaries used by analysis; request history remains inspectable. Completed
budget boundaries retain their manifests and endpoint snapshots so an earlier
request can still be read after extension.

Sparse recording follows scheduled step multiples. For example, interval 10
records steps 10 and 20 at budget 23, without adding a special step-23 record.
Direct execution and budget extension therefore produce the same recorded steps.

## Recording and artifacts

Recording selects numerical fields and logical steps. `step` is always included,
and each field must keep a consistent numerical dtype and shape. Immutable NumPy
`.npz` result chunks preserve every recorded observation, without a duplicate
trajectory CSV. They flush at a per-run byte target (16 MiB by default), before
checkpoints, and at completion. A chunk flush alone is not a resumable
checkpoint. Buffering is per active run, in addition to component state and
serialization memory.

```text
output/
  artifacts.json                 semantic role catalog for external tools
  metadata.json                  active request and artifact schema
  config.resolved.yml             resolved configuration
  requests/                      retained requests
  instances/<content-id>/         immutable scientific inputs
  runs/<variant-id>/
    metadata.json                 identity and selected variant
    results/                      numerical chunks
    checkpoints/                  component and RNG snapshots
    progress.json                 committed boundaries and budgets
    trajectory.json               intentional pruning proof, when pruned
    failure.json                  latest failure, when present
    run.log                       rotating diagnostics
  analysis/
    cache/                        metric, aggregate, and figure caches
    <metric>.csv                  bounded curves, original x coordinates
    <metric>.npz                  the same bounded summary arrays
    figures/                      current exports
  compute/
    invocations/<uuid>.json       immutable finished invocation records
    attempts/<uuid>.start.json    immutable attempt start records
    attempts/<uuid>.json          immutable finished attempt records
    summary.json                  regenerated aggregate data
    summary.md                    concise human-readable report
```

External tools consume [the semantic catalog](ARTIFACTS.md), not this layout
diagram. `storage/artifact_index.py` owns discovery publication: fixed optional
locations, explicit schema/version, deterministic atomic writes, and no tree
inventory. Execution requests, analysis/plot workflows, and compute report
regeneration publish it independently. Exact reviewed discovery hooks preserve
existing scientific and analysis identities through `execution/compatibility.py`;
unrecognized source changes still invalidate conservatively.

The public storage readers validate and expose selected results; exact filenames
within a generation are internal. `iter_completed_runs` yields run
specifications and `RunResult` objects, each with recorded fields, the saved
instance, a completed step count, revision, and final outputs. For single-step
offline or trial runs, `final_outputs` exposes the last recorded values.
Completed-run reuse and `inspect` validate the saved instance as well as the
trajectory chunks. Missing or damaged scientific inputs are reported as corruption;
intact result chunks alone are insufficient for successful reuse.

## Checkpoints, completion, and interruption

A checkpoint is committed only after a complete protocol step. The executor
flushes observations and captures a logical mapping of `algorithm`, `data`,
`protocol`, and `rngs` state. It passes those values to the selected backend
without first forcing native values into NumPy. The backend writes a new
generation; storage validates and synchronizes the files before atomically
publishing their hashes and matching result boundary.
It keeps preceding valid generations so recovery can fall back if the newest
checkpoint is damaged. A directory or partial file is never treated as completed
work.

The backend's `save(state, directory)` returns finite JSON-compatible metadata,
and `load(directory, metadata)` reconstructs the complete logical mapping.
Constructors receive configured parameters, without a simulation RNG. Payloads
may have multiple files and subdirectories, all confined to the supplied
generation without symbolic links. `checkpoint.json` is reserved for EWS's
envelope. Save must finish all writes and close handles before returning; files
then remain immutable. `RunStore` inventories, checksums, and synchronizes every
payload, owns the commit point in progress metadata, and prunes only after
publication. Neither a backend's successful return nor a completed payload file
alone makes a checkpoint resumable.

Restore verifies the saved envelope and payload inventory before loading state
with that generation's backend descriptor. The current configured backend applies
to subsequent writes, so an invocation can resume one representation and publish
another without changing scientific state. Saved backend code and dependencies
must remain available, and loaders must support the representation versions they
created. If no retained generation can be decoded, missing or incompatible loaders
report a continuation problem while preserving artifacts. Corrupt generations
can fall back to a preceding valid checkpoint, subject to the existing
completed-prefix boundary.

Without a custom backend, NumPy studies keep the same supported state values and
need no new configuration. Legacy generations with the original `state.json`
and `arrays.npz` hashes remain readable. This read compatibility does not bypass
code provenance, automatically migrate artifacts, or promise that a different
library revision selects the same stored variants. Numerical results and instances
retain their existing format and remain usable for analysis without checkpoint
backend imports.

Recovery validates committed state and result chunks, discards uncommitted
observations, and replays from the last valid boundary without duplicating
records. Graceful interruption requests a checkpoint at the next safe step;
forced termination can lose work since the previous commit. An offline `fit()`
or callable trial is one indivisible step and cannot checkpoint inside that
call.

Execution and cleanup share a local lock. Workers write separate run
directories, and task submission is bounded. Failures retain inspectable error
information; per-run logs rotate. The live output filesystem needs process
locking and atomic replacement, which an object-store URL does not supply.

## Compute observation and reporting

Resource ownership stays in `execution/resources.py`. Observation belongs to
`execution/compute.py`, hardware queries to `execution/compute_environment.py`,
and durable compute records and aggregation to `execution/compute_report.py`.
These modules do not interpret scientific observations or modify checkpoints.

The CLI and Python execution API observe one complete pipeline invocation. Its
elapsed time includes orchestration and overlapping task execution. Each actual
simulation, metric, aggregate, or figure attempt has its own elapsed and process
CPU cost; totals by task kind are cumulative worker costs, not sequential stage
durations. Historical records with sequential stages remain readable. Invocation
records retain UTC timestamps, worker count, run outcomes and derived failures.

A decorator on `execute_run` brackets each worker call, including restoration,
initialization, checkpoint publication, and component cleanup. Its process CPU
user/system deltas use `resource.getrusage(RUSAGE_SELF)`: threads in that process
contribute, descendants do not. Reused worker processes are measured by deltas,
not lifetime totals. Unrelated threads can contribute in an embedded application
using the single in-process worker. No process-lifetime peak RSS is presented as
an individual run's peak; memory reporting is machine RAM capacity only.

Simulation attempts reference their scientific run, stored variant, group,
algorithm, and invocation. Derived attempts identify their task kind and subject. A reused completion contributes no new simulation compute.
GPU assignments come from the worker's existing fixed resource allocation;
attempt elapsed seconds multiplied by allocated GPU count gives GPU-seconds.
This excludes utilization claims and is not a measurement of the worker's idle
GPU reservation between attempts. Worker time sums elapsed attempt durations;
it differs both from end-to-end elapsed time and from CPU process time. EWS does
not claim CPU-core-hours because it does not allocate CPU cores to workers.

Starts and completed records have unique IDs and are published atomically in
the separate compute tree. A forced worker or coordinator exit can leave a
start without a finish; its duration remains unknown. Normal failure, pause,
and resume preserve separate attempt records. Aggregation keeps completed
attempts, failed/paused work, and missing history distinguishable. Timing
statistics summarize attempts, so a successful resumed attempt need not describe
the cost of the complete scientific run. Older artifacts without recorded timing
remain unknown. Immutable records are retained across invocations; summaries
are replaceable views of the observed history.

Hardware collection is best effort on Linux/macOS, with unsupported Windows
reporting omitted silently. Small bounded system queries can provide NVIDIA
model/VRAM/driver or Apple chip information without framework imports. Unknown
fields stay unavailable. No username, hostname, home-directory name, or network
address is needed in the hardware report. Existing scientific provenance remains
unchanged. Filesystem capacity and availability are start snapshots. One
finalization scan sums regular-file lengths without following symlinks before
publishing the new report; this is logical artifact size, not allocated blocks.
Reporting failures are logged and cannot invalidate completed scientific results.

The human report helps researchers write a computational-resources paragraph,
but describes only records known in this output directory. Its totals may omit
unrecorded or deleted history and are not a whole-project compute estimate.
Researchers must disclose external experiments and other machines themselves.
See [measurement definitions and an example](CONFIGURATION.md#automatic-compute-reporting).

## Analysis and cache invalidation

Metrics operate on recorded observations and saved instances. A metric requiring
every action or reward rejects an incomplete trajectory; unrecorded data cannot
be reconstructed from a figure or cache. Aggregation combines compatible
repetitions with aligned coordinates and an explicit uncertainty convention. One
repetition has no defined sample standard deviation or standard error.
Supplied regret metrics subtract and accumulate in float64, independently of the
saved reward dtype, so boolean and integer inputs retain signed regret gaps.

Per-run metric caches retain full resolution in losslessly compressed `.npz`
files. Existing uncompressed cache generations stay valid and are reused without
rewriting them. After full aggregation and coordinate validation, the
pipeline selects at most `analysis.points` positions per summary (default 100). Rounded evenly spaced
row positions include both endpoints and preserve original x coordinates.
Short curves remain intact; `null` disables the limit, and integers must be at
least 2. CSV, summary NPZ, and figure consumers share these reduced summaries.
The limit affects aggregate and figure cache identities only, so changing it
reuses metric calculations and raw simulations. Existing full-resolution cache
generations remain retained until analysis cleanup.

The analysis pipeline does not define a universal regret measure. The supplied
`pseudo_regret` and `realized_regret` metrics live in `builtins/metrics.py` because
they assume bandit actions, arm rewards, and a particular comparator. Their short
YAML names are convenience aliases; a study can supply its own metric for a
different scientific definition.

Metric, aggregate, and figure artifacts have separate validated caches. Their
identities include relevant inputs, requested result boundaries, implementation,
configuration, and declared dependencies. A figure edit need not rerun a
simulation or unaffected metric; a metric edit invalidates its result and
downstream work. Analysis remains independent of simulation imports, although
custom metrics and plotters may import their own code.

## Dependency scheduling and retained ancestors

The central rule is **recompute backwards only until the nearest retained valid
ancestor**. The graph contains a simulation input for each selected run, a metric
artifact for each run/metric pair, an aggregate for each metric/group, and a figure
for each configured output. Aggregator `group_by` and repetition reduction define
the units; there are no paper-specific scheduling assumptions. Figure dependencies
cover every group of the selected metric, matching the existing plotter contract.

Workers share these task classes. Ready raw-consuming metrics have priority,
with bounded aging for aggregates and figures so a metric backlog cannot starve
ready outputs. Simulations favor analysis units with completed
or active members; a bounded derivation streak ensures forward simulation progress.
Scientific IDs and RNG streams never include admission order or progress observations.
Aggregation reads metrics in canonical run order, so worker completion order cannot
change floating-point reduction order. Derived publication synchronizes files before
an atomic generation rename. Only that durable publication releases dependents.

Each cache identity includes the exact upstream revisions and relevant science,
code, options and declared dependencies. Expected downstream identities can be
calculated without physically reading an ancestor. A retained valid aggregate
therefore does not require its metrics or raw observations to remain present.
A figure change traverses only as far as its retained summaries; an aggregator
change traverses to metrics; a metric change traverses to its raw trajectory.
Absence of proof requires the ancestor. Corrupt retained artifacts are reported,
not silently accepted as reuse. Combined summary tables are exported on successful
pipeline completion; individual aggregate generations and ready figure exports
become durable earlier.

`recording.retention: keep` preserves raw observations. With `until_analyzed`,
all currently configured per-run metrics must first have valid durable artifacts.
EWS then publishes and syncs a checksummed pruning receipt preserving completed
input revisions before removing result chunks. It atomically clears all live
checkpoint references from progress before deleting checkpoint generations. A
crash may leave unreferenced directories, which repeated pruning or explicit
`ews clean OUTPUT --scope settled --yes` removes. Pruned trajectories retain zero
resumable checkpoints because their raw prefixes are gone. Active, paused, and
completed unpruned runs keep their configured checkpoint fallback. Historical
checkpoint counts and timing remain diagnostic. An interrupted deletion is still
an intentional state; unmarked missing or corrupt chunks remain damage. Instances,
provenance, and compute history are retained. Exact budget references
survive rematerialization so earlier retained derivations can remain reusable
while a later budget is being replayed. A prefix-only request
never deletes the unanalysed tail of a longer completed run.

Storage evolution must not add migration work to the normal execution hot path.
Old durable artifacts are read in place when supported; disposable artifacts
are recomputed from the nearest retained valid ancestor; provably obsolete
artifacts are pruned. Format migration is explicit and reserved for genuinely
incompatible future changes.
Recovery v1 selects no checkpoint payloads for a run with a valid current
pruning receipt, including old outputs with stale progress references. It skips
their payload traversal and leaves the source output unchanged.

With `recording.metric_retention: until_aggregated`, each validated aggregate
authorizes deletion of exactly the per-run metric generations it consumed.
Deletion first renames each generation out of the visible cache and then removes
it; recovery excludes interrupted hidden deletions. A later run completes any
leftover deletion. The aggregate identity remains a durable dependency proof, so
figures and exact requests do not need the removed metrics. Changing aggregation
walks backward to raw observations or, if those too were pruned, to simulation.
An aggregate contains the already subsampled per-group CSV and NPZ. Optional
custom figure `partition_by` creates a dependent task for each label partition,
so plots can publish as soon as their selected aggregate groups are complete.
The combined CSV and any global figure still require all their groups.

New metric work may need an intentionally pruned ancestor. Only `run` can
rematerialize it with fresh component state and the original streams. Endpoint
snapshots whose recorded prefix was removed do not authorize continuation; budget
extension falls back to a fresh simulation. Metrics are recomputed from full
requested trajectories rather than assumed incrementally composable. Standalone
`analyze` and `plot` report the missing dependency and direct the user to `ews run`.

## Recovery snapshots

The EWS-owned [cloud recovery contract](CLOUD.md#versioned-recovery-snapshots)
seals a snapshot under output ownership, inventories committed regular files,
and publishes a versioned checksum manifest. Cloud wrappers transfer those bytes
and publish the manifest last. Restoration verifies the complete inventory into
a new output tree; normal EWS compatibility and checkpoint fallback remain
responsible for reuse. Pruning receipts are preserved and excluded temporary or
uncommitted objects cannot be mistaken for durable ancestors. The semantic
[artifact catalog](ARTIFACTS.md) continues to describe discovery locations, not
recovery safety. Live-writer snapshots and object-store execution are unsupported.

## Operational boundaries

The CLI exposes `count-runs`, `run`, `analyze`, `plot`, `inspect`, and `clean`.
`run` coordinates the dependency graph; failed or paused work blocks only its
dependent tasks. Independent ready work can complete and remain reusable.
`count-runs` optionally reports the study name and number of planned runs; it is
never a prerequisite for execution. Internal `plan_runs` remains the shared run
expansion and validation operation. `analyze` and `plot` also work independently
on saved numerical results, allowing expensive simulations to be reused while
metrics and figures change. The Python `run_experiment` API schedules the same
configured graph. Analysis science remains separate from scheduling and reads
only persisted inputs.

Discord progress delivery runs in the coordinator, outside workers and protocol
steps. Its settings do not enter scientific identities or RNG streams, and
network errors do not fail simulations. Delivery is best effort; saved artifacts
remain authoritative.

`ews clean` previews deletion and requires `--yes` to apply it. Removing
checkpoints leaves numerical observations but loses continuation state. Execution, standalone analysis/plotting, cleanup, and recovery snapshots share
the output lock. The
[configuration guide](CONFIGURATION.md#run-inspect-and-clean) lists cleanup
scopes. Compute records survive selective run cleanup, including records of
removed variants; the `all` scope removes them together with other artifacts.

Artifact schema 2 supports retained variants and budget continuation. Supported
analysis of legacy schema 1 artifacts remains available, but execution requires
a new schema 2 output directory; no in-place conversion occurs. Distributed
scheduling and arbitrary mid-function recovery are outside this local design.
EWS does not provide a universal serializer for native framework objects;
study-owned checkpoint backends implement those representations explicitly.

## Portable CPU workers

`execution/portability.py` owns the explicit `portable-numpy-v1` environment
signature. `collect_provenance` chooses it only for `execution.continuation:
portable_numpy`; the policy's own source digest is part of the implementation
signature. Unknown implementation changes conservatively select new variants; historical
exact-digest bridges do not exempt this scheduler redesign. Portable mode remains distinguished
by its environment and additional policy digest. All ordinary scientific,
source/input, recording, budget, checkpoint and variant validation still applies.
See [PORTABILITY.md](PORTABILITY.md) for exact fields and unsupported backends.
External cloud tooling restores a complete verified output tree and recreates
its environment before EWS selects variants. Cloud attempts, provider locks,
and object storage publication remain outside EWS.
