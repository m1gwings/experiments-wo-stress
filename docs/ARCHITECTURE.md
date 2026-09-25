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
| `execution/` | Request coordination, worker lifecycle, provenance, logging, and notifications. |
| `storage/` | Immutable artifact models, atomic files, run checkpoints, experiment state, and cleanup. |
| `analysis/` | General metric contracts and field metrics, repeated-run aggregation, validated caches, and figure export. |

`ExecutionCoordinator` plans one invocation, publishes its prepared request, and
schedules bounded work while owning signals and notifications. Each worker creates
a `RunSession`, which owns its components, RNGs, recorder, and checkpoint lifecycle.
The process-pool entry point remains a module-level function.

`ExperimentStore` owns the output root, lock, retained variants, and request
publication. Execution computes scientific identities before passing them to the
store. `RunStore` owns one variant's durable boundaries; `Recorder` buffers its
observations. `AnalysisCache` handles immutable cache generations and exports,
while metric computation and aggregation remain in the analysis pipeline.

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
independent of worker count, scheduling, grid traversal order, recording
frequency, and requested budget. Data and instance identities exclude algorithm
choice so competing algorithms can share a problem instance. Action-dependent
environments can still produce different trajectories.

Metadata records the resolved configuration, seeds, component and library code,
tracked inputs, and numerical environment. Source fingerprints cover complete
component modules and base-class modules, plus declared dependency files. This
conservative boundary prevents code changes from silently reusing old simulation
results. It also means an edit to a metric in the same module as an algorithm
can invalidate the algorithm's run; separate modules make independent reuse
possible. A long-running Python process retains imported modules, so restart or
reload after editing already imported study code.

Library fingerprints cover the implementation files in the new packages, rather
than their compatibility facades. The architecture refactor therefore selects new
simulation and analysis variants conservatively. Existing artifact schemas and
readers are unchanged; retained results remain inspectable and analyzable with
their matching saved configuration.

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
result chunks flush at a per-run byte target (16 MiB by default), before
checkpoints, and at completion. A chunk flush alone is not a resumable
checkpoint. Buffering is per active run, in addition to component state and
serialization memory.

```text
output/
  metadata.json                  active request and artifact schema
  config.resolved.yml             resolved configuration
  requests/                      retained requests
  instances/<content-id>/         immutable scientific inputs
  runs/<variant-id>/
    metadata.json                 identity and selected variant
    results/                      numerical chunks
    checkpoints/                  component and RNG snapshots
    progress.json                 committed boundaries and budgets
    failure.json                  latest failure, when present
    run.log                       rotating diagnostics
  analysis/
    cache/                        metric, aggregate, and figure caches
    figures/                      current exports
```

The public storage readers validate and expose selected results; exact filenames
within a generation are internal. `iter_completed_runs` yields run
specifications and `RunResult` objects, each with recorded fields, the saved
instance, a completed step count, revision, and final outputs. For single-step
offline or trial runs, `final_outputs` exposes the last recorded values.

## Checkpoints, completion, and interruption

A checkpoint is committed only after a complete protocol step. The executor
flushes observations, captures component and RNG state, writes a new generation,
and atomically publishes its hashes and result boundary after synchronization.
It keeps preceding valid generations so recovery can fall back if the newest
checkpoint is damaged. A directory or partial file is never treated as completed
work.

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

## Analysis and cache invalidation

Metrics operate on recorded observations and saved instances. A metric requiring
every action or reward rejects an incomplete trajectory; unrecorded data cannot
be reconstructed from a figure or cache. Aggregation combines compatible
repetitions with aligned coordinates and an explicit uncertainty convention. One
repetition has no defined sample standard deviation or standard error.

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

## Operational boundaries

Discord progress delivery runs in the coordinator, outside workers and protocol
steps. Its settings do not enter scientific identities or RNG streams, and
network errors do not fail simulations. Delivery is best effort; saved artifacts
remain authoritative.

`ews clean` previews deletion and requires `--yes` to apply it. Removing
checkpoints leaves numerical observations but loses continuation state. Analysis
and plotting do not hold the execution lock, so cleanup should run while those
operations are idle. The
[configuration guide](CONFIGURATION.md#build-inspect-and-clean) lists cleanup
scopes.

Artifact schema 2 supports retained variants and budget continuation. Supported
analysis of legacy schema 1 artifacts remains available, but execution requires
a new schema 2 output directory; no in-place conversion occurs. Distributed
scheduling, arbitrary mid-function recovery, and generic object serialization
are outside this local design.
