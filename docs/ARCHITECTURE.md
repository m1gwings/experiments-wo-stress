# Architecture

Experiments W/O Stress separates scientific behavior from the infrastructure of a
numerical study. A paper supplies algorithms, data generators, protocols, and any
custom metrics. The library plans independent runs, persists their inputs and
observations, resumes work, and produces analysis artifacts. The target is a laptop
or modest multicore machine, with local files and proportionate dependencies.

## Data flow and responsibilities

```text
YAML → validated experiment → run planner → run specifications
     → saved instance + protocol + algorithm + data generator
     → recorded observations + checkpoints + run logs
     → metrics → aggregation → figures
```

| Component | Owns |
| --- | --- |
| `RunPlanner` | Parameter expansion, repetitions, and deterministic run specifications. |
| `Instance` | Immutable scientific input, represented by metadata and numerical arrays. |
| `DataGenerator` | Evolving environment state and generation of observable feedback. |
| `Algorithm` | Learned state and scientific decisions. |
| `InteractionProtocol` | Order of interactions, information flow, and step progress. |
| Executor and recorder | Resource use, selected observations, persistence, recovery, logging. |
| `Metric` | Scientific quantities computed from recorded observations and the saved instance. |
| Aggregator and plotter | Comparisons across independent repetitions and figure presentation. |

Classes are selected explicitly with built-in aliases or `module:Class` paths.
They need not inherit framework base classes. The configuration directory supplies
an import context for paper-owned code. Imported classes are trusted project code;
YAML itself is loaded as data.

## Instances and runtime components

A generator can implement `create_instance(*, rng, **params) -> Instance`. The
executor creates the instance with a dedicated random stream, persists its metadata
and arrays, and supplies it to the generator's `instance=` constructor argument.
The instance remains available to analysis without importing the generator.
Identical instance content is stored once and referenced by runs.

Examples include bandit means, a nonstationary schedule, a graph, an optimization
problem matrix, or a dataset description. This container is generic: the scientific
meaning belongs to the generator and metric. CSV input identity is tracked; large
external inputs need not be copied into every checkpoint.

A fixed nonstationary schedule can support continuation only within its saved
coverage. Editing or appending that schedule selects a new instance and run; prefix
migration is not implemented. For an environment whose hidden state evolves
stochastically during execution, save initial conditions or its law in the instance
and record hidden quantities needed by metrics as evaluator measurements. These
measurements need not be exposed to the learner, and the instance remains immutable.

Runtime constructors receive `rng=` and configured parameters. Each run has fresh
component instances. The online algorithm contract is `act(context=None)` and
`observe(action, feedback)`. The generator optionally provides `context()` and
responds to `generate(action)` with `Feedback(value, measurements)`. The learner
receives only `value`; the recorder receives selected numerical measurements.

Offline algorithms expose `fit(dataset)` returning numerical output or named
measurements. The offline protocol obtains the dataset through the same generator
interface. A callable-trial protocol supports a complete custom computation as one
step. Optional settings cover common bandit problems and Gymnasium environments;
scientific algorithms remain outside the library.

The protocol lifecycle is `initialize(algorithm, data)`,
`advance(algorithm, data)`, `is_finished()`, and a completed-step counter `step`.
Each advance finishes exactly one logical step. The protocol, algorithm, and data
generator expose `state_dict()` and `load_state_dict(state)` for changing state.
State supports numerical arrays, scalars, lists, tuples, and string-keyed mappings;
pickled objects and live resources are outside the contract.

## Reproducibility

Independent streams serve instance generation, the algorithm, the data generator,
and the protocol. Stable scientific identities, component seed overrides, and the
experiment seed feed NumPy `SeedSequence` and explicit PCG64 generators. Stream
identities do not depend on execution order, workers, recording settings, or the
requested execution budget.

Data and instance identities exclude algorithm choice, allowing competing
algorithms to use the same scientific instance. Equal initial streams do not imply
equal trajectories in an environment affected by actions. Stronger coupling is a
scientific choice in the generator.

Metadata records resolved configuration, stream identities, component/library
source fingerprints, tracked inputs, numerical environment, and available code
revision information. Source fingerprints cover complete component module files
and their base-class modules, together with explicitly declared dependency files.
These prevent different scientific implementations from being silently treated
as the same work. Unlisted external dependencies still require management by the
paper project. Exact replay assumes compatible code, inputs, and numerical software.

Module-level tracking is conservative. Editing a custom metric inside the same
module as a generator also changes that generator's source fingerprint. Put
simulation and analysis extensions in separate modules when independent reuse
matters. Python retains imported modules in a live process; start a fresh process
or explicitly reload modules after editing already imported scientific source.

## Budget and reusable work

`budget.steps` is the requested amount of execution. It is separate from scientific
parameters, such as a learning rule that intrinsically uses a fixed horizon.
Increasing the budget can continue an existing run when the protocol, algorithm,
and generator all declare `supports_extension = True`. The executor restores the
final state and RNGs, sets the larger protocol budget, and performs only new steps.
A component that cannot preserve this meaning must leave extension disabled.

A nonextendable budget change selects another run variant. Changes to scientific
parameters, code, or recorded fields also select distinct variants as appropriate.
Previously generated artifacts are retained; unaffected runs can still be reused.
The active experiment request identifies exactly which variants and budgets its
analysis should use, while request history remains inspectable.

Completed budget boundaries retain their result manifests and endpoint snapshots.
This allows a previously requested budget to remain meaningful after an extension.
For extendable runs, sparse recording uses scheduled step multiples; it does not
insert an extra endpoint just because a particular request stopped there. This
makes recording at a larger budget identical whether execution was direct or
extended. A sparse run may therefore have no record exactly at its stopping step.

## Recording and artifacts

Recording is an explicit selection of numerical fields and logical steps. It is
separate from scientific identity and from operational buffering settings. A
recording variant includes field selection and interval. Buffer size, compression,
worker count, and checkpoint frequency do not change random streams.

The recorder establishes a fixed numerical dtype and shape for each selected
field and always includes `step`. It writes immutable NumPy chunks when its byte
budget is reached, before a checkpoint, and at completion. The default buffer is
16 MiB per active run; component state, Python objects, and serialization require
additional memory. Compression is optional and disabled by default.

```text
output/
  metadata.json                  active request and artifact schema
  config.resolved.yml             current resolved configuration
  requests/                      retained request descriptions
  instances/<content-id>/         instance metadata and numerical arrays
  runs/<variant-id>/
    metadata.json                 scientific identity, instance, recording variant
    results/                      immutable numerical chunks
    checkpoints/                  evolving state and RNG snapshots
    progress.json                 committed boundaries and completed budgets
    failure.json                  most recent failure, when present
    run.log                       rotating diagnostics
  analysis/
    cache/                        validated metric, aggregate, and figure caches
    figures/                      current exported figures
```

Exact filenames within an artifact generation are an implementation detail; the
public readers validate and expose them. Use `iter_completed_runs` to read selected
results. Each item contains a run specification and a `RunResult`, a mapping of
recorded fields with its saved instance, completed step count, revision, and final
outputs available as attributes.

For a single-step offline or trial result, `final_outputs` exposes the last recorded
values. Multiple-step results use their recorded arrays; there is no separate
arbitrary final-output callback in this version.

## Checkpoints, completion, and interruption

At a safe boundary the executor flushes recorded observations, captures all
component and RNG state, and writes a new checkpoint generation. Files are flushed
and synchronized before an atomic commit publishes their hashes and result
boundary. The previous valid generation remains available. Filesystem hardware
and durability guarantees still apply.

Recovery considers committed generations only. It validates state and recorded
chunks, can fall back from a damaged latest checkpoint to a valid predecessor,
and discards uncommitted observations before replay. Completion is validated,
never inferred from a directory or a partial file. Final state is retained to
support continuation; completion snapshots are additional to rolling checkpoints.

Checkpoint triggers use elapsed monotonic time or complete steps. A graceful
interrupt requests a checkpoint at the next safe boundary. A forced termination
resumes from the previous committed boundary. An indivisible offline `fit()` or
trial can only checkpoint before or after that operation; incremental algorithms
need a protocol that exposes smaller safe steps.

Execution and cleanup use a local experiment lock. Each worker writes its own run
directory, and task submission is bounded. Failures
retain explanations and traceback information. Run loggers use bounded rotating
files so debugging output does not grow without limit.

## Analysis and cache invalidation

Metrics compute from a `RunResult`: recorded observations plus the saved instance.
The generator does not accumulate regret or other analysis metrics. For example,
pseudo-regret combines every recorded action with stationary or nonstationary arm
means from the instance. Metrics that require a complete trajectory reject sparse
or missing observations. Saving fewer observations necessarily limits future
analysis; no cache can recreate data that was never recorded.

A metric returns a scalar or one-dimensional `MetricResult(x, values)`. Aggregation
groups scientifically compatible runs and reduces independent repetitions using
a mean and explicit uncertainty convention. Coordinates must agree within a group.
Sample standard deviation and standard error are undefined for one repetition and
are represented as such, not converted into an unsupported confidence claim.

Metric, aggregate, and figure artifacts have separate content-addressed caches.
Keys account for relevant saved inputs, requested result boundaries, implementation
and configuration. Checksums validate cached files. Changing a plot only invalidates
the affected plotting work; changing a metric invalidates its metric and downstream
artifacts. Custom metrics can declare additional dependency files.

Analysis does not construct algorithms or generators. PDF and JPG export uses
optional Matplotlib. TikZ/PGFPlots export produces editable source without requiring
LaTeX. Custom metrics and plotters use their own explicitly imported code.

## Optional operational notifications

Discord delivery belongs to the coordinator, outside the scientific step loop and
worker processes. An optional daemon thread sends start, periodic, and final
run-stage summaries. It takes small counter snapshots under a lock and reads only
bounded progress metadata for a few active runs. No trajectory validation or data
upload is needed to report checkpoint coverage.

A webhook URL is resolved from an environment variable only when execution is
requested. Settings are operational and do not enter scientific identities or RNG
streams. HTTP errors, rate limits, and delayed delivery cannot fail a simulation;
shutdown waits only for a bounded interval. Notifications are best effort and are
not a durable monitoring service. The configuration guide defines their scope.

## Operations and limits

`ews build` performs execution and requested analysis in one command; `run`,
`analyze`, and `plot` remain available independently. `inspect` explains stored work.
`clean` previews removal by scope and deletes only with `--yes`; removing checkpoints
preserves numerical results but removes the state needed to continue those runs.
Analysis and plotting publish cache files atomically but do not acquire the
execution lock; run cleanup when those operations are idle.
See [the configuration guide](CONFIGURATION.md) for commands and settings.

Artifact schema 2 implements instances, retained variants, and budget continuation.
Legacy schema 1 artifacts remain readable for supported analysis; execution uses
a new output directory. There is no silent in-place conversion or deletion.

A container on one cloud VM can use the same executor and a durable local/block
filesystem. The current artifact contract needs atomic replacement and process
locking; object storage is suitable for backups, not a direct output-directory
replacement. Provenance includes platform and resolved input paths, so a container
alone does not guarantee resume compatibility across hosts. See [CLOUD.md](CLOUD.md).

The API is experimental. Distributed scheduling, arbitrary mid-function recovery,
generic object serialization, and arbitrary plotting primitives are outside scope.
A real paper experiment should validate the authoring contract before a stable
release is declared.
