# Architecture

This document specifies the initial implementation of Experiments W/O Stress.
It incorporates the design discussion: YAML configuration, extensible components,
an environment state owned by the data generator, independent random streams,
periodic run checkpoints, and analysis independent of execution.

## Scope and ownership

The library supplies planning, execution, persistence, analysis, and plotting.
A paper supplies its algorithms, data generators, metrics, and any custom protocol.
The initial target is a laptop or modest multicore machine. There is no service,
database, distributed scheduler, or dependency on a particular paper package.

Python package: `experiments_wo_stress`; distribution: `experiments-wo-stress`;
command: `ews`. NumPy and PyYAML are core dependencies; Matplotlib is optional.
Python 3.10 and newer are supported.

## Data flow

```text
YAML → validated ExperimentConfig → RunPlanner → immutable RunSpec objects
     → Executor → protocol + algorithm + data generator + Recorder
     → saved numerical results → Metric → Aggregator → Plotter
```

The diagram names architectural responsibilities. The public execution entry point
is `run_experiment`; a run's live objects are coordinated inside its worker rather
than through a required public `Run` base class.

The executor owns infrastructure; the interaction protocol owns scientific ordering.
A run is one parameter combination, algorithm, and independent repetition. Each run
gets fresh component instances. Immutable source datasets may be shared.

## Component contracts

Components are ordinary Python classes resolved from `module:Class` paths or
built-in aliases. Constructors receive `rng=` and YAML `params` as keyword
arguments. They must accept the RNG even when deterministic. Runtime checks fail
early for incompatible interfaces; Python typing protocols document capabilities.
YAML is loaded as data, and import paths explicitly identify trusted project code.

### DataGenerator and Algorithm

`DataGenerator.generate(request)` produces data and owns evolving environment state.
An offline implementation can return a dataset. The online protocol expects a
`Feedback(value, measurements)` object: `value` is observable by the algorithm;
`measurements` contains evaluator-only numerical observations.

The built-in online algorithm contract is `act(context=None)` and
`observe(action, feedback)`. Offline algorithms implement `fit(dataset)` returning
a numerical mapping (or an array represented as `output`). An algorithm does not
hold or select its interaction protocol. Custom protocols can define other
capability contracts without enlarging every algorithm's interface.

Every stateful component implements `state_dict()` and `load_state_dict(state)`.
State contains string-keyed mappings, lists, tuples, scalar values, and numerical
NumPy arrays. The storage layer encodes these without pickling Python objects.
Stateless components return an empty mapping. Constructors may initialize state;
restoration replaces that state and restores RNGs afterward.

### InteractionProtocol

Constructors receive `rng=` and protocol parameters. The lifecycle is:

- `initialize(algorithm, data)` validates/prepares a fresh interaction.
- `advance(algorithm, data)` completes one atomic scientific step and returns a
  mapping of numerical observations.
- `step` is the number of completed steps, initially zero.
- `is_finished()` reports completion.
- `state_dict()` / `load_state_dict()` persist progress and other protocol state.

An online step optionally calls `data.context()`, requests an action using that
context (or `None`), calls `data.generate(action)`, gives only the observable
feedback to the algorithm, and returns evaluation measurements. An
offline step generates a dataset and fits the algorithm. Custom protocols may
support contextual interaction or incremental training. A callable trial helper
supports a complete custom computation as one step.

A checkpoint is taken only between complete steps. A long indivisible `fit()` or
trial cannot be checkpointed internally; incremental algorithms must expose safe
steps through a custom protocol.

## Configuration and planning

An experiment contains `name`, `seed`, `runs`, `execution`, `recording`, and
`analysis`. Each run group supplies `name`, `planner: grid`, `repetitions`,
`protocol`, `data`, `algorithms`, and an optional dotted-path `grid`.
Each component has `type`, `params`, and an optional seed override. Each algorithm
also has a human-readable `name`. Grid axes expand as a Cartesian product; each
point is crossed with algorithms and repetitions. Dotted paths address
`data.params.*`, `protocol.params.*`, or `algorithm.params.*`.

Built-in defaults: one repetition, one local worker, a 120-second checkpoint
interval, optional step checkpoints, a 16 MiB result buffer, every-step recording,
two retained checkpoints, and uncompressed numerical storage.

`ComponentSpec(type, params, seed)` and
`RunSpec(run_id, group, repetition, algorithm_name, algorithm, data, protocol, seed)`
are frozen dataclasses with `to_dict()` / `from_dict()` serialization. Run labels
include `group`, `repetition`, `algorithm.name`, and flattened component parameters.
`ExperimentConfig` exposes `name`, `seed`, `runs`, `execution`, `recording`,
`analysis`, `source_dir`, `to_dict()`, and `simulation_dict()`.
`load_config(path)` validates YAML; `plan_runs(config)` returns stable RunSpecs.

`RunPlanner.plan(group, seed)` is the custom planning contract, with a zero-argument
constructor. `GridPlanner` supplies the default Cartesian expansion. A group's
`planner` can name a custom `module:Class`; custom planners use `make_run_spec()`
to assign canonical IDs, preserve the group and experiment seed, and deterministically
return individual RunSpecs. Planning validates identities and rejects duplicates.

The directory containing the YAML file is added to the import search path so a
paper can keep custom components beside its configuration. Component paths and
constructor parameters are checked before the sweep starts, along with capabilities
required by built-in protocols. Custom protocols validate their own scientific
compatibility. There is no dynamic
class guessing from parameter names.

Run identity uses canonical JSON and SHA-256, never Python's process-dependent
`hash()`. Execution settings, output locations, and analysis do not change run
identities. Recording resolution is part of the persisted simulation contract;
changing it requires a new output directory. Analysis and worker/checkpoint
settings can change when resuming. Scientific configuration mismatches fail
explicitly instead of silently mixing artifacts.

Provenance includes library modules, component/planner/trial source files, Python
files beside the configuration, software versions, and Git revision/dirty status
when available. The contents of built-in CSV inputs are fingerprinted. These checks
are deliberately local: imports do not create a recursive source snapshot, and
custom generators must manage the identity of additional inputs. Pin transitive
dependencies and preserve custom input files in the paper project. Execution
rejects changed tracked code or environment; saved-data analysis does not require
those simulation modules to be available.

## Randomness

`make_rngs(spec)` returns named `algorithm`, `data`, and `protocol` generators.
Stable component identities plus a root seed feed NumPy `SeedSequence` and an
explicit `PCG64` bit generator. A component seed overrides the experiment root
for that component. The data identity excludes algorithm choice and algorithm
parameters, allowing competing algorithms to start with the same generated
instance. Algorithm and protocol streams are separated and stable per run.
Repetition and run-group identities distinguish repetitions and groups.

Changing worker count, scheduling, grid order, or recording/checkpoint frequency
does not change random streams. Matching data seeds alone does not guarantee equal
action-dependent trajectories; custom generators define any stronger coupling.
Components use injected generators rather than global NumPy random functions.
Exact replay assumes unchanged component code, inputs, and numerical environment.

## Recording and storage

Metadata is JSON; the effective configuration is also saved as YAML. Results are
immutable `.npz` chunks of named arrays. Each record contains `step` and explicitly
selected numerical fields. The first record establishes a fixed dtype/shape
schema for subsequent rows. Pickled/object arrays are not accepted.

`Recorder` samples observations at fixed configured step intervals (always including the
final step), buffers them, and writes when its approximate array-byte budget is
reached. It also flushes before checkpointing and completion. The budget is per
active run; Python bookkeeping, temporary buffer growth, and serialization add
overhead. Storage reads chunks into preallocated arrays when loading a complete
run for a metric. Previous result history is never rewritten at each flush.

Sparse recording loses information. A cumulative quantity needed at sparse steps
must be accumulated on every scientific step by the protocol/generator/algorithm,
with its accumulator included in checkpoint state. Postprocessing cannot recover
observations that were not saved. The example demonstrates this explicitly.

```text
output/
  config.resolved.yml
  metadata.json
  runs/<run-id>/
    metadata.json
    results/000000.npz
    checkpoints/<generation>/state.json
    checkpoints/<generation>/arrays.npz
    progress.json
    failure.json                 # most recent failure, when present
  analysis/                      # rebuildable tables and figures
```

Numerical chunks are uncompressed by default. Compression is optional; its value
depends on CPU cost and data compressibility. Reusable input files are referenced
instead of copied into every checkpoint. Large changing state must still be saved
if the component requires it to continue.

## Checkpoint transaction and recovery

At a safe boundary, the executor flushes observations and captures algorithm,
data-generator, and protocol state, all RNG states, and the result boundary/schema.
It writes a new checkpoint generation to temporary files, flushes and syncs them,
then publishes a commit record through an atomic same-filesystem replacement.
The previous valid generation is retained until the new one is committed.
Metadata includes schema versions, content checksums, and configuration/code
identities. This protects against process interruption and detects corruption;
hardware durability still depends on local filesystem guarantees.

On restart, only committed generations are considered. A valid previous generation
can be used if the latest checkpoint is corrupt. Results beyond the restored
boundary are removed before replay, avoiding duplicates. Results before that
boundary are validated. With no valid checkpoint, an unfinished run restarts.
Completed results are validated before a run is skipped; damaged completed output
is reported rather than silently accepted. Missing/partial temporary files never
indicate success. Completion is a separate commit after final results are flushed.

Failures save traceback information. A later run command retries failed runs from
their latest valid checkpoint. Graceful interruption checkpoints at the next safe
boundary; forced process termination resumes from the preceding commit. Time
intervals are measured with a monotonic clock and cannot preempt a scientific step.
Checkpoint count, duration, and bytes are recorded for inspection.

An operating-system advisory lock prevents concurrent executors from modifying
one output directory. It releases automatically when its process exits, including
after abrupt termination; the persistent `.lock` file does not indicate a live
lock and does not need manual deletion. Each
worker writes its own run directory; workers never append to one shared result
file. Submission is bounded to avoid queueing every live component in memory.

## Analysis and figures

`Metric.compute(results)` consumes a mapping of saved numerical arrays and returns
`MetricResult(x, values)`, a scalar or one-dimensional curve. Built-ins include a
saved field and cumulative sum. Custom metrics are importable classes with `params`.
Each metric receives one complete run in memory. Storage fills preallocated columns
from chunks to avoid retaining both chunk lists and concatenated copies. Default
aggregation uses running means and squared deviations, holding grouped summaries
rather than all repetitions. Memory therefore scales with the largest loaded run
and the number/size of group summaries, plus metric-specific workspaces.

The built-in `repeated_runs` aggregator groups metric curves by configured run labels and computes a mean
and either sample standard deviation, standard error, or no uncertainty. Independent
runs are the statistical units. Curves must use identical x coordinates in a group;
misaligned data is rejected. One repetition has undefined sample uncertainty,
represented explicitly rather than as a confidence claim.
Distinct scientific configurations and duplicate repetition identifiers cannot be
pooled accidentally. Only validated completed runs enter analysis; summary tables
report the actual count, which can be lower than planned in an unfinished study.

`analyze(config, output_dir)` validates and reads completed artifacts without
instantiating simulation components, writes summary tables, and returns summaries.
`plot(config, output_dir)` renders configured figures from those summaries.
Default line plots support algorithm colors, parameter panels, uncertainty bands,
and linear/log axes. Scalar metrics use markers and error bars. PDF and JPG use
optional Matplotlib; editable PGFPlots/TikZ
source is generated directly for these supported primitives, without requiring
LaTeX at export time. Custom plotters receive summaries and a figure configuration.
Changing analysis or figures does not rerun simulations.

Tables are saved as CSV and NPZ under `analysis/`, with group metadata in JSON.
Figures live under `analysis/figures/`. Custom metrics implement `compute(results)`;
custom plotters implement `plot(summaries, figure, output_dir)`. Both constructors
receive configured parameters without an RNG. See [CONFIGURATION.md](CONFIGURATION.md)
for supported options and extension signatures.

## CLI and Python API

The CLI is a thin wrapper around `load_config`, `plan_runs`, `run_experiment`,
`inspect_experiment`, `analyze`, and `plot`:

```text
ews plan CONFIG
ews run CONFIG --output DIR [--workers N] [--max-steps N]
ews inspect DIR
ews analyze CONFIG --output DIR
ews plot CONFIG --output DIR
```

`--max-steps` is a deliberate pause after a bounded number of new steps per run,
useful for validating resumption. It checkpoints before returning paused status.
Public return values report completed, skipped, paused, and failed runs.
Failures produce a nonzero CLI exit status. Plotting can operate with custom metric
modules available, but does not need to execute algorithms or data generators.

## Verification and limits

The canonical example compares two algorithms on a synthetic sequential problem,
with three problem sizes and repeated seeds. Tests compare complete numerical
outputs for uninterrupted, interrupted/resumed, and differently parallelized runs;
check seed separation and stable planning; reject configuration/code mismatches;
exercise incomplete/corrupt storage recovery; and regenerate all figure formats.
An offline CSV example verifies the shared generator interface.

Public APIs are initially experimental. Generic object serialization, arbitrary
mid-function checkpoints, distributed execution, and every plotting primitive are
outside this version. The package should be validated on a real paper experiment
before its API is declared stable or a public release is published.
