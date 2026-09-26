# Configuration and extension guide

Start with the [complete sequential configuration](../examples/sequential_study/experiment.yml).
YAML describes a study; Python classes supply scientific behavior. Load a file with
`load_config(path)` or an `ews` command. Classes are selected explicitly through
built-in aliases or `module:Class` paths. Keep study-specific classes in an
`experiment_code/` package beside the YAML file. The configuration directory is
added to Python's import context.

## Experiment and run groups

```yaml
name: gaussian_bandit_comparison
seed: 2026
runs:
  - name: main
    planner: grid
    repetitions: 20
    budget: {steps: 1000}
    protocol: {type: online}
    data:
      type: experiment_code.data:GaussianBandit
      params: {n_arms: 10, noise_std: 0.1}
    algorithms:
      - name: ucb
        type: experiment_code.algorithms:UCB
        params: {exploration: 0.1}
      - name: epsilon_greedy
        type: experiment_code.algorithms:EpsilonGreedy
        params: {epsilon: 0.1}
    grid:
      data.params.n_arms: [10, 20, 50]
```

Place this file beside the example's `experiment_code/` package. Its grid
expands to 3 sizes × 2 algorithms × 20 repetitions. A group has these settings:

| Setting | Meaning |
| --- | --- |
| `name` | Unique label; defaults to `group_0`, `group_1`, etc. |
| `planner` | `grid` by default, or a custom planner import path. |
| `repetitions` | Positive independent repetition count; defaults to `1`. |
| `budget.steps` | Positive requested number of protocol steps. |
| `protocol`, `data` | Component descriptions. |
| `algorithms` | Nonempty list of named component descriptions. |
| `grid` | Parameter paths mapped to candidate lists; defaults to `{}`. |

At the experiment level, `name` is required and `seed` defaults to `0`.
Components have `type`, `params` (default `{}`), an optional `seed` override, and
optional `dependencies`: paths to scientific input files, relative to the YAML
file. Algorithm names default to the final part of their type path and must be
unique within a group. Parameters use plain finite YAML values, lists, and
string-keyed mappings. Duplicate keys and unknown framework settings are rejected.

Declare scientific input files in component `dependencies` (relative to the YAML)
or class `dependency_files` (relative to the module). These files participate in
reuse decisions. Keep metrics and simulation code in separate modules when their
changes should be independent; [the architecture](ARCHITECTURE.md#reproducibility)
explains source fingerprints.

Grid paths address `algorithm.params.*`, `data.params.*`, or `protocol.params.*`.
An algorithm axis applies to every algorithm in its group; separate groups can
sweep different parameters. Grid axes are a Cartesian product. Duplicate candidates
and overlapping paths are rejected. Custom planners must produce deterministic
run specifications.

Execution budget is separate from a scientific horizon parameter. An algorithm
that uses a fixed horizon in its learning rule must retain that as a scientific
parameter.
For older online configurations, `protocol.params.horizon` is normalized into the
execution budget; conflicting settings are rejected.

## Instances and scientific components

A generated instance is represented by `Instance(metadata, arrays, kind=...,
schema_version=1)`. Import it from `experiments_wo_stress.artifacts`. Metadata must
be serializable, and arrays must be numerical. An instance describes immutable
scientific input; component checkpoints describe evolving state.

A generator can expose:

```python
@classmethod
def create_instance(cls, *, rng, **params):
    return Instance(
        kind="my_problem",
        metadata={"description": "scientific input"},
        arrays={"matrix": rng.normal(size=(10, 10))},
    )
```

The runner persists that instance and injects it into the generator's `instance=`
constructor parameter. Instance generation uses its own stream, separately from
runtime environment randomness. This permits metrics to inspect the actual problem
instance without reconstructing a generator or relying on its implementation.

| Component | Runtime contract |
| --- | --- |
| Generator | `generate(request)`, checkpoint methods, optional `context()` and `create_instance`. |
| Online algorithm | `act(context=None)`, `observe(action, feedback)`, checkpoint methods. |
| Offline algorithm | `fit(dataset)` returning numerical output, checkpoint methods. |
| Protocol | `initialize`, `advance`, `is_finished`, `step`, checkpoint methods; budget support when extendable. |

Constructors receive `rng=` and configured parameters. `rng` is reserved.
Stateful objects implement `state_dict()` and `load_state_dict(state)` with all
changing values needed to continue. The default checkpoint backend supports
numerical NumPy arrays, ordinary scalars, lists, tuples, and string-keyed mappings.
A [custom backend](#checkpoint-backends) can support native framework values
without a mandatory NumPy conversion in the component. EWS includes its injected
RNG states alongside component state. `StateMixin` supplies empty state for
stateless components.

Each protocol advance completes one logical step and increments `step` once.
Initialization runs only for a fresh execution, so restored components must recover
from their saved state without repeating it.

`online` obtains optional context, asks for an action, and calls the generator.
`Feedback(value, measurements)` exposes only `value` to the algorithm. Measurements
are evaluator observations, not an invitation to put scientific metrics in the
generator. Compute regret, cumulative return, and other metrics from saved records
and the saved instance.

`offline` optionally accepts `request`, obtains `data.generate(request)`, and calls
`algorithm.fit(dataset)` as one step. A numerical array becomes `output`; a mapping
provides named fields. Built-in data aliases include `normal`, `csv`, and `'null'`.
CSV paths resolve relative to the YAML file and are tracked by content.

The CSV instance saves its numerical matrix once as `arrays['data']`, together
with source metadata and its content hash. Checkpoints retain only evolving cursor
state. The `normal` instance describes distribution parameters; its samples are
drawn with the data RNG during execution. Record those samples separately if later
metrics need their realizations rather than the algorithm's returned summaries.

### Common settings and reinforcement learning

Reusable settings are available in `experiments_wo_stress.settings`, including
stationary Gaussian bandits, nonstationary bandits, and a Gymnasium adapter. They
supply environments and persisted instances, not a catalog of research algorithms.
The example retains a paper-owned generator to demonstrate customization.

| Data alias | Principal parameters |
| --- | --- |
| `stationary_bandit` | Explicit `means`, or `n_arms` for generated means; `distribution` defaults to `bernoulli`. |
| `gaussian_bandit` | Explicit `means` or `n_arms`, with Gaussian rewards and `noise_std`. |
| `nonstationary_bandit` | Required `schedule` of time-by-arm means, `distribution`, and `noise_std`. |
| `gymnasium` | Required `state_adapter`, optional `factory`, `env_params`, and `reset_options`. |

Stationary means are generated uniformly on `[0, 1]` when omitted, using 10 arms
unless `n_arms` is set. An explicit vector determines its own arm count. Bandit
instances save means as a vector or time-by-arm matrix; generated measurements
contain actions and rewards.

For hidden state that evolves stochastically during a run, keep immutable initial
conditions or model parameters in the instance and record any realized hidden
quantities needed for evaluation as measurements. A custom metric can combine
those fields through `RunResult`. The built-in `pseudo_regret` reads a fixed means
vector or schedule from the instance; it does not infer stochastic hidden means.

Install `.[gym]` for Gymnasium. The adapter uses a configured environment factory
and an explicit state adapter with `snapshot(env)` and `restore(env, state)`.
The snapshot must cover the environment, wrappers, and their RNGs: a generic
Gymnasium environment does not automatically provide complete checkpoint support.
The RL protocol keeps termination and truncation distinct, resets at episode
boundaries, and delivers transition information to the learner.

Select the RL protocol and a paper-owned state adapter:

```yaml
protocol: {type: rl}
data:
  type: gymnasium
  params:
    factory: gymnasium:make
    env_params: {id: CartPole-v1}
    state_adapter: experiment_code.data:CartPoleState
```

`CartPoleState` must have a zero-argument constructor and implement the snapshot
contract above; the library does not supply a universal environment serializer.
The learner's `act(context=observation)` receives the current observation.
Its `observe(action, transition)` receives `observation`,
`next_observation`, `reward`, `terminated`, `truncated`, and `info`. Numeric
observation mappings are flattened into recorded fields; recorded shapes must
remain fixed.

### A callable trial

For a complete numerical computation, use the `trial` protocol:

```yaml
name: scalar_trial
seed: 42
runs:
  - repetitions: 10
    protocol:
      type: trial
      params:
        function: experiment_code.trials:trial
        params: {size: 100}
    data: {type: 'null'}
    algorithms:
      - {name: trial, type: null_algorithm}
```

For this example, implement `trial(*, algorithm, data, rng, size)` returning numerical
output or a mapping. Quote `'null'` in YAML so it remains a component alias. The
whole function is one step; an interruption inside it restarts that step.

## Execution, logs, and recording

```yaml
execution:
  workers: 1
  checkpoint_seconds: 120.0
  checkpoint_steps: null
  keep_checkpoints: 2
  compression: false
  logging_level: INFO
  log_max_bytes: 2097152
  log_backups: 2

recording:
  every_steps: 1
  fields: null
  buffer_bytes: 16777216
```

`--workers` and the Python API's `workers=` override the worker setting. Ordinary
CPU experiments require no GPU configuration: with `gpu_ids` omitted, one worker
executes sequentially in the current process and multiple workers use spawned
processes. Python scripts starting process workers need the usual
`if __name__ == "__main__":` guard. GPU execution also needs this guard with one
worker.

`ews run` automatically monitors execution on stderr: an interactive terminal
gets a Rich dashboard with one row per worker process, while redirected output
gets plain updates at most every 30 seconds. Failures are reported immediately;
every invocation prints a final execution summary. `--quiet` disables continuous
monitoring and its worker messages, keeping final summaries and errors. JSON
stdout, file logging, and Discord delivery retain their separate roles. Rich
respects `NO_COLOR`; terminal width controls name truncation and bar visibility.

Progress uses the protocol's completed `step` and requested `budget.steps`.
Online/RL budgets count rounds/environment transitions; built-in offline fits
and trials are one indivisible operation, with no invented progress inside it.
A custom protocol without a budget may optionally expose a positive integer
`total_steps` in the same units as `step`. Without a known total the display is
indeterminate. No terminal logic belongs in study components.

Per-run ETA uses work completed since initialization/restoration, after at least
two seconds and 1% of new work. Global ETA uses the median of up to 32 observed
completed-run durations, unfinished active fractions, queued work, and available
worker concurrency. It waits at least five seconds and one completed execution;
reused and failed runs do not train the estimate. Estimates are approximate,
especially for heterogeneous runs or resumed prefixes. Dashboard completed
counts include reused results; the final summary lists reuse separately.

The Python `run_experiment` API remains silent by default. Its optional
`progress=` argument accepts an entered `TerminalProgress(name, workers)` context
from `experiments_wo_stress.execution.progress`; the CLI supplies this observer
automatically. Monitoring state is temporary and never changes run identities,
RNG streams, checkpoint contents, or failure/cancellation policy.

Checkpoints use time and step triggers; `null` disables either trigger.
A pause, graceful interruption, or completion also saves state. Compression trades
CPU time for disk space; `execution.compression` controls recorded arrays and the
default checkpoint backend unless that backend has an explicit compression
parameter. See [checkpoint recovery](ARCHITECTURE.md#checkpoints-completion-and-interruption)
for commit boundaries and durability.

Run components receive a standard Python logger as `self.logger`. Use parameterized
messages such as `self.logger.debug("round=%d", self.round)` rather than worker
prints. Logs rotate at the configured byte threshold and retain the specified
backups. These operational settings do not change scientific identities or streams.

`fields: null` saves all measurements; a list selects fields. The recorder adds
`step` automatically and requires consistent numerical dtype and shape for each
field. Field names contain letters, digits, and underscores, beginning with a
letter or underscore. A numerical action is provided by the online protocol unless
explicitly present in measurements.

`every_steps` samples logical steps, independently of machine speed. For extendable
runs, only scheduled multiples are recorded: stopping at step 23 with an interval
of 10 produces records at 10 and 20. This avoids inserting different observations
when the same run is extended later. Full trajectories use interval `1`.

`buffer_bytes` is an approximate per-active-run array limit; component state
needs additional memory. The default is 16 MiB. Results are flushed before a
checkpoint and at completion.

A different recording selection retains a separate variant. It may require new
execution to obtain missing observations. Metrics that need every action or reward
reject sparse trajectories; the library does not infer omitted data.

### GPU execution

For a study whose components use CUDA, add this execution fragment:

```yaml
execution:
  workers: 2
  gpu_ids: [0, 1]
```

`gpu_ids` must be a nonempty list of distinct, nonnegative integers. Booleans,
strings, `null`, negative IDs, and duplicates are rejected at configuration load.
`workers` must equal the number of IDs, including after a CLI or Python API
override. For one GPU, use `workers: 1` and `gpu_ids: [0]`; execution still takes
place in a spawned process.

IDs identify devices in the host or container runtime's GPU namespace, before
CUDA masking. When `gpu_ids` is configured, EWS rejects an already-set
`CUDA_VISIBLE_DEVICES`, including an empty value. Unset it before launch rather
than passing indices relative to an existing mask. EWS validates configuration,
but does not detect available hardware; the study's GPU framework reports an
unavailable device. With `gpu_ids` omitted, EWS leaves the environment unchanged.

Each configured GPU belongs to one worker for that worker's entire lifetime.
EWS establishes its single-device `CUDA_VISIBLE_DEVICES` before the spawned
interpreter imports the main script or study components. Planning, component
validation, and source inspection needed by execution also run in a GPU worker.
Runs assigned later to that worker use the same GPU. The algorithm selects its
worker-local CUDA device, normally `cuda:0`; it needs neither the physical GPU
ID nor scheduling code. The coordinator's environment is restored after launch.
This boundary covers EWS execution imports; caller code imported before
`run_experiment` and a custom planner imported by `ews count-runs` remain
outside it. Embedded applications should avoid concurrent environment changes,
GPU initialization, and unrelated subprocess launches while workers start; see
[the launch boundary](ARCHITECTURE.md#source-organization).

GPU assignment is operational configuration. The resolved configuration retains
`gpu_ids`; request `provenance.execution` records worker PIDs and assignments,
and run logs report the worker's PID and CUDA mask at INFO level. GPU IDs do not
change scientific run IDs, injected RNG streams, or stored variant selection. This
does not promise bitwise equality across hardware or GPU frameworks; study code
must seed and checkpoint its own framework RNGs. Use supported numerical values
with the default backend or supply a backend for native framework state.

EWS imports no GPU framework and installs none. Assignments prevent concurrent
workers in the same invocation from sharing a configured GPU; they do not
reserve GPUs against other processes or experiments. GPU memory packing,
fractional GPUs, multiple workers per GPU, multi-GPU training, and distributed
scheduling are outside this execution mode.

### Automatic compute reporting

On Linux and macOS, each `ews run` automatically records resources and timing
under `OUTPUT/compute/`. No YAML setting or additional dependency is required.
Windows runs silently omit reporting. Missing hardware fields stay unavailable;
reporting failures are logged without turning successful scientific execution
into failure.

`compute/summary.md` is a starting point for a computational-resources or
reproducibility paragraph. `summary.json` holds machine-readable aggregates;
immutable invocation and attempt records preserve the underlying history. The
CLI prints a short resource summary to stderr, keeps its result JSON on stdout,
and includes the report location. `ews inspect OUTPUT` also points to a saved
summary. Separate `ews analyze` and `ews plot` commands do not create run
invocation records. The Python `run_experiment` API records execution only;
subsequent analysis calls are outside that invocation.

Environment records include the OS/kernel, architecture, CPU model, reliable
physical and logical CPU counts, total machine RAM, effective worker count, and
output-filesystem capacity and available bytes at invocation start. Hardware
capacity is not a promise about an allocated VM/container quota or memory used
by a run. GPU reporting follows EWS's existing assignments and collects model,
VRAM, and driver information when reliable system queries allow it. Physical
IDs remain in JSON for diagnostics. NVIDIA hardware inventory is labelled
separately: `nvidia-smi` indices need not match CUDA device ordering, so discovered
models are not silently assigned to configured IDs. Apple chip information is best effort;
discovering an accelerator does not mean EWS knows it was used. No framework
is imported for inspection, and no cloud metadata service is queried.

Interpret the timing fields separately:

| Measurement | Meaning |
| --- | --- |
| Invocation wall time | Elapsed time through execution and configured analysis/plotting, with UTC start/finish timestamps. Execution, analysis, and plotting durations are also recorded separately. |
| Worker time | Sum of elapsed execution-attempt durations, including initialization/restoration, checkpointing, and component cleanup. Concurrent attempts add together: four workers active for one hour contribute about four worker-hours. |
| CPU user/system time | Differences of process `resource.getrusage(RUSAGE_SELF)` counters around each attempt, including its process threads and excluding child processes. In an embedded single-worker application, unrelated threads in the same process can contribute. |
| GPU time | Attempt wall time multiplied by the number of GPUs assigned by EWS. This is allocated accelerator time, not measured utilization; CPU-only execution contributes zero. |

Worker-hours and process CPU-hours are not CPU-core-hours. No continuous
utilization or memory sampling is performed, and the report does not claim an
individual run's peak memory. With configured figures, `run` measures analysis
first, then plots from those summaries without repeating analysis. Omitted
stages have no duration.

Resuming a run adds a new attempt; it never overwrites its earlier paused or
failed attempt. Reusing a valid completion adds no simulation attempt compute.
Aggregates distinguish completed attempts from failed/paused work and summarize
attempt durations by group and algorithm. A resumed run's final successful
attempt is only its remaining work, so these attempt statistics are not a
full-run benchmark. Older completions without records and abruptly terminated
attempts without a finish record have unknown timing. Totals are observed
amounts, potentially lower bounds, rather than invented estimates for gaps.

Output size is measured once during finalization, before publishing the new
report, by summing regular-file lengths without following symlinks. It includes
retained variants and analysis artifacts; it is not allocated filesystem blocks.
The figure can include earlier compute reports and excludes the newly published
report's bytes. Filesystem availability is a start snapshot, not peak disk use.

An illustrative report excerpt might read:

```text
Compute environment
Platform: Linux x86_64
CPU cores: 16 physical / 32 logical; RAM: 64 GiB
GPU: 1 × NVIDIA RTX 4090, 24 GiB; Workers: 1

Observed experiment compute
Completed runs with timing: 120; timing unavailable: 0
Invocation wall time: 1h 14m
Cumulative worker time: 1.2 worker-hours
Cumulative allocated GPU time: 1.2 GPU-hours
Failed/paused attempt time: 1m
Attempt timing: count 122 / median 34s / mean 35s / min 12s / max 80s
```

The report covers only compute records retained in this output directory. It
does not measure the whole research project: other directories, deleted compute
records, other machines, external tools, and unobserved exploration remain the
researcher's responsibility to disclose. Hardware and timing metadata do not
change scientific run IDs, RNG streams, stored variants, or analysis cache keys.

### Checkpoint backends

Ordinary NumPy studies need no checkpoint backend configuration. The built-in
`numpy` backend uses explicit JSON state descriptions and NPZ numerical arrays,
without pickle. It accepts numerical NumPy arrays and scalars, ordinary scalars,
lists, tuples, and string-keyed mappings. Configure it explicitly only when
needed, for example to compress checkpoints independently of recorded results:

```yaml
execution:
  compression: false
  checkpoint_backend:
    type: numpy
    params: {compression: true}
```

For native framework state or a different file layout, select a paper-owned
class with the same `type`/`params` form:

```yaml
execution:
  checkpoint_backend:
    type: experiment_code.checkpointing:NestedNumpyBackend
    params: {compression: true}
```

The backend description accepts only `type` and `params`. `type` must be `numpy`
or a `module:Class` import path; `params` is a mapping of finite YAML values and
defaults to `{}`. The built-in backend accepts only a boolean `compression`
parameter. Custom constructor parameters are validated when execution prepares
the backend.

`CheckpointBackend` and `NumPyCheckpointBackend` are public imports from
`experiments_wo_stress.storage`. The interface is structural: inheritance is
optional. Constructors receive only configured keyword parameters, with no
injected scientific RNG. Implement both methods:

| Method | Responsibility |
| --- | --- |
| `save(state, directory)` | Synchronously write one generation's payload and return a finite JSON-compatible metadata mapping. |
| `load(directory, metadata)` | Read that payload using the saved metadata and return the complete logical state mapping. |

The state mapping contains `algorithm`, `data`, `protocol`, and `rngs`. Component
values come directly from `state_dict()`; EWS does not pack or convert them before
calling a custom backend. The backend must preserve all four parts, including
every injected RNG stream, without mutating live state. A native backend can
handle opaque framework values or shard a large network across files; it owns
the representation, not the scientific meaning of those values. Recorded
observations and saved instances remain numerical and are independent of this
extension.

EWS supplies a new generation directory. Write all payload files beneath it,
using subdirectories when useful. Do not write outside it, create symbolic links,
or use the reserved root filename `checkpoint.json`. Finish asynchronous work
and flush and close every file handle before `save` returns. Files are immutable
after return, and `load` must not change the generation. EWS then enumerates,
checksums, and synchronizes every payload file, writes its envelope, and commits
the matching state, step, and result boundary atomically. The backend never
publishes progress, chooses checkpoint timing, prunes generations, or changes
result records.

For example, put this dependency-free wrapper in `experiment_code/checkpointing.py`:

```python
from experiments_wo_stress.storage import NumPyCheckpointBackend


class NestedNumpyBackend:
    """Store the default representation under a private payload directory."""

    def __init__(self, *, compression=False):
        self.codec = NumPyCheckpointBackend(compression=compression)

    def save(self, state, directory):
        """Finish all payload writes and return versioned representation metadata."""
        payload = directory / "payload"
        payload.mkdir()
        return {"version": 1, "payload": self.codec.save(state, payload)}

    def load(self, directory, metadata):
        """Recover the complete logical state without modifying saved files."""
        if metadata["version"] != 1:
            raise ValueError("Unsupported nested checkpoint version")
        return self.codec.load(directory / "payload", metadata["payload"])
```

This example demonstrates nested files while retaining NumPy's supported values.
A framework-native implementation can use its own serializer in the same
boundary; EWS adds no framework dependency. Validate direct execution against
pause/resume with the actual backend, including framework RNGs and device
mapping. A GPU loader should restore onto the worker-local device, normally
`cuda:0`, without assuming the original physical GPU.

Each generation records its backend type, parameters, source fingerprints, and
representation metadata. Restore uses that saved backend, even after the current
configuration selects a different backend for new checkpoints. Keep saved
backend modules and dependencies available, and make loaders understand their
older representation versions. If no retained generation can be decoded, missing
or incompatible loaders cause a continuation error while preserving artifacts.

Backend classes may declare `dependency_files` relative to their module for
source diagnostics. These fingerprints identify the implementation used to write
a generation; a changed fingerprint alone does not reject it. Representation
version compatibility remains the loader's job.

Backend choice is operational: its parameters and separately tracked source do
not change scientific run IDs, injected RNG streams, or stored variant selection.
Resolved configuration and `provenance.checkpoint_backend` retain the selected
backend for diagnosis.
Keep backend code separate from scientific component modules, because changing
a shared file still changes the tracked scientific source. Library implementation
fingerprints remain conservative; upgrading EWS can select new variants. Reading
legacy JSON/NPZ checkpoints remains supported, without automatically bypassing
provenance or migrating old experiments.

## Budget continuation and retained artifacts

A component declares `supports_extension = True` only when its saved state can
continue correctly under a larger execution budget. All three runtime components
must support this for the run to extend. The built-in online protocol and the
sequential example do. Offline and indivisible trial operations are not generally
extendable.

Increasing an extendable run's budget restores its final state and RNGs and executes
only the additional steps. Decreasing the requested budget selects an appropriate
recorded prefix; completed budget boundaries remain recorded. A nonextendable
budget change selects another variant. Scientific, code, dependency, and recording
changes retain distinct affected variants rather than mixing incompatible results.
Other matching runs remain reusable.

Extension still respects the scientific input: a finite nonstationary schedule
cannot provide observations beyond its last row. Changing or appending the schedule
selects a new instance and run; existing schedule prefixes are not migrated.

The output directory retains request history, while analysis follows the active
request's variants and result boundaries. Use `ews inspect OUTPUT` to see selected
work. [The architecture](ARCHITECTURE.md#budget-and-reusable-work) describes the
artifact model.

## Metrics, aggregation, and cache reuse

```yaml
analysis:
  points: 100  # Default; null keeps full analysis resolution.
  metrics:
    - name: regret
      type: pseudo_regret
  aggregator:
    type: repeated_runs
    group_by: [data.params.n_arms, algorithm.name]
    reduce_over: repetition
    summary: mean
    uncertainty: standard_error
  figures:
    - name: regret
      type: line
      metric: regret
      color: algorithm.name
      panel: data.params.n_arms
      panel_label: Number of arms
      xlabel: Round
      ylabel: Cumulative pseudo-regret
      formats: [pdf, jpg, tikz]
```

A metric implements `compute(result) -> MetricResult(x, values)`. The input is a
`RunResult`, which behaves as a mapping of recorded arrays and also exposes
`instance`, `spec`, `completed_steps`, `revision`, and `final_outputs`. Constructors
receive configured `params`, without an injected simulation RNG. Output is a finite
real scalar or matching one-dimensional coordinates and values.

`final_outputs` exposes final recorded fields for single-step offline/trial results.
Multiple-step results use `records`; an arbitrary final-output hook is not provided.

The general field metrics and optional bandit metrics are available by short name:

- `field`: select a saved field using `params.field`, optionally choosing `params.x`.
- `cumulative_sum`: sum a field across a complete recorded trajectory.
- `pseudo_regret`: combine actions with saved `instance.arrays['means']`. Means can
  be stationary or a time-by-arm schedule. The comparator is `dynamic` by default;
  `best_fixed` compares with the best fixed arm over each observed prefix.
- `realized_regret`: compare observed rewards with a saved time-by-arm
  `counterfactual_rewards` matrix, defaulting to the best fixed arm. Recorded
  rewards must agree with the selected matrix entries; arm means alone are
  insufficient to determine this quantity.

The two regret implementations are supplied bandit metrics in `builtins/metrics.py`,
not requirements of the analysis interface. Use a custom metric when the study's
regret definition or comparator differs.

Custom metrics can declare `required_fields`, `required_instance_fields`,
`requires_complete_trajectory`, and `dependency_files`. Dependency paths are
relative to the metric module. Use `module:Class` to select a custom metric.

The `repeated_runs` aggregator computes a mean across independent repetitions.
Choose `standard_error`, sample `std`, or `none` for uncertainty. One repetition
has undefined sample uncertainty, stored as NaN and omitted from the figure.
Standard error is not a confidence interval. Include all varying scientific
parameters in `group_by`; incompatible configurations, duplicate repetitions, and
misaligned coordinates are rejected.

`analysis.points` bounds each metric/group curve in returned `Summary` objects,
CSV tables, summary `.npz` files, and figures (including custom plotters). The
default is 100; set an integer of at least 2, or `null` to retain all points.
`1` is rejected because it cannot preserve both endpoints. Scalar curves and
curves no longer than the limit retain every point. Longer curves use
deterministic, approximately equally spaced row positions with no duplicates,
including the first and last; CSV column `x` retains the metric's actual
coordinates, usually one-based protocol steps, rather than renumbering samples.

Raw trajectories remain full-resolution binary `.npz` chunks at the requested
recording frequency; no raw trajectory CSV is written. Metrics (including
cumulative reward and regret), coordinate validation, and aggregation compute
over every recorded observation before reducing the representation. Per-run
metric caches also retain full-resolution `.npz` curves. Changing `points`
reuses these metrics and simulations and versions aggregate/figure outputs.
Existing configurations now get compact curves by default; `points: null`
restores the previous full-resolution analysis behavior. A table with multiple
groups has up to `points` rows per group. For cumulative metrics, keep
`recording.every_steps: 1`; sparse recording discards inputs and is independent
of this export setting.

`ews analyze` writes summary tables under `analysis/` and reuses valid cached
work. Only completed, validated runs contribute; summaries report their actual
count. Input, code, parameter, or declared dependency changes invalidate affected
analysis. See [analysis and cache invalidation](ARCHITECTURE.md#analysis-and-cache-invalidation)
for cache identities and [cleanup](#run-inspect-and-clean) for removing derived
artifacts.

## Figures

A default figure has `type: line` and a metric name. Options include `name`,
`formats` (`pdf`, `jpg`, `tikz`), `color`, `panel`, `panel_label`, `xlabel`, `ylabel`,
`title`, and linear or logarithmic `xscale`/`yscale`. The `x` setting supplies a
default axis label; numerical coordinates come from the metric.

Uncertainty is a band or a scalar error bar. Undefined uncertainty is omitted;
nonpositive lower bounds are omitted on logarithmic y axes. Series and panel
labels must distinguish curves. PDF/JPG use the `plot` extra; TikZ is an editable
PGFPlots document requiring LaTeX only when compiled.

Custom plotters implement `plot(summaries, figure, output_dir) -> Iterable[Path]`
and receive configured `params`. A `Summary` contains the metric, group labels,
coordinates, mean, uncertainty, repetition count, and uncertainty convention.
Custom figure options belong in `params`.

## Run, inspect, and clean

```bash
ews run examples/sequential_study/experiment.yml --output outputs/bandits --workers 2
```

`run` executes or resumes the study, reuses compatible completed work, and then
produces configured analysis and figures. This final stage runs only when there
are no failed, paused, or pending runs. An interrupted or deliberately paused
invocation can resume with the same command.

CPU or GPU studies without analysis configuration perform execution only. A
metrics-only configuration computes summaries; configured figures trigger the
analysis they need and are then exported.

An optional count helps estimate the work before an expensive study:

```bash
ews count-runs examples/sequential_study/experiment.yml
```

The command validates configuration and returns only `name` and `runs`, for example
`{"name": "gaussian_bandit_comparison", "runs": 120}`. For a grid group, the
count is grid combinations × algorithm count × repetitions; counts add across
groups. Custom planners determine their own set of concrete runs. A count is useful with
a representative run's time and memory measurements, but is never a prerequisite
for `run`.

`analyze` and `plot` work independently from saved numerical results, so changes
to metrics or figure presentation do not require repeating expensive simulations.
`analyze` computes or reuses metrics and summary tables; `plot` computes or reuses
the analysis required for configured figures. `inspect` reports stored status.
The public commands are `count-runs`, `run`, `analyze`, `plot`, `inspect`, and
`clean`. Choose saved-data operations independently as needed:

| Command | Use |
| --- | --- |
| `ews analyze examples/sequential_study/experiment.yml --output outputs/bandits` | Update metrics and summary tables. |
| `ews plot examples/sequential_study/experiment.yml --output outputs/bandits` | Update figures using saved results. |
| `ews inspect outputs/bandits` | Read stored status and validate completed artifacts. |
| `ews clean outputs/bandits --scope inactive` | Preview removal of inactive retained artifacts. |

Cleanup requires `--yes` to delete the previewed selection.

| Cleanup scope | Removes |
| --- | --- |
| `analysis` | Rebuildable analysis outputs and caches. |
| `checkpoints` | Saved execution state; numerical observations remain, continuation is lost. |
| `inactive` | Variants outside the active request; this is the default scope. |
| `runs` | Run artifacts, optionally restricted by repeated `--run-id ID`. |
| `all` | Experiment contents, retaining the output directory itself. |

`--run-id` also restricts checkpoint cleanup. Run cleanup includes newly unreferenced
instances and request records that refer to removed variants.
Compute history survives `runs` and `inactive` cleanup, so it may still describe
attempts whose scientific artifacts were removed. The `all` scope removes that
history too. Aggregate files describe their last regeneration.

Review the preview before applying deletion. Keep datasets, scientific source,
configurations, and generated outputs in distinct locations so cleanup scope is
clear. Execution and cleanup share a lock; analysis and plotting do not, so clean
artifacts while those operations are idle. See [the architecture](ARCHITECTURE.md)
for recovery and artifact boundaries.

## Discord notifications

Add an enabled Discord block to receive progress summaries during the execution
stage of `run`:

```yaml
notifications:
  discord:
    enabled: true
    webhook_env: EWS_DISCORD_WEBHOOK_URL
    interval_seconds: 300
    timeout_seconds: 5
```

`webhook_env` names an environment variable containing the HTTPS webhook URL; keep
the URL out of YAML and source control. An enabled block with a missing or invalid
URL fails before execution. Omit the block or set `enabled: false` to run without a
secret. Analysis and plotting do not send messages.

The update interval must be at least one second; the socket timeout must be
positive and no longer than 30 seconds. Defaults are shown above.

Messages report run counts and recent durable checkpoint progress. Delivery is best
effort: network errors do not fail simulations, and even the final summary may be
missed during an outage or forced stop. Notification settings do not change run
identities or random streams. Messages omit numerical data, parameters, paths, and
error details; inspect stored artifacts for authoritative status.

## Standalone authoring and deployment

The [LLM guide](LLM_GUIDE.md) is a self-contained document for generating a paper's
experiment repository from these interfaces. The [cloud guide](CLOUD.md) shows how
to install a pinned library revision in a container and keep output on durable
storage, with no distributed scheduler or storage-backend change.

## External artifact discovery

No YAML option is needed: run/analyze/plot workflows publish `OUTPUT/artifacts.json`.
Its explicit schema version and semantic roles let external tools find figures,
analysis, compute reports, and raw runs without hardcoding internal paths. Optional
roles can have no files. This does not change recording, checkpoint, or analysis
settings. See [the artifact contract](ARTIFACTS.md) for fields, publication, and
legacy outputs; `inspect` continues to read old outputs without this file.
