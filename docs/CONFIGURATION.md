# Configuration and extension guide

Start with the [complete sequential configuration](../examples/sequential_study/experiment.yml).
YAML describes a study; Python classes supply scientific behavior. Load a file with
`load_config(path)` or an `ews` command. Classes are selected explicitly through
built-in aliases or `module:Class` paths. The configuration directory is added to
Python's import context.

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
      type: paper:GaussianBandit
      params: {n_arms: 10, noise_std: 0.1}
    algorithms:
      - name: ucb
        type: paper:UCB
        params: {exploration: 0.1}
      - name: epsilon_greedy
        type: paper:EpsilonGreedy
        params: {epsilon: 0.1}
    grid:
      data.params.n_arms: [10, 20, 50]
```

Place this file beside the example's `paper.py`. Its grid expands to 3 sizes ×
2 algorithms × 20 repetitions. A group has these settings:

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

`ComponentSpec.dependencies` holds the resolved dependency paths. A scientific
class can also declare `dependency_files`, resolved relative to its source module.
Source tracking covers full module files and base-class modules. Keep metrics and
simulation components in separate files when a metric edit should preserve the
simulation cache. Start a fresh Python process or explicitly reload modules after
editing source that a live process has already imported.

Grid paths address `algorithm.params.*`, `data.params.*`, or `protocol.params.*`.
An algorithm axis applies to every algorithm in its group; separate groups can
sweep different parameters. Grid axes are a Cartesian product. Duplicate candidates
and overlapping paths are rejected. Custom planners must produce deterministic
run specifications.

Execution budget is separate from a scientific horizon parameter. Built-in online
protocol configurations written with legacy `params.horizon` are normalized into
the budget; conflicting explicit settings are rejected. An algorithm that uses a
fixed horizon in its learning rule must retain that as a scientific parameter.

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

`--workers` overrides the worker setting. One worker executes sequentially in the
current process. Python scripts launching multiple workers need the usual
`if __name__ == "__main__":` guard.

A checkpoint is due when either enabled time or step trigger is reached. `null`
disables that trigger. A pause, graceful interruption, and completion also save
state. Rolling checkpoint retention is additional to snapshots pinned by completed
budget boundaries. Compression trades CPU work for storage space.

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

Buffering is bounded by an approximate array-byte target, 16 MiB per active run by
default. Component state and serialization need additional memory. Results flush
at that target, before checkpoints, and at completion. A result flush alone does
not save resumable execution state.

A different recording selection retains a separate variant. It may require new
execution to obtain missing observations. Metrics that need every action or reward
reject sparse trajectories; the library does not infer omitted data.

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

The output directory stores an active request and request history. Analysis follows
the active requested variants and result boundaries. `ews inspect OUTPUT` describes
available work. Execution requires schema 2 artifacts; supported analysis of older
schema 1 outputs remains available without silently converting their execution state.

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
changing values needed to continue. Supported state includes numerical arrays,
ordinary scalars, lists, tuples, and string-keyed mappings. The library saves RNG
states separately. `StateMixin` supplies empty state for stateless components.

Each protocol advance increments `step` exactly once and returns measurements.
Initialization happens only for a fresh run. Restoration constructs components,
sets the requested protocol budget, loads component state, and restores RNGs; essential state
must not depend on repeating initialization.

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
    state_adapter: my_environments:CartPoleState
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
        function: paper:trial
        params: {size: 100}
    data: {type: 'null'}
    algorithms:
      - {name: trial, type: null_algorithm}
```

For this example, implement `trial(*, algorithm, data, rng, size)` returning numerical
output or a mapping. Quote `'null'` in YAML so it remains a component alias. The
whole function is one step; an interruption inside it restarts that step.

## Metrics, aggregation, and cache reuse

```yaml
analysis:
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

Built-in metrics include:

- `field`: select a saved field using `params.field`, optionally choosing `params.x`.
- `cumulative_sum`: sum a field across a complete recorded trajectory.
- `pseudo_regret`: combine actions with saved `instance.arrays['means']`. Means can
  be stationary or a time-by-arm schedule. The comparator is `dynamic` by default;
  `best_fixed` compares with the best fixed arm over each observed prefix.
- `realized_regret`: compare observed rewards with a saved time-by-arm
  `counterfactual_rewards` matrix, defaulting to the best fixed arm. Recorded
  rewards must agree with the selected matrix entries; arm means alone are
  insufficient to determine this quantity.

Custom metrics can declare `required_fields`, `required_instance_fields`,
`requires_complete_trajectory`, and `dependency_files`. Dependency paths are
relative to the metric module. Use `module:Class` to select a custom metric.

The `repeated_runs` aggregator computes a mean across independent repetitions.
Choose `standard_error`, sample `std`, or `none` for uncertainty. One repetition
has undefined sample uncertainty, stored as NaN and omitted from the figure.
Standard error is not a confidence interval. Include all varying scientific
parameters in `group_by`; incompatible configurations, duplicate repetitions, and
misaligned coordinates are rejected.

`ews analyze` writes current summary tables under `analysis/`. Immutable caches
live under `analysis/cache/metrics/`, `aggregates/`, and `plots/`; checksums protect
cached files. Relevant input, code, parameter, and dependency changes invalidate
the affected artifacts. Replotting from a valid cache avoids metric recomputation.
Only completed, validated runs contribute, and summaries report their actual count.

Derived output is not always smaller than raw data. Dense CSV summaries, numerical
summary files, cached curves, and current exported copies can together exceed the
trajectory's storage size. Preview `ews clean OUTPUT --scope analysis` to reclaim
these rebuildable artifacts while retaining instances and run observations.

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

## Build, inspect, and clean

```bash
ews build examples/sequential_study/experiment.yml --output outputs/bandits --workers 2
ews inspect outputs/bandits
ews clean outputs/bandits --scope inactive
```

`build` executes work then produces requested analysis. Individual `run`, `analyze`,
and `plot` commands remain useful when working on one stage. Cleanup previews its
selection and requires `--yes` to delete files.

| Cleanup scope | Removes |
| --- | --- |
| `analysis` | Rebuildable analysis outputs and caches. |
| `checkpoints` | Saved execution state; numerical observations remain, continuation is lost. |
| `inactive` | Variants outside the active request; this is the default scope. |
| `runs` | Run artifacts, optionally restricted by repeated `--run-id ID`. |
| `all` | Experiment contents, retaining the output directory itself. |

`--run-id` also restricts checkpoint cleanup. Run cleanup includes newly unreferenced
instances and request records that refer to removed variants.

Review the preview before applying deletion. Keep datasets, scientific source,
configurations, and generated outputs in distinct locations so cleanup scope is
clear. Execution and cleanup share a lock; analysis and plotting do not, so clean
artifacts while those operations are idle. See [the architecture](ARCHITECTURE.md)
for recovery and artifact boundaries.
