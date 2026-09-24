# Configuration and extension guide

Use [the complete sequential study](../examples/sequential_study/experiment.yml)
as a starting point. YAML describes the experiment; ordinary Python classes hold
the scientific behavior. Load it with `load_config(path)` or an `ews` command.

## Experiment and run groups

```yaml
name: gaussian_bandit_comparison
seed: 2026
runs:
  - name: main
    planner: grid
    repetitions: 20
    protocol:
      type: online
      params: {horizon: 1000}
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

Place this configuration beside the example's `paper.py`. The loader adds the
configuration directory to Python's import search path. Import paths identify
trusted Python code and follow `module:Class`; built-ins also have short aliases.
The library does not infer classes from parameter names.

| Setting | Meaning and default |
| --- | --- |
| `name` | Required nonempty experiment name. |
| `seed` | Nonnegative root seed; defaults to `0`. |
| `runs` | Required nonempty list of run groups. |
| Group `name` | Unique label; defaults to `group_0`, `group_1`, etc. |
| Group `planner` | `grid` by default, or a custom `module:Class`. |
| Group `repetitions` | Positive independent repetition count; defaults to `1`. |
| Group `protocol`, `data` | Required component descriptions. |
| Group `algorithms` | Required nonempty list of component descriptions. |
| Group `grid` | Mapping of parameter paths to nonempty value lists; defaults to `{}`. |

Components have `type`, `params` (default `{}`), and `seed` (default `null`, using
the experiment seed). Algorithms additionally have `name`, defaulting to the final
part of their type path; names must be distinct within a group. Constructors receive
`rng=` plus `params` as keyword arguments. `rng` is reserved and cannot be configured.

Grid axes use `algorithm.params.*`, `data.params.*`, or `protocol.params.*` and
expand as a Cartesian product, crossed with algorithms and repetitions. An
algorithm grid axis applies to every algorithm in that group. Use separate groups
when algorithms need different sweep parameters. Duplicate candidates and
overlapping parameter paths are rejected. The example produces 3 × 2 × 20 runs.

YAML uses plain finite values, lists, and string-keyed mappings. Duplicate keys,
unknown framework options, unsupported YAML objects, and invalid settings produce
errors. Custom component parameters are validated against their constructors at
execution preflight; analysis options are validated when analysis or plotting runs.

## Execution and recording

These are the effective defaults:

```yaml
execution:
  workers: 1
  checkpoint_seconds: 120.0
  checkpoint_steps: null
  keep_checkpoints: 2
  compression: false

recording:
  every_steps: 1
  fields: null
  buffer_bytes: 16777216
```

`workers` chooses local processes; `--workers` overrides it for one invocation.
With one worker, execution is sequential in the current process. Protect Python
scripts using multiple workers with `if __name__ == "__main__":`.

A checkpoint is due when either the time interval or the step interval is reached.
Set an interval to `null` to disable that trigger. Checkpoints occur after complete
protocol steps, and a deliberate pause or graceful interruption also checkpoints.
`keep_checkpoints` is a positive count. Compression applies to numerical result
chunks and checkpoint arrays; it trades CPU work for disk space.

`every_steps` records every positive integer interval and always includes the final
step. This version supports fixed intervals, not logarithmic recording schedules.
`fields: null` records all returned measurements; a list selects fields. The library
adds `step` automatically. Fields must retain their numerical dtype and shape across
records. Names contain letters, digits, and underscores and cannot start with a digit.

The buffer target is 16 MiB **per active run**, with additional component and
serialization memory. Results flush at that target, before checkpoints, and at
completion. Flushing a result chunk does not itself establish a resumable checkpoint.

For sparse curves of cumulative quantities, update the accumulator every scientific
step and checkpoint it. Recording every tenth reward and summing afterward cannot
recover the other nine rewards. The example records a cumulative regret field.

## Metrics and aggregation

```yaml
analysis:
  metrics:
    - name: regret
      type: field
      params: {field: regret, x: step}
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

Analysis requires at least one metric. `field` selects a saved field;
`cumulative_sum` sums the saved values. Both accept required `params.field` and
optional `params.x` (default `step`). Custom metrics use `module:Class` and `params`,
with no injected RNG. Names must be safe filenames using letters, digits, underscores,
dots, or hyphens, beginning with a letter or digit.

The only built-in aggregator is `repeated_runs`, with `summary: mean` and
`reduce_over: repetition`. Default grouping is `[group, algorithm.name]`; include
all parameters that vary within a scientific comparison. Available labels include
`group`, `algorithm.name`, each component's `.type`, and flattened `.params.*`.
Grouping cannot pool different scientific configurations or duplicate repetitions.
All repetitions in a group must have identical x coordinates.

Uncertainty is `standard_error` (default), `std` for sample standard deviation,
or `none`. Independent runs are the statistical units. One repetition has undefined
sample uncertainty, saved as NaN and omitted from the figure. Standard error is
not a confidence interval. Partial studies include only validated completed runs;
each table reports the actual repetition count.

`ews analyze` writes `analysis/<metric>.csv`, `analysis/<metric>.npz`, and an index
in `analysis/metadata.json`. Metrics receive a full run's recorded arrays in memory;
the aggregator retains running summaries instead of all repetitions.

## Figures

Default figures use `type: line` and a metric name. Optional settings are:

| Setting | Behavior |
| --- | --- |
| `name` | Output stem; defaults to `<metric>-<figure index>`, counting from 1. |
| `formats` | Nonempty subset of `pdf`, `jpg`, `tikz`; defaults to `[pdf]`. |
| `color`, `panel` | Grouping labels used for series and panels. |
| `panel_label` | Human-readable panel label; otherwise the last parameter name is used. |
| `x` | Default x-axis label, `step`; coordinates come from the metric. |
| `xlabel`, `ylabel`, `title` | Display labels; the default y label is the metric name. |
| `xscale`, `yscale` | `linear` (default) or `log`; log axes require positive coordinates/means. |

Uncertainty is drawn as a band, or an error bar for a scalar metric. On logarithmic
y axes, bands whose lower bound is nonpositive are omitted. A figure must distinguish
its curves through its series and panel labels. Additional plotting primitives need
a custom plotter.

`ews plot` recomputes summaries and saves figures under `analysis/figures/`.
PDF/JPG need the `plot` extra. TikZ is a self-contained, editable PGFPlots document;
export needs no LaTeX installation, while compilation does.

## Scientific extensions

Inheritance from library classes is optional. Structural interfaces are documented
in `experiments_wo_stress.components`; see the example Python files for complete
implementations.

| Component | Required behavior |
| --- | --- |
| Data generator | `generate(request)` and checkpoint methods; owns evolving environment state. |
| Online algorithm | `act(context=None)`, `observe(action, feedback)`, and checkpoint methods. |
| Offline algorithm | `fit(dataset)` returning numerical measurements, and checkpoint methods. |
| Protocol | `initialize(algorithm, data)`, `advance(algorithm, data)`, `is_finished()`, `step`, and checkpoint methods. |

All scientific constructors accept `rng=`. Use that generator for randomness.
Stateful objects implement `state_dict()` and `load_state_dict(state)`; include
everything needed to continue. State supports string-keyed mappings, lists, tuples,
ordinary scalar values, and numerical NumPy arrays. Do not store live resources or
Python objects requiring pickle. RNG state is saved separately by the executor.
`StateMixin` supplies an empty-state implementation for stateless components only.

Protocol initialization runs only for a fresh run. Restoration constructs the
components, loads their state, and restores RNGs. Put persistent state in a place
that can be restored without rerunning `initialize()`. Each `advance()` must increment
`step` by exactly one and return a numerical mapping. `step` is reserved for the recorder.

The built-in `online` protocol accepts `horizon`. Each round optionally calls
`data.context()`, invokes `algorithm.act(context=...)`, and passes the action to
`data.generate(action)`. Return `Feedback(value, measurements)`: only `value` reaches
`algorithm.observe(action, value)`. Measurements are evaluator-only numerical
fields; a numerical action is recorded automatically unless explicitly supplied.

The `offline` protocol accepts optional `request` and performs
`algorithm.fit(data.generate(request))` as one step. A numerical array becomes an
`output` measurement; a mapping supplies named measurements.

Built-in data aliases are `normal` (`size=100`, `loc=0`, `scale=1`), `csv`
(`path`, `delimiter=','`, `skip_header=0`, `dtype='float64'`), and `null`.
CSV paths resolve relative to the YAML file. `csv.generate(None)` returns the full
read-only matrix; a positive integer request consumes that many rows, with a cursor
saved in checkpoints. `normal.generate(request)` can override the configured shape.

### A complete callable trial

For a small computation, use the `trial` protocol and placeholder components:

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

Quote the alias as `type: 'null'` in YAML: unquoted `null` is a missing value.
The function contract is `trial(*, algorithm, data, rng, size)` for this example,
returning numerical output or a numerical mapping. Trial parameters cannot replace
the reserved `algorithm`, `data`, or `rng` arguments. The entire function is one
step, so interruption within it restarts that step.

### Custom planning, metrics, and plotting

A planner has a zero-argument constructor and
`plan(group, seed) -> Iterable[RunSpec]`. Set `planner: paper:MyPlanner` on a group.
Use the public `make_run_spec(group=..., repetition=..., algorithm_name=...,
algorithm=..., data=..., protocol=..., seed=...)` helper to create stable run IDs.
Component arguments are mappings or `ComponentSpec` objects. Preserve the supplied
group name and experiment seed; planning must be deterministic. `GridPlanner` can
be reused when a custom strategy only filters its output.

A metric implements `compute(results) -> MetricResult(x, values)`. The result must
be a scalar or matching one-dimensional finite real arrays. Import `MetricResult`
from `experiments_wo_stress`; the optional `Metric` typing interface describes this
contract. Constructors receive only their configured `params`.

A custom figure uses `type: paper:MyPlotter`, `metric`, optional `name`, and `params`.
The class receives `params` at construction and implements
`plot(summaries, figure, output_dir) -> Iterable[Path]`. Each `Summary` contains
`metric`, `labels`, `x`, `mean`, `uncertainty`, `count`, and `uncertainty_kind`.
All custom figure options belong in `params`.

## Reproducibility and resumption

Run IDs and streams derive from scientific configuration, group, and repetition.
Algorithm, data, and protocol receive separate RNGs. Data streams exclude algorithm
choice, allowing algorithms to start from the same generated instance. The meaning
of shared randomness in action-dependent environments belongs to the generator.

Changing workers, checkpoint settings, compression, buffer size, or analysis can
reuse an output directory. Changing the scientific run plan, experiment name,
recording interval, or selected fields requires a new directory. The runner also
checks code/environment fingerprints and contents of built-in CSV inputs.

Tracked code includes library modules, component/planner/trial source files, and
Python files beside the YAML. This is not a recursive dependency snapshot: pin
external dependencies and track additional files consumed by custom generators in
the paper project. Exact replay assumes unchanged code, inputs, and numerical
environment. See [the architecture](ARCHITECTURE.md) for artifact validation and
checkpoint recovery details.
