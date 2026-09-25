# Build a paper experiment with Experiments W/O Stress

This file is self-contained. Give it to an LLM together with a paper PDF and ask
it to implement an experiment in a **separate research repository**. The library
provides execution and artifact infrastructure; the research repository contains
the paper's scientific choices. The example below demonstrates the API and does
not claim to reproduce any paper.

## Instructions for the implementing assistant

1. Read the paper, its appendices, and any supplied experiment specifications.
   Identify the exact experiment, algorithm, problem distribution, observable
   feedback, stopping rule, comparator, and figure to reproduce. Record page,
   equation, or algorithm references next to each implementation decision.
2. Write a short evidence table: paper statement, source location, code component,
   and unresolved detail. Distinguish specified facts from proposed defaults.
   **Do not invent missing equations, hyperparameters, dataset preprocessing, or
   statistical conventions.** Ask about material gaps; label runnable provisional
   choices explicitly when the user authorizes them.
3. Implement a small faithful study first. Keep algorithms, generators, metrics,
   and external-library adapters readable and separate. Do not modify the library
   merely to encode paper-specific behavior.
4. Verify scientific invariants and direct-versus-resumed execution before scaling.
   Show the actual configuration, validation, results, and remaining deviations
   from the paper. A successful infrastructure test is not evidence that the paper
   has been reproduced.

## Install a pinned library version

Use Python 3.10 or newer and Git. Choose a full Git commit containing the API
described here, replace the placeholder, and record that commit in the research
repository. Do not assume a package-index release exists.

```bash
python -m venv .venv
source .venv/bin/activate
EWS_COMMIT=REPLACE_WITH_FULL_40_CHARACTER_COMMIT_SHA
python -m pip install "experiments-wo-stress @ git+https://github.com/m1gwings/experiments-wo-stress.git@${EWS_COMMIT}"
python -m pip freeze > requirements.lock.txt
```

The library repository is public. For PDF/JPG figures, add the `plot` extra to
the requirement; for Gymnasium, add `gym`. NumPy and PyYAML are core dependencies.
On Windows, activate with `.venv\Scripts\activate`.

A minimal external repository can look like this:

```text
paper-study/
  README.md                  paper references, assumptions, run instructions
  requirements.lock.txt      exact library commit and dependency versions
  experiment.yml             scientific choices and execution settings
  experiment_code/
    __init__.py
    algorithms.py            learning rules and their checkpoint state
    data.py                  custom generators/adapters when needed
    metrics.py               paper metrics, independent of simulation modules
  verify.py                  repeatability and recovery checks
  outputs/                   generated artifacts, excluded from Git
```

Keep `experiment_code/` beside the YAML file, or install the research package.
The YAML directory is an import context, so
`experiment_code.algorithms:MyAlgorithm` names an ordinary Python class.
Separate simulation code from metrics so a metric edit need not select a new
simulation variant. Restart Python after editing already imported modules.

## A runnable API example

Copy the following files into the indicated paths, creating an empty
`experiment_code/__init__.py` to make the package explicit. The scientific code needs only
NumPy in addition to the library. It compares two exploration probabilities on two
Gaussian bandit sizes, with three independent repetitions: 12 short runs. These
are illustrative settings, not paper-derived recommendations.

`experiment_code/algorithms.py`:

<!-- file: experiment_code/algorithms.py -->
```python
import numpy as np


class EpsilonGreedy:
    """Illustrative fixed-probability exploration with sample-mean estimates."""

    supports_extension = True

    def __init__(self, *, rng, epsilon=0.1):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must lie in [0, 1]")
        self.rng = rng
        self.epsilon = epsilon
        self.counts = np.zeros(0, dtype=np.int64)
        self.sums = np.zeros(0, dtype=np.float64)

    def act(self, context=None):
        n_arms = context["n_arms"]
        if not self.counts.size:
            self.counts = np.zeros(n_arms, dtype=np.int64)
            self.sums = np.zeros(n_arms, dtype=np.float64)
        if self.counts.size != n_arms:
            raise ValueError("the action space changed")
        unplayed = np.flatnonzero(self.counts == 0)
        if unplayed.size:
            return int(unplayed[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(n_arms))
        return int(np.argmax(self.sums / self.counts))

    def observe(self, action, feedback):
        self.counts[action] += 1
        self.sums[action] += float(feedback)
        self.logger.debug("action=%d reward=%.6g", action, feedback)

    def state_dict(self):
        return {"counts": self.counts, "sums": self.sums}

    def load_state_dict(self, state):
        self.counts = np.array(state["counts"], copy=True)
        self.sums = np.array(state["sums"], copy=True)
```

`experiment_code/metrics.py`:

<!-- file: experiment_code/metrics.py -->
```python
import numpy as np

from experiments_wo_stress import MetricResult


class MeanReward:
    """Mean reward over the entire recorded run."""

    required_fields = ("step", "reward")
    requires_complete_trajectory = True

    def compute(self, result):
        return MetricResult(
            x=np.array([result.completed_steps]),
            values=np.array([np.mean(result["reward"])]),
        )
```

`experiment.yml`:

<!-- file: experiment.yml -->
```yaml
name: illustrative_bandits
seed: 2026
runs:
  - name: main
    planner: grid
    repetitions: 3
    budget: {steps: 200}
    protocol: {type: online}
    data:
      type: gaussian_bandit
      params: {n_arms: 3, noise_std: 0.1}
    algorithms:
      - name: epsilon_010
        type: experiment_code.algorithms:EpsilonGreedy
        params: {epsilon: 0.1}
      - name: epsilon_030
        type: experiment_code.algorithms:EpsilonGreedy
        params: {epsilon: 0.3}
    grid:
      data.params.n_arms: [3, 5]
execution:
  workers: 1
  checkpoint_seconds: 120
  checkpoint_steps: 50
  keep_checkpoints: 2
  logging_level: INFO
recording:
  every_steps: 1
  fields: [action, reward]
  buffer_bytes: 1048576
analysis:
  metrics:
    - {name: regret, type: pseudo_regret}
    - {name: mean_reward, type: 'experiment_code.metrics:MeanReward'}
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
      formats: [tikz]
```

Run from the research repository:

```bash
ews run experiment.yml --output outputs/study --workers 2
```

`run` executes the study and produces its configured analysis and figures after
all requested runs complete or are reused. Optionally use
`ews count-runs experiment.yml` to validate configuration and check the experiment
size before execution; it is never required. This example returns only
`{"name": "illustrative_bandits", "runs": 12}`: two grid values × two algorithms
× three repetitions. Counts add across run groups and help estimate work from
a representative run.

The generator saves the drawn means in an immutable instance. It emits actions
and rewards; the algorithm sees the reward and arm count, not hidden means.
`pseudo_regret` combines saved actions with saved means after execution. Summary
tables appear under `analysis/`; the figure appears under `analysis/figures/`.
With the `plot` extra installed, use `formats: [pdf, jpg, tikz]`.

## Implement the paper's scientific components

No framework inheritance is required. Runtime constructors accept `rng=` plus the
YAML `params`; `rng`, `instance`, and `logger` are reserved injected names.
A normal run component receives a logger as `self.logger`; an explicit `logger=`
constructor parameter also supports slotted or frozen classes.

| Component | Required behavior |
| --- | --- |
| Online algorithm | `act(context=None)` and `observe(action, feedback)`. |
| Offline algorithm | `fit(dataset)` returning a numerical array or numerical mapping. |
| Generator | `generate(request)`; owns evolving environment state. |
| Protocol | `initialize(algorithm, data)`, `advance(algorithm, data)`, `is_finished()`, and integer `step`. Each advance completes one step and increments `step` once. |
| Every runtime component | `state_dict()` and `load_state_dict(state)`. |

Include all mutable scientific state in checkpoints: learned parameters, counters,
cursors, pending contexts, queues, and any independent foreign-library RNG state.
The default checkpoint backend supports numerical NumPy arrays, scalars, lists,
tuples, and mappings with string keys, without pickle. A custom backend can
serialize native framework values supplied by `state_dict()`; no mandatory NumPy
conversion occurs before that backend receives state. Include enough information
to reconstruct resources rather than relying on live handles surviving restart.
EWS adds its injected RNG snapshots alongside component state. Truly stateless
components may inherit `StateMixin`; stateful subclasses must override both state
methods.

For a custom instance, import `Instance` and optionally `Feedback` from
`experiments_wo_stress`. Define the generator's class method
`create_instance(*, rng, **params) -> Instance(metadata=..., arrays=..., kind=...)`.
Metadata holds finite structured values; arrays hold immutable numerical inputs.
The generator receives the saved result as an explicit `instance=` constructor
argument. The runner does not expose it to the learner automatically.

Save the actual scientific information analysis needs: a matrix, graph, arm means,
or problem definition. A generator without this hook gets a descriptor, not an
automatic capture of every sample it draws. Immutable instance content is shared
on disk when identical. A CSV instance stores its input matrix and source hash;
a `normal` generator's instance describes its distribution, so sample realizations
must be recorded separately when needed later.

For stochastic hidden-state evolution, save initial conditions/law in the instance
and required realized hidden quantities as evaluator measurements. Do not mutate
the instance or calculate scientific metrics inside the generator. A fixed
nonstationary schedule can only extend within saved coverage; appending the
schedule creates a new scientific instance and run.

### Choose the interaction

- **Online:** `protocol: {type: online}` with `budget.steps`. Optional
  `data.context()` is passed to `act`. `generate(action)` returns
  `Feedback(value, measurements)`; only `value` reaches `observe(action, value)`.
  Measurements are numerical recorder fields, with numerical action added when
  absent. Context and feedback structure beyond this are defined by the study.
- **Offline:** `protocol: {type: offline, params: {request: ...}}`, omitting `request`
  when unnecessary. It calls `fit(data.generate(request))` as one step; omit an
  online budget. `csv` accepts `path`, `delimiter`, `skip_header`, and `dtype`;
  paths are relative to YAML. `normal` accepts `size`, `loc`, and `scale`.
- **Callable trial:** use `protocol.type: trial`, with
  `params.function: experiment_code.trials:run`
  and optional nested `params.params`. The function is
  `run(*, algorithm, data, rng, **params)` and returns numerical output. Placeholder
  components are `data: {type: 'null'}` and an algorithm with `type: null_algorithm`.
  Quote `'null'` in YAML. A whole trial or offline fit is indivisible for recovery.
- **RL:** `protocol: {type: rl}` uses a generator with `reset()` returning
  `(observation, info)` and `generate(action)` returning `Feedback` whose value is
  a mapping of `observation`, `reward`, `terminated`, `truncated`, and optional
  `info`. The learner gets `act(context=observation)` and
  `observe(action, transition)`, where transition has `observation`,
  `next_observation`, `reward`, both terminal flags, and `info`. The protocol resets
  after termination or truncation; the learner chooses the scientific bootstrapping
  rule. Numeric observations/actions are flattened into fixed-shape record fields.

For another interaction, implement a custom protocol. If it accepts an execution
budget, provide `set_budget(steps)`. Initialization runs only for fresh execution;
on restore, the requested budget is set before state and RNGs are restored.
Checkpoint boundaries are between complete advances, never inside arbitrary calls.

### Adapt an existing scientific library

Wrap foreign algorithms/environments behind these contracts. Translate inputs and
outputs, keep device and array conversions explicit, and checkpoint every state
needed for faithful continuation. Inject or derive foreign RNGs from the provided
stream and serialize their state. Global random calls or omitted wrapper state can
break replay even when the framework checkpoints correctly.

For GPU algorithms, EWS assigns the device through execution configuration as
described below. Use the worker-local device, normally `cuda:0`; do not select a
physical GPU or change `CUDA_VISIBLE_DEVICES` in scientific code. With the default
checkpoint backend, convert GPU tensors and framework RNG state to supported
NumPy arrays and structured values. A native checkpoint backend can instead
handle framework values directly. In either case restore onto the local device,
not the original physical GPU. Backend choice is independent of GPU allocation.

The optional Gymnasium adapter is selected as follows; this is a run-group fragment,
not a complete study:

```yaml
protocol: {type: rl}
data:
  type: gymnasium
  params:
    factory: gymnasium:make
    env_params: {id: CartPole-v1}
    state_adapter: experiment_code.data:CartPoleState
```

The paper supplies `CartPoleState`, with a zero-argument constructor,
`snapshot(env) -> state`, and `restore(env, state)`. It must cover environment,
wrappers, episode counters, and their RNGs. There is no universal Gym checkpoint
serializer. If faithful state capture is unavailable, do not claim resumability;
use whole episodes/trials as explicit restartable units or another supported model.

## Seeds, budgets, and recording

The experiment seed supplies separate streams for instance generation, data,
algorithm, and protocol. A component may override its root with YAML `seed`.
Streams stay stable across worker counts, scheduling, grid order, budgets, and
recording frequency. Algorithms can share a saved instance while producing
different trajectories if their actions affect the environment.

`budget.steps` is execution length. A horizon used by a learning rule belongs in
its scientific parameters. To continue at a larger budget, the algorithm,
generator, and protocol must all declare `supports_extension = True` and save
sufficient state. Otherwise a changed budget selects a new variant.

`recording.fields` selects measurements; `null` saves all, and `step` is automatic.
Record every step when a metric needs a complete trajectory. A sparse recording
cannot later supply missing actions or rewards. Recorded values need stable
numerical dtypes and shapes. Checkpoints occur between complete protocol steps;
an offline fit or trial restarts its step if interrupted inside the call.

## Choose checkpoint representation when needed

Ordinary NumPy studies, including the runnable example, need no backend
configuration. The default `NumPyCheckpointBackend` saves explicit JSON state
and NPZ numerical arrays without pickle. `execution.compression` controls result
arrays and the default checkpoint compression. To choose checkpoint compression
separately, use `checkpoint_backend: {type: numpy, params: {compression: true}}`
inside `execution`.

For native framework state or a representation split across files, select a
paper-owned backend:

```yaml
execution:
  checkpoint_backend:
    type: experiment_code.checkpointing:MyBackend
    params: {}
```

The public `CheckpointBackend` protocol and `NumPyCheckpointBackend` class are
importable from `experiments_wo_stress.storage`. Inheritance is optional. A custom
backend constructor receives only configured keyword `params`, without an RNG.
Implement `save(state, directory)` and `load(directory, metadata)` as follows:

- `state` is the complete logical mapping with `algorithm`, `data`, `protocol`,
  and `rngs` entries. The first three values come from the components' state
  methods; `rngs` contains every injected stream's snapshot. Preserve all parts
  without changing the live state.
- `save` writes its representation under the supplied fresh generation directory
  and returns finite JSON-compatible metadata with string keys. Metadata should
  include a representation version. Multiple files and nested directories are
  supported. Do not create symbolic links, write outside the generation, or use
  the reserved root filename `checkpoint.json`. Finish all work and flush and
  close handles before return; payloads are immutable afterward.
- `load` receives the saved metadata and returns the complete logical state
  without modifying files. It must support its older representation versions.
  Native GPU values must map to the assigned local device. EWS has no framework
  dependency and supplies no universal tensor serializer.

EWS owns checkpoint timing, complete-step boundaries, checksums for every payload,
file synchronization, atomic progress publication, fallback, and retention. A
backend does not manage result chunks or progress metadata. Native serialization
can avoid compulsory conversion of a large network to NumPy; actual memory and
I/O behavior depends on the backend and should be measured.

Every generation records its backend type, constructor parameters, source
fingerprints, and representation metadata. Restore uses that saved descriptor;
the current configuration selects the writer for subsequent generations. Keep
previous backend modules and dependencies available. If no retained generation
can be decoded, missing or incompatible loaders block continuation while
preserving saved artifacts.
Source fingerprints identify the writer for diagnosis; changes alone do not
reject its payload. The loader owns representation-version compatibility.
Legacy NumPy checkpoints remain readable, subject to normal provenance checks;
an EWS upgrade may still select a new variant, without automatic migration.

Backend configuration and separately tracked source are operational provenance,
excluded from run IDs, injected RNG streams, and stored variant selection. A
backend can declare module-relative `dependency_files` for source diagnostics. Keep
backends in separate modules from scientific components so an implementation
edit does not also change tracked scientific source. Recorded results and saved
instances remain numerical; analysis does not need to import checkpoint backends.
Test direct versus resumed execution using the actual backend, including all
component and framework RNG state, damaged payload recovery, and changed writers.

## CPU and GPU workers

Ordinary CPU experiments, including the runnable example above, require no GPU
configuration. With `execution.gpu_ids` omitted, one worker runs in the current
process and multiple workers use spawned processes. EWS leaves existing device
visibility unchanged in this mode.

For a study using CUDA, add this fragment:

```yaml
execution:
  workers: 2
  gpu_ids: [0, 1]
```

`gpu_ids` must be a nonempty list of distinct nonnegative integers; booleans,
strings, `null`, negatives, and duplicates are invalid. Worker count must equal
the number of IDs, including CLI `--workers` and Python `workers=` overrides.
For one GPU use `workers: 1` and `gpu_ids: [0]`. GPU execution always spawns a
fresh process per configured device, including that single-worker case. Python
entry scripts need an `if __name__ == "__main__":` guard for every process mode.

IDs refer to devices in the host or container runtime's GPU namespace before
CUDA masking. EWS requires `CUDA_VISIBLE_DEVICES` to be unset before GPU execution;
an existing value, including an empty string, is rejected. Each worker inherits
its one-device mask before the spawned interpreter imports the main script or
study components. Execution planning and component/source inspection also run
in a GPU worker. A worker keeps its assignment across runs, and the coordinator's
environment is restored after launch. The algorithm sees only the local device,
normally `cuda:0`, so it needs no physical-device knowledge or scheduling code.
This covers EWS execution imports, not study code imported earlier by the caller
or a custom planner loaded by `ews count-runs`. EWS temporarily adjusts
the parent environment during process launch; embedding applications should
avoid concurrent environment changes, GPU initialization, or unrelated
subprocess launches while workers start.

GPU assignments are operational settings recorded in resolved configuration,
provenance, and logs. They do not change scientific run IDs, injected RNG streams,
or stored variant selection. The study must seed and checkpoint any framework
RNGs; bitwise equality across different hardware or frameworks is not guaranteed.
EWS has no GPU-framework dependency and does not check available hardware. Device
exclusivity applies only to workers in one invocation; arrange disjoint devices
for concurrent experiments. GPU memory packing, shared or fractional GPUs,
multi-GPU training, and distributed scheduling are outside this mode.

## Metrics, figures, and custom planning

A metric receives configured `params` and implements
`compute(result) -> MetricResult(x, values)`. `RunResult` exposes recorded fields,
`instance`, `spec`, `completed_steps`, `revision`, and `final_outputs`. The latter
contains final recorded values for single-step offline and trial runs. Declare
`required_fields`, `required_instance_fields`, or `requires_complete_trajectory`
when the calculation needs them.

Built-in metrics include `field`, `cumulative_sum`, `pseudo_regret`, and
`realized_regret`. Pseudo-regret uses saved actions and instance `means`, with a
`dynamic` or `best_fixed` comparator. Realized regret instead requires saved
counterfactual rewards for every arm; means alone cannot determine it. Inspect the
paper's comparator and what data it requires before choosing a metric. The regret
aliases refer to supplied bandit metrics, not a general analysis rule; implement
a paper-specific metric when their definitions do not match.

`repeated_runs` averages compatible independent repetitions. Include varying
scientific parameters in `group_by`; coordinates must align. Uncertainty may be
`standard_error`, sample `std`, or `none`. One repetition has undefined sample
uncertainty, and a standard-error band is not a confidence interval. Line figures
can group curves by `color` and panels by `panel`; PDF/JPG need the `plot` extra,
while TikZ export does not require LaTeX until compilation. A custom plotter uses
`type: experiment_code.figures:MyPlotter` and implements
`plot(summaries, figure, output_dir) -> Iterable[Path]`.
Each summary provides its metric, group labels, coordinates, mean, uncertainty,
repetition count, and uncertainty convention.

Grid axes under `algorithm.params.*`, `data.params.*`, or `protocol.params.*`
combine with algorithms and repetitions. Separate groups when algorithms need
different sweeps. A custom planner selected with
`planner: experiment_code.planning:MyPlanner` implements
`plan(group, seed) -> Iterable[RunSpec]`. Use the public `make_run_spec` helper
with `group`, `repetition`, `algorithm_name`, `algorithm`, `data`, `protocol`,
`seed`, and optional `budget_steps`. Keep planning deterministic.

## Run, reuse, inspect, and clean

```bash
ews run experiment.yml --output outputs/study --workers 2
```

`run` executes or resumes the study, then produces configured analysis and
figures only if there are no failed, paused, or pending runs. Rerun the same
command after interruption. `analyze` and `plot` work independently from saved
results, so metrics and figures can change without repeating expensive
simulations; `plot` computes or reuses the analysis it needs. Compatible completed
work is reused, while changes to scientific settings, code, tracked inputs, or recording
retain separate variants. The active request selects the results for analysis.
Declare external inputs in YAML component `dependencies` or class
`dependency_files` so their changes are tracked.

With no analysis configuration, `run` performs execution only. A metrics-only
configuration produces summaries; configured figures are exported after their
required analysis. For saved artifacts, choose any of these independent
operations as needed:

| Command | Use |
| --- | --- |
| `ews analyze experiment.yml --output outputs/study` | Recompute or reuse metrics and summaries. |
| `ews plot experiment.yml --output outputs/study` | Regenerate figures from saved data. |
| `ews inspect outputs/study` | Read saved status and validate completed artifacts. |
| `ews clean outputs/study --scope inactive` | Preview cleanup of inactive variants. |

The six public commands are `count-runs`, `run`, `analyze`, `plot`, `inspect`, and
`clean`. The optional `ews count-runs experiment.yml` validates configuration and
returns only the study name and number of runs. It does not execute the study or
require an output directory, and never needs to precede `run`. From Python,
`run_experiment` performs execution;
call the separate `analyze` or `plot` API when derived outputs are wanted.

Cleanup is a preview until `--yes` is supplied. Scopes are `analysis`,
`checkpoints`, `inactive`, `runs`, and `all`. Removing checkpoints loses the
ability to continue those runs, although saved observations remain. Keep source,
inputs, and generated output in separate locations.

For a small cloud deployment, run the same study on one VM with persistent local
or block storage and a pinned environment. Begin and resume execution in the
environment where compatibility will be checked; a container image alone does
not guarantee exact checkpoint reuse across hosts. Notifications are optional
and use a webhook supplied through an environment variable.

## Verify direct execution against recovery

Copy `verify.py` beside the other files and run `python verify.py`. The main guard
is required for process workers. This check performs real simulations, pauses,
resumes with two workers, and compares every saved field and the generated means.

<!-- file: verify.py -->
```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments_wo_stress import load_config, run_experiment
from experiments_wo_stress.storage import iter_completed_runs


def run_check():
    config = load_config(Path(__file__).with_name("experiment.yml"))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        direct, resumed = root / "direct", root / "resumed"
        report = run_experiment(config, direct, workers=1)
        assert report.completed == 12 and report.failed == 0, report.to_dict()
        report = run_experiment(config, resumed, max_steps=73)
        assert report.paused == 12 and report.failed == 0, report.to_dict()
        report = run_experiment(config, resumed, workers=2)
        assert report.completed == 12 and report.failed == 0, report.to_dict()
        expected = {spec.run_id: result for spec, result in iter_completed_runs(direct)}
        actual = {spec.run_id: result for spec, result in iter_completed_runs(resumed)}
        assert expected.keys() == actual.keys()
        for run_id, result in expected.items():
            assert result.keys() == actual[run_id].keys()
            for field in result:
                np.testing.assert_array_equal(result[field], actual[run_id][field])
            np.testing.assert_array_equal(
                result.instance.arrays["means"], actual[run_id].instance.arrays["means"]
            )
        print("Direct and resumed results match for all 12 runs.")


if __name__ == "__main__":
    run_check()
```

For a paper implementation, also test scientific edge cases, compatible budget
extension against fresh execution, invalid input handling, and analysis after
simulation modules are unavailable. Check foreign-library adapters against an
uninterrupted trajectory. Inspect figures and confirm the paper's axes, comparator,
units, and uncertainty definition. Pin dependencies and document unresolved
scientific details before running an expensive study.
