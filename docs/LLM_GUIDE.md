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

No paper-specific facts are supplied by this guide. Never describe its illustrative
bandit algorithm as an implementation of an unseen paper.

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

On Windows, activate with `.venv\Scripts\activate`. NumPy and PyYAML are core
dependencies. For PDF/JPG, change the requirement to
`experiments-wo-stress[plot] @ git+...`; for Gymnasium, add the `gym` extra.
Use the complete Git URL above in place of `git+...`. TikZ export needs neither
Matplotlib nor LaTeX; compiling the exported document needs LaTeX.

A minimal external repository can look like this:

```text
paper-study/
  README.md                  paper references, assumptions, run instructions
  requirements.lock.txt      exact library commit and dependency versions
  algorithms.py              learning rules and their checkpoint state
  environments.py            custom generators/adapters when needed
  metrics.py                 paper metrics, independent of simulation modules
  experiment.yml             scientific choices and execution settings
  verify.py                  repeatability and recovery checks
  outputs/                   generated artifacts, excluded from Git
```

Keep the Python modules beside the YAML file, or install the research package.
The YAML directory is an import context; `algorithms:MyAlgorithm` names an ordinary
class. Keep simulation and metrics in different modules: source fingerprints cover
whole module files and their base-class modules. A metric edit in an algorithm's
module also invalidates that algorithm's simulation variant. Restart Python or
explicitly reload modules after editing already imported source.

## A runnable API example

Copy the following three files beside each other. The scientific code needs only
NumPy in addition to the library. It compares two exploration probabilities on two
Gaussian bandit sizes, with three independent repetitions: 12 short runs. These
are illustrative settings, not paper-derived recommendations.

`algorithms.py`:

<!-- file: algorithms.py -->
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

`metrics.py`:

<!-- file: metrics.py -->
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
        type: algorithms:EpsilonGreedy
        params: {epsilon: 0.1}
      - name: epsilon_030
        type: algorithms:EpsilonGreedy
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
    - {name: mean_reward, type: 'metrics:MeanReward'}
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
ews plan experiment.yml
ews build experiment.yml --output outputs/study --workers 2
ews inspect outputs/study
```

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
Supported values are numerical NumPy arrays, scalars, lists, tuples, and mappings
with string keys. Do not return live objects, handles, or values requiring pickle.
The library stores its injected RNG states separately. Truly stateless components
may inherit `StateMixin`; stateful subclasses must override both state methods.

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
- **Callable trial:** use `protocol.type: trial`, with `params.function: trials:run`
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

The optional Gymnasium adapter is selected as follows; this is a run-group fragment,
not a complete study:

```yaml
protocol: {type: rl}
data:
  type: gymnasium
  params:
    factory: gymnasium:make
    env_params: {id: CartPole-v1}
    state_adapter: environments:CartPoleState
```

The paper supplies `CartPoleState`, with a zero-argument constructor,
`snapshot(env) -> state`, and `restore(env, state)`. It must cover environment,
wrappers, episode counters, and their RNGs. There is no universal Gym checkpoint
serializer. If faithful state capture is unavailable, do not claim resumability;
use whole episodes/trials as explicit restartable units or another supported model.

## Seeds, budgets, and recording

The experiment seed creates four separated streams: instance, data, algorithm,
and protocol. Each component may override its root with YAML `seed`. Streams are
stable across workers, scheduling, grid order, budgets, and recording frequency.
Instance/data identities exclude algorithm choice, permitting shared problem
instances across algorithms. Equal seeds alone do not force equal trajectories
when actions affect the environment.

`budget.steps` is execution length. A paper's horizon used inside a learning rule
belongs in scientific parameters, for example `algorithm.params.design_horizon`.
Do not confuse those two values. Budget continuation requires all three runtime
components to declare `supports_extension = True` and behave consistently when
continued. Otherwise a changed budget selects a new variant. The example's policy
is independent of budget, so continuation is appropriate.

`recording.fields` selects measurements; `null` saves all. `step` is automatic.
Values must maintain a numerical dtype and shape. Record every step when a metric
needs a full trajectory. Extendable sparse runs record scheduled multiples only:
an interval of 10 at budget 23 saves steps 10 and 20. Missing actions cannot be
recovered by summing a subsample or by regenerating a figure.

Buffers default to 16 MiB per active run, plus scientific state and serialization.
Results flush by byte budget, before checkpoints, and at completion. Checkpoints
default to 120 seconds; `checkpoint_steps` adds a step trigger. Compression is
optional and defaults to false. Rolling checkpoints are retained in addition to
completed-budget endpoint snapshots. Dense summary CSV/cache/export copies can
consume more disk than raw trajectories.

## Metrics, figures, and custom planning

A metric constructor receives only its configured `params`, and
`compute(result) -> MetricResult(x, values)` returns a finite real scalar or
matching one-dimensional arrays. `RunResult` is a mapping of recorded arrays with
attributes `instance`, `spec`, `completed_steps`, `revision`, and `final_outputs`.
Final outputs expose the final recorded fields of single-step offline/trial runs;
there is no arbitrary final-output hook for multistep algorithms.

Declare `required_fields`, `required_instance_fields`, and
`requires_complete_trajectory` as needed. Built-ins include `field`,
`cumulative_sum`, `pseudo_regret`, and `realized_regret`. Pseudo-regret requires
complete actions and instance `means` of shape arms or time-by-arms. Its `dynamic`
comparator uses the best arm each round; `best_fixed` uses the best fixed arm over
each prefix. Realized regret instead requires saved counterfactual rewards for all
arms, consistent with the observed rewards; means alone cannot determine it.

`repeated_runs` groups compatible results and averages independent repetitions.
Include varying scientific parameters in `group_by`. Coordinates must align;
duplicate repetitions and incompatible pooling are rejected. Uncertainty is
`standard_error`, sample `std`, or `none`. One repetition has undefined sample
uncertainty. A standard-error band is not a confidence interval.

Line figures accept `metric`, `color`, `panel`, `panel_label`, `xlabel`, `ylabel`,
`title`, `formats`, and `xscale`/`yscale` (`linear` or `log`). Custom plotters use
`type: figures:MyPlotter`, configured `params`, and
`plot(summaries, figure, output_dir) -> Iterable[Path]`. Summary attributes include
`metric`, `labels`, `x`, `mean`, `uncertainty`, `count`, and `uncertainty_kind`.

Grid paths are `data.params.*`, `algorithm.params.*`, and `protocol.params.*`,
expanded as a Cartesian product with algorithms and repetitions. Separate groups
when algorithms need different sweep parameters. A custom planner has a zero-arg
constructor and `plan(group, seed) -> Iterable[RunSpec]`; select it with
`planner: planning:MyPlanner`. Use the public `make_run_spec` helper with
`group`, `repetition`, `algorithm_name`, `algorithm`, `data`, `protocol`, `seed`,
and optional `budget_steps`; component arguments are `ComponentSpec` or mappings.
Preserve the supplied group and seed and keep planning deterministic.

## Run, reuse, inspect, and clean

```bash
ews plan experiment.yml
ews run experiment.yml --output outputs/study --workers 2
ews build experiment.yml --output outputs/study --workers 2
ews analyze experiment.yml --output outputs/study
ews plot experiment.yml --output outputs/study
ews inspect outputs/study
ews clean outputs/study --scope inactive
```

`build` runs execution and configured analysis together. Execution returns counts
of completed, skipped, paused, failed, and pending runs. Failed runs have inspectable
errors. Increasing a compatible budget reuses saved final state; a smaller request
selects a recorded prefix. Scientific, code, environment, input, and recording
changes select retained variants rather than silently mixing results. The active
request identifies which variants and boundaries analysis uses.

Artifacts include request metadata, immutable instances, per-run chunks,
checkpoints, progress, rotating logs, and analysis. Completed work is validated
before reuse; corrupt checkpoints can fall back to a preceding valid generation.
Recovery removes uncommitted observations before replay. A forced termination can
lose work since the previous commit. Stopping normally saves at the next safe step.

Metrics, aggregates, and figures have separate validated caches. Figure changes
need not rerun simulation or unaffected metrics. YAML component `dependencies`
are paths relative to the configuration. Class `dependency_files` are relative to
the class module; declare helper code and external inputs the framework cannot
infer. Untracked external-library changes can invalidate scientific reproducibility
even when declared cache inputs are unchanged.

Cleanup defaults to a preview. Add `--yes` only for intended deletion. Scopes are
`analysis`, `checkpoints`, `inactive`, `runs`, and `all`; repeat `--run-id ID` to
restrict `runs` or `checkpoints`. Removing checkpoints loses continuation state;
analysis artifacts can be rebuilt from retained observations and instances.
Execution and cleanup share a lock; analysis does not, so clean while analysis is
idle. This is local execution/storage, not a distributed scheduler.

## A small cloud machine and optional Discord updates

A cloud VM can run the same commands as a laptop. Use persistent local/block
storage with ordinary filesystem locking and atomic replacement, a pinned software
environment, and a process/session manager so an SSH disconnect does not end the
job. Start with one worker; add workers only within the VM's CPU and memory budget.
An object-store URL is not an output directory. Copy completed or quiescent whole
artifact directories to external storage for backup.

Exact portability is narrower than file portability. Provenance includes Python,
NumPy, PyYAML, platform identity, code, and tracked input paths/content. Moving to
a different VM or path layout can select new variants instead of resuming old
ones. Do not promise arbitrary cross-host exact continuation; plan cloud runs in
their intended environment and validate any migration.

Notifications are optional. Add this root YAML section only when requested:

```yaml
notifications:
  discord:
    enabled: true
    webhook_env: EWS_DISCORD_WEBHOOK_URL
    interval_seconds: 300
    timeout_seconds: 5
```

Supply the webhook through that environment variable using the host's secret
configuration. Do not place its URL in YAML, source, logs, or the repository.
Omitting the section or setting `enabled: false` disables notifications without
reading the variable. An enabled configuration with a missing or invalid webhook
fails before execution. HTTP delivery failures do not fail scientific runs.

Updates cover the execution stage of `run` and `build`: start, periodic summaries,
and completion, pause, failure, or abort. Separate analysis/plotting is not notified.
Periodic progress uses counts and recent durable checkpoints, not continuously
sampled in-memory step counts. Payloads omit parameters, paths, numerical data,
tracebacks, and secrets. Delivery is best effort, including the final message;
rate limits or outages can prevent it. Persisted artifacts remain authoritative.
Notification settings do not change simulation identities or seeds.

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
