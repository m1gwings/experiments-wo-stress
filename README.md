# Experiments W/O Stress

*Experiments W/O Stress* is a Python library designed to ease the process of
running numerical experiments for Machine Learning papers and producing plots.
The rationale is simple: you specify the baselines, the data-generating
mechanism, and the interaction protocol; the library takes care of the rest,
providing stable **YAML configurations**, efficient **checkpoint management**,
**parallel execution** over a pool of workers, **reproducibility** through
explicit seeding of each component, *etc.*.

## What it provides

- **A study in YAML.** Define algorithms, parameter grids, repetitions,
  recording, metrics, and figures in a version-controlled configuration.
- **Reproducible local execution.** Run independent simulations with explicit
  random streams, one worker or a small process pool, with optional exclusive
  GPU assignments within an invocation.
- **Checkpoints and reuse.** Resume interrupted work and reuse compatible runs
  and analysis when you return to a study. NumPy checkpoints work by default;
  custom backends can serialize native framework state or split large payloads.
- **Several interaction styles.** Use built-in online, offline, trial, or
  reinforcement-learning protocols with your own scientific components. Bandit,
  CSV, and Gymnasium data helpers are available.
- **Analysis after execution.** Compute metrics from saved observations and
  instances, aggregate repetitions, and export PDF, JPG, or editable TikZ
  figures.
- **Operational tools.** Count and inspect runs from the CLI, receive optional
  Discord progress messages, and use the same workflow on a Linux VM.

## Install

Python 3.10 or newer is required. Install the library from its public Git
repository:

```bash
python -m pip install 'experiments-wo-stress[plot] @ git+https://github.com/m1gwings/experiments-wo-stress.git@main'
```

Replace `main` with a full commit hash for a reproducible study. Git must be
available during installation. The package name uses hyphens; Python imports use
underscores (`import experiments_wo_stress`). Installation also provides the
`ews` command.

## A quick-starter

The [sequential example](examples/sequential_study/README.md) compares UCB and
epsilon-greedy on Gaussian bandits with 10, 20, and 50 arms. Its configuration
selects the scientific components and describes how to run and analyze them:

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

execution: {workers: 1}
recording: {every_steps: 1, fields: [action, reward]}
analysis:
  metrics:
    - {name: regret, type: pseudo_regret}
  aggregator:
    group_by: [data.params.n_arms, algorithm.name]
    uncertainty: standard_error
  figures:
    - type: line
      metric: regret
      color: algorithm.name
      panel: data.params.n_arms
      formats: [pdf, jpg, tikz]
```

The grid combines three bandit sizes with two algorithms and 20 independent
repetitions, producing 120 runs. The classes under `experiment_code/` define the
bandit and learning rules; the library handles execution and storage. Recorded
actions and rewards, together with the saved bandit instance, support regret
analysis afterward.

From the repository root, optionally check the run count, then run the study:

```bash
# Optional: check the run count before launching.
ews count-runs examples/sequential_study/experiment.yml
ews run examples/sequential_study/experiment.yml --output outputs/bandits --workers 2
```

`run` executes or resumes the simulations, then produces configured analysis
and figures once all requested runs have completed or been reused. Figures appear
in `outputs/bandits/analysis/figures/`. Failed, paused, or interrupted execution
leaves analysis for a later successful invocation. Compatible work is reused
when you run the command again.

`count-runs` validates configuration and returns only the study name and run count, here
`{"name": "gaussian_bandit_comparison", "runs": 120}`. Counts combine grid
combinations × algorithms × repetitions within each group, summed across groups;
they help estimate cost from a representative run. This check is never required
before `run`.

With no analysis configured, `run` performs execution only. A metrics-only
configuration produces summaries; configured figures are exported after their
required analysis completes.

Use `ews analyze` or `ews plot` to revise metrics or figures from saved data after
expensive simulations. Neither reruns the simulations; `plot` computes or reuses
the analysis it needs. The [configuration guide](docs/CONFIGURATION.md) explains
recording, budgets, and reuse in detail.

To inspect saved status without running anything:

```bash
ews inspect outputs/bandits
```

The [offline CSV example](examples/offline_csv/README.md) uses a stored dataset
and performs one estimator fit per run.

Ordinary CPU experiments require no GPU configuration. For GPU study components,
assign one GPU to each worker:

```yaml
execution:
  workers: 2
  gpu_ids: [0, 1]
```

Each GPU worker is a fresh spawned process, even with one worker. It sees only
its assigned GPU through `CUDA_VISIBLE_DEVICES`, established before study imports;
the algorithm uses its local device, normally `cuda:0`. Worker count must equal
the number of distinct GPU IDs, and `CUDA_VISIBLE_DEVICES` must be unset before
launch. EWS manages visibility without importing a GPU framework. See
[GPU execution](docs/CONFIGURATION.md#gpu-execution) for validation and scope.

Checkpoint serialization is also an execution setting. Ordinary NumPy studies
need no backend configuration. A paper can supply a backend when its model state
needs a native or multiple-file representation:

```yaml
execution:
  checkpoint_backend:
    type: experiment_code.checkpointing:MyBackend
    params: {}
```

The backend saves and loads logical state; EWS still owns complete-step
checkpoint boundaries, checksums, atomic publication, recovery, and retention.
See [checkpoint backends](docs/CONFIGURATION.md#checkpoint-backends) for the
interface and a small implementation example.

## Suggested setup

Keep the paper's scientific code and YAML in a separate repository, with this
library installed at a pinned revision:

```text
my-paper/
├── experiment.yml
├── experiment_code/
│   ├── __init__.py
│   ├── algorithms.py
│   ├── data.py
│   └── metrics.py
└── data/
```

Use only the modules your study needs. A component path such as
`experiment_code.algorithms:MyAlgorithm` refers to a class in that package.
See [the extension interfaces](docs/CONFIGURATION.md#instances-and-scientific-components)
and the [standalone LLM guide](docs/LLM_GUIDE.md) if you are implementing a
study from a paper PDF.

### Cloud execution

Run the same commands on a Linux VM with persistent storage for the output
directory. The [cloud guide](docs/CLOUD.md) walks through a pinned container
build, execution, and recovery on one VM.

### LLM-assisted implementation

The [standalone LLM guide](docs/LLM_GUIDE.md) can be given to an assistant
alongside a paper PDF. It includes a runnable study, the scientific component
contracts, and validation steps without requiring the rest of this repository.

## Discord notifications

Long runs can send optional progress summaries to a Discord webhook. Configure
the webhook through an environment variable; keep its URL out of YAML and source
control. See [Discord notifications](docs/CONFIGURATION.md#discord-notifications) for setup
and delivery behavior.

## Python API

From Python, call `run_experiment` for execution and `plot` for analysis and
figures:

```python
from experiments_wo_stress import load_config, run_experiment
from experiments_wo_stress.plotting import plot

if __name__ == "__main__":
    config = load_config("examples/sequential_study/experiment.yml")
    report = run_experiment(config, "outputs/bandits", workers=2)
    print(report.to_dict())
    if not report.failed and not report.pending and not report.paused:
        plot(config, "outputs/bandits")
```

The main guard is needed for multiple CPU workers and for any GPU execution,
including one GPU worker.

## Commands

| Command | Purpose |
| --- | --- |
| `ews count-runs CONFIG` | Optionally validate configuration and count runs without executing them. |
| `ews run CONFIG --output DIR` | Execute or resume, then produce configured analysis and figures after successful completion. |
| `ews analyze CONFIG --output DIR` | Compute or reuse metrics and summaries. |
| `ews plot CONFIG --output DIR` | Regenerate configured figures from saved results. |
| `ews inspect DIR` | Read stored status and validate completed artifacts without executing runs. |
| `ews clean DIR --scope inactive` | Preview cleanup of retained artifacts. |

## Documentation and development

- [Reading the code](docs/CODE_GUIDE.md) — where to start, how a run flows
  through the implementation, and which tests explain each behavior.
- [Configuration and extension guide](docs/CONFIGURATION.md) — YAML, component
  contracts, and user-facing rules.
- [Architecture](docs/ARCHITECTURE.md) — identities, artifacts, storage, and
  recovery.
- [Cloud guide](docs/CLOUD.md) — deployment of an existing study on one VM.
- [Standalone LLM guide](docs/LLM_GUIDE.md) — context for implementing a paper
  study.
- [Project brief](docs/PROJECT_BRIEF.md) — goals and scope.
- [Contributing](CONTRIBUTING.md) and
  [development workflow](docs/BUILD_WORKFLOW.md) — working on the library.
