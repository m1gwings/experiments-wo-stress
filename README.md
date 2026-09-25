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
  random streams, one worker or a small process pool.
- **Checkpoints and reuse.** Resume interrupted work and reuse compatible runs
  and analysis when you return to a study.
- **Several interaction styles.** Use built-in online, offline, trial, or
  reinforcement-learning protocols with your own scientific components. Bandit,
  CSV, and Gymnasium data helpers are available.
- **Analysis after execution.** Compute metrics from saved observations and
  instances, aggregate repetitions, and export PDF, JPG, or editable TikZ
  figures.
- **Operational tools.** Plan and inspect runs from the CLI, receive optional
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

From the repository root, inspect the plan, execute the study, and inspect its
output:

```bash
ews plan examples/sequential_study/experiment.yml
ews build examples/sequential_study/experiment.yml --output outputs/bandits --workers 2
ews inspect outputs/bandits
```

`plan` validates and expands the study without running it. `build` runs the
simulations and produces the configured analysis and figures in
`outputs/bandits/analysis/figures/`. Compatible work is reused on a later build.
Use `ews run` when you want execution alone, then `ews analyze` or `ews plot` to
work with saved results. The [configuration guide](docs/CONFIGURATION.md)
explains recording, budgets, and reuse in detail.

The [offline CSV example](examples/offline_csv/README.md) uses a stored dataset
and performs one estimator fit per run.

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

The same study can be run from Python:

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

The main guard is needed when starting multiple worker processes.

## Commands

| Command | Purpose |
| --- | --- |
| `ews plan CONFIG` | Validate the configuration and show its runs. |
| `ews build CONFIG --output DIR` | Execute the study and produce configured analysis. |
| `ews run CONFIG --output DIR` | Execute or resume runs. |
| `ews inspect DIR` | Inspect stored status and artifacts. |
| `ews analyze CONFIG --output DIR` | Compute or reuse metrics and summaries. |
| `ews plot CONFIG --output DIR` | Produce configured figures from analysis. |
| `ews clean DIR --scope inactive` | Preview cleanup of retained artifacts. |

## Documentation and development

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
