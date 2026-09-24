# Experiments W/O Stress

Repeatable numerical experiments, from a YAML study to publication figures.

Define algorithms and scientific interactions in ordinary Python. Experiments W/O
Stress manages independent runs, generated instances, checkpointing, saved
observations, and analysis. The intended scale is a laptop or modest multicore
machine.

**Status: experimental implementation.** The examples and tests exercise the full
workflow; the public API still needs validation in a real paper project.

## What it provides

- A common interface for synthetic data, stored CSV datasets, and evolving
  environments, with online, offline, callable-trial, and reinforcement-learning
  interaction helpers.
- Parameter grids, independent repetitions, and custom run planners.
- Independent instance, algorithm, environment, and protocol RNGs, reproducible
  across execution order and worker counts.
- Saved instances, selected numerical observations, rotating run logs, and recovery
  from committed checkpoints.
- Continued execution at larger budgets when components support it, with previous
  requests and recording variants retained.
- Metrics computed from observations and instances, cached aggregation, and PDF,
  JPG, and editable TikZ/PGFPlots figures.
- Reusable bandit settings and an optional Gymnasium adapter.

## Install

Python 3.10 or newer is required. From this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[plot]'
```

On Windows, activate with `.venv\Scripts\activate`. The core needs NumPy and
PyYAML. The `plot` extra supplies Matplotlib for PDF/JPG; TikZ source export needs
neither Matplotlib nor LaTeX. The `gym` extra installs Gymnasium for its adapter.

## Build a study

The [sequential example](examples/sequential_study/README.md) compares UCB and
epsilon-greedy on three Gaussian bandit sizes, with 20 repetitions per combination.
It saves arm means as an instance and records actions and rewards. Regret is
computed afterward by a metric, independently of the simulation.

```bash
ews plan examples/sequential_study/experiment.yml
ews build examples/sequential_study/experiment.yml --output outputs/bandits --workers 2
ews inspect outputs/bandits
```

`build` executes the selected runs and produces the analysis configured in YAML.
Figures are in `outputs/bandits/analysis/figures/`. Repeating the command validates
and reuses available execution and analysis artifacts.

For deliberate interruption and resumption:

```bash
ews run examples/sequential_study/experiment.yml --output outputs/resume-demo --max-steps 250
ews run examples/sequential_study/experiment.yml --output outputs/resume-demo --workers 2
ews plot examples/sequential_study/experiment.yml --output outputs/resume-demo
```

Increase a group's `budget.steps` to continue compatible components from their saved
final state. A different recording selection retains another variant. Scientific
or implementation changes select affected work again while preserving existing
artifacts. The active request tells analysis which results to use.

The [CSV example](examples/offline_csv/README.md) demonstrates the same generator
interface with a stored dataset and an offline estimator.

## Python API

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

The main guard is needed when starting worker processes. Use one worker for a
simple sequential session. Paper components receive their RNGs explicitly and can
write diagnostics through the logger attached to each run component.

## Commands

| Command | Purpose |
| --- | --- |
| `ews plan CONFIG` | Validate configuration and display the run plan. |
| `ews build CONFIG --output DIR` | Execute and produce configured analysis. |
| `ews run CONFIG --output DIR` | Execute or resume; accepts `--workers` and `--max-steps`. |
| `ews inspect DIR` | Inspect active work and stored result validity. |
| `ews analyze CONFIG --output DIR` | Compute or reuse metric and aggregate artifacts. |
| `ews plot CONFIG --output DIR` | Compute or reuse analysis and configured figures. |
| `ews clean DIR --scope inactive` | Preview cleanup of variants outside the active request. |

Cleanup is a dry run unless `--yes` is supplied. Scopes include analysis caches,
checkpoints, inactive variants, selected runs, or all artifacts. Removing state
prevents continuing those runs, even if their numerical results remain available.

Recording controls which future analyses are possible. A cumulative metric needs
the complete underlying trajectory; sparse observations cannot recover missing
actions or rewards. Checkpoints occur between protocol steps, so a single long
`fit()` or trial needs incremental support to resume within that operation.

## Documentation and development

- [Configuration and extension guide](docs/CONFIGURATION.md)
- [Architecture and artifact contracts](docs/ARCHITECTURE.md)
- [Project goals](docs/PROJECT_BRIEF.md)
- [Development workflow](docs/BUILD_WORKFLOW.md)
- [Contributing](CONTRIBUTING.md)

Keep code, documentation, and executable examples consistent. Run
`python scripts/check_docs.py --examples --cli` alongside the relevant tests;
this verifies mechanical references and supported examples, while review verifies
scientific meaning. Generated experiments, caches, and build products stay out of
version control.
