# Experiments W/O Stress

Reproducible numerical experiments, from repeated runs to publication figures.

Define the algorithms, data sources, and scientific interaction in your paper's
Python code. Describe the study in YAML. Experiments W/O Stress handles deterministic
random streams, local execution, checkpoints, result storage, and analysis.

**Status: experimental initial implementation.** The examples and tests exercise
the complete workflow; the API still needs validation in a real paper project.
The intended scale is a laptop or a modest multicore machine.

## What it provides

- A shared data-generator interface for synthetic data, CSV datasets, and evolving
  environments; offline, online, and callable-trial protocols.
- Cartesian parameter grids, independent repetitions, and custom run planners.
- Separate reproducible RNGs for algorithm, data, and protocol, independent of
  worker count and execution order.
- Periodic checkpoints, validated completed outputs, and recovery after interruption.
- Buffered NumPy artifacts, inspectable metadata and failures, and optional compression.
- Metrics and repeated-run summaries computed from saved results, with PDF, JPG,
  and editable TikZ/PGFPlots figures.

## Install

Python 3.10 or newer is required. Install from this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[plot]'
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
The core package needs NumPy and PyYAML. The `plot` extra adds Matplotlib for
PDF/JPG export; generating TikZ source does not require Matplotlib or LaTeX.

## Run the example

The [sequential study](examples/sequential_study/README.md) compares UCB and
epsilon-greedy on three Gaussian bandit sizes, with 20 repetitions per algorithm
and size: 120 runs in total.

```bash
ews plan examples/sequential_study/experiment.yml
ews run examples/sequential_study/experiment.yml --output outputs/bandits --workers 2
ews inspect outputs/bandits
ews plot examples/sequential_study/experiment.yml --output outputs/bandits
```

Figures appear in `outputs/bandits/analysis/figures/`; summary tables are in
`outputs/bandits/analysis/`. Run the same `ews run` command again to validate and
skip completed runs. After an interruption, it continues unfinished runs from
their last valid checkpoints.

To try resumption deliberately, use a fresh output directory:

```bash
ews run examples/sequential_study/experiment.yml --output outputs/resume-demo --max-steps 250
ews run examples/sequential_study/experiment.yml --output outputs/resume-demo --workers 2
```

The [offline CSV example](examples/offline_csv/README.md) demonstrates the same
data interface for a stored dataset. Both examples keep their scientific code
outside the library.

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

The main guard is needed when starting local worker processes. Use `workers=1`
for a simple sequential session.

## Commands

| Command | Purpose |
| --- | --- |
| `ews plan CONFIG` | Validate the configuration and display the run plan. |
| `ews run CONFIG --output DIR` | Execute or resume; accepts `--workers` and `--max-steps`. |
| `ews inspect DIR` | Read progress and validate completed results. |
| `ews analyze CONFIG --output DIR` | Recompute metrics and summary tables from saved results. |
| `ews plot CONFIG --output DIR` | Recompute summaries and regenerate configured figures. |

Recording resolution is part of the scientific configuration. Sparse recording
cannot recover discarded observations later: accumulate cumulative quantities
during the simulation if their curves will be recorded sparsely. Checkpoints occur
between protocol steps; an indivisible `fit()` or trial cannot resume midway.

## Documentation and development

- [Configuration and custom components](docs/CONFIGURATION.md)
- [Architecture and persistence guarantees](docs/ARCHITECTURE.md)
- [Project goals and scope](docs/PROJECT_BRIEF.md)
- [Development workflow](docs/BUILD_WORKFLOW.md)
- [Contributing](CONTRIBUTING.md)

The source package is in `src/experiments_wo_stress/`, executable studies are in
`examples/`, and behavior tests are in `tests/`. Generated outputs are excluded
from version control.
