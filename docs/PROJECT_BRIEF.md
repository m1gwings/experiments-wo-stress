# Project brief

## Purpose

Experiments W/O Stress helps researchers run numerical experiments for
theory-oriented machine learning papers. A researcher specifies instances,
algorithms, observable feedback, measurements, and figures. The library handles
the repeated infrastructure needed to execute and inspect the study.

The target is moderate simulations on a laptop or a modest multicore machine.
Efficiency matters: use bounded worker submission, buffered numerical storage,
explicit checkpoint state, and analysis that does not retain every repetition.

## Design criteria

1. **Reproducibility.** Record effective configuration, seeds, code provenance, and
   software versions. Equivalent runs produce the same results under the same code,
   inputs, and environment, including when worker count changes.
2. **Recoverability.** Skip validated completed runs, resume unfinished runs at safe
   protocol boundaries, and report incompatible or corrupt artifacts explicitly.
3. **Small scientific interface.** Ordinary Python classes implement the capabilities
   a protocol needs. A callable trial also supports a complete custom computation.
4. **Independent analysis.** Save numerical results so researchers can change metrics
   and figures without repeating simulation, provided the required observations exist.
5. **Proportionate infrastructure.** NumPy and PyYAML are core dependencies; Matplotlib
   is optional. Execution and storage remain local.
6. **Inspectability.** Configuration, run progress, numerical arrays, and tracebacks
   remain accessible through ordinary files and a small CLI.

## Current implementation

The experimental package supports explicit algorithms, stateful data generators,
offline/online/trial protocols, grids and custom planners, repeated runs, local
processes, checkpoints, and saved-data analysis. It exports PDF, JPG, and editable
TikZ/PGFPlots figures.

The [sequential study](../examples/sequential_study/README.md) compares two algorithms
on three synthetic Gaussian bandit sizes with 20 repetitions each. Tests verify
that interruption and resumption with a different worker count preserve complete
numerical results, and that figures regenerate without importing simulation code.
The [offline CSV study](../examples/offline_csv/README.md) verifies stored-data use
through the same generator interface.

These are acceptance examples, not a claim that the public API has been validated
in a published paper. The next product milestone is adoption in a genuine research
project, recording friction and simplifying the interface before declaring it stable.

## Researcher workflow

1. Install a pinned version or commit in the paper repository.
2. Define scientific components and explicit checkpoint state in Python.
3. Write YAML specifying parameter combinations, repetitions, recording, and analysis.
4. Execute locally; inspect, interrupt, and resume as needed.
5. Regenerate summary tables and publication figures from the saved results.

## Scope limits

Distributed scheduling, dashboards, hosted storage, a benchmark catalog, arbitrary
Python object serialization, and arbitrary checkpoints inside a function are outside
the initial scope. Default figures cover numerical curves, parameter panels, and
uncertainty; custom plotting remains an extension point.

The source lives in a standalone GitHub repository. A stable release, license
choice, and package-index publication remain separate decisions.
