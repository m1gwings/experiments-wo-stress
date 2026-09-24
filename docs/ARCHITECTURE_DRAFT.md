# Architecture draft

Status: proposal for discussion, 2026-09-24. Names and signatures below are illustrative, not frozen API.

## Boundary between library and paper

The library owns orchestration and artifacts. The paper repository owns scientific choices: instances/data generators, algorithms, feedback models, metrics, and figure definitions. The library must not need to import a particular paper package at installation time.

## Two levels of experiment authoring

### General trial interface

The minimal unit is a Python callable that receives an explicit job specification, its own random generator(s), and a way to record metrics; it returns numerical results. This admits noninteractive Monte Carlo studies, offline evaluations, and custom loops.

### Optional interactive helper

For sequential learning, offer small protocols such as `Environment.reset`, `Environment.step(action)`, `Algorithm.reset`, `Algorithm.act`, and `Algorithm.observe`. A feedback model may live inside the environment or be a separate component if the use case requires it. A runner can drive this loop and collect metrics. The exact methods and ownership of randomness need agreement through the first example.

The general trial stays available even if a study does not fit the interactive helper.

## Components

| Component | Responsibility | Initial direction |
| --- | --- | --- |
| Config | Parse, validate, normalize, and snapshot a study | TOML as a candidate; paper-specific parameters remain typed or validated by paper code |
| Planner | Expand a study into stable job identities | One identity per parameter setting × instance/repetition × algorithm; explicitly decide whether competing algorithms share an instance |
| Randomness | Give each job/component a deterministic random stream | Stable seed derivation independent of scheduling; record actual seeds and RNG state where needed |
| Runner | Execute pending jobs and report progress | Sequential first; bounded local process parallelism next |
| Storage | Save outputs and status safely | Immutable completed-job artifacts; temporary write and atomic commit; validate before skip |
| Aggregation | Combine per-job numerical metrics | Keep raw observations, distinguish trials from time steps, make uncertainty convention explicit |
| Figures | Render curves from stored results | PDF and JPG; editable TikZ/PGFPlots for a defined subset of figures; plotting dependency optional |
| CLI | Run, resume, inspect, and plot | Thin commands calling the same Python API |

## Proposed data flow

`study configuration → validated jobs → isolated trials → per-job artifacts → aggregation → figures`

The stored data must be sufficient to redraw figures independently. Each completed job carries a schema version, effective configuration fingerprint, job identity, seed information, result arrays, and provenance (library version and paper code revision when available). Human-readable metadata plus a NumPy array format is one candidate; avoid pickled algorithm instances as a default artifact format.

## Resume semantics

The first implementation should guarantee **job-level resume**: a successfully committed job is skipped, and missing or incomplete jobs rerun. This handles interrupted sweeps and repetitions. Checkpointing *within* a very long sequential trial requires a separate optional contract for serializing the algorithm/environment state and RNG state; it should be designed from a concrete need, because a generic checkpoint of arbitrary Python objects is brittle.

An output location belongs to a specific effective configuration and experiment implementation. When a job's identity or schema differs, the runner reports a mismatch and does not silently consume old output. Failures preserve error information and allow an explicit retry.

## Scale path

Keep jobs independent so the same plan can run sequentially or on local processes. Do not let the worker index or completion order affect seeds, result names, or numerical aggregation. Support streaming/progress recording for long trajectories without requiring all metrics in memory; decide the first storage representation from the acceptance example.

## Candidate repository layout

```text
pyproject.toml
README.md
docs/
  PROJECT_BRIEF.md
  ARCHITECTURE_DRAFT.md
  BUILD_WORKFLOW.md
src/<package_name>/
  config.py
  jobs.py
  rng.py
  runner.py
  storage.py
  results.py
  plotting.py
  cli.py
examples/
  sequential_study/
tests/
```

These modules are logical responsibilities, not a demand to create one module per row immediately. A narrow vertical slice may start with fewer files.

## Open architecture questions

1. Which first real paper experiment should shape the API? We need its instance/algorithm/feedback loop and desired figure.
2. Should algorithms on the same instance share an environment seed (common random numbers), and which stochastic effects should remain independent?
3. Is a general trial callable enough for v0, or should the sequential protocols ship immediately with the first example?
4. Do configurations need only TOML plus Python definitions, or YAML and command-line overrides too?
5. What minimum checkpoint frequency is required for a single long trial, beyond job-level resume?
6. Which figure styles need editable TikZ output in v0 (lines, confidence bands, log axes, facets)?
