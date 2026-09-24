# Project guidance

## Purpose and status

Experiments W/O Stress is an experimental Python library for numerical experiments
in theory-oriented machine learning papers, targeting laptops and modest multicore
machines. Paper code supplies scientific components; the library supplies planning,
reproducible execution, persistence, aggregation, and figure export.

An initial end-to-end implementation exists. The documented API is the current
contract, with real-paper validation still needed before declaring it stable.

## Repository structure

- `src/experiments_wo_stress/`: package and `ews` CLI.
  - `config.py`, `jobs.py`, `rng.py`: validated YAML, run planning, stable random streams.
  - `components.py`, `protocols.py`, `data.py`: extension contracts and built-in helpers.
  - `runner.py`, `storage.py`: local execution, buffering, checkpoints, recovery.
  - `metrics.py`, `analysis.py`, `plotting.py`: analysis independent of simulation.
- `examples/sequential_study/`: repeated Gaussian bandit study with custom components.
- `examples/offline_csv/`: stored-data example.
- `tests/`: configuration, execution, persistence, analysis, and CLI checks.
- `docs/ARCHITECTURE.md`: component boundaries, reproducibility, and artifact contract.
- `docs/CONFIGURATION.md`: configuration reference and extension guide.
- `docs/PROJECT_BRIEF.md`, `docs/BUILD_WORKFLOW.md`: goals and development process.

## Working rules

- Keep scientific choices in paper code. The library must not depend on an example
  or a particular research package.
- Keep environment state in `DataGenerator`, algorithm state in the algorithm,
  and interaction order/progress in the protocol. Each run gets fresh instances.
- Inject separate RNGs; do not use global randomness. Seeds and run identities
  must remain independent of worker count, scheduling, and grid traversal order.
- Checkpoint only at complete protocol steps. Preserve RNG state and result
  boundaries together; never accept a partial artifact as a completed run.
- Reject incompatible scientific settings, code, or tracked input files explicitly.
  Preserve recovery from a preceding valid checkpoint and avoid duplicate records.
- Keep stored numerical results separate from figures. Analysis must work without
  importing simulation components; custom metrics/plotters may import their own code.
- Keep memory and dependencies proportionate. Buffering is per active run, and
  analysis retains one run plus grouped summaries. Distributed execution is outside scope.
- Update the architecture/configuration documentation when an authorized change
  changes the public contract. Surface material design choices rather than silently
  changing seed, resume, or statistical semantics.
- Verify behaviors that carry risk: deterministic streams, interruption/resumption,
  completed-output validation, mismatches, and figure regeneration. Prefer focused
  tests over tests that duplicate implementation details.
- Keep generated results, caches, environments, and build artifacts out of commits.
  Do not choose a license or publish a release without an explicit request.

## Development checks

```bash
python -m pip install -e '.[dev,plot]'
python -m pytest
ruff check src tests examples
ruff format --check src tests examples
python -m build --no-isolation
```

Run the checks relevant to a change; complete the full set for a release-sized
implementation. See `docs/BUILD_WORKFLOW.md` for end-to-end example verification.
