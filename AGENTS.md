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
  - `study/`: configuration, run descriptions, planning, and stable random streams.
  - `components/`: extension contracts and component loading.
  - `builtins/`: supplied protocols, data generators, bandits, bandit metrics, and Gymnasium adapter.
  - `execution/`: coordination, GPU resources, compute reporting, run sessions, provenance, logs, and notifications.
  - `storage/`: artifact models, experiment ownership, checkpoint backends, and cleanup.
  - `analysis/`: metrics, aggregation, caches, and figures.
  - Legacy top-level modules re-export public names; internal imports use their owners.
- `examples/sequential_study/`: repeated Gaussian bandit study with custom components.
- `examples/offline_csv/`: stored-data example.
- `tests/`: configuration, execution, persistence, analysis, and CLI checks.
- `docs/CODE_GUIDE.md`: guided source reading order and a map of behavior-focused tests.
- `docs/ARTIFACTS.md`: versioned semantic discovery contract for external tooling.
- `docs/ARCHITECTURE.md`: component boundaries, reproducibility, and artifact contract.
- `docs/CONFIGURATION.md`: configuration reference and extension guide.
- `docs/LLM_GUIDE.md`: self-contained public authoring contract for use with a paper PDF.
- `docs/CLOUD.md`, `deploy/`: single-host cloud guidance and standalone-project templates.
- `docs/PROJECT_BRIEF.md`, `docs/BUILD_WORKFLOW.md`: goals and development process.
- `scripts/check_docs.py`: mechanical checks for documentation and examples.
- `scripts/check_container.py`: standalone-container integration check, run by CI.

## Working rules

- Keep scientific choices in paper code. The library must not depend on an example
  or a particular research package.
- Keep environment state in `DataGenerator`, algorithm state in the algorithm,
  and interaction order/progress in the protocol. Each run gets fresh instances.
- Persist generated instances separately from evolving state. Keep scientific
  metrics in analysis code, computed from recorded observations and the saved
  instance. Do not hide metric accumulators in the data generator.
- Inject separate RNGs; do not use global randomness. Seeds and run identities
  must remain independent of worker count, scheduling, and grid traversal order.
- Keep GPU allocation in execution infrastructure and out of scientific identities
  and RNG streams. A GPU worker owns one device for its lifetime; establish CUDA
  visibility before the spawned interpreter imports study code. Scientific
  algorithms use the local device, never schedule physical GPUs. CPU studies
  require no GPU configuration; test allocation without a GPU framework.
- Checkpoint only at complete protocol steps. Preserve RNG state and result
  boundaries together; never accept a partial artifact as a completed run.
- Checkpoint backends own state representation only. Pass complete component and
  RNG state without mandatory conversion; keep generation integrity, commit
  publication, fallback, and retention in storage infrastructure. Default NumPy
  studies need no backend configuration. Restore with the saved backend and keep
  backend selection operational rather than part of scientific identity.
- Never silently mix incompatible scientific settings, code, or tracked inputs.
  Select matching retained variants and report damaged artifacts. Preserve recovery
  from a preceding valid checkpoint and avoid duplicate records.
- Treat run budget and recording selection as explicit execution requests. Extend
  a completed run only when all involved components support continuation; retain
  distinct recording variants and identify the selected variant in metadata.
- Keep stored numerical results separate from figures. Analysis must work without
  importing simulation components; custom metrics/plotters may import their own code.
- Keep the public CLI to `count-runs`, `run`, `analyze`, `plot`, `inspect`, and
  `clean`. `run` schedules simulation, per-run metrics, aggregation, and figures
  when their durable dependencies are ready. Counting runs is optional;
  `analyze` and `plot` operate on saved results without scheduling simulations.
- Recompute backwards only until the nearest retained valid ancestor. Derived
  identities must prove reuse without requiring intentionally pruned ancestors.
  Publish each run variant lazily before use; a compact durable request covers
  the complete invocation. Prefer ready metrics and closing actual analysis groups,
  with bounded fairness for simulations. Scheduling never enters scientific IDs.
- `recording.retention: until_analyzed` removes only trajectories after every
  configured per-run metric has a validated durable artifact. Publish intentional
  pruning proof before deletion. Missing unmarked data remains corruption;
  missing required raw ancestors may be rematerialized only by `run`. Pruned
  prefixes/endpoints cannot authorize budget continuation. Default `keep` retains data.
- `recording.metric_retention: until_aggregated` deletes a group's metric cache
  generations only after its matching aggregate is validated. The aggregate
  proves reuse and figure regeneration without its discarded metrics. Preserve
  exact unchanged requests, scoped recomputation, and interrupted deletion safety.
- Terminal timezone and task progress are observational. Display explicit zones;
  keep persisted timestamps UTC and all display options outside identities.
- Keep recovery in `storage/recovery.py`: versioned sealed snapshots under output
  ownership, committed boundaries only, checksums, safe paths, and version rejection.
  Cloud wrappers consume this API, never private checkpoint layout assumptions.
- Keep memory and dependencies proportionate. Buffering is per active run, and
  analysis retains one run plus grouped summaries. Distributed execution is outside scope.
- Deliver professional-quality code for every implementation task: clear names,
  focused responsibilities, useful errors, complete docstrings at public boundaries,
  consistent formatting, and proportionate verification. Remove stale comments,
  dead code, and scaffolding introduced by the change.
- Keep classes and tests readable for repository inspection. Class docstrings
  explain responsibility, owned state, and lifecycle; comments explain non-obvious
  ordering and invariants. Group tests by behavior, document module/class scope,
  and name each scenario by the result it verifies. Keep the code-reading guide
  current when implementation owners or test organization change.
- Keep documentation part of the implementation. Update every affected reference
  in the README, architecture/configuration docs, examples, and contributor guidance
  when behavior or a public contract changes. Surface material design choices
  rather than silently changing seed, resume, or statistical semantics.
- Keep `docs/LLM_GUIDE.md` self-contained and current whenever public entry points,
  YAML, CLI commands, or component contracts change. Its runnable example must
  remain valid without access to the other documentation.
- After changing a file or a tightly related batch, run
  `python scripts/check_docs.py` and review affected prose against the actual code.
  Before delivery, also run its `--examples --cli` checks. The script checks links,
  fences, repository paths, YAML planning, and command availability; it does not
  replace review of scientific claims or API semantics.
- Verify behaviors that carry risk: deterministic streams, interruption/resumption,
  completed-output validation, mismatches, and figure regeneration. Prefer focused
  tests over tests that duplicate implementation details.
- Keep notification transport outside workers and protocol steps. Notifications
  must not change scientific identities or expose webhook credentials in artifacts
  or errors; test delivery with mocked transports rather than real messages.
- Keep compute observation separate from resource allocation and scientific state.
  Reporting is best effort on Linux/macOS and silently unavailable on Windows.
  Preserve invocation and attempt history outside result chunks and checkpoints;
  distinguish pipeline elapsed time from cumulative task worker, process CPU,
  and allocated GPU time. Never present overlapping work as sequential stages.
  Reuse consumes no new simulation compute; missing historical timing stays unknown.
  Reporting metadata and reporting-only edits must not change scientific or cache
  identities. Exact compatibility mappings require review; unknown execution-source
  edits must continue to invalidate conservatively. Test clocks and system queries
  without requiring particular hardware. Reports cover observed output-directory
  history, never the entire research project's compute.
- Keep external discovery in `storage/artifact_index.py` and `docs/ARTIFACTS.md`.
  When changing output locations, update their semantic roles in the same change.
  Publish deterministic optional locations without rescanning results or changing
  scientific/cache identities. Cloud consumers must use this catalog, not copied
  internal paths; old outputs remain readable without it.
- Keep generated results, caches, environments, and build artifacts out of commits.
  Do not choose a license or publish a release without an explicit request.

## Development checks

```bash
python -m pip install -e '.[dev,plot]'
python -m pytest
python scripts/check_docs.py --examples --cli
ruff check src tests examples scripts
ruff format --check src tests examples scripts
python -m build --no-isolation
```

Run the checks relevant to a change; complete the full set for a release-sized
implementation. See `docs/BUILD_WORKFLOW.md` for end-to-end example verification.

## Portable workers

Keep the explicit CPU/NumPy portability policy in `execution/portability.py` and
its contract in `docs/PORTABILITY.md`. Strict compatibility remains the default.
Do not remove scientific/source/input, package/runtime, architecture, recording,
or checkpoint checks to make cloud continuation work. Unknown native/GPU and
custom backends are unsupported. Cloud leases, persistence, and environment
recreation belong to cloud-experiments; EWS remains authoritative for variants
and checkpoints. Test portable resume against uninterrupted trajectories.
