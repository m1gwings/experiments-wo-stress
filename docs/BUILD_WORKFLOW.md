# Development workflow

This is the maintainer workflow for changes to the library. For a researcher's
first study, start with the [README](../README.md) or an
[example](../examples/sequential_study/README.md).

## Start from a concrete study

Identify the scientific inputs, interaction, recorded observations, and intended
analysis before changing a public contract. The
[configuration guide](CONFIGURATION.md) describes the authoring interface;
[architecture](ARCHITECTURE.md) explains identities and persistence. Keep
paper-specific choices in study code.

Update affected examples and documentation with the code. After each file or
related edit batch, run:

```bash
python scripts/check_docs.py
```

This checks local links, fences, and repository paths. Review the prose against
the implementation as well; the script cannot judge scientific meaning.

## Verify the affected boundary

Install development and optional integration dependencies with
`python -m pip install -e '.[dev,plot,gym]'`. Choose focused tests for the risk:
deterministic streams and worker changes for execution, interruption and replay
for checkpoints, variant selection for compatibility changes, and regeneration
without simulation imports for analysis. Mock notification delivery instead of
sending real webhooks.

For a substantial implementation or release-sized change, run the full checks:

```bash
python -m pytest
python scripts/check_docs.py --examples --cli
ruff check src tests examples scripts
ruff format --check src tests examples scripts
python -m build --no-isolation
```

The documentation check also loads and plans example YAML and probes documented
CLI commands. CI runs the full set. A documentation-only change normally needs
the documentation check; executable example edits need focused execution tests.

## Exercise the studies

For an end-to-end acceptance run, use a fresh output directory:

```bash
ews plan examples/sequential_study/experiment.yml
ews run examples/sequential_study/experiment.yml --output outputs/verification --max-steps 250
ews inspect outputs/verification
ews build examples/sequential_study/experiment.yml --output outputs/verification --workers 2
ews build examples/sequential_study/experiment.yml --output outputs/verification
ews run examples/offline_csv/experiment.yml --output outputs/offline-verification
ews analyze examples/offline_csv/experiment.yml --output outputs/offline-verification
```

The paused run should resume with two workers; the second build should reuse
completed runs and analysis. Inspect logs and figures when those outputs are
affected. To test extension, increase `budget.steps` in a temporary
configuration beside its `experiment_code/` package and compare the extended
result with a fresh run at the larger budget. Do not commit temporary output.

The separate container check builds the deployment template, then exercises
non-root execution, a persistent mount, pause/resume, inspection, TikZ export,
and reuse. With Docker and pip-tools available, run it against an accessible
full Git revision:

```bash
python scripts/check_container.py --revision FULL_40_CHARACTER_GIT_COMMIT
```

If that revision is private, pass a repository-read token through
`--github-token-env EWS_BUILD_GITHUB_TOKEN`; the helper uses a temporary
BuildKit secret. Python-only checks do not establish that the container builds
successfully.

## Review and deliver

Review source, documentation, examples, and actual test results together. Record
material limits, especially incomplete recording, indivisible steps, and
external state that cannot be restored. Keep a clear distinction between
implemented behavior and future work.
