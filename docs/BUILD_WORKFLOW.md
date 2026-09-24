# Development workflow

## Start from a concrete study

Use a research need to guide implementation. The current public contracts are in
[ARCHITECTURE.md](ARCHITECTURE.md) and [CONFIGURATION.md](CONFIGURATION.md).
Describe scientific inputs, interactions, recorded observations, continuation,
and analysis before changing those boundaries. Keep speculative features out of
the operational documentation.

Every implementation task includes code polish, affected documentation, and
examples. After each changed file or tightly related batch, run the quick
consistency check and inspect affected prose for stale claims:

```bash
python scripts/check_docs.py
```

This catches broken local links, fences, and repository references. Human review
and behavior tests establish whether the documented scientific meaning is correct.

## Verify the relevant boundaries

Install development tools and optional integrations:

```bash
python -m pip install -e '.[dev,plot,gym]'
```

For a substantial change, run:

```bash
python -m pytest
python scripts/check_docs.py --examples --cli
ruff check src tests examples scripts
ruff format --check src tests examples scripts
python -m build --no-isolation
```

The full documentation check also loads and plans example configurations and probes
available CLI commands. CI runs these checks with the tests. Focus smaller changes
on their actual risks; avoid tests that merely repeat implementation details.

Execution verification compares fresh runs with pause/resume, process interruption,
changed worker counts, and compatible budget extension. Storage verification covers
commit boundaries, corrupt artifacts, retained variants, and cleanup scope.
Analysis verification checks saved-instance access, complete-trajectory requirements,
cache hits and invalidation, statistical grouping, and figure regeneration without
simulation imports.

## Exercise the example

Use a fresh output directory for a clean acceptance run:

```bash
ews plan examples/sequential_study/experiment.yml
ews run examples/sequential_study/experiment.yml --output outputs/verification --max-steps 250
ews inspect outputs/verification
ews build examples/sequential_study/experiment.yml --output outputs/verification --workers 2
ews build examples/sequential_study/experiment.yml --output outputs/verification
ews run examples/offline_csv/experiment.yml --output outputs/offline-verification
ews analyze examples/offline_csv/experiment.yml --output outputs/offline-verification
```

The first execution pauses after 250 new steps per run. The first build continues
with two workers and produces configured analysis. The repeated build should reuse
valid completed work and analysis caches. Inspect logs and generated figures;
formats should contain the same curves, labels, and uncertainty meaning.

To exercise extension, increase `budget.steps` in a temporary copy of the sequential
configuration kept beside its scientific module, then run against the same output.
Compare its complete numerical trajectory with a fresh run at that larger budget.
Do not keep the temporary configuration or generated outputs in the commit.

To inspect cleanup without deleting anything:

```bash
ews clean outputs/verification --scope inactive
```

Apply cleanup only when the displayed selection is intended. Removing checkpoints
preserves observations but removes continuation state. TikZ compilation is an
optional local check; source export does not require a TeX installation.

## Review and deliver

Review the final source and documentation together. Record tests and material
limits, such as incomplete observations, an indivisible offline step, or an
external environment that cannot snapshot all state. Preserve the distinction
between implemented behavior and future work.

The next product milestone is adoption in a genuine paper project. Stable release,
license choice, and publication remain separate owner decisions.
