# Development workflow

## Start from an experiment

Use a concrete research need to guide changes. The current contracts are recorded
in [ARCHITECTURE.md](ARCHITECTURE.md) and [CONFIGURATION.md](CONFIGURATION.md).
When an authorized feature changes those contracts, describe its scientific inputs,
interaction, stored observations, resume behavior, and resulting figure before
implementing it. Keep unrelated design proposals out of the operational docs.

The initial implementation is complete enough for local end-to-end studies. The
next milestone is adapting a real paper experiment and using the experience to
refine the experimental API.

## Implement and verify

Keep the smallest useful vertical slice: configuration through execution to saved
results and analysis. For persistence changes, compare uninterrupted execution
with interrupted/resumed execution, including changed worker counts. For analysis
changes, work from existing artifacts and avoid requiring simulation imports.

Install the development and plotting extras in a virtual environment:

```bash
python -m pip install -e '.[dev,plot]'
```

Run the full verification set for substantial implementation changes:

```bash
python -m pytest
ruff check src tests examples
ruff format --check src tests examples
python -m build --no-isolation
```

Tests cover stable run planning, independent RNG streams, configuration validation,
sequential/parallel equivalence, interruption, checkpoint recovery, artifact and
input mismatches, aggregation, figure export, and CLI behavior. Add tests for
concrete new risks, and use focused checks for smaller changes.

## Exercise the complete example

Use a new output directory if one from a previous code version already exists:

```bash
ews plan examples/sequential_study/experiment.yml
ews run examples/sequential_study/experiment.yml --output outputs/verification --max-steps 250
ews inspect outputs/verification
ews run examples/sequential_study/experiment.yml --output outputs/verification --workers 2
ews run examples/sequential_study/experiment.yml --output outputs/verification
ews plot examples/sequential_study/experiment.yml --output outputs/verification
ews run examples/offline_csv/experiment.yml --output outputs/offline-verification
ews analyze examples/offline_csv/experiment.yml --output outputs/offline-verification
```

The first run pauses each trial after 250 new steps. The second continues it with
two workers. The third validates and skips completed results. Inspect the generated
figures when changing the exporter; all requested formats should contain the same
curves, labels, and uncertainty semantics. TikZ compilation is an optional local
check and is not required for source export.

## Review and deliver

Review the final behavior and documentation together. Keep generated artifacts out
of commits, describe validation in the PR, and state material limitations such as
an indivisible offline step or missing raw observations. Preserve a clear distinction
between implemented features and possible future extensions.

After a real paper validates the workflow, the project owner can choose licensing,
versioning, and publication. Implementation work alone does not authorize a release.
