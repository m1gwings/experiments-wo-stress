# Contributing

Keep the scientific authoring interface small and preserve reproducibility,
continuation, and independent analysis. Read [the architecture](docs/ARCHITECTURE.md)
before changing those boundaries. The package is experimental and should be
validated in a real paper before its API is declared stable.

## Set up a checkout

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,plot,gym]'
```

Use Python 3.10 or newer. On Windows, activate with `.venv\Scripts\activate`.
The `dev` extra supplies tests, style checks, and build tooling. `plot` and `gym`
install optional integrations exercised by the full development workflow.

## Make a change

1. Identify the concrete experiment or failure the change addresses.
2. Keep scientific algorithms in paper code; add reusable settings or infrastructure
   to the library only when their contract is clear.
3. Write polished code with focused responsibilities, clear public contracts,
   actionable errors, consistent formatting, and useful diagnostics.
4. Update every affected reference in the README, documentation, and examples.
   A changed parameter, artifact, command, or continuation rule must not leave an
   old explanation that still looks authoritative.
5. After each file change or tightly related edit batch, run the documentation
   checker and review affected explanations against the implementation.
6. Add focused tests for new risks. Persistence changes need fresh, interrupted,
   resumed, and extended executions; cache changes need invalidation checks.
7. Describe the resulting behavior, validation, and material limits in the PR.

## Verify

```bash
python scripts/check_docs.py
python -m pytest
python scripts/check_docs.py --examples --cli
ruff check src tests examples scripts
ruff format --check src tests examples scripts
python -m build --no-isolation
```

The quick documentation check verifies local Markdown links, balanced fences, and
referenced repository paths. `--examples` loads and plans example YAML files;
`--cli` checks help for documented commands. It does not execute every code snippet
or prove semantic agreement: review and behavior tests are still required.

Use `ruff format src tests examples scripts` to apply formatting. The
[development workflow](docs/BUILD_WORKFLOW.md) covers the end-to-end example.
PDF/JPG checks run with Matplotlib. TikZ export needs no TeX installation; optional
local compilation is useful when modifying the exporter.

## Preserve scientific meaning

Instance generation, environment randomness, algorithms, and protocols use distinct
streams. Metrics consume stored observations and instances. A sparse trajectory
must not silently become a full cumulative statistic. Budget extension requires
explicit component support, and cached analysis must account for its inputs and
implementation.

Retain existing variants when requests change. Cleanup is a separate explicit
operation with a reviewable dry run. Generated experiments, caches, environments,
and build products stay out of commits; source, configurations, and small input
fixtures belong in version control.

Licensing and release publication remain project-owner decisions.
