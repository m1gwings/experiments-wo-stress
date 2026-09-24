# Contributing

The package is experimental. Contributions should keep the authoring interface
small and preserve reproducibility, recovery, and analysis of saved results.
Read [the architecture](docs/ARCHITECTURE.md) before changing those boundaries.

## Set up a checkout

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,plot]'
```

Use Python 3.10 or newer. On Windows, activate with `.venv\Scripts\activate`.
The `dev` extra supplies pytest, Ruff, and build tooling; `plot` adds Matplotlib.

## Make a change

1. Reproduce the problem or identify the concrete experiment the change supports.
2. Keep paper-specific algorithms and environment models in `examples/` or the
   research project. Extend the library where orchestration is shared.
3. Update the relevant public contract in `docs/` when changing configuration,
   component interfaces, artifact formats, or reproducibility semantics.
4. Add focused checks for meaningful failure cases. Validate both fresh execution
   and recovery when changing state or persistence.
5. Summarize the resulting behavior, validation, and remaining limits in the PR.

## Verify

```bash
python -m pytest
ruff check src tests examples
ruff format --check src tests examples
python -m build --no-isolation
```

Use `ruff format src tests examples` to apply formatting. An end-to-end example
check is documented in [the development workflow](docs/BUILD_WORKFLOW.md).
PDF/JPG tests run when Matplotlib is installed. TikZ source export needs no TeX
installation; optional local compilation can validate changes to the exporter.

Use a new output directory when the scientific configuration, component code,
tracked input, or recorded fields change. Commit source, configurations, and small
input fixtures; leave generated experiments and build products out of the repository.

The API should be exercised in a real paper project before a stable release.
Licensing and package publication remain project-owner decisions.
