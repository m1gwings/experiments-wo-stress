# Contributing

Experiments W/O Stress is experimental. Changes should serve a concrete study or
failure, keep paper-specific science in study code, and preserve reproducibility
and independent analysis. Read the [project brief](docs/PROJECT_BRIEF.md) for
scope and the [architecture](docs/ARCHITECTURE.md) before changing storage or
execution boundaries.

For a guided first pass through the source and tests, use
[Reading the code](docs/CODE_GUIDE.md). It follows one study from configuration
through execution and saved-result analysis.

## Set up a checkout

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,plot,gym]'
```

On Windows, activate with `.venv\Scripts\activate`. The `dev` extra supplies
tests, style checks, and build tools; `plot` and `gym` install optional
integrations.

## Make a change

Start with the behavior a researcher needs. Keep new scientific algorithms in
the paper repository; add library settings only when their reuse across studies
is clear. Write focused code with useful errors and public docstrings. Update
affected documentation and examples in the same change, then test the behavior
at the boundary it touches.

Group tests by the behavior they explain. Give each test module and class a
clear scope, name scenarios by their expected outcome, and keep setup close to
the assertions. Use small documented fixtures and comments to explain unusual
failure injection or recovery steps. Class docstrings should describe what the
class owns, how it is used, and any ordering or state constraints a reader needs.

Run `python scripts/check_docs.py` after a documentation edit or related batch.
It checks links, fences, and repository paths. The
[development workflow](docs/BUILD_WORKFLOW.md) gives the full check commands,
example exercise, and container test. Use those checks in proportion to the
change; a documentation edit does not require a full simulation suite.

In the PR, describe the behavior, scientific or artifact semantics affected,
checks run, and any remaining limitation. Commit source, configuration, and
small fixtures; keep generated experiments, caches, virtual environments, and
build products out of Git. Licensing and release publication remain
project-owner decisions.
