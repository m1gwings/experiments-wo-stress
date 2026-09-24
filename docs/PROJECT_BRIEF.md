# Experiments W/O Stress — project brief (draft)

Status: discussion draft, 2026-09-24. This document describes the goal; it is not an implementation instruction.

## Purpose

Build a lightweight Python library for numerical experiments in theory-oriented machine learning papers. A researcher should spend most of their time specifying an instance, an algorithm, a feedback process when needed, and the figures that answer the paper's questions. The library handles recurring experiment infrastructure.

Typical target: moderate simulations with many independent instances, algorithms, and seeds, run on a laptop or a modest multicore machine. The design should allow growth to larger runs without making the first version a cluster framework.

## Repeated problems to solve

- Deterministic, isolated random streams across instances, algorithms, repetitions, and workers.
- A clear interaction loop for sequential experiments, without imposing that loop on every numerical study.
- Configuration parsing and validation; a saved snapshot of the effective configuration.
- Independent work units that can be skipped after success, retried after failure, and resumed after interruption.
- Raw numerical results and metadata that survive plot changes.
- Aggregation and export of publication figures to PDF, JPG, and TikZ/PGFPlots source.
- A simple way to use the library from a paper repository, including an example that an LLM can adapt.

## Design criteria

1. **Reproducibility:** a run records its configuration, seeds, software versions, and code revision when available. Re-running the same job produces the same result under the same code and environment, including when parallel worker count changes.
2. **Safe resumption:** completed jobs are recognized from validated outputs; partial outputs do not count as completed. An incompatible configuration does not silently reuse old results.
3. **Small authoring surface:** simple studies should work with a Python trial function. Interactive studies may use explicit environment/algorithm protocols. Users should not have to subclass a large framework.
4. **Separation:** simulation writes raw data; plotting reads saved data. Changing a figure must not rerun simulations.
5. **Proportionate dependencies:** numeric computing can use NumPy; plotting and other formats can be optional extras. Avoid mandatory distributed services or databases initially.
6. **Inspectability:** files, configuration, and errors should be easy to inspect; the CLI should report which jobs ran, were skipped, or failed.

## Intended user journey

1. Create a paper-specific repository and install a pinned library version.
2. Define a trial or interactive environment and algorithms in that repository.
3. Write a configuration specifying the instance family, algorithms, budgets, repetitions, and metrics.
4. Run the experiment; stop and resume it; optionally use local parallel workers.
5. Generate figures from stored data, and include the PDF or TikZ output in the paper.

## First acceptance example

A small sequential learning experiment compares two algorithms across several problem sizes and independent seeds. The researcher can interrupt a run halfway, restart with a different number of workers, and obtain the same complete set of per-job results. A plotting command generates a PDF, JPG, and editable TikZ/PGFPlots curve from those saved results. The example also documents how to add a new algorithm and a custom feedback rule.

## Out of scope for the initial release

- A distributed scheduler, web dashboard, or hosted results service.
- A catalog of research algorithms or benchmarks.
- Automatic support for every plotting primitive or arbitrary Python object serialization.
- A framework-specific dependency in each paper's public code beyond the library itself.

## Distribution assumption to review

Develop a standalone Python package in a GitHub repository. Keep one canonical example paper project. Use tagged versions or pinned commits in paper repositories; consider PyPI after a real paper experiment validates the API.
