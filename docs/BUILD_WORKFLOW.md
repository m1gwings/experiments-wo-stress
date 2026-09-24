# Collaborative build workflow

Status: proposal for discussion, 2026-09-24.

## Phase 1 — Agree on the contract in Markdown

Use `PROJECT_BRIEF.md` for goals and acceptance criteria, `ARCHITECTURE_DRAFT.md` for boundaries and data flow, and this file for the work sequence. Chat about one open question at a time and edit the documents to record conclusions. Mark unresolved choices explicitly rather than letting an implementing agent decide them implicitly.

Select one genuine paper experiment as the design example. Write a short walkthrough showing the experiment's inputs, interaction protocol, repetitions, stored measurements, and final figure. This is the strongest test of whether the proposed API is small enough.

## Phase 2 — Freeze a small v0 implementation brief

Once the example is agreed, specify the public API, config schema, artifact layout, and exact semantics of seeds and resume. Convert open questions into recorded decisions. Define one end-to-end acceptance check: an interrupted run resumes without changing completed results, and all requested figure formats can be regenerated from saved data.

Only then add a concise `AGENTS.md` in the repository: point Codex to the agreed documents; state the current milestone and lightweight verification commands; require discussion before changing the public contract. The design documents remain the source of product requirements; `AGENTS.md` supplies implementation workflow.

## Phase 3 — Implement one vertical slice

Create the package and example in one repository. Implement config loading, deterministic job planning, sequential execution, committed results, job-level resume, aggregation, and one line figure. Exercise it with two algorithms and multiple seeds. Test the behaviors with meaningful failure risk: seed invariance, interruption/resume, config mismatch, and figure regeneration without rerunning.

Add local process parallelism after sequential correctness is demonstrated, then check that a different worker count yields identical per-job outputs. Add more protocol conveniences only when the example exposes a real repetition.

## Phase 4 — Use it in a paper and release

Adapt one existing paper experiment to the package. Record friction and simplify the API. Tag a version and pin it in that paper repository. Decide on a package name, license, Python support range, and PyPI publication when the first paper workflow is convincing.

## Suggested Codex prompts

**Design session (no implementation):**

> Read `docs/PROJECT_BRIEF.md`, `docs/ARCHITECTURE_DRAFT.md`, and `docs/BUILD_WORKFLOW.md`. Help me refine the architecture using a real paper experiment. Identify ambiguities and propose the smallest public API. Edit only the design Markdown files after we agree on each decision; do not implement the library yet.

**First implementation session (after agreement):**

> Read `AGENTS.md` and the approved design documents. Implement only the first vertical slice and its example. Preserve the documented seed and resume semantics. Run the specified acceptance checks, summarize deviations from the design, and show the diff.

## Where files live

During discussion, these files may be kept outside a repository. When the repository is created, place them under `docs/`, add `AGENTS.md`, and commit the reviewed versions before implementation. Each paper keeps its own code, configs, results policy, and dependency lock or version pin.
