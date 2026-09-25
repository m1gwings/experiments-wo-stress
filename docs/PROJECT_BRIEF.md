# Project brief

## Purpose

Experiments W/O Stress supports numerical experiments in theory-oriented machine
learning papers. Researchers define scientific components and the observations
needed to evaluate them. The library provides repeatable planning, local
execution, saved results, and analysis, without taking ownership of the paper's
scientific choices.

The intended scale is a laptop or a modest multicore VM. A separate research
repository holds the study's YAML, `experiment_code/` package, and inputs; this
library remains a dependency.

## Design principles

1. **Make studies inspectable.** A configuration should show what was
   requested, and artifacts should explain what ran and what was saved.
2. **Separate scientific state.** Instances hold immutable problem input;
   algorithms and generators hold changing state; protocols own interaction
   order. Metrics use recorded observations after execution.
3. **Preserve reproducibility.** Random streams and run identities should not
   change with worker count or scheduling. Code, environment, and tracked input
   changes must not silently reuse incompatible work.
4. **Recover and reuse proportionately.** Checkpoint at complete steps, retain
   valid previous work, and cache analysis without demanding a distributed
   system.
5. **Keep the interface small.** Ordinary Python classes and a few built-in
   protocols should cover common studies while leaving research-specific
   decisions in the paper repository.

## Scope and status

The package is experimental. The sequential and offline examples exercise its
workflow, but a real paper project still needs to validate the authoring
interface before it can be called stable. Adoption in such a project is the next
product milestone.

Distributed scheduling, hosted storage, dashboards, a benchmark catalog,
arbitrary Python-object serialization, and checkpoints inside an indivisible
function call are outside the intended initial scope. The design favors local
files, bounded memory, and optional plotting dependencies. Release publication
and license choice remain separate owner decisions.

See the [README](../README.md) for a first run, the
[configuration guide](CONFIGURATION.md) for public contracts, and
[architecture](ARCHITECTURE.md) for storage and identity details.
