# Running a study on a small cloud VM

An existing study can run on one Linux VM using the same `ews` commands as on a
laptop. This guide uses a container for the software environment and a
persistent block disk for experiment output. The library still runs local
workers and writes ordinary files; it does not need a distributed service.

Start with a small VM, then size it from a representative run. Choose storage
that survives VM replacement and back it up
separately. Provider prices and shared-CPU performance change, so check the
chosen region and plan before committing to a long run.

## Disposable VMs without containers

The separate [cloud-experiments](https://github.com/m1gwings/cloud-experiments)
workflow uses system Python and a venv, with automatic repeated `cloud-run`
continuation through Object Storage and disposable VM attempts. It opts into
EWS's [portable CPU/NumPy policy](PORTABILITY.md), recreates a saved environment
lock, restores the complete output, and lets EWS select compatible variants.
No Docker is needed for that workflow. The remaining sections describe the
alternative standalone container deployment, whose strict defaults are unchanged.

## Prepare the study

Keep the paper repository separate from the library. A container build needs the
study, its pinned dependencies, and the deployment template:

```text
my-paper/
  experiment.yml
  experiment_code/
    __init__.py
    algorithms.py
    data.py
    metrics.py
  data/
  requirements.in
  requirements.lock
  .ews-revision
  Dockerfile
  .dockerignore
```

Use only the `experiment_code/` modules your study needs. With `experiment.yml`
at the repository root, paths such as `experiment_code.algorithms:MyAlgorithm`
import from that package. The [configuration guide](CONFIGURATION.md) covers
component interfaces.

Put the full library Git commit hash in `.ews-revision`. List the study's
runtime packages in `requirements.in`, including NumPy and PyYAML, plus
`setuptools>=68` and `wheel` for the library wheel build. Add optional plotting,
Gymnasium, or scientific dependencies your study uses. Generate and commit a
version-and-hash lock for the Python version and platform you intend to deploy:

```bash
python -m pip install pip-tools
python -m piptools compile --generate-hashes --allow-unsafe \
  --output-file requirements.lock requirements.in
```

The template installs the lock with `--require-hashes`, so it must include
transitive packages and their hashes. Copy
[Dockerfile.example](../deploy/Dockerfile.example) to `Dockerfile` and
[.dockerignore.example](../deploy/.dockerignore.example) to `.dockerignore` in
the paper repository.

## Build the image

Build from the **paper repository**, using its full Git revision and the pinned
library revision:

```bash
docker build \
  --build-arg EWS_REV="$(cat .ews-revision)" \
  --build-arg STUDY_REV="$(git rev-parse HEAD)" \
  --tag paper-study:locked .
```

For repeatable rebuilds, pin the Python base image by digest as `PYTHON_IMAGE`
and retain the image or its registry digest with the lock and study commit. The
template fetches the public library during the build; the running container
needs no Git credentials. If a dependency is private, use a Docker build secret
or SSH mount rather than putting a token in a URL or build argument. The
optional `github_token` secret is supported by the template.

The repository's [container check](../scripts/check_container.py) builds this
template and exercises pause, resume, inspection, figure export, and reuse with
a temporary standalone study.

## Run with durable output

Mount a persistent block disk with a normal Linux filesystem and give the Docker
user write access. The following example assumes that `/srv/ews-output` is on
that disk:

```bash
docker run --detach --name paper-study \
  --init --stop-timeout 120 \
  --cpus 2 --memory 4g \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src=/srv/ews-output,dst=/output \
  paper-study:locked \
  run /workspace/experiment.yml --output /output/study --workers 2

docker logs --follow paper-study
```

The bind mount keeps output outside the container; the host path must itself be
on persistent storage. Only one executor should write a given output directory.
Use separate directories for separate studies.

`run` schedules simulation, metric, aggregation, and figure work when their
required artifacts are ready. Completed groups can produce summaries while
unrelated simulations continue. A paused or failed invocation keeps its committed
artifacts, including any completed derivations, for resumption.
`count-runs` is an optional preview of study size; you do not need to invoke it
before `run`.

To stop and resume with the same image and mount:

```bash
docker stop --timeout 120 paper-study
docker start --attach paper-study
```

The stop grace period should allow a normal protocol step and checkpoint write.
A long offline `fit()` or callable trial is one step and may restart from its
previous checkpoint if stopped during that call. See the
[checkpoint model](ARCHITECTURE.md#checkpoints-completion-and-interruption).

Inspect results or regenerate figures after changing plot settings, using the
same image and output mount:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src=/srv/ews-output,dst=/output \
  paper-study:locked inspect /output/study

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src=/srv/ews-output,dst=/output \
  paper-study:locked plot /workspace/experiment.yml --output /output/study
```

`plot` uses saved numerical results without scheduling simulations. Use `analyze`
instead when only metrics or aggregation settings have changed and you want
updated summaries from those results.

Use the recovery snapshot API below to persist a consistent output boundary.
Object storage is suitable for snapshots, but the live output directory needs
filesystem locking and atomic file replacement. With
`recording.retention: until_analyzed`, snapshots preserve the intentional deletion
of trajectories alongside their retained metric artifacts. A changed metric
whose raw input was deleted requires `ews run`; standalone `analyze` and `plot`
never schedule simulations.

## Versioned recovery snapshots

EWS owns the recovery contract in
[`storage/recovery.py`](../src/experiments_wo_stress/storage/recovery.py).
Cloud wrappers, including `cloud-experiments`, should consume this API instead of
parsing run directories or checkpoint layouts. The semantic catalog in
[ARTIFACTS.md](ARTIFACTS.md) remains useful for selecting downloaded figures or
reports; it is not a recovery inventory.

After gracefully stopping the writer, create a snapshot on the local filesystem:

```python
from experiments_wo_stress.storage import (
    create_snapshot,
    restore_snapshot,
    validate_snapshot,
)

snapshot = create_snapshot("/work/output", "/work/snapshots/attempt-001")
manifest = validate_snapshot(snapshot)
# Transfer recovery.json, listed output files, and recreate listed directories.
# Download them into an isolated snapshot directory before restoring.
restored = restore_snapshot("/work/downloaded-snapshot", "/work/restored-output")
```

Both destination directories must be new and outside their source tree. Creation
holds EWS's exclusive experiment lock for validation and copying, and fails if an
executor, standalone analysis/plot operation, or cleanup owns that lock. It does
not pause a running invocation or provide snapshots of live writers. Direct file
edits and custom writers must also be stopped. Snapshotting needs space for a
second copy of retained artifacts; remote transfer can reuse identical checksums.

A snapshot contains an `output/` directory and `recovery.json`. The manifest has:

| Field | Meaning |
| --- | --- |
| `schema` | Exact identifier `experiments-wo-stress/recovery`. |
| `schema_version` | Integer `1`; unsupported versions must be rejected. |
| `snapshot_id` | SHA-256 of the canonical manifest contents without this field. |
| `files` | Mapping of canonical POSIX paths relative to `output/` to `sha256` and byte `size`. |
| `directories` | Canonical relative directory paths, including empty checkpoint payload directories. |
| `omitted_checkpoints` | Run/generation identifiers and reasons for invalid committed checkpoint payloads excluded from this snapshot. |
| `pruned_runs` | Storage identifiers whose valid intentional-pruning receipts are retained. |

Paths are literal relative files, without empty, `.` or `..` components,
backslashes, colons, nonprintable characters, or absolute roots. Payloads must be
regular files; symbolic links and special files are rejected. The manifest and
file hashes detect incomplete or damaged transfer; they are not an authentication
mechanism. `validate_snapshot` verifies the version, manifest identity, exact
file/directory inventory, sizes, and hashes. Cloud tools must not guess the meaning of a future
version; use a compatible EWS reader instead. Additional manifest fields are
included in `snapshot_id` and may be ignored by version-1 consumers.

The snapshot is published atomically only after its files and manifest have been
synchronized. Upload all listed payloads before publishing `recovery.json` as the
remote commit marker, and expose a snapshot to consumers only after that upload
succeeds. Failed transfers cannot replace the last complete remote snapshot.
A downloader verifies the entire inventory before restoring. Restore writes to a
temporary directory and atomically publishes a new output tree, never merges into
an existing tree; this prevents old remote files from reviving pruned trajectories.

EWS chooses which artifacts are committed. Checkpoint publication commits matching
component/RNG state and result boundaries together. The snapshot retains validated
referenced generations and their committed result chunks, including older fallback
generations; unpublished generations and result tails are excluded. It validates
checkpoint payload integrity without importing or invoking its saved backend.
Unknown checkpoint envelope versions fail snapshot creation instead of being
misclassified as missing state.
An invalid newest generation is omitted and reported while unchanged progress
metadata lets EWS recover a valid predecessor. Completed retained trajectories,
instances, and published analysis caches must validate; corruption makes snapshot
creation fail instead of certifying damaged results. Temporary files and pending
analysis generations are never committed snapshot objects.

Intentional-pruning receipts, required retained derivations, provenance, instances,
and compute records travel together. Deleted result chunks remain absent. Retained
checkpoint payloads for a pruned run are historical data, not permission to extend
a run without its required prefix. A changed metric or larger budget may need a
fresh simulation. References for previously analyzed budgets survive that replay,
including interruption, so their retained derivations remain reusable after restore.
After restore, EWS still performs its ordinary scientific,
environment, dependency, variant, and checkpoint validation; byte integrity alone
does not promise compatibility with a replacement VM or checkpoint decoder.

## Resources and compatibility

Start with one worker and increase it after measuring CPU and memory use. The
template limits BLAS/OpenMP threads to one per worker; a run dominated by large
matrix operations may work better with fewer workers and more numerical threads.
Allow room for each worker's result buffer (16 MiB by default), component state,
checkpoints, retained run variants, and analysis artifacts.

Ordinary NumPy studies need no checkpoint backend configuration. If a large
model needs native framework serialization or multiple checkpoint files, use a
paper-owned `execution.checkpoint_backend` as described in
[checkpoint backends](CONFIGURATION.md#checkpoint-backends). Include its module
and serializer dependencies in the pinned image. EWS passes logical state to the
backend and retains responsibility for checksums, complete-step publication,
fallback, and retention. Native serialization can avoid a compulsory NumPy copy,
but its actual memory and I/O costs depend on the backend; measure them with a
representative model.

Ordinary CPU experiments require no GPU configuration. On a GPU VM, arrange GPU
access and install the study's compatible framework in the container, then use
one worker per GPU:

```yaml
execution:
  workers: 2
  gpu_ids: [0, 1]
```

GPU IDs use the container runtime's unmasked GPU namespace; they are not indices
into a preexisting CUDA mask. `CUDA_VISIBLE_DEVICES` must be unset when EWS starts
GPU execution, even if the container image sets it to an empty value. Each
spawned worker inherits a single-device mask before study imports and keeps it
for its lifetime; the algorithm uses its local device, normally `cuda:0`. This
also applies with one GPU and one worker. Any `--workers` override must match the
number of IDs.

EWS manages visibility: it does not install a framework, validate GPU
availability, or reserve devices against another experiment. Allocate disjoint
GPUs to concurrent invocations. Memory packing, multiple workers sharing a GPU,
multi-GPU training, and distributed scheduling are outside this mode. See
[GPU execution](CONFIGURATION.md#gpu-execution) for the full configuration rules.

**Use the same VM and software environment when you need exact checkpoint
continuation.** Compatibility checks include Python and numerical-library
versions, platform identity, source code, and tracked input paths. Even the same
container image on another host may select a new run variant, especially if the
kernel or resolved paths differ. Keep `/workspace` and `/output` stable and
validate any migration before relying on reuse. Saved results can still be
copied for separate analysis.

Changing the configured checkpoint backend affects new writes. Continuation
loads each saved generation with its recorded backend type and parameters, so
keep previous backend code and representation readers available in the image.
Back up whole generations, including every nested payload file and their envelope.
If no retained generation can be decoded, missing serializer dependencies cause
a continuation error while saved artifacts are preserved. Numerical analysis
does not require loading native checkpoint payloads. Legacy NumPy checkpoints
remain readable, subject to the same scientific and library compatibility checks.

Changing GPU assignment alone does not select a new scientific run or stored
variant. It remains visible in execution diagnostics, but does not guarantee
bitwise reproducibility across GPU models or framework versions. Check framework
state conversion and RNG restoration in the study's own continuation tests.

## Retain compute reports with the artifacts

On Linux/macOS, `ews run` automatically records compute information in
`OUTPUT/compute/`. Keep this directory with the rest of the output mount and
backups. `summary.md` is a concise starting point for a paper's resources
paragraph; immutable JSON records retain invocation and attempt history.
`ews inspect OUTPUT` points to the latest summary. Missing hardware queries or
reporting failures do not make successful scientific execution fail.

The report includes the platform, available CPU/RAM/GPU capacity information,
worker count, output-filesystem capacity/available space at start, and logical
artifact bytes at finalization. A VM or container may expose host capacities
rather than its usable quota. EWS does not query cloud metadata services or infer
a provider, instance SKU, billing rate, or tenancy. Record the provider, machine
type, configured limits, and storage arrangement separately in the paper.

End-to-end elapsed time includes the overlapping pipeline. Cumulative worker
time adds simulation and derived-task attempt durations; separate task-kind
totals describe work that can overlap, rather than consecutive elapsed stages.
Allocated GPU-hours multiply each task attempt's wall time by its assigned GPU
count, including derived tasks occupying an exclusively assigned GPU worker.
Neither is CPU-core-hours or measured accelerator utilization.
Process CPU time excludes descendants.
Failed/paused attempts remain visible across retries, reused completions add no
new simulation compute, and missing historical or abruptly terminated timings
remain unknown. See the
[measurement definitions](CONFIGURATION.md#automatic-compute-reporting).

These totals cover only records retained in this output directory. They cannot
describe experiments in other directories, deleted records, other machines, or
tools outside EWS, and do not include a VM's idle reservation between invocations.
The researcher remains responsible for disclosing broader project compute.

## Optional progress messages

Discord summaries can report simulation progress during `run`.
Configure them as described in
[Discord notifications](CONFIGURATION.md#discord-notifications), set
`EWS_DISCORD_WEBHOOK_URL` in the VM environment, and add
`--env EWS_DISCORD_WEBHOOK_URL` to `docker run`. Keep the webhook out of YAML,
image arguments, and Git. Saved logs and artifacts remain the source of truth
for execution state.

## Semantic retrieval metadata

Keep `OUTPUT/artifacts.json` with the uploaded output. External download tools
should consume [the semantic artifact catalog](ARTIFACTS.md) for figures,
analysis, and compute reports, not hardcode their internal EWS locations. The
cloud-experiments selectors feature-detect that catalog; legacy uploads without
it require a literal `--path` selected from `cloud-results ls RUN_ID`. Full pulls
remain the archival option, and the cloud run manifest retains the exact EWS pin.
