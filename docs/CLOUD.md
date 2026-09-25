# Running a study on a small cloud VM

An existing study can run on one Linux VM using the same `ews` commands as on a
laptop. This guide uses a container for the software environment and a
persistent block disk for experiment output. The library still runs local
workers and writes ordinary files; it does not need a distributed service.

Start with a small VM, then size it from a representative run. Choose storage
that survives VM replacement and back it up
separately. Provider prices and shared-CPU performance change, so check the
chosen region and plan before committing to a long run.

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

To stop and resume with the same image and mount:

```bash
docker stop --timeout 120 paper-study
docker start --attach paper-study
```

The stop grace period should allow a normal protocol step and checkpoint write.
A long offline `fit()` or callable trial is one step and may restart from its
previous checkpoint if stopped during that call. See the
[checkpoint model](ARCHITECTURE.md#checkpoints-completion-and-interruption).

Inspect results or generate figures using the same image and output mount:

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

Back up the complete output tree after stopping execution or at another
consistent point. Object storage is suitable for backups, but the live output
directory needs filesystem locking and atomic file replacement.

## Resources and compatibility

Start with one worker and increase it after measuring CPU and memory use. The
template limits BLAS/OpenMP threads to one per worker; a run dominated by large
matrix operations may work better with fewer workers and more numerical threads.
Allow room for each worker's result buffer (16 MiB by default), component state,
checkpoints, retained run variants, and analysis artifacts.

**Use the same VM and software environment when you need exact checkpoint
continuation.** Compatibility checks include Python and numerical-library
versions, platform identity, source code, and tracked input paths. Even the same
container image on another host may select a new run variant, especially if the
kernel or resolved paths differ. Keep `/workspace` and `/output` stable and
validate any migration before relying on reuse. Saved results can still be
copied for separate analysis.

## Optional progress messages

Discord summaries can report progress during `run` and the execution stage of
`build`. Configure them as described in
[Discord notifications](CONFIGURATION.md#discord-notifications), set
`EWS_DISCORD_WEBHOOK_URL` in the VM environment, and add
`--env EWS_DISCORD_WEBHOOK_URL` to `docker run`. Keep the webhook out of YAML,
image arguments, and Git. Saved logs and artifacts remain the source of truth
for execution state.
