# Running a study on a small cloud VM

The current library fits **one Linux VM running Docker, with a persistent block
disk for experiment artifacts**. Start with 2–4 vCPUs and 4–8 GB RAM, then size the
machine from a representative run. This is a deployment recommendation, not a new
execution backend: the CLI still launches local workers and writes ordinary files.

Nothing in this guide provisions infrastructure or starts a paid resource.

## Provider choices

Pricing checked against official pages on **24 September 2026**. Compare the actual
region and available plan before ordering; storage, public IPs, taxes, and retained
resources can add to the compute price.

| Option | Published reference price | Fit for this project |
| --- | --- | --- |
| Hetzner shared x86 VM in Germany/Finland | June 2026 schedule: CPX22 €19.49/month, excluding VAT and IPv4 | A practical EU starting point. Benchmark sustained throughput on the chosen shared plan. |
| Hetzner lower-cost CX plan, if available | The same schedule lists CX23 at €5.49/month, excluding VAT and IPv4 | Worth checking in the console; a published tariff does not establish current order availability. |
| AWS Lightsail Linux with IPv4 | 2 vCPUs, 4 GB RAM, 80 GB SSD: $24/month | Simple bundled pricing for an existing AWS workflow, with an important sustained-CPU limit. |

The Hetzner numbers come from its [official price adjustment schedule](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/);
check [current Cloud offerings](https://www.hetzner.com/cloud/) for availability.
Lightsail's bundle is listed on [AWS pricing](https://aws.amazon.com/lightsail/pricing/).

I would first compare the smallest available Hetzner shared x86 plan with at least
4 GB RAM. Its shared CPU resources can vary with neighboring workloads; choose
dedicated resources if measured throughput warrants the higher price.
[Hetzner explains shared versus dedicated resources](https://docs.hetzner.com/cloud/servers/faq/).

For long CPU-saturating runs, do not treat Lightsail's two vCPUs as two continuously
available full cores: its $24 general-purpose plan has a 20% baseline per vCPU and
uses burst capacity above that level. Short tests may overstate sustained throughput.
[AWS baseline and burst-capacity documentation](https://docs.aws.amazon.com/lightsail/latest/userguide/baseline-cpu-performance.html)

Attach a block-storage volume when results must survive VM replacement. Hetzner
Volumes attach to one server at a time and are billed separately. Its server backups
do **not** include attached Volumes, so make a separate artifact backup.
[Hetzner Volume documentation](https://docs.hetzner.com/cloud/volumes/overview/)

## Keep the paper in its own repository

The library is a dependency. Your study repository contains its configuration,
scientific extensions, small inputs, and pinned environment:

```text
my-paper/
  experiment.yml
  paper/
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

Use component paths such as `paper.algorithms:MyAlgorithm`. With `experiment.yml`
at the repository root, the loader makes that root available for imports. The
container uses `/workspace` consistently; the output directory is `/output/study`.
Keep metric-only code in its own module so changes do not alter simulation module
fingerprints. See [the configuration guide](CONFIGURATION.md) for the interfaces.

Put the **full 40-character library Git commit** you intend to use in `.ews-revision`.
Choose a commit that contains the features your configuration requests. The container
recipe deliberately requires this value; it does not install a floating `main` branch.
Pip supports commit-pinned Git installations and recommends full hashes.
[Pip VCS documentation](https://pip.pypa.io/en/stable/topics/vcs-support/)

Your `requirements.in` should list the study's runtime packages, including NumPy
and PyYAML, plus `setuptools>=68` and `wheel` for building the pinned library wheel.
Add Matplotlib for PDF/JPG figures, Gymnasium for Gym adapters, and the dependencies
of your own components. The library itself is installed separately from its Git pin.

Generate `requirements.lock` with exact versions and hashes in the intended Python
version/platform, review it, and commit it. For example, in a matching development
environment:

```bash
python -m pip install pip-tools
python -m piptools compile --generate-hashes --allow-unsafe \
  --output-file requirements.lock requirements.in
```

`--allow-unsafe` keeps requested build packages such as setuptools in the lock.
The Dockerfile installs it with `--require-hashes`, so transitive dependencies must
also be listed and hashed. [Pip-tools usage](https://pip-tools.readthedocs.io/en/stable/),
[pip hash checking](https://pip.pypa.io/en/stable/topics/secure-installs/).

## Build the container

Copy [Dockerfile.example](../deploy/Dockerfile.example) to `Dockerfile` and
[.dockerignore.example](../deploy/.dockerignore.example) to `.dockerignore` in the
paper repository. The build context is the **paper repository**, not this library's
checkout. The Dockerfile builds the library wheel from the pinned commit and copies
the study code into the final image.

```bash
docker build \
  --build-arg EWS_REV="$(cat .ews-revision)" \
  --build-arg STUDY_REV="$(git rev-parse HEAD)" \
  --tag paper-study:locked .
```

For repeatable deployments, also pass `--build-arg PYTHON_IMAGE=...` with a pinned
Python image digest, and keep the same CPU architecture. Retain the built image or
its registry digest together with the lock and study commit. The default tag in the
example is convenient for a first build; a tag alone does not freeze the base image.
The final image does not need a copy of the library's Git repository.

The example installs the public library over HTTPS and copies an already checked-out
paper repository into the image, so neither step needs repository credentials in
the runtime container. If you adapt the build for private dependencies, use Docker
build secrets or SSH mounts rather than embedding tokens in URLs, `ARG`, or `ENV`.
[Docker build secrets](https://docs.docker.com/build/building/secrets/)

The repository's container CI job builds this template against a temporary
standalone study and checks non-root execution, a durable output mount,
two-worker pause/resume, inspection, figures, and reuse. Its helper is
[`scripts/check_container.py`](../scripts/check_container.py).

The recipe has an exec-form `ENTRYPOINT`, so the `ews` process receives stop signals.
It defaults to `--help`, exposing no network service.
[Dockerfile reference](https://docs.docker.com/reference/dockerfile/)

## Run with durable output

On the VM, mount the persistent block disk using a normal Linux filesystem such as
ext4. Create an output directory on that disk and give the account running Docker
write access. The example assumes `/srv/ews-output` is that existing directory:

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

Docker bind mounts preserve files on the host independently of container removal.
They do not turn an ephemeral VM disk into persistent storage: the host path must
be on the disk you intend to retain.
[Docker bind-mount documentation](https://docs.docker.com/engine/storage/bind-mounts/)

For a deliberate stop and resume:

```bash
docker stop --timeout 120 paper-study
docker start --attach paper-study
```

The same image, configuration, and output mount let the runner select compatible
checkpoints. Docker sends SIGTERM and later SIGKILL if the grace period expires;
the library handles SIGTERM at the next protocol boundary. Set the grace period
longer than a normal step plus a checkpoint write. An indivisible long `fit()` may
still restart from its preceding checkpoint.
[Docker stop semantics](https://docs.docker.com/reference/cli/docker/container/stop/)

After execution, inspect or plot using the same image and mount:

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

Only one executor may write an output root. Run separate experiments in separate
directories. Use object storage for backups of a stopped/consistent output tree,
including `metadata.json`, `requests/`, `instances/`, and `runs/`. The live storage
contract depends on filesystem locking, atomic rename, and fsync; object-storage
URLs, object-store FUSE mounts, and multiple machines sharing one output directory
are outside its tested guarantees.

## Resources and compatibility

Start with one worker, measure it, then increase up to the VM's useful CPU capacity.
The recipe sets BLAS/OpenMP thread counts to one so each worker does not also spawn
a full numerical thread pool. If one run is dominated by large matrix operations,
measure fewer workers with more numerical threads instead.

Budget memory for each worker's 16 MiB default result buffer **plus** algorithm,
environment, input, and checkpoint state. Analysis loads one run and grouped metric
summaries, so long trajectories and many groups can exceed the simulation's memory.
Disk planning must include instances, retained variants/checkpoints, metric caches,
tables, and figures. Worker count is not a bound on total analysis memory.

**A Docker image does not guarantee portable checkpoint reuse between machines.**
The current simulation compatibility fingerprint includes Python/NumPy versions
and `platform.platform()`, including the host kernel reported inside a container.
Resolved input/dependency paths also participate in identities. Moving from a laptop
to a cloud VM, changing kernels, or changing absolute input paths can create a new
simulation variant even when the image and scientific parameters match.

Keep `/workspace` and `/output` stable, preserve the full artifacts, and expect exact
continuation only when the recorded compatibility inputs match. Saved results remain
available for independent analysis on another machine; there is no supported switch
to ignore the execution compatibility checks. The practical first route is to run
and resume a study on the same VM/image, and use local copies for analysis.

## Optional Discord progress notifications

With a library revision that includes notifications, configure:

```yaml
notifications:
  discord:
    enabled: true
    webhook_env: EWS_DISCORD_WEBHOOK_URL
    interval_seconds: 300
    timeout_seconds: 5
```

Set `EWS_DISCORD_WEBHOOK_URL` in the VM session or secret manager, then add
`--env EWS_DISCORD_WEBHOOK_URL` to `docker run`. Docker copies the existing variable;
the URL need not appear in the command, YAML, Git history, or image build arguments.
The Docker ignore example excludes `.env` files. These messages are optional and
do not replace the saved logs, progress records, or checkpoint state.

Notifications describe the run stage (`run`, or the simulation stage of `build`),
not analysis or plotting. Enabling them without the configured environment variable
is a configuration error before execution; delivery failures during a run produce
warnings without failing the simulation. Omit the block or set `enabled: false` to
run without notification credentials.

## When a managed scheduler becomes useful

AWS Batch on EC2 Spot becomes worth evaluating when there are enough independent
jobs to justify queueing and worker provisioning. Batch itself has no additional
service charge; underlying compute and related services are billed.
[AWS Batch pricing](https://aws.amazon.com/batch/pricing/)

Spot interruption notices normally give two minutes to respond. This library has
checkpoint boundaries, but it does not implement an EC2 interruption listener,
cross-machine artifact transfer, or a distributed scheduler. A Batch/Spot setup
needs explicit durable-storage and restore handling, separate output roots, and a
tested compatibility policy before relying on retries. Managed containers alone
do not solve those requirements. For the current scale, a single VM is less work.
[AWS Spot interruption guidance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-best-practices.html)
