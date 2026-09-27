# Portable CPU/NumPy continuation

Normal EWS compatibility remains conservative. Disposable CPU workers can opt in
with `ews run CONFIG --output OUTPUT --portable`, or set
`execution.continuation: portable_numpy` in YAML. The default is `strict`.
Both policies use ordinary local files and EWS's existing variant selection,
checkpoint validation, generation fallback, and safe protocol boundaries.

## Compatibility contract

Portable mode supports the built-in `numpy` checkpoint backend on CPU. It rejects
`gpu_ids` and other checkpoint backend types before executing runs. A study using
this mode declares that its components' saved state consists of supported NumPy
arrays and ordinary serializable values and can be restored with its normal
`load_state_dict` methods. It does not provide generic native ML, GPU, external
process, device, file-handle, or Gymnasium environment serialization.

The portable environment signature (`portable-numpy-v1`) includes:

- Exact Python version, implementation, and SOABI.
- Operating system family, architecture, endianness, pointer width, and libc
  identity/version.
- Exact installed distribution names/versions, including NumPy and numerical
  dependencies. EWS itself is covered by its execution implementation hashes.

Kernel release, hostname, CPU count, scheduling, and cloud machine identifiers
are not compatibility inputs. Host details remain diagnostic provenance. This
allows compatible x86-64 CPU workers with different hostnames, kernel revisions,
and core counts to reuse the same completed variant and checkpoint.

Scientific specifications, RNG streams, recording selection, tracked source and
declared dependencies/inputs, execution implementation, and budget-extension
rules are unchanged. Declare helper/input dependencies as described in the
configuration guide; EWS cannot discover arbitrary undeclared external state.
Resolved dependency/input paths remain significant: keep `/work/source`,
`/work/.ews`, and `/work/output` stable across cloud attempts. A source or
environment mismatch selects a separate retained variant, never an incompatible
checkpoint. Switching strict/portable policies also selects separate variants.
Old outputs remain inspectable without adopting portable mode.

This is a checkpoint decoding and compatible-state continuation contract, not a
promise of bitwise results across every CPU model, BLAS build, wheel rebuild,
threading configuration, or native-library implementation. EWS does not hash
system shared libraries or recreate environments. Studies requiring stronger
numerical guarantees must control these independently and should retain strict
compatibility until they have validated their own portability.

## Cloud ownership

[`cloud-experiments`](https://github.com/m1gwings/cloud-experiments) owns study
identity, archived source, environment recreation, whole-output restore, VM
leases, and final persistence. Its repeated `cloud-run CONFIG` restores one
logical output lineage, then invokes normal EWS with `--portable`. EWS alone
decides completed-run reuse, compatible checkpoints, and new variants. A cloud
environment lock mismatch fails setup before loading checkpoints; use `--fresh`
deliberately when that environment cannot be recreated.

Object Storage holds recovery snapshots; EWS executes on a VM's ordinary local
filesystem. EWS provides `create_snapshot`, `validate_snapshot`, and
`restore_snapshot` from `experiments_wo_stress.storage` for wrappers to use.
The [versioned recovery contract](CLOUD.md#versioned-recovery-snapshots) defines a
sealed checksum inventory and remote commit marker. It requires a stopped writer
and EWS's exclusive output lock, excludes unpublished objects, preserves valid
checkpoint predecessors, and rejects unsupported contract versions. Restore to a
new directory and let normal EWS validation select compatible state. Never allow
two writers to share one lineage. Use the [semantic artifact catalog](ARTIFACTS.md)
for result selection; cloud code should not duplicate EWS's internal paths.

Persistence compatibility and numerical portability are separate contracts. A
valid recovery snapshot can preserve native checkpoint bytes without loading
their backend; `portable_numpy` still rejects GPU/custom-backend execution. An
intentionally pruned trajectory stays pruned after restore, with its receipt and
retained derivations. Changed analysis reuses the nearest valid retained ancestor;
if the necessary raw trajectory or continuation prefix is gone, `ews run` starts
the required simulation afresh. Portable mode does not relax that rule.

## Interruption and loss bounds

EWS catches SIGINT/SIGTERM and requests checkpointing at complete protocol steps.
Cloud timeout/cancellation first requests graceful interruption, waits a bounded
grace period, then forcibly stops the process group if needed. Normal graceful
shutdown persists a safe boundary; forced termination may lose work since the
previous committed checkpoint. Existing generation checks reject incomplete
checkpoints and can recover a preceding valid generation. A cloud upload failure
can lose the whole attempt's new progress; it must not keep a VM alive forever.

`tests/test_portability.py` checks resumed NumPy trajectories against uninterrupted
execution across simulated host/kernel changes, completed reuse, source/package
invalidation, architecture/Python mismatches, policy separation, and explicit
GPU/custom-backend rejection. These tests create no cloud resources.
