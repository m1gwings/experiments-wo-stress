# Semantic artifact discovery

External tools should read `OUTPUT/artifacts.json` instead of hardcoding EWS
internal paths. EWS owns this catalog and updates its role locations when its
storage layout changes. The filename, schema identifier, version, and role
meanings are the discovery contract; the locations below describe today's layout.

```json
{
  "schema": "experiments-wo-stress/artifacts",
  "schema_version": 1,
  "artifacts": {
    "figures": {"path": "analysis/figures", "kind": "directory", "optional": true},
    "analysis": {"path": "analysis", "kind": "directory", "optional": true},
    "compute_report": {"path": "compute/summary.md", "kind": "file", "optional": true},
    "compute": {"path": "compute", "kind": "directory", "optional": true},
    "runs": {"path": "runs", "kind": "directory", "optional": true},
    "instances": {"path": "instances", "kind": "directory", "optional": true},
    "requests": {"path": "requests", "kind": "directory", "optional": true}
  }
}
```

## Version 1 rules

- `schema` is the exact identifier above; `schema_version` is integer `1`.
  `artifacts` maps semantic role names to descriptors with required `path`,
  `kind`, and boolean `optional` fields. JSON object keys must be unique.
- Paths are literal, nonempty, canonical POSIX paths relative to the directory
  containing `artifacts.json`. No absolute paths, `.`/`..` or empty components,
  trailing slash, backslash, colon, or nonprintable characters are allowed.
  There are no globs, URLs, or executable instructions.
- `kind` is `file` (exact object) or `directory` (its descendants). A directory
  may contain nested groups: `analysis` includes figures and derived caches.
  `runs` includes retained trajectories, checkpoints, progress, and per-run logs;
  these are not separate top-level groups. `instances` holds immutable inputs,
  `requests` retained execution requests, and `compute` all compute history.
- `compute_report` means the human-readable compute summary. Its absence is
  normal when reporting was unavailable; `compute` can also be absent.
- This is a location catalog, **not a presence snapshot, checksum inventory, or
  completion marker**. Every current role is optional. Missing or empty locations
  are valid; readers consult their local filesystem or stored object listing.
  A missing role means it was not published by this producer. A future descriptor
  with `optional: false` requires an error if no matching artifact is available.
- Consumers may ignore unknown roles and additional fields. They must reject
  unsupported schema versions and unsafe paths, without guessing a layout.
  Additive roles retain version 1; incompatible semantics require a new version.

## Publication and compatibility

EWS atomically publishes the catalog when it publishes an execution request,
finishes analysis (also called by plotting), or regenerates a compute report.
The same locations are published regardless of which workflow runs first or
which optional outputs exist. Publication writes deterministic, sorted JSON with
no timestamps, configuration, environment, credentials, or host paths, and never
scans the output tree. Concurrent writers of this EWS revision publish identical
bytes; cleanup and manual file removal cannot leave stale presence flags.
The catalog does not change scientific identities or numerical/cache formats.

Existing outputs remain readable and inspectable without the catalog. Running
analysis or plotting on compatible saved results adds it; inspection alone does
not modify the output. There is no output-layout migration. Existing execution
and checkpoint compatibility rules still apply.

`cloud-experiments` discovers the single catalog under a stored run's captured
`artifacts/` tree, including custom output roots, and joins each role path to that
catalog's parent. `--plots`, `--analysis`, and `--report` select `figures`,
`analysis`, and `compute_report`. Optional missing files are a successful no-op.
For old uploads without a catalog, use `cloud-results ls RUN_ID` followed by
`cloud-results pull RUN_ID --path RELATIVE_PATH`; no legacy path mapping is
silently applied. Multiple catalogs require explicit `--path` selection too.
Plain `cloud-results pull RUN_ID` remains the full archival pull.

Cloud tooling feature-detects this schema per run, not a global EWS commit or
package version. Its separate run manifest continues to record the exact EWS
commit. Consult the current
[cloud-experiments README](https://github.com/m1gwings/cloud-experiments)
and CLI help for retrieval syntax and limits (including its 64 KiB catalog cap).
