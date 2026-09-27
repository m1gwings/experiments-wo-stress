"""Durable experiments, run checkpoints, and immutable scientific artifacts.

The exports retain the historical storage API. Implementation code imports the
owning modules so that persistence dependencies remain visible.
"""

from .checkpoints import CheckpointBackend, NumPyCheckpointBackend
from .experiment import ExperimentStore, iter_completed_runs, load_instance, save_instance
from .files import (
    EXPERIMENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    StorageError,
    atomic_json,
    atomic_text,
    digest_file,
    fingerprint,
    read_json,
    write_arrays,
)
from .recovery import create_snapshot, restore_snapshot, validate_snapshot
from .run import Recorder, RunStore, validate_results

__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "CheckpointBackend",
    "ExperimentStore",
    "NumPyCheckpointBackend",
    "Recorder",
    "RunStore",
    "StorageError",
    "atomic_json",
    "atomic_text",
    "create_snapshot",
    "digest_file",
    "fingerprint",
    "iter_completed_runs",
    "load_instance",
    "read_json",
    "restore_snapshot",
    "save_instance",
    "validate_snapshot",
    "validate_results",
    "write_arrays",
]
