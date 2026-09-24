"""Local execution with independent components, bounded workers, and safe resumption."""

from __future__ import annotations

import inspect
import multiprocessing
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .components import construct, resolve_type
from .config import ExperimentConfig, load_config
from .jobs import RunSpec, plan_runs
from .rng import make_rngs
from .storage import (
    SCHEMA_VERSION,
    Recorder,
    RunStore,
    StorageError,
    atomic_json,
    atomic_text,
    digest_file,
    fingerprint,
    read_json,
)

_STOP_EVENT: Any = threading.Event()


@dataclass
class RunReport:
    """Counts for a single execution request; failures retain per-run explanations."""

    completed: int = 0
    skipped: int = 0
    paused: int = 0
    failed: int = 0
    pending: int = 0
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def add(self, result: tuple[str, str, str | None]) -> None:
        run_id, status, error = result
        setattr(self, status, getattr(self, status) + 1)
        if error:
            self.errors[run_id] = error


@contextmanager
def _experiment_lock(root: Path):
    """An OS advisory lock is automatically released even after process termination."""
    path = root / ".lock"
    with path.open("a+b") as stream:
        if os.name == "posix":
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StorageError(f"Another executor is using {root}") from exc
        else:
            import msvcrt

            stream.seek(0)
            if not stream.read(1):
                stream.write(b" ")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise StorageError(f"Another executor is using {root}") from exc
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            else:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _preflight(specs: list[RunSpec]) -> dict[str, str]:
    from .protocols import OfflineProtocol, OnlineProtocol

    sources: dict[str, str] = {}
    classes: dict[str, Any] = {}
    hashed: dict[str, str] = {}

    def hash_source(label: str, path: str) -> None:
        if path not in hashed:
            hashed[path] = digest_file(Path(path))
        sources[label] = hashed[path]

    for spec in specs:
        for component in (spec.algorithm, spec.data, spec.protocol):
            if component.type not in classes:
                cls = resolve_type(component.type)
                if not isinstance(cls, type):
                    raise ValueError(f"{component.type} must identify a component class")
                for method in ("state_dict", "load_state_dict"):
                    if not callable(getattr(cls, method, None)):
                        raise ValueError(f"{component.type} must implement {method}()")
                source = inspect.getsourcefile(cls)
                if source:
                    hash_source(component.type, source)
                classes[component.type] = cls
            cls = classes[component.type]
            try:
                signature = inspect.signature(cls)
            except ValueError:
                signature = None
            if signature is not None:
                signature.bind(rng=None, **component.params)
        if spec.data.type in {"csv", "experiments_wo_stress.data:CSVDataGenerator"}:
            path = spec.data.params["path"]
            hash_source(f"input:{path}", path)
        if spec.protocol.type in {"trial", "experiments_wo_stress.protocols:TrialProtocol"}:
            function = spec.protocol.params["function"]
            source = inspect.getsourcefile(resolve_type(function))
            if source:
                hash_source(function, source)
        protocol_cls = classes[spec.protocol.type]
        for method in ("initialize", "advance", "is_finished"):
            if not callable(getattr(protocol_cls, method, None)):
                raise ValueError(f"{spec.protocol.type} must implement {method}()")
        algorithm_cls = classes[spec.algorithm.type]
        methods = (
            ("act", "observe")
            if issubclass(protocol_cls, OnlineProtocol)
            else (("fit",) if issubclass(protocol_cls, OfflineProtocol) else ())
        )
        for method in methods:
            if not callable(getattr(algorithm_cls, method, None)):
                raise ValueError(
                    f"{spec.algorithm.type} must implement {method}() for {spec.protocol.type}"
                )
        data_cls = classes[spec.data.type]
        if not callable(getattr(data_cls, "generate", None)):
            raise ValueError(f"{spec.data.type} must implement generate()")
    return sources


def _provenance(config: ExperimentConfig, sources: dict[str, str]) -> dict[str, Any]:
    package = Path(__file__).parent
    implementation = {path.name: digest_file(path) for path in sorted(package.glob("*.py"))}
    # Include adjacent project modules so changes in helper functions invalidate resume too.
    project = Path(config.source_dir)
    for path in sorted(project.glob("*.py")):
        sources[f"project:{path.name}"] = digest_file(path)
    for group in config.runs:
        if group.get("planner", "grid") != "grid":
            planner = group["planner"]
            source = inspect.getsourcefile(resolve_type(planner))
            if source:
                sources[f"planner:{planner}"] = digest_file(Path(source))
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
        "platform": platform.platform(),
    }
    revision = None
    dirty = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=project,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "implementation": implementation,
        "components": sources,
        "environment": environment,
        "revision": revision,
        "dirty": dirty,
        "host": socket.gethostname(),
    }


def _prepare(
    root: Path, config: ExperimentConfig, specs: list[RunSpec], provenance: dict[str, Any]
) -> None:
    signature = fingerprint(config.simulation_dict())
    code_signature = fingerprint(
        {key: provenance[key] for key in ("implementation", "components", "environment")}
    )
    path = root / "metadata.json"
    if path.exists():
        previous = read_json(path)
        if previous.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("The output directory uses an incompatible artifact schema")
        if previous["simulation_fingerprint"] != signature:
            raise ValueError(
                "Configuration mismatch: use a new output directory for this experiment"
            )
        if previous["implementation_fingerprint"] != code_signature:
            raise ValueError(
                "Implementation/environment mismatch: use the original code or a new output directory"
            )
        if previous["run_ids"] != sorted(spec.run_id for spec in specs):
            raise ValueError("Run plan mismatch in saved experiment metadata")
    else:
        # Existing unowned results are not adopted as a fresh experiment.
        if any(path.name != ".lock" for path in root.iterdir()):
            raise ValueError("Output directory is not empty and has no experiment metadata")
        atomic_json(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "name": config.name,
                "simulation_fingerprint": signature,
                "implementation_fingerprint": code_signature,
                "run_ids": sorted(spec.run_id for spec in specs),
                "provenance": provenance,
            },
        )
    atomic_text(root / "config.resolved.yml", yaml.safe_dump(config.to_dict(), sort_keys=False))
    for spec in specs:
        path = root / "runs" / spec.run_id / "metadata.json"
        value = {
            "schema_version": SCHEMA_VERSION,
            "spec": spec.to_dict(),
            "simulation_fingerprint": signature,
            "implementation_fingerprint": code_signature,
        }
        if path.exists():
            if read_json(path) != value:
                raise ValueError(f"Run metadata mismatch for {spec.run_id}")
        else:
            atomic_json(path, value)


def _initialize_worker(event: Any) -> None:
    global _STOP_EVENT
    _STOP_EVENT = event
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _execute(
    spec_dict: dict[str, Any],
    root: str,
    execution: dict[str, Any],
    recording: dict[str, Any],
    max_steps: int | None,
) -> tuple[str, str, str | None]:
    spec = RunSpec.from_dict(spec_dict)
    store = None
    try:
        store = RunStore(
            Path(root) / "runs" / spec.run_id,
            compression=execution.get("compression", False),
            keep_checkpoints=execution.get("keep_checkpoints", 2),
        )
        if store.completed():
            return spec.run_id, "skipped", None
        rngs = make_rngs(spec)
        algorithm = construct(spec.algorithm, rngs["algorithm"])
        data = construct(spec.data, rngs["data"])
        protocol = construct(spec.protocol, rngs["protocol"])
        restored, manifest, restored_step = store.restore()
        components = {"algorithm": algorithm, "data": data, "protocol": protocol}
        if restored is None:
            protocol.initialize(algorithm, data)
        else:
            for name, component in components.items():
                component.load_state_dict(restored[name])
            for name, rng in rngs.items():
                rng.bit_generator.state = restored["rngs"][name]
            if protocol.step != restored_step:
                raise StorageError("Checkpoint protocol step does not match result boundary")
        recorder = Recorder(store, recording, manifest)
        checkpoint_time = time.monotonic()
        checkpoint_step = protocol.step
        started_step = protocol.step

        def save(status: str = "running") -> None:
            nonlocal checkpoint_time, checkpoint_step
            recorder.flush()
            state = {name: component.state_dict() for name, component in components.items()}
            state["rngs"] = {name: rng.bit_generator.state for name, rng in rngs.items()}
            store.checkpoint(state, recorder.manifest, protocol.step, status=status)
            checkpoint_time, checkpoint_step = time.monotonic(), protocol.step

        while not protocol.is_finished():
            if _STOP_EVENT.is_set() or (
                max_steps is not None and protocol.step - started_step >= max_steps
            ):
                save("paused")
                return spec.run_id, "paused", None
            previous_step = protocol.step
            observations = protocol.advance(algorithm, data)
            if protocol.step != previous_step + 1:
                raise ValueError("Protocol.advance() must increment step by exactly one")
            recorder.record(protocol.step, observations, final=protocol.is_finished())
            seconds = execution.get("checkpoint_seconds", 120)
            steps = execution.get("checkpoint_steps")
            due = seconds is not None and time.monotonic() - checkpoint_time >= seconds
            due = due or (steps is not None and protocol.step - checkpoint_step >= steps)
            if due and not protocol.is_finished():
                save()
        recorder.flush()
        store.finish(recorder.manifest, protocol.step)
        return spec.run_id, "completed", None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if store is not None and store.progress.get("status") != "completed":
            try:
                store.fail(error, traceback.format_exc())
            except OSError:
                pass
        return spec.run_id, "failed", error


def run_experiment(
    config: ExperimentConfig | str | Path,
    output_dir: str | Path,
    *,
    workers: int | None = None,
    max_steps: int | None = None,
) -> RunReport:
    """Run or resume an experiment. ``max_steps`` pauses each run after new steps."""
    global _STOP_EVENT
    if not isinstance(config, ExperimentConfig):
        config = load_config(config)
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0
    ):
        raise ValueError("max_steps must be a nonnegative integer")
    execution = dict(config.execution)
    workers = execution.get("workers", 1) if workers is None else workers
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    source_dir = str(config.source_dir)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    specs = plan_runs(config)
    sources = _preflight(specs)
    provenance = _provenance(config, sources)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = RunReport()
    context = multiprocessing.get_context("spawn")
    _STOP_EVENT = context.Event() if workers > 1 else threading.Event()
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: _STOP_EVENT.set())
    try:
        with _experiment_lock(root):
            _prepare(root, config, specs, provenance)
            if workers == 1:
                for index, spec in enumerate(specs):
                    if _STOP_EVENT.is_set():
                        report.pending = len(specs) - index
                        break
                    report.add(
                        _execute(
                            spec.to_dict(), str(root), execution, dict(config.recording), max_steps
                        )
                    )
            else:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=context,
                    initializer=_initialize_worker,
                    initargs=(_STOP_EVENT,),
                ) as pool:
                    remaining = iter(specs)
                    active = {}

                    def submit() -> bool:
                        spec = next(remaining, None)
                        if spec is None:
                            return False
                        future = pool.submit(
                            _execute,
                            spec.to_dict(),
                            str(root),
                            execution,
                            dict(config.recording),
                            max_steps,
                        )
                        active[future] = spec.run_id
                        return True

                    for _ in range(min(workers, len(specs))):
                        submit()
                    while active:
                        done, _ = wait(active, return_when=FIRST_COMPLETED)
                        for future in done:
                            run_id = active.pop(future)
                            try:
                                report.add(future.result())
                            except Exception as exc:
                                report.add((run_id, "failed", f"Worker failure: {exc}"))
                                _STOP_EVENT.set()
                            if not _STOP_EVENT.is_set():
                                submit()
                    report.pending = sum(1 for _ in remaining)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return report


def inspect_experiment(output_dir: str | Path) -> dict[str, Any]:
    """Inspect stored status without importing or constructing scientific components."""
    root = Path(output_dir)
    metadata = read_json(root / "metadata.json")
    counts = dict.fromkeys(("pending", "running", "paused", "completed", "failed", "corrupt"), 0)
    runs = []
    for run_id in metadata["run_ids"]:
        path = root / "runs" / run_id / "progress.json"
        try:
            progress = read_json(path) if path.exists() else {"status": "pending", "step": 0}
            status = progress["status"]
            if status == "completed":
                RunStore(path.parent).completed()
            counts[status] += 1
            runs.append({"run_id": run_id, **progress})
        except (StorageError, KeyError, ValueError) as exc:
            counts["corrupt"] += 1
            runs.append({"run_id": run_id, "status": "corrupt", "error": str(exc)})
    return {"name": metadata["name"], "counts": counts, "runs": runs}
