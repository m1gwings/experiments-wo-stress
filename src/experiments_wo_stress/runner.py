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

from .components import construct, create_instance, resolve_type
from .config import ExperimentConfig, load_config
from .jobs import RunSpec, plan_runs
from .notifications import ExperimentNotifier
from .rng import make_rngs
from .storage import (
    EXPERIMENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Recorder,
    RunStore,
    StorageError,
    atomic_json,
    atomic_text,
    digest_file,
    fingerprint,
    load_instance,
    read_json,
    save_instance,
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
                kwargs = {"rng": None, **component.params}
                for injected in ("instance", "logger"):
                    if injected in signature.parameters:
                        kwargs[injected] = None
                signature.bind(**kwargs)
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
    simulation_modules = (
        "artifacts.py",
        "components.py",
        "protocols.py",
        "rng.py",
        "runner.py",
        "storage.py",
    )
    implementation = {name: digest_file(package / name) for name in simulation_modules}
    project = Path(config.source_dir)
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


def _component_sources(component: Any) -> dict[str, str]:
    """Fingerprint defining/base modules and explicitly declared helper/input files."""
    cls = resolve_type(component.type)
    sources = {}
    for base in getattr(cls, "__mro__", (cls,)):
        if base is object:
            continue
        try:
            source = inspect.getsourcefile(base)
        except TypeError:
            source = None
        if source:
            sources[f"module:{base.__module__}"] = digest_file(Path(source))
            for dependency in getattr(base, "dependency_files", ()):
                path = (Path(source).parent / dependency).resolve()
                sources[f"dependency:{path}"] = digest_file(path)
    for dependency in getattr(component, "dependencies", ()):
        path = Path(dependency).resolve()
        sources[f"dependency:{path}"] = digest_file(path)
    if component.type in {"csv", "experiments_wo_stress.data:CSVDataGenerator"}:
        path = Path(component.params["path"])
        sources[f"input:{path}"] = digest_file(path)
    if component.type in {"trial", "experiments_wo_stress.protocols:TrialProtocol"}:
        function = resolve_type(component.params["function"])
        source = inspect.getsourcefile(function)
        if source:
            sources[f"function:{component.params['function']}"] = digest_file(Path(source))
    if component.type in {"gymnasium", "experiments_wo_stress.settings:GymnasiumAdapter"}:
        import importlib.metadata

        for parameter, default in (("factory", "gymnasium:make"), ("state_adapter", None)):
            target = component.params.get(parameter, default)
            if not target:
                continue
            factory = resolve_type(target)
            for base in getattr(factory, "__mro__", (factory,)):
                if base is object:
                    continue
                source = inspect.getsourcefile(base)
                if source:
                    sources[f"adapter:{target}:{base.__module__}"] = digest_file(Path(source))
                    for dependency in getattr(base, "dependency_files", ()):
                        path = (Path(source).parent / dependency).resolve()
                        sources[f"dependency:{path}"] = digest_file(path)
            package = target.split(":", 1)[0].split(".", 1)[0]
            try:
                sources[f"package:{package}"] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
    return sources


def _prepare(
    root: Path, config: ExperimentConfig, specs: list[RunSpec], provenance: dict[str, Any]
) -> dict[str, str]:
    """Select compatible retained variants; changing a request never deletes results."""
    path = root / "metadata.json"
    if path.exists():
        previous = read_json(path)
        if previous.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(
                "Legacy output is readable for analysis; use a new directory for schema-2 execution"
            )
    elif any(path.name != ".lock" for path in root.iterdir()):
        raise ValueError("Output directory is not empty and has no experiment metadata")
    locations = {}
    requests = {}
    recording = {key: config.recording[key] for key in ("every_steps", "fields")}
    if recording["fields"] is not None:
        recording["fields"] = sorted(recording["fields"])
    source_cache = {}
    for spec in specs:
        sources = {}
        extendable = spec.budget_steps is not None
        for component in (spec.algorithm, spec.data, spec.protocol):
            key = fingerprint(component.to_dict())
            if key not in source_cache:
                source_cache[key] = _component_sources(component)
            sources.update(source_cache[key])
            extendable = extendable and bool(
                getattr(resolve_type(component.type), "supports_extension", False)
            )
        scientific = spec.to_dict()
        scientific.pop("budget_steps", None)
        code_signature = fingerprint(
            {
                "implementation": provenance["implementation"],
                "components": sources,
                "environment": provenance["environment"],
            }
        )
        identity = {
            "science": scientific,
            "code": code_signature,
            "recording": recording,
            "budget": None if extendable else spec.budget_steps,
        }
        storage_id = fingerprint(identity)[:32]
        locations[spec.run_id] = storage_id
        requests[storage_id] = spec.to_dict()
        run_path = root / "runs" / storage_id / "metadata.json"
        if not run_path.exists():
            atomic_json(
                run_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "spec": spec.to_dict(),
                    "identity": identity,
                    "extendable": extendable,
                    "implementation_fingerprint": code_signature,
                },
            )
        elif read_json(run_path).get("identity") != identity:
            raise StorageError(f"Stored identity mismatch for {storage_id}")
    request = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "name": config.name,
        "run_ids": sorted(requests),
        "requests": requests,
        "provenance": provenance,
    }
    request_id = fingerprint({"requests": requests, "recording": recording})
    atomic_json(root / "requests" / f"{request_id}.json", request)
    atomic_json(path, request)
    atomic_text(root / "config.resolved.yml", yaml.safe_dump(config.to_dict(), sort_keys=False))
    return locations


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
    storage_id: str,
) -> tuple[str, str, str | None]:
    from .logging import run_logging

    spec = RunSpec.from_dict(spec_dict)
    store = None
    components = {}
    with run_logging(Path(root) / "runs" / storage_id, spec.run_id, execution) as logger:
        try:
            store = RunStore(
                Path(root) / "runs" / storage_id,
                compression=execution.get("compression", False),
                keep_checkpoints=execution.get("keep_checkpoints", 2),
            )
            metadata_path = store.directory / "metadata.json"
            metadata = read_json(metadata_path)
            if store.completion(spec.budget_steps) is not None:
                logger.info("Reusing completed results")
                return spec.run_id, "skipped", None
            rngs = make_rngs(spec)
            if metadata.get("instance_id"):
                instance = load_instance(Path(root), metadata["instance_id"])
            else:
                instance = create_instance(spec.data, rngs["instance"])
                metadata["instance_id"] = save_instance(
                    Path(root), instance, compression=store.compression
                )
                atomic_json(metadata_path, metadata)
            algorithm = construct(
                spec.algorithm, rngs["algorithm"], logger=logger.getChild("algorithm")
            )
            components["algorithm"] = algorithm
            data = construct(
                spec.data, rngs["data"], instance=instance, logger=logger.getChild("data")
            )
            components["data"] = data
            protocol = construct(
                spec.protocol, rngs["protocol"], logger=logger.getChild("protocol")
            )
            components["protocol"] = protocol
            if spec.budget_steps is not None:
                setter = getattr(protocol, "set_budget", None)
                if not callable(setter):
                    raise ValueError("A budget requires protocol.set_budget(steps)")
                setter(spec.budget_steps)
            restored, manifest, restored_step = store.restore()
            if restored is None:
                protocol.initialize(algorithm, data)
            else:
                for name, component in components.items():
                    component.load_state_dict(restored[name])
                for name, rng in rngs.items():
                    rng.bit_generator.state = restored["rngs"][name]
                if protocol.step != restored_step:
                    raise StorageError("Checkpoint protocol step does not match result boundary")
            logger.info(
                "Executing from step %s to %s", protocol.step, spec.budget_steps or "completion"
            )
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
                logger.debug("Checkpoint committed at step %s", protocol.step)

            while not protocol.is_finished():
                if _STOP_EVENT.is_set() or (
                    max_steps is not None and protocol.step - started_step >= max_steps
                ):
                    save("paused")
                    logger.info("Paused at step %s", protocol.step)
                    return spec.run_id, "paused", None
                previous_step = protocol.step
                observations = protocol.advance(algorithm, data)
                if protocol.step != previous_step + 1:
                    raise ValueError("Protocol.advance() must increment step by exactly one")
                # A request-specific final sample would break prefix invariance when extended.
                final = protocol.is_finished() and not metadata["extendable"]
                recorder.record(protocol.step, observations, final=final)
                seconds = execution.get("checkpoint_seconds", 120)
                steps = execution.get("checkpoint_steps")
                due = seconds is not None and time.monotonic() - checkpoint_time >= seconds
                due = due or (steps is not None and protocol.step - checkpoint_step >= steps)
                if due and not protocol.is_finished():
                    save()
            save()
            store.finish(recorder.manifest, protocol.step)
            logger.info("Completed at step %s", protocol.step)
            return spec.run_id, "completed", None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Run failed: %s", error)
            if store is not None:
                try:
                    store.fail(error, traceback.format_exc())
                except OSError:
                    pass
            return spec.run_id, "failed", error
        finally:
            for component in components.values():
                close = getattr(component, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        logger.exception("Component cleanup failed")


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
    notifier = ExperimentNotifier(config.notifications, config.name, len(specs))
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
            locations = _prepare(root, config, specs, provenance)
            notifier.start()
            aborted = True
            try:
                if workers == 1:
                    for index, spec in enumerate(specs):
                        if _STOP_EVENT.is_set():
                            report.pending = len(specs) - index
                            break
                        notifier.run_started(
                            spec.run_id, root / "runs" / locations[spec.run_id], spec.budget_steps
                        )
                        result = _execute(
                            spec.to_dict(),
                            str(root),
                            execution,
                            dict(config.recording),
                            max_steps,
                            locations[spec.run_id],
                        )
                        report.add(result)
                        notifier.run_finished(result)
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
                                locations[spec.run_id],
                            )
                            active[future] = spec.run_id
                            notifier.run_started(
                                spec.run_id,
                                root / "runs" / locations[spec.run_id],
                                spec.budget_steps,
                            )
                            return True

                        for _ in range(min(workers, len(specs))):
                            submit()
                        while active:
                            done, _ = wait(active, return_when=FIRST_COMPLETED)
                            for future in done:
                                run_id = active.pop(future)
                                try:
                                    result = future.result()
                                    report.add(result)
                                    notifier.run_finished(result)
                                except Exception as exc:
                                    result = (run_id, "failed", f"Worker failure: {exc}")
                                    report.add(result)
                                    notifier.run_finished(result)
                                    _STOP_EVENT.set()
                                if not _STOP_EVENT.is_set():
                                    submit()
                        report.pending = sum(1 for _ in remaining)
                aborted = False
            finally:
                notifier.finish(report.to_dict(), aborted=aborted)
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
            if path.exists():
                target = metadata.get("requests", {}).get(run_id, {}).get("budget_steps")
                completion = RunStore(path.parent).completion(target)
                if completion is not None:
                    progress = {
                        **progress,
                        "stored_status": status,
                        "stored_step": progress["step"],
                        "status": "completed",
                        "step": target or completion["step"],
                    }
                    status = "completed"
            counts[status] += 1
            runs.append({"run_id": run_id, **progress})
        except (StorageError, KeyError, ValueError) as exc:
            counts["corrupt"] += 1
            runs.append({"run_id": run_id, "status": "corrupt", "error": str(exc)})
    variants = sum(1 for path in (root / "runs").iterdir() if path.is_dir())
    return {
        "name": metadata["name"],
        "counts": counts,
        "runs": runs,
        "retained_variants": variants,
        "active_variants": len(metadata["run_ids"]),
    }
