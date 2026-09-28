"""Validate runnable components and describe the code behind retained run variants."""

from __future__ import annotations

import inspect
import platform
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..builtins.protocols import OfflineProtocol, OnlineProtocol
from ..components.loading import resolve_type
from ..storage.checkpoints import describe_checkpoint_backend
from ..storage.files import (
    EXPERIMENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    digest_file,
    fingerprint,
    read_json,
)
from ..study.config import ExperimentConfig
from ..study.planning import plan_runs
from ..study.specs import ComponentSpec, RunSpec
from .compatibility import implementation_digest, legacy_implementations
from .portability import portable_environment


def preflight(specs: list[RunSpec]) -> dict[str, str]:
    """Check constructor and protocol contracts before publishing an execution request."""
    sources: dict[str, str] = {}
    classes: dict[str, Any] = {}
    hashed: dict[str, str] = {}
    signatures: dict[str, Any] = {}
    validated: set[str] = set()

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
            key = fingerprint(component.to_dict())
            if key in validated:
                continue
            if component.type not in signatures:
                try:
                    signatures[component.type] = inspect.signature(cls)
                except ValueError:
                    signatures[component.type] = None
            signature = signatures[component.type]
            if signature is not None:
                kwargs = {"rng": None, **component.params}
                for injected in ("instance", "logger"):
                    if injected in signature.parameters:
                        kwargs[injected] = None
                signature.bind(**kwargs)
            validated.add(key)
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


def collect_provenance(config: ExperimentConfig, sources: dict[str, str]) -> dict[str, Any]:
    """Describe the executing implementation, tracked components, and numerical environment."""
    package = Path(__file__).resolve().parents[1]
    # Hash the code that executes and persists runs, never compatibility facades.
    # Analysis and notification edits must remain independent of simulation identity.
    simulation_modules = (
        "study/specs.py",
        "study/planning.py",
        "study/rng.py",
        "components/contracts.py",
        "components/loading.py",
        "builtins/protocols.py",
        "execution/coordinator.py",
        "execution/worker.py",
        "execution/scheduler.py",
        "execution/provenance.py",
        "execution/resources.py",
        "storage/models.py",
        "storage/files.py",
        "storage/run.py",
        "storage/experiment.py",
        "storage/trajectories.py",
    )
    implementation = {
        name: implementation_digest(name, digest_file(package / name))
        for name in simulation_modules
    }
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
    if config.execution.get("continuation") == "portable_numpy":
        environment = portable_environment(config)
        implementation["execution/portability.py"] = digest_file(
            package / "execution/portability.py"
        )
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
        # Representation is operational. Its sources are diagnostic metadata,
        # excluded from prepare_request's simulation compatibility signature.
        "checkpoint_backend": describe_checkpoint_backend(
            config.execution.get("checkpoint_backend"),
            compression=config.execution.get("compression", False),
        ),
    }


def _component_sources(
    component: ComponentSpec, hashed: dict[Path, str] | None = None
) -> dict[str, str]:
    """Fingerprint defining/base modules and explicitly declared helper/input files."""
    hashed = {} if hashed is None else hashed

    def digest(path: Path) -> str:
        path = path.resolve()
        if path not in hashed:
            hashed[path] = digest_file(path)
        return hashed[path]

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
            sources[f"module:{base.__module__}"] = digest(Path(source))
            for dependency in getattr(base, "dependency_files", ()):
                path = (Path(source).parent / dependency).resolve()
                sources[f"dependency:{path}"] = digest(path)
    for dependency in getattr(component, "dependencies", ()):
        path = Path(dependency).resolve()
        sources[f"dependency:{path}"] = digest(path)
    if component.type in {"csv", "experiments_wo_stress.data:CSVDataGenerator"}:
        path = Path(component.params["path"])
        sources[f"input:{path}"] = digest(path)
    if component.type in {"trial", "experiments_wo_stress.protocols:TrialProtocol"}:
        function = resolve_type(component.params["function"])
        source = inspect.getsourcefile(function)
        if source:
            sources[f"function:{component.params['function']}"] = digest(Path(source))
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
                    sources[f"adapter:{target}:{base.__module__}"] = digest(Path(source))
                    for dependency in getattr(base, "dependency_files", ()):
                        path = (Path(source).parent / dependency).resolve()
                        sources[f"dependency:{path}"] = digest(path)
            package = target.split(":", 1)[0].split(".", 1)[0]
            try:
                sources[f"package:{package}"] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
    return sources


@dataclass(frozen=True)
class PreparedRequest:
    """Carry the result of variant selection to the experiment store for publication.

    ``locations`` maps scientific run IDs to retained storage IDs. ``variants``
    holds each selected run directory's metadata, while ``metadata`` describes
    the complete active request saved under ``request_id``. Computing this value
    does not modify artifacts; the coordinator publishes it while holding a lock.
    """

    locations: dict[str, str]
    variants: dict[str, dict[str, Any]]
    request_id: str
    metadata: dict[str, Any]


def prepare_request(
    config: ExperimentConfig,
    specs: list[RunSpec],
    provenance: dict[str, Any],
    retained_root: Path | None = None,
) -> PreparedRequest:
    """Select variants from scientific inputs, implementation, recording, and budget.

    Continuation-capable components share one variant across budget extensions.
    This calculation performs no writes; ExperimentStore owns publication.
    """
    locations = {}
    requests = {}
    variants = {}
    recording = {key: config.recording[key] for key in ("every_steps", "fields")}
    if recording["fields"] is not None:
        recording["fields"] = sorted(recording["fields"])
    source_cache = {}
    hashed: dict[Path, str] = {}
    extension_support: dict[str, bool] = {}
    legacy_options = (
        legacy_implementations(provenance["implementation"])
        if retained_root is not None and not config.execution.get("gpu_ids")
        else ()
    )
    for spec in specs:
        sources = {}
        extendable = spec.budget_steps is not None
        for component in (spec.algorithm, spec.data, spec.protocol):
            key = fingerprint(component.to_dict())
            if key not in source_cache:
                source_cache[key] = _component_sources(component, hashed)
            sources.update(source_cache[key])
            if component.type not in extension_support:
                extension_support[component.type] = bool(
                    getattr(resolve_type(component.type), "supports_extension", False)
                )
            extendable = extendable and extension_support[component.type]
        scientific = spec.to_dict()
        scientific.pop("budget_steps", None)
        code_signature = fingerprint(
            {
                "implementation": provenance["implementation"],
                "components": sources,
                "environment": provenance["environment"],
            }
        )
        # A larger budget can reuse a variant only if every component declares
        # continuation support. Recording and code still select distinct variants.
        identity = {
            "science": scientific,
            "code": code_signature,
            "recording": recording,
            "budget": None if extendable else spec.budget_steps,
        }
        storage_id = fingerprint(identity)[:32]
        current_variant = (
            retained_root / "runs" / storage_id / "metadata.json" if retained_root else None
        )
        legacy_candidates = (
            legacy_options if current_variant is None or not current_variant.is_file() else ()
        )
        for legacy_implementation in legacy_candidates:
            legacy_code = fingerprint(
                {
                    "implementation": legacy_implementation,
                    "components": sources,
                    "environment": provenance["environment"],
                }
            )
            legacy_identity = {**identity, "code": legacy_code}
            legacy_id = fingerprint(legacy_identity)[:32]
            legacy_path = retained_root / "runs" / legacy_id / "metadata.json"
            if legacy_path.is_file() and read_json(legacy_path).get("identity") == legacy_identity:
                identity, code_signature, storage_id = legacy_identity, legacy_code, legacy_id
                break
        locations[spec.run_id] = storage_id
        requests[storage_id] = spec.to_dict()
        variants[storage_id] = {
            "schema_version": SCHEMA_VERSION,
            "spec": spec.to_dict(),
            "identity": identity,
            "extendable": extendable,
            "implementation_fingerprint": code_signature,
        }
    request = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "name": config.name,
        "run_ids": sorted(requests),
        "requests": requests,
        "provenance": provenance,
    }
    request_id = fingerprint({"requests": requests, "recording": recording})
    return PreparedRequest(locations, variants, request_id, request)


def prepare_gpu_request(
    config: ExperimentConfig,
) -> tuple[list[RunSpec], dict[str, Any], PreparedRequest]:
    """Inspect study code within an assigned GPU worker and return plain metadata.

    Planners, contract inspection, and source fingerprints can import scientific
    modules. Keeping all three here prevents GPU-enabled execution from loading
    those modules in the coordinator, which does not own a device.
    """
    specs = plan_runs(config)
    provenance = collect_provenance(config, preflight(specs))
    return specs, provenance, prepare_request(config, specs, provenance)
