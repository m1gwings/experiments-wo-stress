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
from ..storage.files import EXPERIMENT_SCHEMA_VERSION, SCHEMA_VERSION, digest_file, fingerprint
from ..study.config import ExperimentConfig
from ..study.specs import ComponentSpec, RunSpec


def preflight(specs: list[RunSpec]) -> dict[str, str]:
    """Check constructor and protocol contracts before publishing an execution request."""
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
        "execution/provenance.py",
        "storage/models.py",
        "storage/files.py",
        "storage/run.py",
        "storage/experiment.py",
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


def _component_sources(component: ComponentSpec) -> dict[str, str]:
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


@dataclass(frozen=True)
class PreparedRequest:
    """Variant identities and request metadata ready for durable publication."""

    locations: dict[str, str]
    variants: dict[str, dict[str, Any]]
    request_id: str
    metadata: dict[str, Any]


def prepare_request(
    config: ExperimentConfig, specs: list[RunSpec], provenance: dict[str, Any]
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
