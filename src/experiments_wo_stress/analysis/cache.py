"""Immutable analysis caches, source identities, and convenience exports."""

from __future__ import annotations

import inspect
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..storage.files import atomic_json, digest_file, fingerprint, read_json


def safe_name(value: Any, description: str = "Name") -> str:
    """Validate names used as analysis artifact filenames."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{description} must use letters, digits, underscores, dots, or hyphens")
    return value


def source_identity(component: type) -> dict[str, Any]:
    """Fingerprint analysis code and explicitly declared external dependencies."""
    sources = {}
    dependencies = {}
    for base in component.__mro__:
        if base is object:
            continue
        source = inspect.getsourcefile(base)
        if source and Path(source).is_file():
            sources[base.__module__] = digest_file(Path(source))
        declared = base.__dict__.get("dependency_files", ())
        if isinstance(declared, (str, Path)):
            raise TypeError("dependency_files must be a sequence of paths")
        for value in declared:
            path = Path(value)
            if not path.is_absolute():
                if not source:
                    raise ValueError("Relative metric dependencies require a source file")
                path = Path(source).resolve().parent / path
            if not path.is_file():
                raise ValueError(f"Missing analysis dependency: {path}")
            dependencies[str(path.resolve())] = digest_file(path)
    if not sources:
        raise ValueError("Cached analysis components must have inspectable Python source")
    return {"sources": sources, "dependencies": dependencies}


def cache_identity(kind: str, **values: Any) -> dict[str, Any]:
    """Include runtime and cache implementation versions in an output identity."""
    return {
        "schema_version": 1,
        "kind": kind,
        "cache_implementation": digest_file(Path(__file__)),
        "numpy": np.__version__,
        "python": list(sys.version_info[:3]),
        **values,
    }


class AnalysisCache:
    """Own immutable cache generations and exported views for one experiment.

    Writers populate a temporary directory. A checksum manifest is published with
    it through a directory rename, so readers see either a complete generation or
    no generation. Existing generations are validated before reuse.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.root = Path(output_dir) / "analysis"
        self.current_path = self.root / "current.json"
        self.figures_dir = self.root / "figures"

    def find(self, collection: str, identity: Mapping[str, Any]) -> Path | None:
        """Return a validated generation, or None when it has not been produced."""
        directory = self.root / "cache" / collection / fingerprint(identity)
        return directory if self._validate(directory, identity) else None

    @staticmethod
    def _validate(directory: Path, identity: Mapping[str, Any]) -> bool:
        if not directory.exists():
            return False
        manifest_path = directory / "cache.json"
        if not manifest_path.is_file():
            raise ValueError(f"Incomplete analysis cache: {directory}")
        manifest = read_json(manifest_path)
        if manifest.get("identity") != identity or not isinstance(manifest.get("files"), dict):
            raise ValueError(f"Analysis cache identity mismatch: {directory}")
        actual_files = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if actual_files != set(manifest["files"]):
            raise ValueError(f"Analysis cache file manifest mismatch: {directory}")
        for filename, expected in manifest["files"].items():
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid analysis cache path: {filename}")
            path = directory / relative
            if not path.is_file() or digest_file(path) != expected:
                raise ValueError(f"Corrupt analysis cache artifact: {path}")
        return True

    def publish(
        self, collection: str, identity: dict[str, Any], writer: Callable[[Path], None]
    ) -> Path:
        """Publish immutable, checksum-validated files through a directory rename."""
        parent = (self.root / "cache" / collection).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        destination = parent / fingerprint(identity)
        if self._validate(destination, identity):
            return destination
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
        try:
            writer(temporary)
            checksums = {
                path.relative_to(temporary).as_posix(): digest_file(path)
                for path in sorted(temporary.rglob("*"))
                if path.is_file()
            }
            atomic_json(temporary / "cache.json", {"identity": identity, "files": checksums})
            try:
                os.rename(temporary, destination)
            except OSError:
                if not self._validate(destination, identity):
                    raise
            return destination
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def export(self, source: Path, filenames: list[str], *, figures: bool = False) -> list[Path]:
        """Refresh convenience paths while preserving all immutable cache generations."""
        destination = self.figures_dir if figures else self.root
        destination.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            target = destination / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".export-", dir=target.parent)
            os.close(descriptor)
            try:
                shutil.copyfile(source / filename, temporary)
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)

        return [destination / filename for filename in filenames]
