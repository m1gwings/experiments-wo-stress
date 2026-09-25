"""Inspect non-identifying compute hardware without changing resource allocation.

All queries are optional and bounded. This module never imports a scientific
framework or contacts a metadata service. Capacities describe the operating
system's visible machine and filesystem, not a scheduler reservation or a
container's memory/CPU limit. GPU assignments come from execution/resources.py;
hardware discovery here does not validate or change those assignments.
"""

from __future__ import annotations

import csv
import io
import json
import os
import platform
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any


def supported() -> bool:
    """Return whether automatic compute reporting is supported on this platform."""
    return platform.system() in {"Linux", "Darwin"}


def _read_text(path: str) -> str | None:
    """Read an optional system file without making inspection an execution error."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _query(arguments: list[str]) -> str | None:
    """Run a small local system query with no shell and a short timeout."""
    try:
        completed = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def _positive_int(value: str | None) -> int | None:
    """Normalize positive integer capacities and counts, leaving unknowns null."""
    try:
        number = int(value) if value is not None else 0
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def _linux_cpu() -> tuple[str | None, int | None]:
    """Count physical package/core pairs only when every processor supplies them."""
    text = _read_text("/proc/cpuinfo") or ""
    processors = []
    for block in text.strip().split("\n\n"):
        fields = {}
        for line in block.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip()] = value.strip()
        if "processor" in fields:
            processors.append(fields)
    model = next((fields["model name"] for fields in processors if fields.get("model name")), None)
    cores = None
    if processors and all(
        fields.get("physical id") and fields.get("core id") for fields in processors
    ):
        cores = len({(fields["physical id"], fields["core id"]) for fields in processors})
    return model, cores


def _linux_ram() -> int | None:
    """Read Linux's system-visible physical memory capacity, expressed in bytes."""
    for line in (_read_text("/proc/meminfo") or "").splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == "MemTotal:" and fields[2] == "kB":
            kibibytes = _positive_int(fields[1])
            return kibibytes * 1024 if kibibytes is not None else None
    return None


def _linux_version() -> str | None:
    """Keep only the distribution description, without machine identifiers."""
    for line in (_read_text("/etc/os-release") or "").splitlines():
        key, separator, value = line.partition("=")
        if separator and key == "PRETTY_NAME":
            return value.strip().strip("\"'") or None
    return None


def _filesystem_capacity(output: Path) -> tuple[int | None, int | None]:
    """Inspect the output filesystem before its directory necessarily exists."""
    try:
        location = output.absolute()
        while not location.exists() and location.parent != location:
            location = location.parent
        usage = shutil.disk_usage(location)
        return usage.total, usage.free
    except OSError:
        return None, None


def _nvidia_inventory() -> list[dict[str, Any]]:
    """Describe NVIDIA inventory without assuming its indices equal CUDA IDs.

    CUDA's enumeration order can differ from nvidia-smi's on heterogeneous
    systems. Keep discovery and configured allocation separate instead of
    assigning an unverified model or VRAM capacity to a worker's CUDA ID.
    """
    output = _query(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    devices = []
    try:
        for row in csv.reader(io.StringIO(output or ""), skipinitialspace=True):
            if len(row) != 4:
                continue
            index, model, memory, driver = (value.strip() for value in row)
            try:
                index = int(index)
            except ValueError:
                continue
            if index < 0:
                continue
            mebibytes = _positive_int(memory)
            devices.append(
                {
                    "nvidia_smi_index": index,
                    "model": model if model not in {"", "[N/A]", "N/A"} else None,
                    "vram_bytes": mebibytes * 1024**2 if mebibytes is not None else None,
                    "driver_version": driver if driver not in {"", "[N/A]", "N/A"} else None,
                }
            )
    except csv.Error:
        pass
    return devices


def _apple_accelerator() -> list[dict[str, Any]]:
    """Read display-controller descriptions without storing serials or displays.

    Apple silicon uses unified memory, so machine RAM is retained separately;
    it is not presented as dedicated GPU VRAM or an EWS GPU allocation.
    """
    output = _query(["system_profiler", "-json", "SPDisplaysDataType"])
    try:
        payload = json.loads(output or "{}")
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    devices = payload.get("SPDisplaysDataType", [])
    if not isinstance(devices, list):
        return []
    result = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        model = device.get("sppci_model")
        if isinstance(model, str) and model:
            result.append({"model": model, "cores": _positive_int(device.get("sppci_cores"))})
    return result


def collect_machine(output: Path, workers: int, gpu_ids: list[int] | None) -> dict[str, Any]:
    """Collect a POSIX invocation's optional machine and resource information.

    Unknown fields stay null. Unsupported platforms return an empty mapping
    without warnings. The effective worker count and GPU IDs are supplied by the
    coordinator, while discovered GPUs are a separate hardware inventory.
    No usernames, hostnames, addresses, or filesystem paths are retained.
    """
    system = platform.system()
    if system not in {"Linux", "Darwin"}:
        return {}
    if system == "Linux":
        cpu_model, physical_cores = _linux_cpu()
        logical_count = os.cpu_count()
        ram = _linux_ram()
        version = _linux_version()
        apple_accelerator = []
    else:
        cpu_model = _query(["sysctl", "-n", "machdep.cpu.brand_string"])
        if cpu_model is None:
            cpu_model = _query(["sysctl", "-n", "hw.model"])
        physical_cores = _positive_int(_query(["sysctl", "-n", "hw.physicalcpu"]))
        logical_count = _positive_int(_query(["sysctl", "-n", "hw.logicalcpu"]))
        ram = _positive_int(_query(["sysctl", "-n", "hw.memsize"]))
        version = _query(["sw_vers", "-productVersion"])
        apple_accelerator = _apple_accelerator()
    total, available = _filesystem_capacity(output)
    return {
        "platform": system,
        "os_version": version,
        "kernel_version": platform.release() or None,
        "architecture": platform.machine() or None,
        "cpu_model": cpu_model,
        "physical_cpu_cores": physical_cores,
        "logical_cpu_count": logical_count,
        "ram_bytes": ram,
        "workers": workers,
        "filesystem_total_bytes": total,
        "filesystem_available_bytes": available,
        "allocated_gpu_count": len(gpu_ids or []),
        "gpu_ids": list(gpu_ids or []),
        "gpus": _nvidia_inventory() if gpu_ids else [],
        "gpu_inventory_scope": "NVIDIA inventory; nvidia-smi indices may differ from CUDA IDs",
        "apple_accelerator": apple_accelerator,
    }


def artifact_size(root: Path) -> int | None:
    """Sum regular-file lengths once, excluding symbolic links and their targets.

    This is apparent artifact size, not allocated disk blocks or a peak storage
    requirement. An unreadable or concurrently disappearing entry makes the
    measurement unavailable instead of returning a misleading partial total.
    """
    try:
        if root.is_symlink() or not root.is_dir():
            return None
        pending = [root]
        total = 0
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(Path(entry.path))
                    elif stat.S_ISREG(info.st_mode):
                        total += info.st_size
        return total
    except OSError:
        return None
