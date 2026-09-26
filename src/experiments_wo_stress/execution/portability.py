"""Explicit CPU/NumPy continuation compatibility across disposable hosts."""

from __future__ import annotations

import platform
import re
import struct
import sys
import sysconfig
from importlib.metadata import distributions
from typing import Any

from ..study.config import ExperimentConfig


def portable_environment(config: ExperimentConfig) -> dict[str, Any]:
    """Describe decoding/runtime compatibility without hostname or kernel identity.

    Opt-in supports the built-in non-pickle NumPy backend on CPU only. Scientific
    source, inputs, recording, RNG and result/checkpoint validation remain owned
    by the existing provenance and storage machinery. Exact installed package
    versions are conservative compatibility inputs, not a promise of bitwise
    equality for native libraries on different CPU implementations.
    """
    backend = config.execution.get("checkpoint_backend", {}).get("type", "numpy")
    if config.execution.get("gpu_ids") is not None or backend != "numpy":
        raise ValueError(
            "Portable continuation supports CPU studies with the built-in numpy checkpoint backend only"
        )
    packages = {}
    for distribution in distributions():
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        if name != "experiments-wo-stress":
            packages[name] = distribution.version
    return {
        "continuation": "portable-numpy-v1",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "abi": sysconfig.get_config_var("SOABI"),
        "system": platform.system(),
        "machine": platform.machine().lower(),
        "byteorder": sys.byteorder,
        "pointer_bits": struct.calcsize("P") * 8,
        "libc": list(platform.libc_ver()),
        "packages": dict(sorted(packages.items())),
    }
