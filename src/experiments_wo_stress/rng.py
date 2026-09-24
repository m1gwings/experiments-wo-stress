"""Order-independent random streams for each mutable run component."""

from __future__ import annotations

import hashlib

import numpy as np

from .jobs import _BUDGET_PROTOCOLS, RunSpec, canonical_json


def make_rngs(spec: RunSpec) -> dict[str, np.random.Generator]:
    """Create separate PCG64 generators, with paired data across algorithms.

    Stream identities use scientific inputs, never worker numbers, process IDs,
    Python hashes, or traversal positions. Component seeds override the root seed.
    """
    descriptions = {
        name: {"type": getattr(spec, name).type, "params": dict(getattr(spec, name).params)}
        for name in ("algorithm", "data", "protocol")
    }
    if spec.protocol.type in _BUDGET_PROTOCOLS:
        descriptions["protocol"]["params"].pop("horizon", None)
    streams = {}
    for name in (*descriptions, "instance"):
        component = spec.data if name == "instance" else getattr(spec, name)
        identity = {
            "version": 1,
            "stream": name,
            "group": spec.group,
            "repetition": spec.repetition,
            "components": {"data": descriptions["data"]}
            if name in {"data", "instance"}
            else descriptions,
        }
        digest = hashlib.sha256(canonical_json(identity).encode()).digest()
        words = np.frombuffer(digest, dtype="<u4").tolist()
        root = spec.seed if component.seed is None else component.seed
        sequence = np.random.SeedSequence([root, *words])
        streams[name] = np.random.Generator(np.random.PCG64(sequence))
    return streams
