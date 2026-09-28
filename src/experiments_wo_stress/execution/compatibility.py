"""Preserve scientific compatibility for the reviewed compute, terminal, and artifact discovery hooks.

Only the exact observation revisions below inherit their preceding source
digest. Any further edit falls back to its real digest, so scientific execution
changes still invalidate reuse. Discovery and reporting modules themselves are operational and
are not part of the simulation fingerprint inventory.
"""

from __future__ import annotations

from collections.abc import Mapping

# Reviewed edits: observational decorators, progress transport, and this digest
# bridge and semantic catalog publication; no scientific state, scheduling,
# checkpoint persistence, RNG, or numerical result-format changes. The portable
# environment branch is separately opt-in and fingerprints its policy module;
# strict-mode provenance retains its preceding compatibility digest.
# The metric cache bridge changes ZIP storage only; decoded arrays and keys are unchanged.
_OBSERVATIONAL_REVISIONS = {
    "analysis/cache.py": (
        "67cc05a21899d99a9e0edc2409341733546ebde6013fd5be359a28dfe4ad08e4",
        "c09be47b7c6b2d28c8e08095e3369a09ea1fbc8b7e701afcc836e16c7372444f",
    ),
    "analysis/pipeline.py": (
        "a5bd774ccbfae39e10a4d8251fa11463155ac8e3b27ad8812a7be61d2f8c31bd",
        "06df1e1844a8b1d3331b6c85fbb15f27c771a7328b5dc3f158bec9a69b56df51",
    ),
    "storage/experiment.py": (
        "7a90349cd560b36a66a95c67d0a801204d3135f1a22547d4e81dbeb0fde8a913",
        "001a4f83ab819b7d32ba14fb3e12f0c47b0247d96bb18c6906aded295bdb7ac6",
    ),
    # Startup credit for retained analysis work changes only terminal observation.
    "execution/coordinator.py": (
        "5afe16f797c7948be3bc502206b53701378e72021631b259c2e8e32d48bd131d",
        "0ae37851113dcf1b8b3de08ff0a86c749696b9c9d5bf52d67caeb910413c62dc",
    ),
    "execution/resources.py": (
        "0969b5c26b530ccec9dc414803f8ae3c6f52cbe8aafb6c9b6c9cc39befdb2602",
        "c59620c438e444bf6bc21c60af190ace4433eee1d7a24f9d0a0c3335ca8e4d95",
    ),
    "execution/provenance.py": (
        "4a39808844f4734014827095a9fc3333bcece81e21bd6373f2800f6edca97cdd",
        "e31edfdb26117bdd868814bb1350056cef6666861cab2a95d4c4df823376d1a2",
    ),
    "analysis/figures.py": (
        "02d4bd1df72b3f5d53dbab25dcfde1392984e5c3ad2c17b2fd88f17719334a0c",
        "5a63ebaa2b911919155b346c1898260af7f3e844148914361a9f9ac77b1b6583",
    ),
    # Retrying a pruning receipt after snapshot restore only skips fsync of an
    # already absent, intentionally empty results directory.
    "storage/trajectories.py": (
        "e6fa4f80a52ca460ae814946ac7be28a74bab48fd700bae089fa90350d429407",
        "1845acf3e62ac0b324323056f6612f759a799e525e4abb11cc09241492614dea",
    ),
}

# Exact retention-only revision: receipt publication, result bytes, checkpoint
# encoding/decoding, RNG state, and scientific execution are unchanged. Bridge
# the former effective digest so committed CPU variants stay selectable.
_SETTLED_TRAJECTORY_REVISION = "0f4caf90a11a55f8f4b8437e59ff4833de413414359531f7cbf0d3ec6a0936ca"


def implementation_digest(name: str, digest: str) -> str:
    """Bridge exact reviewed operational edits; conservatively hash every other edit."""
    if name == "storage/trajectories.py" and digest == _SETTLED_TRAJECTORY_REVISION:
        return _OBSERVATIONAL_REVISIONS[name][1]
    reviewed = _OBSERVATIONAL_REVISIONS.get(name)
    return reviewed[1] if reviewed and digest == reviewed[0] else digest


# The 047fbc5 CPU writer used the same run/checkpoint code, but its fingerprint
# included operational coordinator and provenance modules before the DAG and
# recovery catalog were split out. These exact revisions have been reviewed and
# exercised against a paused 047fbc5 CMAB run. A later edit disables the bridge.
_LEGACY_047F_BRIDGE: dict[str, tuple[str, str | None]] = {
    "execution/coordinator.py": (
        "0ae37851113dcf1b8b3de08ff0a86c749696b9c9d5bf52d67caeb910413c62dc",
        "415420bf9df2ddef783c3d07efb5ace788b13b25479607d6ae73b16fa3a89423",
    ),
    "execution/provenance.py": (
        "1db252f3ce15945dd8f592a9673cedbc2e2cc93f2299adc6cfad57b001249ae3",
        "e31edfdb26117bdd868814bb1350056cef6666861cab2a95d4c4df823376d1a2",
    ),
    "execution/resources.py": (
        "9a5d9207473eb0a9e2974b80c78300d184e8d67284c2cb0c9b174a3b8ec5aeb3",
        "c59620c438e444bf6bc21c60af190ace4433eee1d7a24f9d0a0c3335ca8e4d95",
    ),
    "storage/experiment.py": (
        "04070880aacd2f0355ad1d2dd9df99114599490ed512e14d05d7cb0a8a0ab963",
        "001a4f83ab819b7d32ba14fb3e12f0c47b0247d96bb18c6906aded295bdb7ac6",
    ),
    "execution/scheduler.py": (
        "76f5fcb89e6dad0dae0d37469ca8c2587ec0fd9bda2382999064e25afd29d0d4",
        None,
    ),
    "storage/trajectories.py": (
        "1845acf3e62ac0b324323056f6612f759a799e525e4abb11cc09241492614dea",
        None,
    ),
}


# e60b9bb already contains the DAG and recovery catalog. The only strict
# simulation-fingerprint differences are these reviewed coordinator/provenance
# edits; the run writer, decoder, and scientific component code are unchanged.
_LEGACY_E60_BRIDGE: dict[str, tuple[str, str | None]] = {
    "execution/coordinator.py": (
        "0ae37851113dcf1b8b3de08ff0a86c749696b9c9d5bf52d67caeb910413c62dc",
        "a935d6a7a3bdc80f125d2c9dfcc6bf7674e6990cda1f864d594c4dfb5fef7eb6",
    ),
    "execution/provenance.py": (
        "1db252f3ce15945dd8f592a9673cedbc2e2cc93f2299adc6cfad57b001249ae3",
        "e389cc73db33ad0e9c3727a564405b5f00d8de6964327a93b2560e92f865753c",
    ),
}


def _legacy_implementation(
    implementation: Mapping[str, str], bridge: Mapping[str, tuple[str, str | None]]
) -> dict[str, str] | None:
    if any(implementation.get(name) != current for name, (current, _) in bridge.items()):
        return None
    legacy = dict(implementation)
    for name, (_, old) in bridge.items():
        if old is None:
            legacy.pop(name)
        else:
            legacy[name] = old
    return legacy


def legacy_implementations(implementation: Mapping[str, str]) -> tuple[dict[str, str], ...]:
    """Offer only reviewed 047fbc5/e60b9bb signatures for retained CPU variants."""
    return tuple(
        result
        for bridge in (_LEGACY_E60_BRIDGE, _LEGACY_047F_BRIDGE)
        if (result := _legacy_implementation(implementation, bridge)) is not None
    )
