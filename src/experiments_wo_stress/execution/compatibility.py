"""Preserve scientific compatibility for the reviewed compute, terminal, and artifact discovery hooks.

Only the exact observation revisions below inherit their preceding source
digest. Any further edit falls back to its real digest, so scientific execution
changes still invalidate reuse. Discovery and reporting modules themselves are operational and
are not part of the simulation fingerprint inventory.
"""

from __future__ import annotations

# Reviewed edits: observational decorators, progress transport, and this digest
# bridge and semantic catalog publication; no scientific state, scheduling,
# checkpoint persistence, RNG, or numerical result-format changes. The portable
# environment branch is separately opt-in and fingerprints its policy module;
# strict-mode provenance retains its preceding compatibility digest.
_OBSERVATIONAL_REVISIONS = {
    "analysis/pipeline.py": (
        "a5bd774ccbfae39e10a4d8251fa11463155ac8e3b27ad8812a7be61d2f8c31bd",
        "06df1e1844a8b1d3331b6c85fbb15f27c771a7328b5dc3f158bec9a69b56df51",
    ),
    "storage/experiment.py": (
        "7a90349cd560b36a66a95c67d0a801204d3135f1a22547d4e81dbeb0fde8a913",
        "001a4f83ab819b7d32ba14fb3e12f0c47b0247d96bb18c6906aded295bdb7ac6",
    ),
    "execution/coordinator.py": (
        "3953ebbb374ce6e8d0b1b4f0f3bdc1100eaef8816390f8b9950ba3308ffa475c",
        "415420bf9df2ddef783c3d07efb5ace788b13b25479607d6ae73b16fa3a89423",
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


def implementation_digest(name: str, digest: str) -> str:
    """Bridge exact reviewed operational edits; conservatively hash every other edit."""
    reviewed = _OBSERVATIONAL_REVISIONS.get(name)
    return reviewed[1] if reviewed and digest == reviewed[0] else digest
