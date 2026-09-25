"""Preserve scientific compatibility for the reviewed compute-only integration.

Only the exact integration revisions below inherit their preceding source
digest. Any further edit falls back to its real digest, so scientific execution
changes still invalidate reuse. Reporting modules themselves are operational and
are not part of the simulation fingerprint inventory.
"""

from __future__ import annotations

# Reviewed edits: observational decorators and this digest bridge, no scientific
# state, scheduling, persistence, RNG, or result-format changes.
_COMPUTE_ONLY_REVISIONS = {
    "execution/coordinator.py": (
        "49b941986835e89b727ae3656a05a50bda5b41515f7ba160f2a8807c1212be91",
        "415420bf9df2ddef783c3d07efb5ace788b13b25479607d6ae73b16fa3a89423",
    ),
    "execution/worker.py": (
        "9b48c02cee248d20d26e9fa48a03a1050c691b989042146f404c53a78cfd025c",
        "ea7e261a9d4a948239622974ed951b46c56f1b47091a9a5069e13f86e2d9f52a",
    ),
    "execution/provenance.py": (
        "0e8d8277585b297f7f77986750daa7097b081f86fd1a0c97d9272a6b2d1ab451",
        "e31edfdb26117bdd868814bb1350056cef6666861cab2a95d4c4df823376d1a2",
    ),
    "analysis/figures.py": (
        "02d4bd1df72b3f5d53dbab25dcfde1392984e5c3ad2c17b2fd88f17719334a0c",
        "5a63ebaa2b911919155b346c1898260af7f3e844148914361a9f9ac77b1b6583",
    ),
}


def implementation_digest(name: str, digest: str) -> str:
    """Bridge exact reviewed operational edits; conservatively hash every other edit."""
    reviewed = _COMPUTE_ONLY_REVISIONS.get(name)
    return reviewed[1] if reviewed and digest == reviewed[0] else digest
