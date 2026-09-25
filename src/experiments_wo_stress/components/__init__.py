"""Public extension contracts and component construction.

NullAlgorithm remains available here for existing component identifiers; its
implementation belongs to the built-in trial protocol support.
"""

from typing import TYPE_CHECKING

from .contracts import (
    Checkpointable,
    DataGenerator,
    EnvironmentStateAdapter,
    Feedback,
    InteractionProtocol,
    OfflineAlgorithm,
    OnlineAlgorithm,
    StateMixin,
)
from .loading import (
    construct,
    constructor_kwargs,
    create_instance,
    resolve_type,
    validate_component,
)

__all__ = [
    "Checkpointable",
    "DataGenerator",
    "EnvironmentStateAdapter",
    "Feedback",
    "InteractionProtocol",
    "NullAlgorithm",
    "OfflineAlgorithm",
    "OnlineAlgorithm",
    "StateMixin",
    "construct",
    "constructor_kwargs",
    "create_instance",
    "resolve_type",
    "validate_component",
]


if TYPE_CHECKING:
    from ..builtins.protocols import NullAlgorithm


def __getattr__(name: str):
    # Resolve the legacy alias lazily so importing contracts does not import
    # their implementations or give analysis a dependency on simulation code.
    if name == "NullAlgorithm":
        from ..builtins.protocols import NullAlgorithm

        return NullAlgorithm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
