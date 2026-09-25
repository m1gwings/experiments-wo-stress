"""Compatibility imports for reusable bandit and Gymnasium environments."""

from .builtins.bandits import GaussianBandit, NonstationaryBandit, StationaryBandit
from .builtins.gymnasium import GymnasiumAdapter
from .components.contracts import EnvironmentStateAdapter

__all__ = [
    "EnvironmentStateAdapter",
    "GaussianBandit",
    "GymnasiumAdapter",
    "NonstationaryBandit",
    "StationaryBandit",
]
