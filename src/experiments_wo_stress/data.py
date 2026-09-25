"""Compatibility imports for built-in data generators."""

from .builtins.data import (
    CSVDataGenerator,
    NormalDataGenerator,
    NullDataGenerator,
)

__all__ = ["CSVDataGenerator", "NormalDataGenerator", "NullDataGenerator"]
