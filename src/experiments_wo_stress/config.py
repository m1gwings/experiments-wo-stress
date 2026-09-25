"""Compatibility imports for experiment configuration."""

from .study.config import (
    ComponentSpec,
    ExperimentConfig,
    RunSpec,
    load_config,
)

__all__ = ["ComponentSpec", "ExperimentConfig", "RunSpec", "load_config"]
