"""Reproducible numerical experiments with explicit state and independent analysis."""

from .analysis import Summary, analyze
from .artifacts import Instance, RunResult
from .cleanup import clean_experiment
from .components import (
    DataGenerator,
    Feedback,
    InteractionProtocol,
    OfflineAlgorithm,
    OnlineAlgorithm,
    StateMixin,
)
from .config import ExperimentConfig, load_config
from .jobs import ComponentSpec, GridPlanner, RunPlanner, RunSpec, make_run_spec, plan_runs
from .metrics import Metric, MetricResult
from .plotting import plot
from .runner import RunReport, inspect_experiment, run_experiment

__version__ = "0.1.0"

__all__ = [
    "ComponentSpec",
    "DataGenerator",
    "ExperimentConfig",
    "Feedback",
    "GridPlanner",
    "InteractionProtocol",
    "Instance",
    "Metric",
    "MetricResult",
    "OfflineAlgorithm",
    "OnlineAlgorithm",
    "RunPlanner",
    "RunReport",
    "RunResult",
    "RunSpec",
    "StateMixin",
    "Summary",
    "analyze",
    "clean_experiment",
    "inspect_experiment",
    "load_config",
    "make_run_spec",
    "plan_runs",
    "plot",
    "run_experiment",
]
