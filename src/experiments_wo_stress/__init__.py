"""Reproducible numerical experiments with explicit state and independent analysis."""

from .analysis.figures import plot
from .analysis.metrics import Metric, MetricResult
from .analysis.pipeline import Summary, analyze
from .components.contracts import (
    DataGenerator,
    Feedback,
    InteractionProtocol,
    OfflineAlgorithm,
    OnlineAlgorithm,
    StateMixin,
)
from .execution.coordinator import RunReport, run_experiment
from .storage.cleanup import clean_experiment
from .storage.experiment import inspect_experiment
from .storage.models import Instance, RunResult
from .study.config import ExperimentConfig, load_config
from .study.planning import GridPlanner, RunPlanner, plan_runs
from .study.specs import ComponentSpec, RunSpec, make_run_spec

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
