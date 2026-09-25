"""Component aliases, constructor validation, and optional dependency injection."""

from __future__ import annotations

import logging
import unittest

import numpy as np

from experiments_wo_stress.artifacts import Instance
from experiments_wo_stress.components import (
    StateMixin,
    construct,
    create_instance,
    resolve_type,
    validate_component,
)
from experiments_wo_stress.data import NormalDataGenerator
from experiments_wo_stress.jobs import ComponentSpec
from tests.sample_components import RecordingLearner


class InjectedComponent(StateMixin):
    """Request all optional injected objects so their identities can be checked."""

    def __init__(self, *, rng, instance, logger):
        self.rng, self.instance, self.logger = rng, instance, logger


class ForwardingLearner(RecordingLearner):
    """Model a legacy constructor that forwards arbitrary keyword arguments."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ComponentLoadingTests(unittest.TestCase):
    """Resolve public component paths and validate construction/injection contracts."""

    def test_legacy_component_paths_preserve_alias_resolution(self):
        """Legacy import paths resolve to the same components as their short YAML aliases."""
        component_paths = {
            "online": "experiments_wo_stress.protocols:OnlineProtocol",
            "offline": "experiments_wo_stress.protocols:OfflineProtocol",
            "trial": "experiments_wo_stress.protocols:TrialProtocol",
            "rl": "experiments_wo_stress.protocols:RLProtocol",
            "csv": "experiments_wo_stress.data:CSVDataGenerator",
            "normal": "experiments_wo_stress.data:NormalDataGenerator",
            "null": "experiments_wo_stress.data:NullDataGenerator",
            "null_algorithm": "experiments_wo_stress.components:NullAlgorithm",
            "stationary_bandit": "experiments_wo_stress.settings:StationaryBandit",
            "gaussian_bandit": "experiments_wo_stress.settings:GaussianBandit",
            "nonstationary_bandit": "experiments_wo_stress.settings:NonstationaryBandit",
            "gymnasium": "experiments_wo_stress.settings:GymnasiumAdapter",
        }
        for alias, qualified_path in component_paths.items():
            with self.subTest(alias=alias):
                self.assertIs(resolve_type(alias), resolve_type(qualified_path))
                component_spec = ComponentSpec(qualified_path)
                self.assertEqual(component_spec.to_dict()["type"], qualified_path)

    def test_component_resolution_and_parameters(self):
        """Loading validates aliases, constructor arguments, and required component methods."""
        self.assertIs(resolve_type("normal"), NormalDataGenerator)
        instance = construct(ComponentSpec("normal", {"size": 5}), np.random.default_rng(1))
        self.assertEqual(instance.generate(None).shape, (5,))
        with self.assertRaisesRegex(TypeError, "Invalid parameters"):
            construct(ComponentSpec("normal", {"typo": 5}), np.random.default_rng(1))
        with self.assertRaisesRegex(ValueError, "Unknown component"):
            resolve_type("missing")
        with self.assertRaisesRegex(TypeError, "missing required methods"):
            validate_component(instance, ["act", "observe"])

    def test_optional_injection_preserves_legacy_forwarding_constructors(self):
        """Optional instance/logger injection also supports older forwarding constructors."""
        logger = logging.getLogger("test.component")
        rng = np.random.default_rng(1)
        instance = Instance(metadata={"value": 7})
        aware = construct(
            ComponentSpec(f"{__name__}:InjectedComponent"), rng, instance=instance, logger=logger
        )
        self.assertIs(aware.instance, instance)
        self.assertIs(aware.logger, logger)
        legacy = construct(ComponentSpec(f"{__name__}:ForwardingLearner"), rng, logger=logger)
        self.assertIs(legacy.rng, rng)
        self.assertIs(legacy.logger, logger)
        descriptor = create_instance(ComponentSpec("null"), rng)
        self.assertEqual(descriptor.metadata["data"]["type"], "null")
