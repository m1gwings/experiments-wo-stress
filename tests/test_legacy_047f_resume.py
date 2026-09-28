"""Select an exact old variant without weakening normal provenance checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from experiments_wo_stress.execution.compatibility import legacy_047f_implementation
from experiments_wo_stress.execution.provenance import (
    collect_provenance,
    prepare_request,
)
from experiments_wo_stress.storage.files import atomic_json
from experiments_wo_stress.study.config import load_config
from experiments_wo_stress.study.planning import plan_runs


class LegacyVariantSelectionTests(unittest.TestCase):
    """Require an existing matching 047fbc5 identity before reusing its run path."""

    def test_exact_legacy_variant_is_selected_but_changed_source_is_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "study.yml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "name": "legacy-resume",
                        "seed": 17,
                        "runs": [
                            {
                                "name": "run",
                                "budget": {"steps": 5},
                                "protocol": {"type": "offline"},
                                "data": {"type": "normal", "params": {"size": 2}},
                                "algorithms": [{"name": "null", "type": "null_algorithm"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(config_path)
            specs = plan_runs(config)
            provenance = collect_provenance(config, {})
            legacy = legacy_047f_implementation(provenance["implementation"])
            self.assertIsNotNone(legacy)
            old_provenance = {**provenance, "implementation": legacy}
            old = prepare_request(config, specs, old_provenance)
            current = prepare_request(config, specs, provenance, root)
            self.assertNotEqual(current.locations, old.locations)
            old_id = next(iter(old.variants))
            atomic_json(root / "runs" / old_id / "metadata.json", old.variants[old_id])
            selected = prepare_request(config, specs, provenance, root)
            self.assertEqual(selected.locations, old.locations)
            self.assertEqual(
                selected.variants[old_id]["identity"], old.variants[old_id]["identity"]
            )
            changed = {**provenance, "implementation": {**provenance["implementation"]}}
            changed["implementation"]["execution/scheduler.py"] = "future edit"
            self.assertNotEqual(
                prepare_request(config, specs, changed, root).locations, old.locations
            )
