"""Portable NumPy continuation keeps science and decoding checks across host changes."""

from dataclasses import replace
from unittest.mock import patch

from experiments_wo_stress import run_experiment
from experiments_wo_stress.execution.portability import portable_environment
from experiments_wo_stress.storage import iter_completed_runs
from tests.test_execution import _ExecutionStudyTestCase


class PortableContinuationTests(_ExecutionStudyTestCase):
    """Exercise saved checkpoints with changed host facts and incompatible runtimes."""

    def config(self):
        config = self.load_study_config(self.single_run_settings())
        return replace(config, execution={**config.execution, "continuation": "portable_numpy"})

    def test_new_hostname_and_kernel_resume_same_variant_and_trajectory(self):
        config = self.config()
        output = self.root / "portable"
        with (
            patch("socket.gethostname", return_value="vm-one"),
            patch("platform.platform", return_value="Linux-old-kernel"),
        ):
            self.assertEqual(run_experiment(config, output, max_steps=5).paused, 1)
        with (
            patch("socket.gethostname", return_value="vm-two"),
            patch("platform.platform", return_value="Linux-new-kernel"),
        ):
            self.assertEqual(run_experiment(config, output).completed, 1)
            self.assertEqual(run_experiment(config, output).skipped, 1)
        self.assertEqual(len(list((output / "runs").iterdir())), 1)
        uninterrupted = self.root / "uninterrupted"
        run_experiment(config, uninterrupted)
        self.assert_results_equal(output, uninterrupted)

    def test_architecture_or_python_change_selects_new_variant(self):
        config = self.config()
        for target, old, new in (
            ("platform.machine", "x86_64", "aarch64"),
            ("platform.python_version", "3.12.1", "3.12.2"),
        ):
            output = self.root / target
            with patch(target, return_value=old):
                run_experiment(config, output, max_steps=5)
            with patch(target, return_value=new):
                self.assertEqual(run_experiment(config, output).completed, 1)
            self.assertEqual(len(list((output / "runs").iterdir())), 2)

    def test_package_and_scientific_source_changes_do_not_reuse_checkpoints(self):
        config = self.config()
        output = self.root / "source-change"
        run_experiment(config, output, max_steps=5)
        code = self.study / self.package_name / "algorithms.py"
        code.write_text(code.read_text() + "\n# changed scientific source\n")
        self.assertEqual(run_experiment(config, output).completed, 1)
        self.assertEqual(len(list((output / "runs").iterdir())), 2)
        original = portable_environment(config)
        changed = {**original, "packages": {**original["packages"], "numpy": "incompatible"}}
        with patch(
            "experiments_wo_stress.execution.provenance.portable_environment", return_value=changed
        ):
            self.assertEqual(run_experiment(config, output).completed, 1)
        self.assertEqual(len(list((output / "runs").iterdir())), 3)

    def test_gpu_and_custom_checkpoint_backends_are_rejected_explicitly(self):
        config = self.config()
        for options in ({"gpu_ids": [0]}, {"checkpoint_backend": {"type": "custom:Backend"}}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "CPU studies"):
                portable_environment(replace(config, execution={**config.execution, **options}))

    def test_strict_and_portable_modes_do_not_silently_mix(self):
        portable = self.config()
        strict = replace(portable, execution={**portable.execution, "continuation": "strict"})
        output = self.root / "modes"
        run_experiment(strict, output, max_steps=5)
        self.assertEqual(run_experiment(portable, output).completed, 1)
        self.assertEqual(len(list((output / "runs").iterdir())), 2)
        self.assertEqual(len(list(iter_completed_runs(output))), 1)
