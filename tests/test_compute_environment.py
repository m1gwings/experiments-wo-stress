"""Best-effort POSIX hardware reports without workstation or GPU requirements."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments_wo_stress.execution import compute_environment as environment


class MachineInformationTests(unittest.TestCase):
    """Normalize optional system data and keep unsupported execution silent."""

    def setUp(self) -> None:
        """Isolate filesystem capacities and identifying platform fields."""
        self.root = Path("/unused/experiment")
        for name, value in (("release", "6.8.0"), ("machine", "x86_64")):
            patcher = patch.object(environment.platform, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        capacity = patch.object(environment, "_filesystem_capacity", return_value=(1000, 400))
        capacity.start()
        self.addCleanup(capacity.stop)

    def test_linux_normalizes_cpu_ram_filesystem_and_gpu_inventory(self) -> None:
        processors = "\n\n".join(
            f"processor: {index}\nmodel name: Example CPU\nphysical id: {index // 4}\n"
            f"core id: {(index // 2) % 2}"
            for index in range(8)
        )
        files = {
            "/proc/cpuinfo": processors,
            "/proc/meminfo": "MemTotal:       67108864 kB\nMemFree: 100 kB\n",
            "/etc/os-release": 'NAME="Example Linux"\nPRETTY_NAME="Example Linux 24"\n',
        }
        with (
            patch.object(environment.platform, "system", return_value="Linux"),
            patch.object(environment.os, "cpu_count", return_value=8),
            patch.object(environment, "_read_text", side_effect=files.get),
            patch.object(environment, "_query", return_value="0, Example GPU, 24576, 555.1"),
        ):
            report = environment.collect_machine(self.root, 2, [0, 3])
        self.assertEqual(report["platform"], "Linux")
        self.assertEqual(report["os_version"], "Example Linux 24")
        self.assertEqual(report["kernel_version"], "6.8.0")
        self.assertEqual(report["architecture"], "x86_64")
        self.assertEqual(report["cpu_model"], "Example CPU")
        self.assertEqual(report["physical_cpu_cores"], 4)
        self.assertEqual(report["logical_cpu_count"], 8)
        self.assertEqual(report["ram_bytes"], 64 * 1024**3)
        self.assertEqual(report["workers"], 2)
        self.assertEqual(report["filesystem_total_bytes"], 1000)
        self.assertEqual(report["filesystem_available_bytes"], 400)
        self.assertEqual(report["allocated_gpu_count"], 2)
        self.assertEqual(report["gpu_ids"], [0, 3])
        self.assertEqual(
            report["gpus"],
            [
                {
                    "nvidia_smi_index": 0,
                    "model": "Example GPU",
                    "vram_bytes": 24 * 1024**3,
                    "driver_version": "555.1",
                }
            ],
        )
        self.assertNotIn("hostname", report)
        self.assertNotIn("/unused", json.dumps(report))

    def test_macos_retains_apple_chip_and_unified_ram_without_claiming_allocation(self) -> None:
        commands = {
            ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M3 Pro",
            ("sysctl", "-n", "hw.physicalcpu"): "12",
            ("sysctl", "-n", "hw.logicalcpu"): "12",
            ("sysctl", "-n", "hw.memsize"): str(36 * 1024**3),
            ("sw_vers", "-productVersion"): "14.4",
            ("system_profiler", "-json", "SPDisplaysDataType"): json.dumps(
                {
                    "SPDisplaysDataType": [
                        {
                            "sppci_model": "Apple M3 Pro",
                            "sppci_cores": "18",
                            "serial_number": "must not be retained",
                        }
                    ]
                }
            ),
        }
        with (
            patch.object(environment.platform, "system", return_value="Darwin"),
            patch.object(environment, "_query", side_effect=lambda args: commands.get(tuple(args))),
        ):
            report = environment.collect_machine(self.root, 4, None)
        self.assertEqual(report["cpu_model"], "Apple M3 Pro")
        self.assertEqual(report["physical_cpu_cores"], 12)
        self.assertEqual(report["logical_cpu_count"], 12)
        self.assertEqual(report["ram_bytes"], 36 * 1024**3)
        self.assertEqual(report["os_version"], "14.4")
        self.assertEqual(report["apple_accelerator"], [{"model": "Apple M3 Pro", "cores": 18}])
        self.assertEqual(report["allocated_gpu_count"], 0)
        self.assertEqual(report["gpus"], [])
        self.assertNotIn("serial", json.dumps(report))

    def test_missing_system_files_and_utilities_leave_optional_fields_unavailable(self) -> None:
        with (
            patch.object(environment.platform, "system", return_value="Linux"),
            patch.object(environment, "_read_text", return_value=None),
            patch.object(environment.subprocess, "run", side_effect=FileNotFoundError),
        ):
            report = environment.collect_machine(self.root, 1, [5])
        for key in ("cpu_model", "physical_cpu_cores", "ram_bytes", "os_version"):
            self.assertIsNone(report[key])
        self.assertEqual(report["allocated_gpu_count"], 1)
        self.assertEqual(report["gpus"], [])

    def test_macos_missing_or_malformed_optional_outputs_remain_unavailable(self) -> None:
        for output in (None, "not a number or JSON", "[]", '{"SPDisplaysDataType": 5}'):
            with (
                self.subTest(output=output),
                patch.object(environment.platform, "system", return_value="Darwin"),
                patch.object(environment, "_query", return_value=output),
            ):
                report = environment.collect_machine(self.root, 1, None)
            self.assertIsNone(report["physical_cpu_cores"])
            self.assertIsNone(report["logical_cpu_count"])
            self.assertIsNone(report["ram_bytes"])
            self.assertEqual(report["apple_accelerator"], [])

    def test_windows_returns_no_report_without_queries_or_warnings(self) -> None:
        with (
            patch.object(environment.platform, "system", return_value="Windows"),
            patch.object(environment, "_query") as query,
            patch.object(environment, "_read_text") as read,
        ):
            self.assertFalse(environment.supported())
            self.assertEqual(environment.collect_machine(self.root, 1, None), {})
        query.assert_not_called()
        read.assert_not_called()

    def test_cpu_only_linux_does_not_query_gpu_utilities(self) -> None:
        with (
            patch.object(environment.platform, "system", return_value="Linux"),
            patch.object(environment, "_read_text", return_value=None),
            patch.object(environment, "_query") as query,
        ):
            report = environment.collect_machine(self.root, 3, None)
        query.assert_not_called()
        self.assertEqual(report["allocated_gpu_count"], 0)
        self.assertEqual(report["gpu_ids"], [])

    def test_incomplete_cpu_topology_is_not_reported_as_physical_core_count(self) -> None:
        with patch.object(
            environment,
            "_read_text",
            return_value="processor: 0\nphysical id: 0\ncore id: 0\n\nprocessor: 1\n",
        ):
            self.assertEqual(environment._linux_cpu(), (None, None))

    def test_malformed_gpu_rows_do_not_discard_valid_optional_fields(self) -> None:
        with patch.object(
            environment,
            "_query",
            return_value="garbage\nwrong, GPU, 100, 555\n1, GPU, [N/A], [N/A]\n",
        ):
            self.assertEqual(
                environment._nvidia_inventory(),
                [
                    {
                        "nvidia_smi_index": 1,
                        "model": "GPU",
                        "vram_bytes": None,
                        "driver_version": None,
                    }
                ],
            )


class SystemBoundaryTests(unittest.TestCase):
    """System-query and artifact-walk failures never escape into an experiment."""

    def test_command_failures_and_timeouts_are_optional(self) -> None:
        for failure in (
            FileNotFoundError(),
            subprocess.CalledProcessError(1, "sysctl"),
            subprocess.TimeoutExpired("sysctl", 2),
            UnicodeError(),
        ):
            with (
                self.subTest(failure=type(failure).__name__),
                patch.object(environment.subprocess, "run", side_effect=failure),
            ):
                self.assertIsNone(environment._query(["sysctl", "-n", "hw.memsize"]))

    def test_command_query_has_a_bounded_timeout_and_no_shell(self) -> None:
        with patch.object(
            environment.subprocess, "run", return_value=SimpleNamespace(stdout="  42\n")
        ) as query:
            self.assertEqual(environment._query(["sysctl", "-n", "hw.memsize"]), "42")
        self.assertEqual(query.call_args.kwargs["timeout"], 2)
        self.assertNotIn("shell", query.call_args.kwargs)

    def test_output_capacity_uses_the_nearest_existing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                environment.shutil, "disk_usage", return_value=SimpleNamespace(total=100, free=20)
            ) as usage:
                self.assertEqual(
                    environment._filesystem_capacity(root / "new" / "output"), (100, 20)
                )
            usage.assert_called_once_with(root)

    def test_unreadable_output_filesystem_has_unknown_capacity(self) -> None:
        with patch.object(environment.shutil, "disk_usage", side_effect=PermissionError):
            self.assertEqual(environment._filesystem_capacity(Path(".")), (None, None))

    def test_artifact_size_counts_files_once_without_following_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            (output / "first").write_bytes(b"abc")
            (output / "nested").mkdir()
            (output / "nested" / "second").write_bytes(b"defg")
            (root / "outside").write_bytes(b"should not count")
            (output / "link").symlink_to(root / "outside")
            (output / "loop").symlink_to(output, target_is_directory=True)
            self.assertEqual(environment.artifact_size(output), 7)
            self.assertIsNone(environment.artifact_size(output / "loop"))

    def test_artifact_size_is_unknown_when_the_walk_is_incomplete(self) -> None:
        with patch.object(environment.os, "scandir", side_effect=PermissionError):
            self.assertIsNone(environment.artifact_size(Path(".")))
