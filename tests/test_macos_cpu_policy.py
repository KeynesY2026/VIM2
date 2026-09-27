from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vim2.models import MODEL_SPECS, ModelId
from vim2.paths import AppPaths
from vim2.platform_services import select_platform_profile
from vim2.preflight import PreflightChecker


def _create_minimal_cpu_model(model_dir: Path) -> None:
    for name in (
        "conv_frontend.onnx",
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "tokenizer/merges.txt",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
    ):
        path = model_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


class MacCPUPreflightPolicyTests(unittest.TestCase):
    def test_macos_rejects_cuda_model_without_fallback(self) -> None:
        dependency_checks: list[str] = []

        def unexpected_dependency_check(name: str) -> object:
            dependency_checks.append(name)
            raise AssertionError("runtime dependencies must not be probed")

        with tempfile.TemporaryDirectory() as directory:
            checker = PreflightChecker(
                AppPaths.from_root(Path(directory)),
                python_version=(3, 11, 16),
                dependency_finder=unexpected_dependency_check,
                platform_profile=select_platform_profile("darwin", "arm64"),
            )
            checker._check_cuda = lambda: (_ for _ in ()).throw(
                AssertionError("CUDA runtime must not be probed")
            )

            result = checker.check(ModelId.ACCURATE)

            self.assertFalse(result.ok)
            self.assertTrue(any("supports only" in error for error in result.errors))
            self.assertTrue(any(ModelId.CPU.value in error for error in result.errors))
            self.assertTrue(any("automatically" in error for error in result.errors))
            self.assertFalse(any("CUDA" in error for error in result.errors))
            self.assertEqual(dependency_checks, [])

    def test_macos_cpu_model_requires_native_adapter_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.from_root(Path(directory))
            _create_minimal_cpu_model(
                paths.models_dir / MODEL_SPECS[ModelId.CPU].directory_name
            )
            runtime = {"PySide6", "sounddevice", "sherpa_onnx", "numpy", "soundfile"}
            checker = PreflightChecker(
                paths,
                python_version=(3, 11, 16),
                dependency_finder=lambda name: object() if name in runtime else None,
                platform_profile=select_platform_profile("darwin", "arm64"),
            )

            result = checker.check(ModelId.CPU)

            self.assertEqual(
                result.errors,
                (
                    "Python dependency is missing: AppKit",
                    "Python dependency is missing: Quartz",
                    "Python dependency is missing: ApplicationServices",
                    "Python dependency is missing: AVFoundation",
                    "Python dependency is missing: CoreFoundation",
                    "Install the macOS CPU dependencies with: "
                    "python -m pip install -r requirements-macos-cpu.lock",
                ),
            )

    def test_missing_macos_cpu_model_error_names_location_and_no_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.from_root(Path(directory))
            checker = PreflightChecker(
                paths,
                python_version=(3, 11, 16),
                dependency_finder=lambda name: object(),
                platform_profile=select_platform_profile("darwin", "arm64"),
            )

            result = checker.check(ModelId.CPU)

            model_dir = paths.models_dir / MODEL_SPECS[ModelId.CPU].directory_name
            self.assertTrue(any(str(model_dir) in error for error in result.errors))
            self.assertTrue(any("No CUDA or MPS fallback" in error for error in result.errors))

    def test_macos_requires_the_phase_one_python_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checker = PreflightChecker(
                AppPaths.from_root(Path(directory)),
                python_version=(3, 12, 9),
                dependency_finder=lambda name: object(),
                platform_profile=select_platform_profile("darwin", "arm64"),
            )

            result = checker.check(ModelId.CPU, check_runtime=False)

            self.assertEqual(len(result.errors), 1)
            self.assertIn("requires Python 3.11", result.errors[0])
            self.assertIn("Python 3.12", result.errors[0])

    def test_rosetta_error_is_returned_before_model_runtime_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checker = PreflightChecker(
                AppPaths.from_root(Path(directory)),
                python_version=(3, 11, 16),
                dependency_finder=lambda name: object(),
                platform_profile=select_platform_profile("darwin", "x86_64"),
            )

            result = checker.check(ModelId.CPU)

            self.assertEqual(len(result.errors), 1)
            self.assertIn("Rosetta", result.errors[0])


if __name__ == "__main__":
    unittest.main()
