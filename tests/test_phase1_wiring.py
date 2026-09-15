from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import vim2.application as application_module
import vim2.main as main_module
import vim2.platform_windows as platform_windows_module
from vim2.application import (
    ERROR_ALREADY_EXISTS,
    MUTEX_NAME,
    SingleInstanceGuard,
)
from vim2.config import Settings, SettingsRepository
from vim2.models import ModelId
from vim2.paths import AppPaths
from vim2.platform_services import (
    create_platform_services,
    select_platform_profile,
)


class _PassingChecker:
    def __init__(self, paths: AppPaths, *, platform_profile: object) -> None:
        self.paths = paths
        self.platform_profile = platform_profile

    def check(self, selected_model: ModelId, **kwargs) -> object:
        del selected_model, kwargs
        return SimpleNamespace(ok=True, errors=())


class MainApplicationWiringTests(unittest.TestCase):
    def test_main_application_and_qt_runtime_receive_same_services_instance(self) -> None:
        profile = select_platform_profile("darwin", "arm64")
        created: list[object] = []
        received: list[tuple[object, object, object]] = []
        real_create = create_platform_services

        def recording_create(paths: AppPaths, *, profile: object) -> object:
            services = real_create(paths, profile=profile)
            created.append(services)
            return services

        qt_runtime = types.ModuleType("vim2.qt_runtime")

        def run_qt_application(
            paths: AppPaths,
            settings: object,
            *,
            platform_services: object,
        ) -> int:
            received.append((paths, settings, platform_services))
            return 73

        qt_runtime.run_qt_application = run_qt_application

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config" / "settings.json").write_text(
                json.dumps({"selected_model": ModelId.CPU.value}),
                encoding="utf-8",
            )
            with (
                patch.object(main_module, "select_platform_profile", return_value=profile),
                patch.object(main_module, "PreflightChecker", _PassingChecker),
                patch.object(main_module, "create_platform_services", recording_create),
                patch.dict(sys.modules, {"vim2.qt_runtime": qt_runtime}),
            ):
                result = main_module.run(
                    ["--root", str(root), "--skip-runtime-check"]
                )

        self.assertEqual(result, 73)
        self.assertEqual(len(created), 1)
        self.assertEqual(len(received), 1)
        paths, settings, desktop_services = received[0]
        self.assertIs(desktop_services, created[0])
        self.assertEqual(paths.root, root.resolve())
        self.assertIs(settings.selected_model, ModelId.CPU)
        self.assertEqual(desktop_services.supported_models, (ModelId.CPU,))


class PhaseOneMetadataTests(unittest.TestCase):
    def test_checked_in_settings_and_public_model_values_are_consistent(self) -> None:
        root = Path(__file__).resolve().parents[1]
        checked_in = SettingsRepository(root / "config").load()

        self.assertIs(checked_in.selected_model, ModelId.CPU)
        self.assertEqual(
            {model.value for model in ModelId},
            {
                "qwen3-asr-0.6b-int8-cpu",
                "qwen3-asr-0.6b-fp16",
                "qwen3-asr-1.7b-int8",
            },
        )
        self.assertIs(Settings().selected_model, ModelId.FAST)

    def test_macos_cpu_lock_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        lines = (root / "requirements-macos-cpu.lock").read_text(
            encoding="utf-8"
        ).splitlines()
        requirements = [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertTrue(all(line.count("==") == 1 for line in requirements))
        pins = {
            name.lower().replace("_", "-"): version
            for name, version in (
                requirement.split("==", maxsplit=1)
                for requirement in requirements
            )
        }

        self.assertEqual(pins["numpy"], "2.2.6")
        self.assertEqual(
            pins["pyobjc-framework-applicationservices"], "11.1"
        )
        self.assertEqual(pins["sherpa-onnx"], "1.13.8")
        self.assertTrue(
            {
                "accelerate",
                "bitsandbytes",
                "qwen-asr",
                "torch",
                "transformers",
            }.isdisjoint(pins)
        )
        self.assertFalse(
            any(name.startswith("nvidia-") or "cuda" in name for name in pins)
        )


class _FakeKernel32:
    def __init__(self, last_error: int) -> None:
        self.last_error = last_error
        self.closed: list[int] = []

    def CreateMutexW(self, security, initially_owned, name: str) -> int:
        del security, initially_owned
        self.name = name
        return 99

    def get_last_error(self) -> int:
        return self.last_error

    def CloseHandle(self, handle: int) -> None:
        self.closed.append(handle)


class WindowsFactoryCompatibilityTests(unittest.TestCase):
    def test_factory_preserves_public_mutex_seam_and_three_models(self) -> None:
        profile = select_platform_profile("win32", "AMD64")
        services = create_platform_services(
            AppPaths.from_root(Path.cwd()), profile=profile
        )
        sentinel = object()

        with patch.object(
            platform_windows_module,
            "SingleInstanceGuard",
            return_value=sentinel,
        ):
            guard = services.create_single_instance_guard()

        self.assertIs(guard, sentinel)
        self.assertEqual(services.supported_models, tuple(ModelId))
        self.assertEqual(ERROR_ALREADY_EXISTS, 183)
        self.assertEqual(MUTEX_NAME, r"Local\VIM2.SingleInstance")
        self.assertIs(application_module.SingleInstanceGuard, SingleInstanceGuard)

    def test_public_mutex_accepts_fake_native_api_without_win32(self) -> None:
        api = _FakeKernel32(last_error=0)
        guard = SingleInstanceGuard(api=api)

        self.assertTrue(guard.acquire())
        guard.close()

        self.assertEqual(api.name, MUTEX_NAME)
        self.assertEqual(api.closed, [99])


if __name__ == "__main__":
    unittest.main()
