from __future__ import annotations

import sys
from importlib.util import find_spec
from dataclasses import dataclass
from typing import Callable

from vim2.models import (
    MODEL_SPECS,
    ModelBackend,
    ModelId,
    validate_model_directory,
)
from vim2.paths import AppPaths
from vim2.platform_services import PlatformProfile


@dataclass(frozen=True, slots=True)
class PreflightResult:
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


class PreflightChecker:
    def __init__(
        self,
        paths: AppPaths,
        python_version: tuple[int, int, int] | None = None,
        dependency_finder: Callable[[str], object | None] = find_spec,
        platform_profile: PlatformProfile | None = None,
    ) -> None:
        self._paths = paths
        self._python_version = python_version or sys.version_info[:3]
        self._dependency_finder = dependency_finder
        self._platform_profile = platform_profile

    def check(
        self,
        selected_model: ModelId,
        *,
        check_cuda: bool = True,
        check_runtime: bool = True,
    ) -> PreflightResult:
        errors: list[str] = []
        major, minor, _ = self._python_version
        if major != 3 or minor < 10 or minor > 13:
            errors.append(
                f"Python {major}.{minor} is unsupported; "
                "install Python 3.10 through 3.13."
            )

        profile = self._platform_profile
        if profile is not None and profile.preflight_errors:
            errors.extend(profile.preflight_errors)
            return PreflightResult(tuple(errors))
        if (
            profile is not None
            and profile.required_python is not None
            and self._python_version[:2] != profile.required_python
        ):
            required = ".".join(str(part) for part in profile.required_python)
            current = ".".join(str(part) for part in self._python_version[:2])
            errors.append(
                f"The {profile.name} CPU MVP requires Python {required}; "
                f"the current interpreter is Python {current}."
            )
            return PreflightResult(tuple(errors))
        if profile is not None and selected_model not in profile.supported_models:
            supported = ", ".join(
                model_id.value for model_id in profile.supported_models
            ) or "none"
            errors.append(
                f"The {profile.name} runtime supports only: {supported}. "
                f"Select {ModelId.CPU.value} in config/settings.json. "
                "No alternative model or backend will be selected automatically."
            )
            return PreflightResult(tuple(errors))

        spec = MODEL_SPECS[selected_model]
        model_dir = self._paths.models_dir / spec.directory_name
        model_errors = validate_model_directory(model_dir, spec.backend)
        if model_errors:
            errors.append(f"Model is incomplete: {model_dir}")
            errors.extend(model_errors)
            if profile is not None and profile.name == "macos":
                errors.append(
                    f"Copy the complete sherpa-onnx CPU model to {model_dir}. "
                    "No CUDA or MPS fallback will be attempted."
                )

        if check_runtime:
            if spec.backend is ModelBackend.SHERPA_ONNX_CPU:
                modules = (
                    "PySide6",
                    "sounddevice",
                    "sherpa_onnx",
                    "numpy",
                    "soundfile",
                )
            else:
                modules = (
                    "torch",
                    "PySide6",
                    "qwen_asr",
                    "sounddevice",
                    "bitsandbytes",
                    "numpy",
                    "soundfile",
                    "transformers",
                )
            if profile is not None:
                modules += profile.required_modules
            modules = tuple(dict.fromkeys(modules))
            missing_platform_module = False
            for module in modules:
                if self._dependency_finder(module) is None:
                    errors.append(
                        f"Python dependency is missing: {module}"
                    )
                    if (
                        profile is not None
                        and module in profile.required_modules
                    ):
                        missing_platform_module = True
            if (
                profile is not None
                and profile.name == "macos"
                and missing_platform_module
            ):
                errors.append(
                    "Install the macOS CPU dependencies with: "
                    "python -m pip install -r requirements-macos-cpu.lock"
                )

        if check_cuda and spec.backend is ModelBackend.QWEN_CUDA:
            errors.extend(self._check_cuda())

        return PreflightResult(tuple(errors))

    @staticmethod
    def _check_cuda() -> list[str]:
        try:
            import torch
        except ImportError:
            return [
                "PyTorch is missing from runtime; restore the portable dependencies."
            ]
        if not torch.cuda.is_available():
            return [
                "CUDA is unavailable; install a compatible NVIDIA driver and "
                "confirm that the GPU is enabled."
            ]
        return []
