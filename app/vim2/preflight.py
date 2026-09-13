from __future__ import annotations

import sys
from dataclasses import dataclass

from vim2.models import MODEL_SPECS, ModelId, validate_model_directory
from vim2.paths import AppPaths


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
    ) -> None:
        self._paths = paths
        self._python_version = python_version or sys.version_info[:3]

    def check(
        self, selected_model: ModelId, *, check_cuda: bool = True
    ) -> PreflightResult:
        errors: list[str] = []
        major, minor, _ = self._python_version
        if major != 3 or minor < 10 or minor > 13:
            errors.append(
                f"Python {major}.{minor} is unsupported; "
                "install Python 3.10 through 3.13."
            )

        spec = MODEL_SPECS[selected_model]
        model_dir = self._paths.models_dir / spec.directory_name
        model_errors = validate_model_directory(model_dir)
        if model_errors:
            errors.append(f"Model is incomplete: {model_dir}")
            errors.extend(model_errors)

        if check_cuda:
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
