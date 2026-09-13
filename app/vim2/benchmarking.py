from __future__ import annotations

import json
import os
import subprocess
import sys
import statistics
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from vim2.models import ModelId

RESULT_PREFIX = "RESULT_JSON:"


def format_result_line(result: dict[str, object]) -> str:
    return f"{RESULT_PREFIX}{json.dumps(result, ensure_ascii=True)}"


def select_peak_vram(
    *, nvidia_smi_mib: int | None, torch_peak_bytes: int
) -> tuple[float, str]:
    if nvidia_smi_mib is not None:
        return float(nvidia_smi_mib), "nvidia-smi process used_memory at 100ms"
    return (
        torch_peak_bytes / (1024 * 1024),
        "torch.cuda.max_memory_allocated",
    )


def summarize_sample(
    *,
    audio_path: Path,
    duration_seconds: float,
    inference_seconds: list[float],
    texts: list[str],
) -> dict[str, object]:
    if duration_seconds <= 0:
        raise ValueError("audio duration must be positive")
    if not inference_seconds or len(inference_seconds) != len(texts):
        raise ValueError("each inference timing must have a transcript")
    median_seconds = statistics.median(inference_seconds)
    median_index = min(
        range(len(inference_seconds)),
        key=lambda index: abs(inference_seconds[index] - median_seconds),
    )
    return {
        "audio": str(audio_path),
        "duration_seconds": duration_seconds,
        "inference_seconds": inference_seconds,
        "median_inference_seconds": median_seconds,
        "rtf": median_seconds / duration_seconds,
        "text": texts[median_index],
        "all_texts": texts,
    }


def parse_process_vram_mib(output: str, process_id: int) -> int | None:
    values: list[int] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            pid, memory = (int(field) for field in fields)
        except ValueError:
            continue
        if pid == process_id:
            values.append(memory)
    return max(values, default=None)


class NvidiaSmiMonitor:
    def __init__(self, *, interval_seconds: float = 0.1) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mib: int | None = None
        self.error: str | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="VIM2 VRAM monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=5,
                )
            except (
                FileNotFoundError,
                OSError,
                subprocess.SubprocessError,
            ) as exc:
                self.error = str(exc)
                return
            value = parse_process_vram_mib(completed.stdout, os.getpid())
            if value is not None:
                self.peak_mib = max(self.peak_mib or 0, value)
            self._stop.wait(self._interval_seconds)


def _launch(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"Benchmark worker failed with exit code "
            f"{completed.returncode}:\n{details}"
        )
    return completed.stdout


def _parse_result(output: str) -> dict[str, object]:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_PREFIX):
            value = json.loads(line.removeprefix(RESULT_PREFIX))
            if not isinstance(value, dict):
                raise RuntimeError("RESULT_JSON must contain a JSON object")
            return value
    raise RuntimeError("Worker did not emit a RESULT_JSON result")


def run_model_processes(
    *,
    root: Path,
    audio_paths: Iterable[Path],
    models: Iterable[ModelId],
    runs: int,
    launcher: Callable[[list[str]], str] = _launch,
) -> list[dict[str, object]]:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    audio_arguments = [
        argument
        for path in audio_paths
        for argument in ("--audio", str(path.resolve()))
    ]
    results: list[dict[str, object]] = []
    for model_id in models:
        command = [
            sys.executable,
            "-m",
            "vim2.benchmark_worker",
            "--root",
            str(root.resolve()),
            "--model",
            model_id.value,
            "--runs",
            str(runs),
            *audio_arguments,
        ]
        results.append(_parse_result(launcher(command)))
    return results
