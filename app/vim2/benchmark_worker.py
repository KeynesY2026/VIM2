from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import soundfile
import torch

from vim2.benchmarking import (
    NvidiaSmiMonitor,
    format_result_line,
    select_peak_vram,
    summarize_sample,
)
from vim2.models import ModelId
from vim2.paths import AppPaths
from vim2.recognizer import QwenRecognizer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=ModelId)
    parser.add_argument("--audio", action="append", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    missing = [path for path in args.audio if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Audio file does not exist: {missing[0]}")

    recognizer = QwenRecognizer(AppPaths.from_root(args.root))
    monitor = NvidiaSmiMonitor()
    monitor.start()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    try:
        recognizer.load(args.model)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started

        recognizer.transcribe(args.audio[0], args.model)
        torch.cuda.synchronize()

        samples: list[dict[str, object]] = []
        for audio_path in args.audio:
            duration = float(soundfile.info(str(audio_path)).duration)
            timings: list[float] = []
            texts: list[str] = []
            for _ in range(args.runs):
                torch.cuda.synchronize()
                started = time.perf_counter()
                text = recognizer.transcribe(audio_path, args.model)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
                texts.append(text)
            samples.append(
                summarize_sample(
                    audio_path=audio_path.resolve(),
                    duration_seconds=duration,
                    inference_seconds=timings,
                    texts=texts,
                )
            )
        torch_peak_bytes = torch.cuda.max_memory_allocated()
    finally:
        recognizer.unload()
        monitor.stop()

    peak_vram_mib, measurement = select_peak_vram(
        nvidia_smi_mib=monitor.peak_mib,
        torch_peak_bytes=torch_peak_bytes,
    )
    total_audio = sum(
        float(sample["duration_seconds"]) for sample in samples
    )
    total_inference = sum(
        float(sample["median_inference_seconds"]) for sample in samples
    )
    result = {
        "model": args.model.value,
        "load_seconds": load_seconds,
        "pure_inference_seconds": total_inference,
        "total_audio_seconds": total_audio,
        "rtf": total_inference / total_audio,
        "peak_vram_mib": peak_vram_mib,
        "peak_vram_gib": peak_vram_mib / 1024,
        "vram_measurement": measurement,
        "vram_monitor_error": monitor.error,
        "runs_per_audio": args.runs,
        "samples": samples,
    }
    print(format_result_line(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
