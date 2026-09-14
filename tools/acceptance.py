from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from vim2.benchmarking import run_model_processes
from vim2.metrics import (
    AcceptanceSample,
    ModelTranscript,
    calculate_accuracy,
    evaluate_acceptance,
)
from vim2.models import MODEL_SPECS, ModelBackend, ModelId


def _load_dataset(path: Path) -> tuple[str, list[AcceptanceSample]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Dataset must contain a non-empty version")
    samples = [
        AcceptanceSample(
            audio=str((path.parent / item["audio"]).resolve()),
            reference=item["reference"],
            proper_nouns=tuple(item.get("proper_nouns", ())),
            mixed=bool(item.get("mixed", False)),
        )
        for item in payload.get("samples", ())
    ]
    return version, samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run VIM2 performance and accuracy acceptance."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="allow fewer than the required 100 samples for dry runs",
    )
    args = parser.parse_args()

    version, samples = _load_dataset(args.dataset.resolve())
    if len(samples) < 100 and not args.allow_incomplete:
        raise ValueError(
            f"Acceptance requires at least 100 samples; found {len(samples)}"
        )
    benchmark_results = run_model_processes(
        root=ROOT,
        audio_paths=[Path(sample.audio) for sample in samples],
        models=list(ModelId),
        runs=args.runs,
    )
    sample_by_audio = {sample.audio: sample for sample in samples}
    accuracy_results = {}
    failures: list[str] = []
    for benchmark in benchmark_results:
        model_id = ModelId(str(benchmark["model"]))
        transcripts = [
            ModelTranscript(
                sample=sample_by_audio[str(Path(item["audio"]).resolve())],
                text=str(item["text"]),
            )
            for item in benchmark["samples"]
        ]
        accuracy_results[model_id] = calculate_accuracy(transcripts)
        if float(benchmark["rtf"]) >= 1.0:
            failures.append(
                f"{MODEL_SPECS[model_id].display_name}: RTF must be below 1.0."
            )
        peak = benchmark["peak_vram_gib"]
        if (
            MODEL_SPECS[model_id].backend
            is ModelBackend.SHERPA_ONNX_CPU
        ):
            continue
        limit = 3.8 if model_id is ModelId.ACCURATE else 2.5
        if peak is None:
            failures.append(
                f"{MODEL_SPECS[model_id].display_name}: peak VRAM was not measured."
            )
        elif float(peak) > limit:
            failures.append(
                f"{MODEL_SPECS[model_id].display_name}: peak VRAM exceeds "
                f"{limit} GiB."
            )
    failures.extend(evaluate_acceptance(accuracy_results))
    report = {
        "dataset_version": version,
        "sample_count": len(samples),
        "passed": not failures,
        "failures": failures,
        "accuracy": {
            model.value: result.to_dict()
            for model, result in accuracy_results.items()
        },
        "benchmarks": benchmark_results,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
