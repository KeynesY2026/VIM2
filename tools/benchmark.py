from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from vim2.benchmarking import run_model_processes
from vim2.models import ModelId


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run isolated Qwen3-ASR performance benchmarks."
    )
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument(
        "--model",
        choices=("both", *(model.value for model in ModelId)),
        default="both",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    models = (
        list(ModelId)
        if args.model == "both"
        else [ModelId(args.model)]
    )
    results = run_model_processes(
        root=ROOT,
        audio_paths=args.audio,
        models=models,
        runs=args.runs,
    )
    payload = json.dumps(
        {"models": results}, ensure_ascii=False, indent=2
    )
    print(json.dumps({"models": results}, ensure_ascii=True, indent=2))
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
