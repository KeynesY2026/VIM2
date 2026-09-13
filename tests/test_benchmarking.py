import json
import os
from pathlib import Path

from vim2.benchmarking import (
    RESULT_PREFIX,
    format_result_line,
    parse_process_vram_mib,
    run_model_processes,
    select_peak_vram,
    summarize_sample,
    worker_environment,
)
from vim2.models import ModelId


def test_each_model_runs_in_an_independent_worker_process(
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def launcher(command: list[str]) -> str:
        commands.append(command)
        model = command[command.index("--model") + 1]
        return (
            "worker noise\n"
            + RESULT_PREFIX
            + json.dumps({"model": model, "samples": []})
        )

    results = run_model_processes(
        root=tmp_path,
        audio_paths=[tmp_path / "a.wav"],
        models=[ModelId.FAST, ModelId.ACCURATE],
        runs=3,
        launcher=launcher,
    )

    assert len(commands) == 2
    assert all("--runs" in command for command in commands)
    assert [result["model"] for result in results] == [
        ModelId.FAST.value,
        ModelId.ACCURATE.value,
    ]


def test_worker_output_without_result_marker_is_rejected(
    tmp_path: Path,
) -> None:
    try:
        run_model_processes(
            root=tmp_path,
            audio_paths=[tmp_path / "a.wav"],
            models=[ModelId.FAST],
            runs=3,
            launcher=lambda command: "unexpected output",
        )
    except RuntimeError as exc:
        assert "RESULT_JSON" in str(exc)
    else:
        raise AssertionError("missing worker result should fail")


def test_sample_summary_uses_median_inference_time() -> None:
    summary = summarize_sample(
        audio_path=Path("sample.wav"),
        duration_seconds=10.0,
        inference_seconds=[3.0, 1.0, 2.0],
        texts=["a", "b", "c"],
    )

    assert summary["median_inference_seconds"] == 2.0
    assert summary["rtf"] == 0.2
    assert summary["text"] == "c"
    assert summary["all_texts"] == ["a", "b", "c"]


def test_nvidia_smi_parser_returns_only_current_process_memory() -> None:
    output = "123, 512\n456, 1024\n123, 768\n"

    assert parse_process_vram_mib(output, process_id=123) == 768


def test_worker_result_line_is_safe_for_legacy_windows_console_encoding() -> None:
    line = format_result_line({"text": "中文 transcript"})

    line.encode("cp1252")
    assert json.loads(line.removeprefix(RESULT_PREFIX))["text"] == (
        "中文 transcript"
    )


def test_cuda_peak_is_used_when_nvidia_smi_has_no_process_sample() -> None:
    peak_mib, method = select_peak_vram(
        nvidia_smi_mib=None,
        torch_peak_bytes=1536 * 1024 * 1024,
    )

    assert peak_mib == 1536
    assert method == "torch.cuda.max_memory_allocated"


def test_worker_environment_includes_application_package(
    tmp_path: Path,
) -> None:
    environment = worker_environment(
        tmp_path, {"PYTHONPATH": "existing"}
    )

    assert environment["PYTHONPATH"] == (
        f"{tmp_path.resolve() / 'app'}{os.pathsep}existing"
    )
