from pathlib import Path

import pytest

from vim2.hotwords import HotwordRepository, parse_hotwords


def test_parse_hotwords_normalizes_deduplicates_and_serializes() -> None:
    snapshot = parse_hotwords(
        "\ufeff# One phrase per line\n  VIM2  \n\nQwen3-ASR\nVIM2\n"
    )

    assert snapshot.entries == ("VIM2", "Qwen3-ASR")
    assert snapshot.gpu_context == "VIM2\nQwen3-ASR"
    assert snapshot.cpu_hotwords == "VIM2,Qwen3-ASR"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("contains,comma", "ASCII comma"),
        ("x" * 101, "100 characters"),
        ("\n".join(f"word-{index}" for index in range(101)), "100 entries"),
    ],
)
def test_parse_hotwords_rejects_unsafe_or_excessive_input(
    text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_hotwords(text)


def test_repository_creates_first_run_template_and_returns_empty_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config" / "hotwords.txt"
    repository = HotwordRepository(path)

    snapshot = repository.load()

    assert snapshot.entries == ()
    assert path.read_text(encoding="utf-8").startswith("#")