import pytest

from vim2.transcript import (
    StableCheckpoint,
    StablePrefixTracker,
    format_mixed_language_spacing,
    merge_stable_tail,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("中文EnglishРусский", "中文 English Русский"),
        ("مرحباEnglish中文", "مرحبا English 中文"),
        ("東京タワーでテスト", "東京タワーでテスト"),
        ("𠀀API〇", "𠀀 API 〇"),
        ("版本2API，Русский", "版本2API，Русский"),
        ("中文  English", "中文  English"),
        ("API・仕様A゠B", "API・仕様 A゠B"),
        ("カ\u3099API e\u0301Русский", "カ\u3099 API e\u0301 Русский"),
    ],
)
def test_mixed_language_spacing_uses_unicode_script_boundaries(
    text: str, expected: str
) -> None:
    assert format_mixed_language_spacing(text) == expected


def test_three_identical_complete_prefixes_create_checkpoint() -> None:
    tracker = StablePrefixTracker()

    tracker.observe("第一句。正在变化", 160_000)
    tracker.observe("第一句。仍在变化", 176_000)
    assert tracker.checkpoint is None

    tracker.observe("第一句。最后变化", 192_000)

    assert tracker.checkpoint is not None
    assert tracker.checkpoint.prefix == "第一句。"
    assert tracker.checkpoint.anchor == "第一句。"
    assert tracker.checkpoint.frame_count == 192_000


def test_growing_preview_immediately_confirms_before_last_sentence() -> None:
    tracker = StablePrefixTracker()

    tracker.observe("第一句。第二句。第三句还没说完", 160_000)

    assert tracker.checkpoint is not None
    assert tracker.checkpoint.prefix == "第一句。"
    assert tracker.checkpoint.anchor == "第一句。"
    assert tracker.checkpoint.frame_count == 160_000

    tracker.observe("第一句。第二句。第三句。第四句还没说完", 176_000)

    assert tracker.checkpoint.prefix == "第一句。第二句。"
    assert tracker.checkpoint.anchor == "第二句。"
    assert tracker.checkpoint.frame_count == 176_000


def test_text_without_sentence_ending_never_stabilizes() -> None:
    tracker = StablePrefixTracker()
    for frame_count in (16_000, 32_000, 48_000):
        tracker.observe("还没有结束", frame_count)

    assert tracker.checkpoint is None


def test_changed_candidate_restarts_three_preview_count() -> None:
    tracker = StablePrefixTracker()
    tracker.observe("旧句。", 16_000)
    tracker.observe("旧句。", 32_000)
    tracker.observe("新句。", 48_000)
    tracker.observe("新句。", 64_000)

    assert tracker.checkpoint is None

    tracker.observe("新句。", 80_000)
    assert tracker.checkpoint is not None
    assert tracker.checkpoint.prefix == "新句。"


def test_stable_prefix_only_moves_forward() -> None:
    tracker = StablePrefixTracker()
    for frame_count in (16_000, 32_000, 48_000):
        tracker.observe("第一句。尾巴", frame_count)
    first = tracker.checkpoint

    tracker.observe("冲突句。", 64_000)
    tracker.observe("第一句。第二句。尾巴", 80_000)
    tracker.observe("第一句。第二句。尾巴", 96_000)
    tracker.observe("第一句。第二句。尾巴", 112_000)

    assert first is not None
    assert tracker.checkpoint is not None
    assert tracker.checkpoint.prefix == "第一句。第二句。"
    assert tracker.checkpoint.anchor == "第二句。"
    assert tracker.checkpoint.frame_count == 112_000


def test_unique_anchor_merges_without_duplicate_text() -> None:
    checkpoint = StableCheckpoint(
        prefix="第一句。第二句。",
        anchor="第二句。",
        frame_count=320_000,
    )

    assert merge_stable_tail(
        checkpoint, "第二句。最后一段。"
    ) == "第一句。第二句。最后一段。"


def test_missing_or_repeated_anchor_requires_fallback() -> None:
    checkpoint = StableCheckpoint("第一句。", "第一句。", 160_000)

    assert merge_stable_tail(checkpoint, "不同内容。") is None
    assert merge_stable_tail(
        checkpoint, "第一句。中间。第一句。结尾。"
    ) is None
    assert merge_stable_tail(checkpoint, "   ") is None


@pytest.mark.parametrize(
    ("prefix", "tail", "expected"),
    [
        (
            "Earlier. What?!",
            "Noise?! What?! Next.",
            "Earlier. What?! Next.",
        ),
        (
            "前句。真的吗？！",
            "噪声？！真的吗？！后续。",
            "前句。真的吗？！后续。",
        ),
    ],
)
def test_terminal_punctuation_run_is_part_of_full_anchor(
    prefix: str, tail: str, expected: str
) -> None:
    tracker = StablePrefixTracker()
    for frame_count in (16_000, 32_000, 48_000):
        tracker.observe(prefix, frame_count)

    checkpoint = tracker.checkpoint
    assert checkpoint is not None
    assert checkpoint.anchor == prefix.removeprefix(
        "Earlier. " if prefix.startswith("Earlier") else "前句。"
    )
    assert merge_stable_tail(checkpoint, tail) == expected
