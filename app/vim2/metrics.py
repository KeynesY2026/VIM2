from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from vim2.models import MODEL_SPECS, ModelId

_ENGLISH_TOKEN = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*|\d+(?:[.,]\d+)*")


@dataclass(frozen=True, slots=True)
class AcceptanceSample:
    audio: str
    reference: str
    proper_nouns: tuple[str, ...] = ()
    mixed: bool = False


@dataclass(frozen=True, slots=True)
class ModelTranscript:
    sample: AcceptanceSample
    text: str


@dataclass(frozen=True, slots=True)
class AccuracyResult:
    chinese_characters: int
    chinese_errors: int
    cer: float | None
    english_words: int
    english_errors: int
    wer: float | None
    proper_nouns: int
    correct_proper_nouns: int
    proper_noun_accuracy: float | None
    mixed_sentences: int
    boundary_error_sentences: int

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_item in enumerate(
            hypothesis, start=1
        ):
            substitution = previous[hypothesis_index - 1] + (
                reference_item != hypothesis_item
            )
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


def _chinese_characters(text: str) -> list[str]:
    return [char for char in text if "\u4e00" <= char <= "\u9fff"]


def _english_words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _ENGLISH_TOKEN.finditer(text)]


def _language_runs(text: str) -> tuple[str, ...]:
    runs: list[str] = []
    for character in text:
        language = None
        if "\u4e00" <= character <= "\u9fff":
            language = "zh"
        elif character.isascii() and character.isalpha():
            language = "en"
        if language and (not runs or runs[-1] != language):
            runs.append(language)
    return tuple(runs)


def calculate_accuracy(
    transcripts: Iterable[ModelTranscript],
) -> AccuracyResult:
    chinese_characters = 0
    chinese_errors = 0
    english_words = 0
    english_errors = 0
    proper_nouns = 0
    correct_proper_nouns = 0
    mixed_sentences = 0
    boundary_error_sentences = 0

    for transcript in transcripts:
        reference_zh = _chinese_characters(transcript.sample.reference)
        hypothesis_zh = _chinese_characters(transcript.text)
        chinese_characters += len(reference_zh)
        chinese_errors += _edit_distance(reference_zh, hypothesis_zh)

        reference_en = _english_words(transcript.sample.reference)
        hypothesis_en = _english_words(transcript.text)
        english_words += len(reference_en)
        english_errors += _edit_distance(reference_en, hypothesis_en)

        hypothesis_folded = transcript.text.casefold()
        for proper_noun in transcript.sample.proper_nouns:
            proper_nouns += 1
            if proper_noun.casefold() in hypothesis_folded:
                correct_proper_nouns += 1

        if transcript.sample.mixed:
            mixed_sentences += 1
            if _language_runs(transcript.sample.reference) != _language_runs(
                transcript.text
            ):
                boundary_error_sentences += 1

    return AccuracyResult(
        chinese_characters=chinese_characters,
        chinese_errors=chinese_errors,
        cer=(
            chinese_errors / chinese_characters
            if chinese_characters
            else None
        ),
        english_words=english_words,
        english_errors=english_errors,
        wer=english_errors / english_words if english_words else None,
        proper_nouns=proper_nouns,
        correct_proper_nouns=correct_proper_nouns,
        proper_noun_accuracy=(
            correct_proper_nouns / proper_nouns if proper_nouns else None
        ),
        mixed_sentences=mixed_sentences,
        boundary_error_sentences=boundary_error_sentences,
    )


def evaluate_acceptance(
    results: dict[ModelId, AccuracyResult],
) -> list[str]:
    failures: list[str] = []
    for model_id, result in results.items():
        name = MODEL_SPECS[model_id].display_name
        if result.cer is None or result.cer > 0.05:
            failures.append(f"{name}: CER must not exceed 5%.")
        if result.wer is None or result.wer > 0.10:
            failures.append(f"{name}: WER must not exceed 10%.")
        if (
            result.proper_noun_accuracy is None
            or result.proper_noun_accuracy < 0.90
        ):
            failures.append(
                f"{name}: proper-noun accuracy must be at least 90%."
            )
        if result.mixed_sentences == 0 or (
            result.boundary_error_sentences / result.mixed_sentences > 0.05
        ):
            failures.append(
                f"{name}: mixed-language boundary error rate must not "
                "exceed 5%."
            )

    fast = results.get(ModelId.FAST)
    accurate = results.get(ModelId.ACCURATE)
    if fast and accurate:
        if (
            fast.cer is not None
            and accurate.cer is not None
            and accurate.cer > fast.cer
        ):
            failures.append(
                "Qwen3-ASR 1.7B INT8 CER must not exceed the 0.6B FP16 CER."
            )
        if (
            fast.wer is not None
            and accurate.wer is not None
            and accurate.wer > fast.wer
        ):
            failures.append(
                "Qwen3-ASR 1.7B INT8 WER must not exceed the 0.6B FP16 WER."
            )
        if (
            fast.proper_noun_accuracy is not None
            and accurate.proper_noun_accuracy is not None
            and accurate.proper_noun_accuracy
            < fast.proper_noun_accuracy
        ):
            failures.append(
                "Qwen3-ASR 1.7B INT8 proper-noun accuracy must not be "
                "lower than the 0.6B FP16 result."
            )
    return failures
