from vim2.metrics import (
    AccuracyResult,
    AcceptanceSample,
    ModelTranscript,
    calculate_accuracy,
    evaluate_acceptance,
)
from vim2.models import ModelId


def test_accuracy_reports_chinese_cer_and_english_wer_separately() -> None:
    samples = [
        ModelTranscript(
            sample=AcceptanceSample(
                audio="mixed.wav",
                reference="今天 deploy Kubernetes",
                proper_nouns=("Kubernetes",),
                mixed=True,
            ),
            text="今夭 deploy Kubernete",
        )
    ]

    result = calculate_accuracy(samples)

    assert result.chinese_characters == 2
    assert result.chinese_errors == 1
    assert result.cer == 0.5
    assert result.english_words == 2
    assert result.english_errors == 1
    assert result.wer == 0.5
    assert result.proper_noun_accuracy == 0.0


def test_case_and_punctuation_do_not_inflate_english_wer() -> None:
    samples = [
        ModelTranscript(
            sample=AcceptanceSample(
                audio="english.wav",
                reference="Hello, VS Code!",
            ),
            text="hello vs code",
        )
    ]

    assert calculate_accuracy(samples).wer == 0.0


def test_mixed_language_boundary_mismatch_counts_affected_sentence() -> None:
    samples = [
        ModelTranscript(
            sample=AcceptanceSample(
                audio="mixed.wav",
                reference="部署 Kubernetes 服务",
                mixed=True,
            ),
            text="部署服务",
        )
    ]

    result = calculate_accuracy(samples)

    assert result.mixed_sentences == 1
    assert result.boundary_error_sentences == 1


def test_acceptance_thresholds_and_model_comparison_are_enforced() -> None:
    fast = AccuracyResult(
        chinese_characters=100,
        chinese_errors=4,
        cer=0.04,
        english_words=100,
        english_errors=8,
        wer=0.08,
        proper_nouns=100,
        correct_proper_nouns=92,
        proper_noun_accuracy=0.92,
        mixed_sentences=20,
        boundary_error_sentences=1,
    )
    accurate = AccuracyResult(
        chinese_characters=100,
        chinese_errors=5,
        cer=0.05,
        english_words=100,
        english_errors=9,
        wer=0.09,
        proper_nouns=100,
        correct_proper_nouns=91,
        proper_noun_accuracy=0.91,
        mixed_sentences=20,
        boundary_error_sentences=1,
    )

    failures = evaluate_acceptance(
        {ModelId.FAST: fast, ModelId.ACCURATE: accurate}
    )

    assert any("CER must not exceed" in failure for failure in failures)
    assert any("WER must not exceed" in failure for failure in failures)
    assert any(
        "proper-noun accuracy must not be lower" in failure
        for failure in failures
    )
