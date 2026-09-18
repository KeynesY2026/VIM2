import pytest

from vim2.number_normalizer import NumberNormalizer
from vim2.postprocessing import create_text_postprocessor


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("一百二十三个文件", "123个文件"),
        ("二零二六年九月十八日", "2026年9月18日"),
        ("温度是负三点五度", "温度是-3.5度"),
        ("百分之二十五", "25%"),
        ("twenty five files", "25 files"),
        ("one hundred and six requests", "106 requests"),
        ("version one two three", "version 123"),
        ("code one ten", "code 110"),
        ("negative three point five", "-3.5"),
    ],
)
def test_normalizes_explicit_chinese_and_english_numbers(
    text: str, expected: str
) -> None:
    assert NumberNormalizer().process(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "OneDrive 保存了一心一意.txt",
        "已有 123 个文件，版本 2.5",
        "someone mentioned money",
        "vingt cinq fichiers",
        "двадцать пять файлов",
    ],
)
def test_preserves_non_number_words_and_existing_arabic_numbers(
    text: str,
) -> None:
    assert NumberNormalizer().process(text) == text


def test_number_normalizer_can_be_removed_from_postprocessing_pipeline() -> None:
    enabled = create_text_postprocessor(normalize_numbers=True)
    disabled = create_text_postprocessor(normalize_numbers=False)

    assert enabled.process("one hundred files") == "100 files"
    assert disabled.process("one hundred files") == "one hundred files"


def test_joins_spaced_chinese_digits_when_dictating_a_number_sequence() -> None:
    assert NumberNormalizer().process("验证码是一 二 三 四 五") == "验证码是12345"


def test_concatenates_unseparated_chinese_digit_sequence() -> None:
    assert NumberNormalizer().process("一二三四五六七八九零") == "1234567890"


def test_converts_multi_digit_run_before_ordinary_chinese_text() -> None:
    assert NumberNormalizer().process(
        "三五七九和300（三百）"
    ) == "3579和300（300）"


def test_concatenates_atomic_values_when_chinese_expression_is_invalid() -> None:
    assert NumberNormalizer().process(
        "一二三四五六七八九十"
    ) == "12345678910"


def test_converts_numbers_before_classifiers() -> None:
    normalizer = NumberNormalizer()

    assert normalizer.process("我有一个问题") == "我有1个问题"
    assert normalizer.process("一百个文件") == "100个文件"
    assert normalizer.process("三个人") == "3个人"


def test_additional_language_can_be_injected_without_changing_dispatcher() -> None:
    class ExampleLanguageAdapter:
        def process(self, text: str) -> str:
            return text.replace("deux", "2")

    normalizer = NumberNormalizer(adapters=[ExampleLanguageAdapter()])

    assert normalizer.process("deux fichiers") == "2 fichiers"
