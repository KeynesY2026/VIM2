from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Protocol

_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1_000}
_LARGE_UNITS = {"万": 10_000, "亿": 100_000_000}
_CHINESE_NUMBER = r"[负零〇一二两三四五六七八九十百千万亿点]+"
_CHINESE_NUMBER_RE = re.compile(_CHINESE_NUMBER)
_CHINESE_PERCENT_RE = re.compile(rf"百分之(?P<number>{_CHINESE_NUMBER})")
_SPACED_CHINESE_DIGITS_RE = re.compile(
    r"[零〇一二两三四五六七八九](?:\s+[零〇一二两三四五六七八九])+"
)
_QUANTITY_SUFFIXES = frozenset(
    "年月日号个只本件次岁米斤克吨秒分时天周元角度章页项位名人台套张"
)

_ENGLISH_DIGITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_ENGLISH_SMALL = {
    **_ENGLISH_DIGITS,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_ENGLISH_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_ENGLISH_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_ENGLISH_INTEGER_WORDS = tuple(
    _ENGLISH_SMALL | _ENGLISH_TENS | {"hundred": 100} | _ENGLISH_SCALES
)
_ENGLISH_WORD_PATTERN = "|".join(
    sorted(_ENGLISH_INTEGER_WORDS, key=len, reverse=True)
)
_ENGLISH_DIGIT_PATTERN = "|".join(_ENGLISH_DIGITS)
_ENGLISH_NUMBER_RE = re.compile(
    rf"(?<![A-Za-z])(?:(?:minus|negative)[ -]+)?"
    rf"(?:{_ENGLISH_WORD_PATTERN})"
    rf"(?:(?:[ -]+)(?:and[ -]+)?(?:{_ENGLISH_WORD_PATTERN}))*"
    rf"(?:[ -]+point(?:[ -]+)(?:{_ENGLISH_DIGIT_PATTERN})"
    rf"(?:(?:[ -]+)(?:{_ENGLISH_DIGIT_PATTERN}))*)?(?![A-Za-z])",
    re.IGNORECASE,
)


class NumberLanguageAdapter(Protocol):
    def process(self, text: str) -> str: ...


class ChineseNumberAdapter:
    def process(self, text: str) -> str:
        text = _SPACED_CHINESE_DIGITS_RE.sub(
            _replace_spaced_chinese_digits, text
        )
        text = _CHINESE_PERCENT_RE.sub(_replace_chinese_percent, text)
        text = _CHINESE_NUMBER_RE.sub(
            lambda match: _replace_chinese_number(match, text), text
        )
        return text


class EnglishNumberAdapter:
    def process(self, text: str) -> str:
        return _ENGLISH_NUMBER_RE.sub(_replace_english_number, text)


class NumberNormalizer:
    def __init__(
        self,
        adapters: Iterable[NumberLanguageAdapter] | None = None,
    ) -> None:
        self._adapters = tuple(
            adapters
            if adapters is not None
            else (ChineseNumberAdapter(), EnglishNumberAdapter())
        )

    def process(self, text: str) -> str:
        for adapter in self._adapters:
            text = adapter.process(text)
        return text


def _replace_chinese_percent(match: re.Match[str]) -> str:
    value = _normalize_number_expression(
        match.group("number"),
        _parse_complete_chinese_number,
        _concatenate_chinese_number,
    )
    return match.group(0) if value is None else f"{value}%"


def _replace_spaced_chinese_digits(match: re.Match[str]) -> str:
    return "".join(
        str(_DIGITS[character])
        for character in match.group(0)
        if not character.isspace()
    )


def _replace_chinese_number(match: re.Match[str], source: str) -> str:
    token = match.group(0)
    next_character = source[match.end() : match.end() + 1]
    if (
        len(token) == 1
        and next_character
        and _is_cjk_letter(next_character)
        and next_character not in _QUANTITY_SUFFIXES
    ):
        return token
    value = _normalize_number_expression(
        token,
        _parse_complete_chinese_number,
        _concatenate_chinese_number,
    )
    return token if value is None else value


def _is_chinese_number_enumeration(token: str) -> bool:
    first_unit = next(
        (
            index
            for index, character in enumerate(token)
            if character in _SMALL_UNITS or character in _LARGE_UNITS
        ),
        -1,
    )
    return first_unit >= 2 and all(
        character in _DIGITS for character in token[:first_unit]
    )


def _parse_complete_chinese_number(token: str) -> str | None:
    if _is_chinese_number_enumeration(token):
        return None
    return _parse_chinese_number(token)


def _concatenate_chinese_number(token: str) -> str | None:
    atoms = {**_DIGITS, **_SMALL_UNITS, **_LARGE_UNITS}
    parts: list[str] = []
    for character in token:
        if character == "负":
            parts.append("-")
        elif character == "点":
            parts.append(".")
        elif character in atoms:
            parts.append(str(atoms[character]))
        else:
            return None
    return "".join(parts)


def _parse_chinese_number(token: str) -> str | None:
    negative = token.startswith("负")
    if negative:
        token = token[1:]
    if not token:
        return None

    integer_text, separator, decimal_text = token.partition("点")
    if separator and (not decimal_text or "点" in decimal_text):
        return None
    integer = _parse_chinese_integer(integer_text)
    if integer is None:
        return None
    result = str(integer)
    if separator:
        if any(character not in _DIGITS for character in decimal_text):
            return None
        result += "." + "".join(
            str(_DIGITS[character]) for character in decimal_text
        )
    return f"-{result}" if negative else result


def _parse_chinese_integer(token: str) -> int | None:
    if not token:
        return 0
    if all(character in _DIGITS for character in token):
        return int("".join(str(_DIGITS[character]) for character in token))
    if any(
        character not in _DIGITS
        and character not in _SMALL_UNITS
        and character not in _LARGE_UNITS
        for character in token
    ):
        return None

    total = 0
    section = 0
    digit = 0
    for character in token:
        if character in _DIGITS:
            digit = _DIGITS[character]
        elif character in _SMALL_UNITS:
            section += (digit or 1) * _SMALL_UNITS[character]
            digit = 0
        else:
            section += digit
            total += (section or 1) * _LARGE_UNITS[character]
            section = 0
            digit = 0
    return total + section + digit


def _replace_english_number(match: re.Match[str]) -> str:
    token = match.group(0)
    value = _normalize_number_expression(
        token,
        _parse_complete_english_number,
        _concatenate_english_number,
    )
    return token if value is None else value


def _parse_complete_english_number(token: str) -> str | None:
    words = re.split(r"[ -]+", token.lower())
    negative = words[0] in {"minus", "negative"}
    if negative:
        words = words[1:]

    if "point" in words:
        point_index = words.index("point")
        integer_words = words[:point_index]
        decimal_words = words[point_index + 1 :]
    else:
        integer_words = words
        decimal_words = []
    integer_words = [word for word in integer_words if word != "and"]
    integer = _parse_english_integer(integer_words)
    if integer is None or any(word not in _ENGLISH_DIGITS for word in decimal_words):
        return None
    result = str(integer)
    if decimal_words:
        result += "." + "".join(
            str(_ENGLISH_DIGITS[word]) for word in decimal_words
        )
    return f"-{result}" if negative else result


def _parse_english_integer(words: list[str]) -> int | None:
    if not words:
        return 0
    total = 0
    chunk: list[str] = []
    previous_scale = float("inf")
    for word in words:
        if word not in _ENGLISH_SCALES:
            chunk.append(word)
            continue
        scale = _ENGLISH_SCALES[word]
        chunk_value = _parse_english_chunk(chunk)
        if chunk_value is None or scale >= previous_scale:
            return None
        total += chunk_value * scale
        chunk = []
        previous_scale = scale
    chunk_value = _parse_english_chunk(chunk)
    return None if chunk_value is None else total + chunk_value


def _parse_english_chunk(words: list[str]) -> int | None:
    if not words:
        return None
    if "hundred" not in words:
        if "and" in words:
            return None
        return _parse_english_under_hundred(words)
    if words.count("hundred") != 1 or words.index("hundred") != 1:
        return None
    if words[0] not in _ENGLISH_DIGITS or words[0] == "zero":
        return None
    remainder = words[2:]
    if remainder[:1] == ["and"]:
        remainder = remainder[1:]
    elif "and" in remainder:
        return None
    remainder_value = (
        _parse_english_under_hundred(remainder) if remainder else 0
    )
    if remainder_value is None:
        return None
    return _ENGLISH_DIGITS[words[0]] * 100 + remainder_value


def _parse_english_under_hundred(words: list[str]) -> int | None:
    if len(words) == 1:
        return _ENGLISH_SMALL.get(words[0], _ENGLISH_TENS.get(words[0]))
    if (
        len(words) == 2
        and words[0] in _ENGLISH_TENS
        and words[1] in _ENGLISH_DIGITS
        and words[1] != "zero"
    ):
        return _ENGLISH_TENS[words[0]] + _ENGLISH_DIGITS[words[1]]
    return None


def _concatenate_english_number(token: str) -> str | None:
    atoms = {
        **_ENGLISH_SMALL,
        **_ENGLISH_TENS,
        "hundred": 100,
        **_ENGLISH_SCALES,
    }
    parts: list[str] = []
    for word in re.split(r"[ -]+", token.lower()):
        if word in {"minus", "negative"}:
            parts.append("-")
        elif word == "point":
            parts.append(".")
        elif word == "and":
            continue
        elif word in atoms:
            parts.append(str(atoms[word]))
        else:
            return None
    return "".join(parts)


def _normalize_number_expression(
    token: str,
    parse_complete: Callable[[str], str | None],
    concatenate_atoms: Callable[[str], str | None],
) -> str | None:
    parsed = parse_complete(token)
    return parsed if parsed is not None else concatenate_atoms(token)


def _is_cjk_letter(character: str) -> bool:
    return "\u3400" <= character <= "\u9fff"