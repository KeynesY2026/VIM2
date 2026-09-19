from __future__ import annotations

import unicodedata
from dataclasses import dataclass

SENTENCE_ENDINGS = frozenset("。！？.!?")
_CJK_NAME_PARTS = ("HIRAGANA", "KATAKANA", "BOPOMOFO")
_COMMON_LETTER_PREFIXES = {
    "CIRCLED",
    "MATHEMATICAL",
    "MODIFIER",
    "PARENTHESIZED",
    "SUBSCRIPT",
    "SUPERSCRIPT",
}


@dataclass(frozen=True, slots=True)
class StableCheckpoint:
    prefix: str
    anchor: str
    frame_count: int


class StablePrefixTracker:
    def __init__(self, required_matches: int = 3) -> None:
        self._required_matches = required_matches
        self.reset()

    @property
    def checkpoint(self) -> StableCheckpoint | None:
        return self._checkpoint

    def reset(self) -> None:
        self._candidate = ""
        self._matches = 0
        self._checkpoint: StableCheckpoint | None = None

    def observe(self, text: str, frame_count: int) -> None:
        prefix = _complete_prefix(text.strip())
        if not prefix:
            self._candidate = ""
            self._matches = 0
            return
        if self._checkpoint and not prefix.startswith(self._checkpoint.prefix):
            self._candidate = ""
            self._matches = 0
            return
        confirmed = _prefix_before_last_sentence(prefix)
        if confirmed and (
            self._checkpoint is None
            or len(confirmed) > len(self._checkpoint.prefix)
        ):
            self._checkpoint = StableCheckpoint(
                prefix=confirmed,
                anchor=_last_sentence(confirmed),
                frame_count=frame_count,
            )
        if prefix == self._candidate:
            self._matches += 1
        else:
            self._candidate = prefix
            self._matches = 1
        if self._matches >= self._required_matches and (
            self._checkpoint is None
            or len(prefix) > len(self._checkpoint.prefix)
        ):
            self._checkpoint = StableCheckpoint(
                prefix=prefix,
                anchor=_last_sentence(prefix),
                frame_count=frame_count,
            )


def merge_stable_tail(
    checkpoint: StableCheckpoint, tail_text: str
) -> str | None:
    tail = tail_text.strip()
    if not tail or tail.count(checkpoint.anchor) != 1:
        return None
    _, remainder = tail.split(checkpoint.anchor, maxsplit=1)
    return f"{checkpoint.prefix}{remainder}".strip()


def format_mixed_language_spacing(text: str) -> str:
    formatted: list[str] = []
    active_system: str | None = None
    for character in text:
        system = _writing_system(character)
        if system is not None:
            if active_system is not None and system != active_system:
                formatted.append(" ")
            formatted.append(character)
            active_system = system
        elif (
            active_system is not None
            and unicodedata.category(character).startswith("M")
        ):
            formatted.append(character)
        else:
            formatted.append(character)
            active_system = None
    return "".join(formatted)


def _writing_system(character: str) -> str | None:
    name = unicodedata.name(character, "")
    if (
        name.startswith("CJK UNIFIED IDEOGRAPH")
        or name.startswith("CJK COMPATIBILITY IDEOGRAPH")
        or character == "\u3007"
    ):
        return "CJK"
    if not unicodedata.category(character).startswith("L"):
        return None
    if any(part in name for part in _CJK_NAME_PARTS):
        return "CJK"
    parts = name.split()
    if not parts or parts[0] in _COMMON_LETTER_PREFIXES:
        return None
    if parts[0] in {"FULLWIDTH", "HALFWIDTH"} and len(parts) > 1:
        return parts[1]
    if parts[0] in {"OLD", "LINEAR"} and len(parts) > 1:
        return " ".join(parts[:2])
    return parts[0]


def _complete_prefix(text: str) -> str:
    final_ending = max(
        (index for index, character in enumerate(text) if character in SENTENCE_ENDINGS),
        default=-1,
    )
    return text[: final_ending + 1]


def _prefix_before_last_sentence(prefix: str) -> str:
    anchor = _last_sentence(prefix)
    if not anchor:
        return ""
    return prefix[: prefix.rfind(anchor)].rstrip()


def _last_sentence(prefix: str) -> str:
    terminator_start = len(prefix)
    while (
        terminator_start > 0
        and prefix[terminator_start - 1] in SENTENCE_ENDINGS
    ):
        terminator_start -= 1
    previous_ending = max(
        (
            index
            for index, character in enumerate(prefix[:terminator_start])
            if character in SENTENCE_ENDINGS
        ),
        default=-1,
    )
    return prefix[previous_ending + 1 :].lstrip()
