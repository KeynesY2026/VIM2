from __future__ import annotations

from dataclasses import dataclass

SENTENCE_ENDINGS = frozenset("。！？.!?")


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


def _complete_prefix(text: str) -> str:
    final_ending = max(
        (index for index, character in enumerate(text) if character in SENTENCE_ENDINGS),
        default=-1,
    )
    return text[: final_ending + 1]


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
