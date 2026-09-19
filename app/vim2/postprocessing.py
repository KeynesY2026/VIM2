from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from vim2.transcript import format_mixed_language_spacing


class TextPostProcessor(Protocol):
    def process(self, text: str) -> str: ...


class TextPostProcessingPipeline:
    def __init__(self, processors: Iterable[TextPostProcessor] = ()) -> None:
        self._processors = tuple(processors)

    def process(self, text: str) -> str:
        for processor in self._processors:
            text = processor.process(text)
        return format_mixed_language_spacing(text)


def create_text_postprocessor(
    *, normalize_numbers: bool,
) -> TextPostProcessingPipeline:
    processors: list[TextPostProcessor] = []
    if normalize_numbers:
        from vim2.number_normalizer import NumberNormalizer

        processors.append(NumberNormalizer())
    return TextPostProcessingPipeline(processors)
