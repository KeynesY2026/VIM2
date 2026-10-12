from __future__ import annotations

import threading
import traceback
from types import SimpleNamespace
from typing import Any, Sequence


class GpuGenerationLimitError(RuntimeError):
    def __init__(self, tokens: Sequence[int]) -> None:
        super().__init__("GPU decoding reached the generation limit without EOS")
        self.tokens = tuple(tokens)


def draft_continuation(
    generated: Sequence[int], draft: Sequence[int], maximum: int
) -> list[int]:
    if not generated or maximum <= 0:
        return []
    width = min(4, len(generated))
    suffix = list(generated[-width:])
    for end in range(len(draft) - 1, width - 1, -1):
        if list(draft[end - width:end]) == suffix:
            return list(draft[end:end + maximum])
    return []


def matched_draft_tokens(
    input_tokens: Sequence[int], predictions: Sequence[int]
) -> list[int]:
    if not input_tokens or len(input_tokens) != len(predictions):
        raise ValueError("Invalid draft verification row count")
    matched = []
    for predicted, drafted in zip(predictions, input_tokens[1:]):
        if predicted != drafted:
            break
        matched.append(drafted)
    return matched


class GpuDraftDecoder:
    """Verify drafts without locking text or retaining a live GPU KV cache."""

    def __init__(
        self,
        model: Any,
        torch_module: Any,
        draft: Sequence[int],
        *,
        cancel_event: threading.Event | None = None,
        cancellation_error: type[RuntimeError] = RuntimeError,
    ) -> None:
        self.model = model
        self.torch = torch_module
        self.draft = list(draft)
        self.cancel_event = cancel_event
        self.cancellation_error = cancellation_error
        self.tokens: tuple[int, ...] = ()
        self.forward_calls = 0
        self.verified_tokens = 0

    def _check_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise self.cancellation_error("Transcription cancelled")

    def generate(self, input_ids=None, max_new_tokens=512, **kwargs):
        with self.torch.inference_mode():
            try:
                return self._generate(input_ids, max_new_tokens, **kwargs)
            except (RuntimeError, ValueError, MemoryError) as error:
                # Qt delivers exceptions across threads; their traceback must
                # not keep cancelled GPU outputs/KV alive during finalization.
                traceback.clear_frames(error.__traceback__)
                raise

    def _generate(self, input_ids, max_new_tokens, **kwargs):
        self._check_cancelled()
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Draft decoding requires exactly one input")
        allowed = {"input_features", "feature_attention_mask", "attention_mask"}
        if set(kwargs) - allowed:
            raise ValueError(f"Unsupported draft generation options: {set(kwargs) - allowed}")
        if not 1 <= max_new_tokens <= 512:
            raise ValueError("Draft generation budget must be between 1 and 512")
        torch = self.torch
        thinker = self.model.thinker
        prompt_length = input_ids.shape[1]
        initial_mask = kwargs.pop("attention_mask", torch.ones_like(input_ids))
        if not torch.all(initial_mask == 1):
            raise ValueError("Draft decoding does not support padded inputs")
        # Prefill needs only the last logit row, not a vocabulary projection
        # for every audio placeholder. All hidden states and KV remain intact.
        projection_hook = thinker.lm_head.register_forward_pre_hook(
            lambda _module, args: (args[0][:, -1:, :],)
        )
        try:
            output = thinker(
                input_ids=input_ids, attention_mask=initial_mask,
                cache_position=torch.arange(prompt_length, device=input_ids.device),
                use_cache=True, return_dict=True, **kwargs,
            )
        finally:
            projection_hook.remove()
        self._check_cancelled()
        cache = output.past_key_values
        if cache is None or not callable(getattr(cache, "crop", None)):
            raise RuntimeError("GPU decoder did not provide a rollback-capable cache")
        next_token = int(output.logits[0, -1].argmax())
        del output
        self.forward_calls = 1
        self.verified_tokens = 0
        generated = [next_token]
        position = prompt_length
        force_single = False
        eos = {151645, 151643}
        while generated[-1] not in eos and len(generated) < max_new_tokens:
            self._check_cancelled()
            proposals = [] if force_single else draft_continuation(
                generated, self.draft, min(7, max_new_tokens - len(generated) - 1),
            )
            ids = [generated[-1], *proposals]
            output = thinker(
                input_ids=torch.tensor([ids], device=input_ids.device),
                attention_mask=torch.ones(
                    (1, position + len(ids)), dtype=initial_mask.dtype,
                    device=input_ids.device,
                ),
                cache_position=torch.arange(
                    position, position + len(ids), device=input_ids.device
                ),
                past_key_values=cache, use_cache=True, return_dict=True,
            )
            self._check_cancelled()
            cache = output.past_key_values
            predictions = output.logits[0].argmax(dim=-1).tolist()
            del output
            if len(ids) > 1:
                emitted = matched_draft_tokens(ids, predictions)
                force_single = True
                self.verified_tokens += len(emitted)
            else:
                emitted = predictions
                force_single = False
            for index, token in enumerate(emitted):
                if token in eos:
                    emitted = emitted[:index + 1]
                    break
            self.forward_calls += 1
            position += len(emitted)
            cache.crop(position)
            generated.extend(emitted)
        self._check_cancelled()
        if generated[-1] not in eos:
            raise GpuGenerationLimitError(generated)
        self.tokens = tuple(generated)
        return SimpleNamespace(sequences=torch.cat(
            (input_ids, torch.tensor([generated], device=input_ids.device)), dim=1
        ))
