import threading
from types import SimpleNamespace

import pytest

from vim2.draft import (
    GpuDraftDecoder,
    GpuGenerationLimitError,
    draft_continuation,
    matched_draft_tokens,
)
from vim2.recognizer import TranscriptionCancelled


def test_draft_resynchronizes_after_correction_and_limits_proposals():
    assert draft_continuation([99, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6], 1) == [5]
    assert draft_continuation([9], [1, 2, 3], 7) == []
    assert draft_continuation([1], [1, 2, 1, 3], 7) == [3]
    assert draft_continuation([], [1], 7) == []
    assert draft_continuation([1], [1, 2], 0) == []


def test_only_matching_draft_tokens_are_accepted_not_bonus_or_correction():
    assert matched_draft_tokens([1, 2, 3], [2, 9, 8]) == [2]
    assert matched_draft_tokens([1, 2, 3], [2, 3, 8]) == [2, 3]
    assert matched_draft_tokens([1, 2], [9, 8]) == []
    with pytest.raises(ValueError):
        matched_draft_tokens([1], [])


class FakeCache:
    def __init__(self):
        self.crops = []

    def crop(self, position):
        self.crops.append(position)


class FakeThinker:
    def __init__(self, torch, *, cancel_event=None):
        self.torch = torch
        self.cache = FakeCache()
        self.calls = []
        self.cancel_event = cancel_event
        self.lm_head = torch.nn.Identity()

    def __call__(self, input_ids, **kwargs):
        ids = input_ids[0].tolist()
        self.calls.append((ids, kwargs["cache_position"].tolist()))
        predictions = [{10: 1, 1: 2, 2: 3, 3: 151645}.get(token, 151645) for token in ids]
        logits = self.torch.zeros((1, len(ids), 151646))
        for index, prediction in enumerate(predictions):
            logits[0, index, prediction] = 1
        if self.cancel_event is not None and len(self.calls) == 2:
            self.cancel_event.set()
        return SimpleNamespace(logits=self.lm_head(logits), past_key_values=self.cache)


@pytest.mark.parametrize("draft", [[], [1, 2, 3, 151645], [1, 99, 3, 151645]])
def test_gpu_decoder_verifies_and_rolls_back_to_target_tokens(draft):
    torch = pytest.importorskip("torch")
    thinker = FakeThinker(torch)
    decoder = GpuDraftDecoder(SimpleNamespace(thinker=thinker), torch, draft)
    result = decoder.generate(input_ids=torch.tensor([[10]]))
    assert result.sequences.tolist() == [[10, 1, 2, 3, 151645]]
    assert decoder.tokens == (1, 2, 3, 151645)
    assert thinker.cache.crops[-1] == 4
    if draft and draft[1] == 99:
        assert thinker.cache.crops[0] == 1
        assert thinker.calls[2][0] == [1]


def test_gpu_decoder_cancellation_and_generation_limit_are_explicit():
    torch = pytest.importorskip("torch")
    event = threading.Event()
    thinker = FakeThinker(torch, cancel_event=event)
    decoder = GpuDraftDecoder(
        SimpleNamespace(thinker=thinker), torch, [1, 2, 3],
        cancel_event=event, cancellation_error=TranscriptionCancelled,
    )
    with pytest.raises(TranscriptionCancelled):
        decoder.generate(input_ids=torch.tensor([[10]]))
    assert decoder.tokens == ()
    thinker = FakeThinker(torch)
    decoder = GpuDraftDecoder(SimpleNamespace(thinker=thinker), torch, [])
    with pytest.raises(GpuGenerationLimitError, match="generation limit") as limited:
        decoder.generate(input_ids=torch.tensor([[10]]), max_new_tokens=2)
    assert limited.value.tokens == (1, 2)


def test_gpu_decoder_rejects_unsupported_options():
    torch = pytest.importorskip("torch")
    decoder = GpuDraftDecoder(SimpleNamespace(), torch, [])
    with pytest.raises(ValueError, match="Unsupported"):
        decoder.generate(input_ids=torch.tensor([[10]]), temperature=0.7)


def test_gpu_decoder_rejects_budget_above_verified_limit():
    torch = pytest.importorskip("torch")
    thinker = FakeThinker(torch)
    decoder = GpuDraftDecoder(SimpleNamespace(thinker=thinker), torch, [])
    with pytest.raises(ValueError, match="budget"):
        decoder.generate(input_ids=torch.tensor([[10]]), max_new_tokens=513)


def test_prefill_last_row_projection_leaves_no_hook_or_shortened_kv():
    torch = pytest.importorskip("torch")
    thinker = FakeThinker(torch)
    decoder = GpuDraftDecoder(SimpleNamespace(thinker=thinker), torch, [])
    output = decoder.generate(input_ids=torch.tensor([[10, 10, 10]]))
    assert output.sequences.tolist() == [[10, 10, 10, 1, 2, 3, 151645]]
    assert thinker.calls[0][1] == [0, 1, 2]
    assert thinker.cache.crops[-1] == 6
    assert not thinker.lm_head._forward_pre_hooks


def test_cancelled_decoder_traceback_does_not_retain_output_or_gpu_cache():
    torch = pytest.importorskip("torch")
    event = threading.Event()
    thinker = FakeThinker(torch, cancel_event=event)
    decoder = GpuDraftDecoder(
        SimpleNamespace(thinker=thinker), torch, [1, 2, 3],
        cancel_event=event, cancellation_error=TranscriptionCancelled,
    )
    with pytest.raises(TranscriptionCancelled) as caught:
        decoder.generate(input_ids=torch.tensor([[10]]))
    trace = caught.value.__traceback__
    while trace:
        if trace.tb_frame.f_code.co_name == "_generate":
            assert trace.tb_frame.f_locals == {}
        trace = trace.tb_next
    assert not thinker.lm_head._forward_pre_hooks
