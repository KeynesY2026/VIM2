from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from vim2.attention import audio_sdpa_attention_forward, text_sdpa_attention_forward


@pytest.mark.parametrize("floating_mask", [False, True])
def test_audio_windows_preserve_existing_attention_mask(floating_mask):
    module = SimpleNamespace(training=False, is_causal=False)
    query = torch.zeros(1, 1, 4, 1)
    value = torch.tensor([1.0, 3.0, 10.0, 12.0]).reshape(1, 1, 4, 1)
    mask = torch.tensor([True, False, True, False]).expand(4, 4)
    if floating_mask:
        mask = torch.zeros(4, 4).masked_fill(~mask, float("-inf"))
    boundaries = torch.tensor([0, 2, 4], dtype=torch.int32)
    result, _ = audio_sdpa_attention_forward(
        module, query, query, value, mask,
        cu_seq_lens_q=boundaries, cu_seq_lens_k=boundaries,
    )
    assert torch.equal(result.reshape(-1), torch.tensor([1.0, 1.0, 10.0, 10.0]))


def test_attention_without_audio_boundaries_preserves_causality():
    module = SimpleNamespace(training=False, is_causal=True)
    query = torch.zeros(1, 1, 2, 1)
    value = torch.tensor([1.0, 9.0]).reshape(1, 1, 2, 1)
    result, _ = audio_sdpa_attention_forward(module, query, query, value, None)
    assert torch.equal(result.reshape(-1), torch.tensor([1.0, 5.0]))


@pytest.mark.parametrize(
    "values", [[0, 2, 3], [1, 2, 4], [0, 2, 2, 4], [0.0, 2.5, 4.0]]
)
def test_invalid_audio_boundaries_are_rejected(values):
    module = SimpleNamespace(training=False, is_causal=False)
    query = torch.zeros(1, 1, 4, 1)
    boundaries = torch.tensor(values)
    with pytest.raises(ValueError, match="boundaries"):
        audio_sdpa_attention_forward(
            module, query, query, query, None,
            cu_seq_lens_q=boundaries, cu_seq_lens_k=boundaries,
        )


def test_missing_query_boundaries_are_rejected():
    module = SimpleNamespace(training=False, is_causal=False)
    query = torch.zeros(1, 1, 4, 1)
    with pytest.raises(ValueError, match="boundaries"):
        audio_sdpa_attention_forward(
            module, query, query, query, None,
            cu_seq_lens_k=torch.tensor([0, 2, 4], dtype=torch.int32),
        )


@pytest.mark.parametrize("masked", [False, True])
def test_text_sdpa_preserves_grouped_query_attention_and_causality(masked):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    torch.manual_seed(42)
    module = SimpleNamespace(training=False, is_causal=True, num_key_value_groups=2)
    query = torch.randn(1, 4, 6, 8)
    key = torch.randn(1, 2, 6, 8)
    value = torch.randn_like(key)
    mask = torch.ones(6, 6, dtype=torch.bool).tril() if masked else None
    expected, _ = sdpa_attention_forward(module, query, key, value, mask)
    actual, _ = text_sdpa_attention_forward(module, query, key, value, mask)
    torch.testing.assert_close(actual, expected)


def test_text_sdpa_does_not_select_unbounded_gqa_math_path(monkeypatch):
    calls = []

    def attention(query, key, value, **kwargs):
        calls.append((query.shape[1], key.shape[1], value.shape[1], kwargs))
        return query

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", attention)
    module = SimpleNamespace(is_causal=True, num_key_value_groups=2)
    text_sdpa_attention_forward(
        module, torch.zeros(1, 4, 6, 8), torch.zeros(1, 2, 6, 8),
        torch.zeros(1, 2, 6, 8), None,
    )
    assert calls[0][:3] == (4, 4, 4)
    assert "enable_gqa" not in calls[0][3]
    assert calls[0][3]["is_causal"] is True
