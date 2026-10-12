from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


AUDIO_SDPA_IMPLEMENTATION = "vim2_audio_sdpa"
TEXT_SDPA_IMPLEMENTATION = "vim2_text_sdpa"
WINDOWED_AUDIO_THRESHOLD_SECONDS = 30


def text_sdpa_attention_forward(
    module: Any,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[Tensor, None]:
    import torch
    from transformers.integrations.sdpa_attention import repeat_kv

    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        raise ValueError("Text SDPA does not support attention weights or head masks")
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, :key.shape[-2]]
    if is_causal is None:
        is_causal = (
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )
    # Explicit KV expansion avoids GQA's quadratic math-kernel fallback
    # on GPUs that cannot run the fused GQA kernel.
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout,
        scale=scaling, is_causal=is_causal,
    )
    return output.transpose(1, 2).contiguous(), None


def audio_sdpa_attention_forward(
    module: Any,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    **kwargs: Any,
) -> tuple[Tensor, None]:
    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    boundaries = kwargs.get("cu_seq_lens_q")
    key_boundaries = kwargs.get("cu_seq_lens_k")
    if boundaries is None and key_boundaries is not None:
        raise ValueError("Missing query audio attention window boundaries")
    if boundaries is not None:
        if (
            boundaries.ndim != 1
            or boundaries.dtype not in (torch.int32, torch.int64)
            or len(boundaries) < 2
            or boundaries[0].item() != 0
            or boundaries[-1].item() != query.shape[-2]
            or torch.any(boundaries[1:] <= boundaries[:-1]).item()
            or key_boundaries is None
            or not torch.equal(boundaries, key_boundaries)
            or key.shape[-2] != query.shape[-2]
        ):
            raise ValueError("Invalid audio attention window boundaries")
        # SDPA ignores cumulative lengths, so encode the audio windows as a mask.
        positions = torch.arange(query.shape[-2], device=query.device)
        windows = torch.bucketize(
            positions, boundaries[1:-1].to(query.device), right=True
        )
        allowed = windows[:, None] == windows[None, :]
        if attention_mask is None:
            attention_mask = allowed
        elif attention_mask.dtype == torch.bool:
            attention_mask = attention_mask & allowed
        else:
            attention_mask = attention_mask.masked_fill(~allowed, float("-inf"))
        kwargs["is_causal"] = False
    return sdpa_attention_forward(
        module, query, key, value, attention_mask, **kwargs
    )
