from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from transformers import AutoConfig


@dataclass(frozen=True)
class ModelInfo:
    model_name: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    hidden_size: int
    head_dim: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _must_int(x: Any, name: str) -> int:
    if x is None:
        raise ValueError(f"Missing required model config field: {name}")
    return int(x)


def probe_model(model_name: str, *, trust_remote_code: bool = False) -> ModelInfo:
    """
    Pure-Python model probe via Transformers config.

    IMPORTANT:
    - Prefer explicit config.head_dim when present.
      (For Qwen3-4B-Instruct-2507, head_dim=128 even though hidden_size/num_heads=80.)
    """
    print(f"Probing model config for {model_name}")
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    # Layers
    num_layers = (
            getattr(cfg, "num_hidden_layers", None)
            or getattr(cfg, "n_layer", None)
            or getattr(cfg, "num_layers", None)
    )
    num_layers = _must_int(num_layers, "num_hidden_layers|n_layer|num_layers")

    # Heads
    num_attention_heads = (
            getattr(cfg, "num_attention_heads", None)
            or getattr(cfg, "n_head", None)
            or getattr(cfg, "num_heads", None)
    )
    num_attention_heads = _must_int(num_attention_heads, "num_attention_heads|n_head|num_heads")

    # KV heads (GQA/MQA)
    num_kv_heads = (
            getattr(cfg, "num_key_value_heads", None)
            or getattr(cfg, "num_kv_heads", None)
            or getattr(cfg, "n_kv_head", None)
    )
    if num_kv_heads is None:
        # If a model doesn't specify KV heads, it typically means MHA (kv_heads == attn_heads).
        num_kv_heads = num_attention_heads
    num_kv_heads = int(num_kv_heads)

    # Hidden size
    hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "n_embd", None)
            or getattr(cfg, "d_model", None)
    )
    hidden_size = _must_int(hidden_size, "hidden_size|n_embd|d_model")

    # Head dim: prefer explicit
    explicit_head_dim = getattr(cfg, "head_dim", None) or getattr(cfg, "head_size", None)
    if explicit_head_dim is not None:
        head_dim = int(explicit_head_dim)
    else:
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Cannot derive head_dim: hidden_size={hidden_size} not divisible by "
                f"num_attention_heads={num_attention_heads} for model={model_name}"
            )
        head_dim = hidden_size // num_attention_heads

    return ModelInfo(
        model_name=model_name,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
    )
