from __future__ import annotations

import math
from dataclasses import dataclass

from .model_probe import ModelInfo


def dtype_bytes(dtype_str: str) -> int:
    """
    Byte size per element for KV cache dtype strings.
    Keep it intentionally small & explicit: no external probing, no subprocesses.
    """
    d = dtype_str.lower().strip()

    if d in ("float16", "fp16"):
        return 2
    if d in ("bfloat16", "bf16"):
        return 2

    # vLLM may accept fp8 variants like "fp8", "fp8_e4m3fn", etc.
    if "fp8" in d:
        return 1

    raise ValueError(f"Unsupported dtype string for byte sizing: {dtype_str}")


@dataclass(frozen=True)
class KvSizing:
    kv_bytes_per_token: int
    kv_bytes_per_block: int
    blocks_per_seq: int
    rounded_seq_len: int
    kv_bytes_per_seq: int


def kv_sizing(
        model: ModelInfo,
        seq_len: int,
        block_size: int,
        kv_cache_dtype: str,
        fallback_kv_dtype_if_auto: str,
) -> KvSizing:
    """
    Static KV sizing that matches vLLM's notion:
      kv_bytes_per_token = layers * kv_heads * head_dim * 2(K+V) * bytes_per_elem

    CRITICAL:
      model.head_dim must come from config.head_dim when present
      (do NOT assume hidden_size/num_attention_heads).
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    kv_dtype = fallback_kv_dtype_if_auto if kv_cache_dtype == "auto" else kv_cache_dtype
    bpe = dtype_bytes(kv_dtype)

    kv_bytes_per_token = model.num_layers * model.num_kv_heads * model.head_dim * 2 * bpe

    blocks_per_seq = math.ceil(seq_len / block_size)
    rounded_seq_len = blocks_per_seq * block_size
    kv_bytes_per_block = kv_bytes_per_token * block_size
    kv_bytes_per_seq = kv_bytes_per_token * rounded_seq_len

    return KvSizing(
        kv_bytes_per_token=kv_bytes_per_token,
        kv_bytes_per_block=kv_bytes_per_block,
        blocks_per_seq=blocks_per_seq,
        rounded_seq_len=rounded_seq_len,
        kv_bytes_per_seq=kv_bytes_per_seq,
    )
