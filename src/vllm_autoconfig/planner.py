from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal, Iterable

from huggingface_hub import snapshot_download

from .gpu_probe import probe_gpu, GpuInfo
from .model_probe import probe_model, ModelInfo
from .kv_math import kv_sizing
from .cache import make_cache_key, load_cached_plan, save_cached_plan

PerfMode = Literal["throughput", "latency"]

logger = logging.getLogger(__name__)

GiB = 1024 ** 3


@dataclass(frozen=True)
class Plan:
    vllm_kwargs: dict[str, Any]
    gpu: dict[str, Any]
    model: dict[str, Any]
    notes: dict[str, Any]
    cache_key: str


def _pick_weight_dtype(gpu: GpuInfo) -> str:
    return "bfloat16" if gpu.is_bf16_supported else "float16"


def _is_mistral_repo(model_name: str) -> bool:
    # Your requested heuristic: infer from repo name
    return model_name.startswith("mistralai/")


def _align_down(x: int, align: int) -> int:
    return (x // align) * align


def _iter_weight_candidates(repo_dir: Path) -> list[Path]:
    # Weights are typically in the snapshot root; but be robust.
    files: list[Path] = []
    for p in repo_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix in {".safetensors", ".bin"}:
            files.append(p)
    return files


def _read_index_weight_files(repo_dir: Path) -> list[Path] | None:
    """
    If a weight index exists, sum ONLY files referenced by that index.

    This avoids accidental double-counting if the snapshot directory also contains
    extra files (e.g., a stray consolidated.safetensors).
    """
    index_candidates = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        # keep this last; some repos may have it, but if other index exists it should win
        "consolidated.safetensors.index.json",
    ]

    for name in index_candidates:
        idx_path = repo_dir / name
        if not idx_path.exists():
            continue
        try:
            data = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed reading weight index {idx_path}: {e}") from e

        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            continue

        referenced = sorted(set(weight_map.values()))
        paths: list[Path] = []
        for rel in referenced:
            p = repo_dir / rel
            if p.exists() and p.is_file():
                paths.append(p)
        if paths:
            return paths

    # Also try any "*.index.json" if the above canonical names weren't found.
    # (Deterministic: stable sort)
    for idx_path in sorted(repo_dir.glob("*.index.json")):
        if idx_path.name in index_candidates:
            continue
        try:
            data = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            continue
        referenced = sorted(set(weight_map.values()))
        paths: list[Path] = []
        for rel in referenced:
            p = repo_dir / rel
            if p.exists() and p.is_file():
                paths.append(p)
        if paths:
            return paths

    return None


def _prefer_sharded(files: Iterable[Path]) -> list[Path]:
    safes = [p for p in files if p.suffix == ".safetensors"]
    bins = [p for p in files if p.suffix == ".bin"]

    # Common shard naming: model-00001-of-000XX.safetensors or pytorch_model-00001-of-000XX.bin
    shard_re = re.compile(r".*-\d{5}-of-\d{5}\.(safetensors|bin)$")

    safe_shards = [p for p in safes if shard_re.match(p.name)]
    bin_shards = [p for p in bins if shard_re.match(p.name)]

    if safe_shards:
        return sorted(safe_shards)
    if bin_shards:
        return sorted(bin_shards)

    # If both sharded AND consolidated exist, ignore consolidated by default.
    # If we have multiple safetensors files and at least one looks sharded, we already returned above.
    # Otherwise: return all of one preferred type if present, else all bins.
    if safes:
        return sorted(safes)
    return sorted(bins)


def _estimate_weights_bytes(model_name: str, local_files_only: bool) -> tuple[int, list[str]]:
    """
    Weight-size estimate that avoids double counting:
    - If an index exists: sum only referenced weight files.
    - Else: prefer shard patterns; avoid counting both shards + consolidated.
    Returns (total_bytes, chosen_filenames).
    """
    repo_dir = Path(
        snapshot_download(
            repo_id=model_name,
            local_files_only=local_files_only,
            allow_patterns=[
                "*.safetensors",
                "*.bin",
                "*.index.json",
                "*.json",
            ],
        )
    )

    indexed = _read_index_weight_files(repo_dir)
    if indexed is not None:
        total = sum(p.stat().st_size for p in indexed)
        if total <= 0:
            raise RuntimeError(f"Index-based estimate found 0 bytes for {model_name}.")
        return total, [p.name for p in indexed]

    candidates = _iter_weight_candidates(repo_dir)
    if not candidates:
        raise RuntimeError(f"Could not estimate weights size for {model_name} (no *.safetensors/*.bin found).")

    chosen = _prefer_sharded(candidates)

    # If both shards and a consolidated file exist, _prefer_sharded will select shards if they match pattern.
    # But if pattern doesn't match (some repos), we still want to avoid obvious double-count:
    # If there are multiple files and one is literally "consolidated.safetensors", drop it unless it's the only file.
    if len(chosen) > 1:
        consolidated = [p for p in chosen if p.name == "consolidated.safetensors"]
        if consolidated:
            chosen = [p for p in chosen if p.name != "consolidated.safetensors"]

    total = sum(p.stat().st_size for p in chosen)
    if total <= 0:
        raise RuntimeError(f"Could not estimate weights size for {model_name} (selected files sum to 0 bytes).")
    return total, [p.name for p in chosen]


def make_plan(
        model_name: str,
        context_len: int,
        *,
        device_index: int = 0,
        block_size: int = 16,
        gpu_memory_utilization: float = 0.90,
        perf_mode: PerfMode = "throughput",
        trust_remote_code: bool = False,
        prefer_fp8_kv_cache: bool = False,
        enforce_eager: bool = False,
        reserve_gib: float = 2.5,          # hard VRAM margin
        safety_margin: float = 0.92,       # extra slack multiplier
        local_files_only: bool = False,
        cache: bool = True,
) -> Plan:
    if context_len <= 0:
        raise ValueError("context_len must be positive")
    if not (0.1 <= gpu_memory_utilization <= 0.99):
        raise ValueError("gpu_memory_utilization must be within [0.1, 0.99]")
    if not (0.5 <= safety_margin <= 0.99):
        raise ValueError("safety_margin must be within [0.5, 0.99]")

    gpu = probe_gpu(device_index=device_index)
    model = probe_model(model_name, trust_remote_code=trust_remote_code)
    dtype = _pick_weight_dtype(gpu)

    fp8_ignored_reason: str | None = None
    if prefer_fp8_kv_cache and not gpu.supports_fp8:
        fp8_ignored_reason = (
            f"prefer_fp8_kv_cache=True, but GPU compute capability {gpu.capability} "
            f"({gpu.name}) does not indicate FP8 support; using kv_cache_dtype='auto'."
        )
        logger.warning(fp8_ignored_reason)

    kv_cache_dtype = "fp8" if (prefer_fp8_kv_cache and gpu.supports_fp8) else "auto"

    # KV sizing for reporting + conservative planning
    sizing = kv_sizing(
        model=model,
        seq_len=context_len,
        block_size=block_size,
        kv_cache_dtype=kv_cache_dtype,
        fallback_kv_dtype_if_auto=dtype,
    )

    weights_bytes, weight_files = _estimate_weights_bytes(
        model_name=model_name,
        local_files_only=local_files_only,
    )

    total_bytes = int(gpu.total_memory_bytes)
    budget_bytes = int(total_bytes * gpu_memory_utilization)

    # torch.compile/cudagraph overhead (static heuristic)
    compile_overhead = int(0.5 * GiB) if enforce_eager else int(1.5 * GiB)
    reserve_bytes = int(reserve_gib * GiB)

    kv_budget = budget_bytes - weights_bytes - compile_overhead - reserve_bytes
    kv_budget = int(kv_budget * safety_margin)
    kv_budget = max(0, kv_budget)

    # allocator-friendly: 256 MiB chunks
    kv_cache_memory_bytes = _align_down(kv_budget, 256 * 1024 * 1024)

    if kv_cache_memory_bytes < sizing.kv_bytes_per_seq:
        max_tokens_est = int(kv_cache_memory_bytes // max(1, sizing.kv_bytes_per_token))
        raise ValueError(
            f"Static KV budget too small for context_len={context_len}.\n"
            f"GPU={gpu.name} VRAM={total_bytes/GiB:.2f} GiB util={gpu_memory_utilization}\n"
            f"weights≈{weights_bytes/GiB:.2f} GiB reserve={reserve_gib:.2f} GiB "
            f"compile≈{compile_overhead/GiB:.2f} GiB safety={safety_margin}\n"
            f"kv_cache≈{kv_cache_memory_bytes/GiB:.2f} GiB, estimated max_tokens≈{max_tokens_est}\n"
            f"Fix: reduce context_len or increase gpu_memory_utilization (or reduce reserve/overhead)."
        )

    max_num_seqs_by_kv = max(1, kv_cache_memory_bytes // max(1, sizing.kv_bytes_per_seq))
    cap = 4 if perf_mode == "latency" else 128
    max_num_seqs = min(cap, int(max_num_seqs_by_kv))

    max_num_batched_tokens = min(32768, max(8192, 16 * min(context_len, 2048)))

    plan_payload = {
        "model_name": model_name,
        "context_len": context_len,
        "block_size": block_size,
        "gpu": {"name": gpu.name, "total_memory_bytes": total_bytes, "cap": gpu.capability},
        "dtype": dtype,
        "kv_cache_dtype": kv_cache_dtype,
        "prefer_fp8_kv_cache": prefer_fp8_kv_cache,
        "enforce_eager": enforce_eager,
        "util": gpu_memory_utilization,
        "reserve_gib": reserve_gib,
        "safety_margin": safety_margin,
        "local_files_only": local_files_only,
        "mistral_mode": _is_mistral_repo(model_name),
    }
    key = make_cache_key(plan_payload)

    if cache:
        cached = load_cached_plan(key)
        if cached is not None:
            return Plan(
                vllm_kwargs=cached["vllm_kwargs"],
                gpu=cached["gpu"],
                model=cached["model"],
                notes=cached["notes"],
                cache_key=key,
            )

    vllm_kwargs: dict[str, Any] = {
        "model": model_name,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,

        "disable_log_stats": True,
        "max_model_len": int(context_len),

        "block_size": int(block_size),
        "enable_prefix_caching": True,

        "max_num_seqs": int(max_num_seqs),
        "max_num_batched_tokens": int(max_num_batched_tokens),

        # Manual KV size
        "kv_cache_memory_bytes": int(kv_cache_memory_bytes),
        "kv_cache_dtype": kv_cache_dtype,

        "enforce_eager": bool(enforce_eager),

        # IMPORTANT: keep tokenizer init ON (your pipeline needs tokenizer behavior consistently)
        "skip_tokenizer_init": False,
        "seed": 0,
    }

    # Mistral compatibility inferred from repo name
    if _is_mistral_repo(model_name):
        vllm_kwargs.update(
            {
                "tokenizer_mode": "mistral",
                "load_format": "mistral",
                "config_format": "mistral",
            }
        )

    notes = {
        "weights_gib_est": weights_bytes / GiB,
        "weight_files_used_for_estimate": weight_files,
        "compile_overhead_gib": compile_overhead / GiB,
        "kv_cache_gib": kv_cache_memory_bytes / GiB,
        "kv_bytes_per_token_est": sizing.kv_bytes_per_token,
        "kv_bytes_per_seq_est": sizing.kv_bytes_per_seq,
        "max_num_seqs_by_kv": int(max_num_seqs_by_kv),
        "fp8_requested": bool(prefer_fp8_kv_cache),
        "fp8_enabled": bool(prefer_fp8_kv_cache and gpu.supports_fp8),
        "fp8_ignored_reason": fp8_ignored_reason,
        "mistral_mode": _is_mistral_repo(model_name),
    }

    plan = Plan(
        vllm_kwargs=vllm_kwargs,
        gpu=asdict(gpu),
        model=asdict(model),
        notes=notes,
        cache_key=key,
    )

    if cache:
        save_cached_plan(
            key,
            {
                "vllm_kwargs": plan.vllm_kwargs,
                "gpu": plan.gpu,
                "model": plan.model,
                "notes": plan.notes,
            },
        )

    return plan
