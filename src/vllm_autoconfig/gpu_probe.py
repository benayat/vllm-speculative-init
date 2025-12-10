from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class GpuInfo:
    device_index: int
    name: str
    total_memory_bytes: int
    capability: tuple[int, int]
    multi_processor_count: int
    is_bf16_supported: bool
    supports_fp8: bool  # heuristic via compute capability


def probe_gpu(device_index: int = 0) -> GpuInfo:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available (torch.cuda.is_available() is False).")

    props = torch.cuda.get_device_properties(device_index)
    cap = (int(props.major), int(props.minor))

    # BF16: PyTorch provides a direct probe.
    bf16 = bool(torch.cuda.is_bf16_supported())

    # FP8 hardware support heuristic:
    # - Hopper: SM90 (9.0) supports FP8
    # - Ada:    SM89 (8.9) supports FP8
    supports_fp8 = (cap[0] > 8) or (cap[0] == 8 and cap[1] >= 9)

    return GpuInfo(
        device_index=device_index,
        name=str(props.name),
        total_memory_bytes=int(props.total_memory),
        capability=cap,
        multi_processor_count=int(props.multi_processor_count),
        is_bf16_supported=bf16,
        supports_fp8=supports_fp8,
    )
