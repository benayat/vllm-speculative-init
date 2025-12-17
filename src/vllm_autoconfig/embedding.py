from __future__ import annotations

import os
import logging
from typing import Any, List, Optional, Literal

import torch
import numpy as np

from .planner import make_plan, Plan

PerfMode = Literal["throughput", "latency"]
PoolingType = Literal["MEAN", "CLS", "LAST"]


def _configure_python_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] %(levelname)s %(name)s:%(lineno)d: %(message)s",
        )
    else:
        root.setLevel(level)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2 normalization for numpy arrays."""
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


class AutoVLLMEmbedding:
    """
    Automatic vLLM embedding client with mean pooling support.
    
    This client automatically configures vLLM for embedding tasks, reusing
    the same GPU probing and memory planning infrastructure as AutoVLLMClient.

    Just like AutoVLLMClient, you only need to specify:
    - model_name: Which model to use
    - max_model_len: Maximum sequence length

    Everything else (GPU memory, parallelism, etc.) is automatically configured!
    """
    
    def __init__(
            self,
            model_name: str,
            max_model_len: int = 512,
            *,
            pooling_type: PoolingType = "MEAN",
            normalize: bool = False,  # We'll normalize manually for consistency
            device_index: int = 0,
            perf_mode: PerfMode = "throughput",
            trust_remote_code: bool = False,
            enforce_eager: bool = True,  # Better compatibility for embeddings
            local_files_only: bool = False,
            cache_plan: bool = True,
            debug: bool = False,
            vllm_logging_level: Optional[str] = None,
    ):
        """
        Initialize AutoVLLMEmbedding client.
        
        Args:
            model_name: HuggingFace model name/path
            max_model_len: Maximum sequence length (default: 512, suitable for embeddings)
            pooling_type: Pooling strategy - "MEAN", "CLS", or "LAST" (default: "MEAN")
            normalize: Let vLLM normalize embeddings (default: False, we normalize manually)
            device_index: GPU device index (default: 0)
            perf_mode: Performance mode - "throughput" or "latency" (default: "throughput")
            trust_remote_code: Trust remote code for model loading (default: False)
            enforce_eager: Use eager execution mode (default: True for embeddings)
            local_files_only: Only use local files, no downloads (default: False)
            cache_plan: Cache configuration plan (default: True)
            debug: Enable debug logging (default: False)
            vllm_logging_level: vLLM logging level override (default: None)
        """
        self.model_name = model_name
        self.max_model_len = int(max_model_len)
        self.pooling_type = pooling_type
        self.normalize_in_vllm = normalize
        self.debug = bool(debug)

        _configure_python_logging(self.debug)
        log = logging.getLogger(__name__)

        # Must be set BEFORE importing vllm anywhere
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        if vllm_logging_level is None:
            vllm_logging_level = "DEBUG" if self.debug else "INFO"
        os.environ.setdefault("VLLM_LOGGING_LEVEL", vllm_logging_level.upper())

        # Import vLLM after env vars
        from vllm import LLM  # noqa: WPS433
        from vllm.config import PoolerConfig  # noqa: WPS433

        self._LLM = LLM
        self._PoolerConfig = PoolerConfig

        # Use the autoconfig planner - it will automatically determine:
        # - gpu_memory_utilization
        # - tensor_parallel_size
        # - dtype
        # - max_model_len (if we need to reduce it)
        # - and other optimal settings
        self.plan: Plan = make_plan(
            model_name=model_name,
            context_len=self.max_model_len,
            device_index=device_index,
            perf_mode=perf_mode,
            trust_remote_code=trust_remote_code,
            prefer_fp8_kv_cache=False,  # Not applicable for embeddings (no KV cache)
            enforce_eager=enforce_eager,
            local_files_only=local_files_only,
            cache=cache_plan,
        )

        # Override plan settings for embedding task
        embedding_kwargs = dict(self.plan.vllm_kwargs)
        
        # Set embedding-specific parameters
        embedding_kwargs["task"] = "embed"
        embedding_kwargs["pooler_config"] = self._PoolerConfig(
            pooling_type=self.pooling_type,
            normalize=self.normalize_in_vllm,
        )

        # Debug: let vLLM print more internal stats
        if self.debug:
            embedding_kwargs["disable_log_stats"] = False

        # Model-specific overrides (e.g., Mistral)
        self._apply_model_specific_overrides(embedding_kwargs)

        log.info("AutoVLLMEmbedding initialized for model: %s", model_name)
        log.info("Configuration:")
        log.info("  - Task: embed")
        log.info("  - Pooling type: %s", self.pooling_type)
        log.info("  - Max model length: %d", embedding_kwargs.get("max_model_len", self.max_model_len))
        log.info("  - GPU memory utilization: %.2f (auto-configured)",
                 embedding_kwargs.get("gpu_memory_utilization", 0.9))
        log.info("  - Normalize in vLLM: %s", self.normalize_in_vllm)
        log.info("Plan cache_key=%s", self.plan.cache_key)
        
        if self.plan.notes:
            log.info("Plan notes: %s", self.plan.notes)

        try:
            self.llm = self._LLM(**embedding_kwargs)
            log.info("✓ Model loaded successfully with vLLM")
        except Exception as e:
            log.error("❌ Error loading model with vLLM: %s", str(e))
            log.error("Troubleshooting tips:")
            log.error("  1. Make sure vLLM is installed: pip install vllm")
            log.error("  2. Check if model is compatible with vLLM")
            log.error("  3. Try reducing max_model_len if running out of memory")
            log.error("  4. Enable debug=True for more details")
            raise

    @staticmethod
    def _is_mistral_model_name(model_name: str) -> bool:
        return model_name.lower().startswith("mistralai/")

    def _apply_model_specific_overrides(self, vllm_kwargs: dict[str, Any]) -> None:
        """
        Infer special vLLM init params from model name.
        Currently: Mistral (mistralai/*)
        """
        if self._is_mistral_model_name(self.model_name):
            # vLLM docs/examples: tokenizer_mode/load_format/config_format = "mistral"
            # Keep it non-invasive (only set if user didn't already specify).
            vllm_kwargs.setdefault("tokenizer_mode", "mistral")
            vllm_kwargs.setdefault("load_format", "mistral")
            vllm_kwargs.setdefault("config_format", "mistral")

    def embed(
            self,
            texts: List[str],
            normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed
            normalize: Apply L2 normalization to embeddings (default: True)
        
        Returns:
            numpy array of shape (N, D) where N is number of texts and D is embedding dimension
        """
        log = logging.getLogger(__name__)
        
        if not texts:
            return np.array([])
        
        log.debug("Encoding %d texts with vLLM...", len(texts))
        
        # vLLM encode returns list of outputs
        outputs = self.llm.encode(texts, pooling_task="embed")
        
        # Extract embeddings from outputs
        embeddings = []
        for output in outputs:
            # output.outputs.data is the pooled embedding vector
            vec = np.array(output.outputs.data)
            embeddings.append(vec)
        
        embeddings = np.array(embeddings)  # (N, D)
        
        # L2 normalize if requested and not already normalized by vLLM
        if normalize and not self.normalize_in_vllm:
            embeddings = l2_normalize(embeddings, axis=1)
        
        log.debug("Generated embeddings shape: %s", embeddings.shape)
        
        return embeddings

    def embed_batch(
            self,
            texts: List[str],
            normalize: bool = True,
    ) -> np.ndarray:
        """
        Alias for embed() method for consistency with naming conventions.
        
        Args:
            texts: List of text strings to embed
            normalize: Apply L2 normalization to embeddings (default: True)
        
        Returns:
            numpy array of shape (N, D) where N is number of texts and D is embedding dimension
        """
        return self.embed(texts, normalize=normalize)

    def close(self) -> None:
        """Clean up resources and free GPU memory."""
        try:
            del self.llm
        finally:
            torch.cuda.empty_cache()
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass

