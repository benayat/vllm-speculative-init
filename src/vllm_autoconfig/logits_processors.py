"""
Example logits processor for vLLM autoconfig.

This module provides a minimal example of a vLLM V1 LogitsProcessor
that can be used with AutoVLLMClient.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch

try:
    from vllm.v1.sample.logits_processor import LogitsProcessor
except ImportError:
    # Fallback for older vLLM versions
    try:
        from vllm.logits_processor import LogitsProcessor
    except ImportError:
        # Provide a stub if vLLM is not available
        class LogitsProcessor:
            """Stub LogitsProcessor base class."""
            pass

logger = logging.getLogger(__name__)


class ExampleLogitsProcessor(LogitsProcessor):
    """
    Example logits processor demonstrating the vLLM V1 API.
    
    This processor logs information about the logits it receives
    and can optionally modify them based on environment variables.
    
    Environment Variables:
        EXAMPLE_PROCESSOR_DEBUG: Set to "1" to enable debug logging
        EXAMPLE_PROCESSOR_ENABLED: Set to "0" to disable processing
    """
    
    def __init__(self, vllm_config: Any, device: torch.device, is_pin_memory: bool):
        """
        Initialize the logits processor.
        
        This method is called automatically by vLLM with the engine's configuration.
        
        Args:
            vllm_config: vLLM's model configuration object
            device: The device (CPU/GPU) where tensors will be processed
            is_pin_memory: Whether to use pinned memory for tensors
        """
        logger.info(f"[ExampleLogitsProcessor] Initializing on device: {device}")
        
        self.device = device
        self.is_pin_memory = is_pin_memory
        self.vllm_config = vllm_config
        
        # Read configuration from environment
        self.debug = os.getenv("EXAMPLE_PROCESSOR_DEBUG", "0") == "1"
        self.enabled = os.getenv("EXAMPLE_PROCESSOR_ENABLED", "1") == "1"
        
        if self.debug:
            logger.info(f"[ExampleLogitsProcessor] Configuration:")
            logger.info(f"  - Device: {device}")
            logger.info(f"  - Pin Memory: {is_pin_memory}")
            logger.info(f"  - Enabled: {self.enabled}")
            
        if hasattr(vllm_config, 'model_config'):
            vocab_size = vllm_config.model_config.get_vocab_size()
            logger.info(f"[ExampleLogitsProcessor] Vocab size: {vocab_size}")
    
    def is_argmax_invariant(self) -> bool:
        """
        Return whether this processor preserves the argmax of logits.
        
        If True, vLLM can apply certain optimizations.
        Return False if your processor modifies which token has the highest probability.
        """
        return True  # This example processor doesn't change token rankings
    
    def update_state(self, batch_update: Any) -> None:
        """
        Update processor state based on batch changes.

        This method is called by vLLM when the batch composition changes
        (e.g., new requests added, completed requests removed).

        Args:
            batch_update: Update information from vLLM containing batch changes.
                         Typically has attributes like 'added', 'removed', etc.
        """
        if self.debug and batch_update is not None:
            # Log batch updates for debugging
            if hasattr(batch_update, 'added') and batch_update.added:
                logger.debug(
                    f"[ExampleLogitsProcessor] Batch update: "
                    f"{len(batch_update.added)} request(s) added"
                )
            if hasattr(batch_update, 'removed') and batch_update.removed:
                logger.debug(
                    f"[ExampleLogitsProcessor] Batch update: "
                    f"{len(batch_update.removed)} request(s) removed"
                )

        # Example: Extract per-request configuration from batch_update if needed
        # In the CategoryGate example, this is used to read 'gate_enabled' flag
        # from individual request's extra_args
        if batch_update and hasattr(batch_update, 'added'):
            for entry in batch_update.added:
                # entry is typically a tuple/list: (request_id, request_data, ...)
                if len(entry) >= 2 and hasattr(entry[1], 'extra_args'):
                    extra_args = entry[1].extra_args
                    if self.debug and extra_args:
                        logger.debug(
                            f"[ExampleLogitsProcessor] Request extra_args: {extra_args}"
                        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply processing to the logits.
        
        Args:
            logits: Tensor of shape (batch_size, vocab_size) containing raw logits
            
        Returns:
            Modified logits tensor of the same shape
        """
        if not self.enabled:
            return logits
        
        if self.debug:
            batch_size, vocab_size = logits.shape
            logger.debug(
                f"[ExampleLogitsProcessor] Processing logits: "
                f"shape={logits.shape}, dtype={logits.dtype}, "
                f"device={logits.device}"
            )
            
            # Log some statistics
            max_logit = logits.max().item()
            min_logit = logits.min().item()
            logger.debug(
                f"[ExampleLogitsProcessor] Logit range: "
                f"min={min_logit:.2f}, max={max_logit:.2f}"
            )
        
        # Example: This processor doesn't modify logits, just logs them
        # In a real processor, you would apply your modifications here
        # For example:
        # - Block certain tokens by setting their logits to -inf
        # - Boost certain tokens by adding to their logits
        # - Apply custom sampling transformations
        
        return logits

