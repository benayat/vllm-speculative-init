from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

import torch

from .planner import make_plan, Plan

PerfMode = Literal["throughput", "latency"]


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 32
    n: int = 1
    stop: Optional[List[str]] = None


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


class AutoVLLMClient:
    def __init__(
            self,
            model_name: str,
            context_len: int,
            *,
            device_index: int = 0,
            perf_mode: PerfMode = "throughput",
            trust_remote_code: bool = False,
            prefer_fp8_kv_cache: bool = False,
            enforce_eager: bool = False,
            local_files_only: bool = False,
            cache_plan: bool = True,
            debug: bool = False,
            vllm_logging_level: Optional[str] = None,
    ):
        self.model_name = model_name
        self.context_len = int(context_len)
        self.debug = bool(debug)

        _configure_python_logging(self.debug)
        log = logging.getLogger(__name__)

        # Must be set BEFORE importing vllm anywhere
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        if vllm_logging_level is None:
            vllm_logging_level = "DEBUG" if self.debug else "INFO"
        os.environ.setdefault("VLLM_LOGGING_LEVEL", vllm_logging_level.upper())

        # Import vLLM after env vars
        from vllm import LLM, SamplingParams, TokensPrompt  # noqa: WPS433

        self._LLM = LLM
        self._SamplingParams = SamplingParams
        self._TokensPrompt = TokensPrompt

        self.plan: Plan = make_plan(
            model_name=model_name,
            context_len=self.context_len,
            device_index=device_index,
            perf_mode=perf_mode,
            trust_remote_code=trust_remote_code,
            prefer_fp8_kv_cache=prefer_fp8_kv_cache,
            enforce_eager=enforce_eager,
            local_files_only=local_files_only,
            cache=cache_plan,
        )

        # Debug: let vLLM print more internal stats
        if self.debug:
            self.plan.vllm_kwargs["disable_log_stats"] = False

        # We want vLLM to pick/instantiate the tokenizer for the model.
        # If skip_tokenizer_init=True, vLLM may not initialize tokenizer internals
        # (and you'll see EOS-related warnings / missing IDs).
        self.plan.vllm_kwargs["skip_tokenizer_init"] = False

        # Model-specific knobs (Mistral)
        self._apply_model_specific_overrides(self.plan.vllm_kwargs)

        log.info("AutoVLLMClient plan cache_key=%s", self.plan.cache_key)
        log.info("Plan notes: %s", self.plan.notes)
        if self.plan.notes.get("fp8_ignored_reason"):
            log.warning(self.plan.notes["fp8_ignored_reason"])

        self.llm = self._LLM(**self.plan.vllm_kwargs)

        # Grab the tokenizer that vLLM actually uses (could be a wrapper).
        self.tokenizer = self._get_vllm_tokenizer(self.llm)
        if self.debug:
            log.debug("vLLM tokenizer type=%s", type(self.tokenizer))
            log.debug("vLLM tokenizer is_fast=%s", getattr(self.tokenizer, "is_fast", None))
            log.debug("vLLM tokenizer eos_token_id=%s", getattr(self.tokenizer, "eos_token_id", None))

    @staticmethod
    def _is_mistral_model_name(model_name: str) -> bool:
        return model_name.lower().startswith("mistralai/")

    def _apply_model_specific_overrides(self, vllm_kwargs: Dict[str, Any]) -> None:
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

    @staticmethod
    def _get_vllm_tokenizer(llm: Any) -> Any:
        """
        Prefer llm.get_tokenizer() if present (common in vLLM),
        otherwise fall back to llm.tokenizer attribute.
        """
        if hasattr(llm, "get_tokenizer") and callable(getattr(llm, "get_tokenizer")):
            return llm.get_tokenizer()
        if hasattr(llm, "tokenizer"):
            return llm.tokenizer
        raise RuntimeError("Could not obtain tokenizer from vLLM LLM instance (no get_tokenizer() / .tokenizer).")

    # -----------------------------
    # Chat templating + tokenization
    # -----------------------------

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Keep your design: chat_template -> str, then tokenize separately.
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _encode_text(self, text: str) -> List[int]:
        """
        Encode *one* prompt string into token IDs using the vLLM-selected tokenizer.

        Mistral wrapper note:
        - Its __call__ has a known EOS-fix hack that only reliably triggers for single-string calls.
        - Using .encode(...) is also safe and avoids batch corner-cases.
        """
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        ids = enc["input_ids"]
        return list(ids)

    def _batch_tokenize_messages(self, prompts: List[Dict[str, Any]]) -> List[Any]:
        texts = [self._messages_to_text(p["messages"]) for p in prompts]
        try:
            # import inspect
            # sig = inspect.signature(self.tokenizer)
            # print("Tokenizer __call__ signature:", sig)
            enc = self.tokenizer(
                texts,
                add_special_tokens=False,
                truncation=False,
            )
            input_ids = enc["input_ids"]
            return [self._TokensPrompt(prompt_token_ids=ids) for ids in input_ids]
        except TypeError:
            # Wrapper doesn't like batched call signature -> encode one-by-one
            return [self._TokensPrompt(prompt_token_ids=self._encode_text(t)) for t in texts]

    def generate_batch(
            self,
            prompts: List[Dict[str, Any]],
            sampling: SamplingConfig,
    ):
        stop_token_ids = None
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            stop_token_ids = [int(eos_id)]

        params = self._SamplingParams(
            temperature=float(sampling.temperature),
            top_p=float(sampling.top_p),
            max_tokens=int(sampling.max_tokens),
            n=int(sampling.n),
            stop=sampling.stop,
            stop_token_ids=stop_token_ids,
        )

        tokenized = self._batch_tokenize_messages(prompts)
        return self.llm.generate(
            prompts=tokenized,
            sampling_params=params,
            use_tqdm=self.debug,
        )

    # -----------------------------
    # Public API
    # -----------------------------

    def run_batch(
            self,
            prompts: List[Dict[str, Any]],
            sampling: SamplingConfig,
            output_field: str = "output",
    ) -> List[Dict[str, Any]]:
        outputs = self.generate_batch(prompts, sampling)
        return [
            {
                **prompt.get("metadata", {}),
                output_field: out.outputs[0].text.strip() if sampling.n == 1 else [o.text.strip() for o in out.outputs]
            }
            for prompt, out in zip(prompts, outputs)
        ]

    def close(self) -> None:
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
