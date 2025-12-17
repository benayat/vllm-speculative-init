# vllm-autoconfig

**Automatic configuration planner for vLLM** - Eliminate the guesswork of configuring vLLM by automatically determining optimal parameters based on your GPU hardware and model requirements.

## 🚀 Features

- **Zero-configuration vLLM setup**: Automatically calculates optimal `max_model_len`, `gpu_memory_utilization`, and other vLLM parameters
- **Hardware-aware planning**: Probes GPU memory and capabilities using PyTorch to ensure configurations fit your hardware
- **Model-specific optimizations**: Applies model-family-specific settings (Mistral, Llama, Qwen, etc.)
- **KV cache sizing**: Intelligently calculates memory requirements for attention key-value caches
- **Configuration caching**: Saves computed plans to avoid redundant calculations
- **Performance modes**: Choose between `throughput` and `latency` optimization strategies
- **FP8 KV cache support**: Automatically enables FP8 quantization for KV caches when beneficial
- **Simple API**: Just specify your model name and desired context length - everything else is handled automatically

## 📦 Installation

```bash
pip install vllm-autoconfig
```

**Requirements:**
- Python >= 3.10
- PyTorch with CUDA support
- vLLM
- Access to CUDA-capable GPU(s)

## 🎯 Quick Start

### Python API

```python
from vllm_autoconfig import AutoVLLMClient, SamplingConfig

# Initialize with your model and desired context length
client = AutoVLLMClient(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    context_len=1024,  # The ONLY parameter you need to set!
)

# Prepare your prompts
prompts = [
    {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "metadata": {"id": 1},
    }
]

# Run inference
results = client.run_batch(
    prompts, 
    SamplingConfig(max_tokens=100, temperature=0.7)
)

print(results)
client.close()
```

### Advanced Usage

```python
from vllm_autoconfig import AutoVLLMClient, SamplingConfig

# Fine-tune the configuration
client = AutoVLLMClient(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    context_len=2048,
    perf_mode="latency",          # or "throughput" (default)
    prefer_fp8_kv_cache=True,     # Enable FP8 KV cache if supported
    trust_remote_code=False,       # For models requiring custom code
    debug=True,                    # Enable detailed logging
)

# Check the computed plan
print(f"Plan cache key: {client.plan.cache_key}")
print(f"vLLM kwargs: {client.plan.vllm_kwargs}")
print(f"Notes: {client.plan.notes}")

# Run inference with custom sampling
sampling = SamplingConfig(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
    stop=["###", "\n\n"]
)

results = client.run_batch(prompts, sampling)
client.close()
```

### Embedding Generation

```python
from vllm_autoconfig import AutoVLLMEmbedding
import numpy as np

# Initialize embedding client - only specify model and max length!
# GPU settings are automatically configured
client = AutoVLLMEmbedding(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_model_len=512,  # Typical for embeddings
)

# Generate embeddings for texts
texts = [
    "The cat sits on the mat",
    "A feline rests on the carpet",
    "Dogs are playing in the park",
]

embeddings = client.embed(texts, normalize=True)
print(f"Embeddings shape: {embeddings.shape}")  # (3, embedding_dim)

# Compute cosine similarity (embeddings are already normalized)
def cosine_similarity(a, b):
    return np.dot(a, b)

sim = cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity between 'cat' and 'feline': {sim:.4f}")

client.close()
```

## 🛠️ How It Works

1. **GPU Probing**: Detects available GPU memory and capabilities (BF16 support, compute capability)
2. **Model Analysis**: Downloads model configuration from HuggingFace Hub and analyzes architecture
3. **Weight Calculation**: Computes actual model weight size from checkpoint files
4. **Memory Planning**: Calculates KV cache memory requirements based on context length and batch size
5. **Configuration Generation**: Produces optimal vLLM initialization parameters within hardware constraints
6. **Caching**: Saves the computed plan for reuse with the same configuration

## 📊 Configuration Parameters

The `AutoVLLMClient` automatically configures:

- `model`: Model name/path
- `max_model_len`: Maximum sequence length
- `gpu_memory_utilization`: GPU memory usage fraction
- `dtype`: Weight precision (bfloat16 or float16)
- `kv_cache_dtype`: KV cache precision (including FP8 when beneficial)
- `enforce_eager`: Whether to use eager mode (affects compilation)
- `trust_remote_code`: Whether to trust remote code execution
- Model-specific parameters (e.g., `tokenizer_mode`, `load_format` for Mistral)

## 🎛️ API Reference

### `AutoVLLMClient`

```python
AutoVLLMClient(
    model_name: str,              # HuggingFace model name or local path
    context_len: int,             # Desired context length
    device_index: int = 0,        # GPU device index
    perf_mode: str = "throughput", # "throughput" or "latency"
    trust_remote_code: bool = False,
    prefer_fp8_kv_cache: bool = False,
    enforce_eager: bool = False,
    local_files_only: bool = False,
    cache_plan: bool = True,      # Cache computed plans
    debug: bool = False,          # Enable debug logging
    vllm_logging_level: str = None, # vLLM logging level
)
```

### `SamplingConfig`

```python
SamplingConfig(
    temperature: float = 0.0,     # Sampling temperature
    top_p: float = 1.0,           # Nucleus sampling threshold
    max_tokens: int = 32,         # Maximum tokens to generate
    stop: List[str] = None,       # Stop sequences
)
```

#### Methods

- `run_batch(prompts, sampling, output_field="output")`: Run inference on a batch of prompts
- `close()`: Clean up resources and free GPU memory

### `AutoVLLMEmbedding`

```python
AutoVLLMEmbedding(
    model_name: str,              # HuggingFace model name or local path
    max_model_len: int = 512,     # Maximum sequence length (embeddings typically need less)
    pooling_type: str = "MEAN",   # "MEAN", "CLS", or "LAST"
    normalize: bool = False,      # Normalize in vLLM (default: False, normalized manually)
    device_index: int = 0,        # GPU device index
    perf_mode: str = "throughput", # "throughput" or "latency"
    trust_remote_code: bool = False,
    enforce_eager: bool = True,   # Better compatibility for embeddings
    local_files_only: bool = False,
    cache_plan: bool = True,      # Cache computed plans
    debug: bool = False,          # Enable debug logging
    vllm_logging_level: str = None,
)
```

**Note:** GPU memory utilization, tensor parallelism, dtype, and other hardware-specific settings are **automatically configured** by the planner based on your GPU and model. You don't need to specify them!

#### Methods

- `embed(texts, normalize=True)`: Generate embeddings for a list of texts, returns numpy array (N, D)
- `embed_batch(texts, normalize=True)`: Alias for `embed()`
- `close()`: Clean up resources and free GPU memory

## 🏗️ Project Structure

```
vllm-autoconfig/
├── src/vllm_autoconfig/
│   ├── __init__.py          # Package exports
│   ├── client.py            # AutoVLLMClient implementation (text generation)
│   ├── embedding.py         # AutoVLLMEmbedding implementation (embeddings)
│   ├── planner.py           # Configuration planning logic
│   ├── gpu_probe.py         # GPU detection and probing
│   ├── model_probe.py       # Model analysis utilities
│   ├── kv_math.py           # KV cache memory calculations
│   └── cache.py             # Plan caching utilities
├── examples/
│   ├── simple_run.py        # Text generation example
│   ├── embedding_simple.py  # Basic embedding example
│   └── embedding_similarity.py  # Advanced embedding similarity analysis
└── pyproject.toml
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [vLLM](https://github.com/vllm-project/vllm) - the high-performance LLM inference engine
- Uses [HuggingFace Transformers](https://github.com/huggingface/transformers) for model configuration

## 📚 Citation

If you use vllm-autoconfig in your research or production systems, please cite:

```bibtex
@software{vllm_speculative_autoconfig,
  title = {vllm-autoconfig: Automatic Configuration Planning for vLLM},
  author = {Benaya Trabelsi},
  year = {2025},
  url = {https://github.com:benayat/vllm-speculative-init}
}
```

## 🐛 Issues and Support

For issues, questions, or feature requests, please open an issue on [GitHub Issues](https://github.com/yourusername/vllm-autoconfig/issues).

## 🔗 Links

- [Documentation](https://github.com/benayat/vllm-speculative-init)
- [PyPI Package](https://pypi.org/project/vllm-autoconfig/)
- [vLLM Documentation](https://docs.vllm.ai/)

