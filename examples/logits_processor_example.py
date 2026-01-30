"""
Example demonstrating logits processor usage with AutoVLLMClient.

This example shows how to:
1. Pass a custom logits processor class to AutoVLLMClient
2. Configure the processor via environment variables
3. Use the processor during inference
"""
import os
from vllm_autoconfig import AutoVLLMClient, SamplingConfig
from vllm_autoconfig.logits_processors import ExampleLogitsProcessor


def main():
    # Configure the example processor via environment variables
    os.environ["EXAMPLE_PROCESSOR_DEBUG"] = "1"
    os.environ["EXAMPLE_PROCESSOR_ENABLED"] = "1"

    print("=" * 80)
    print("Logits Processor Example")
    print("=" * 80)

    # Initialize client with logits processor
    # The processor class (not instance) is passed to AutoVLLMClient
    # vLLM will instantiate it automatically with proper configuration
    client = AutoVLLMClient(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        context_len=512,
        logits_processors=[ExampleLogitsProcessor],  # Pass the class, not an instance
        debug=True,
    )

    print("\n" + "=" * 80)
    print("Running Inference with Logits Processor")
    print("=" * 80 + "\n")

    # Prepare prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about programming.",
    ]

    # Run inference
    results = client.run_batch_raw(
        prompts,
        SamplingConfig(max_tokens=50, temperature=0.7)
    )

    # Display results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    for i, (prompt, result) in enumerate(zip(prompts, results), 1):
        print(f"\n{i}. Prompt: {prompt}")
        print(f"   Output: {result['output']}")

    print("\n" + "=" * 80)
    client.close()
    print("Done!")


if __name__ == "__main__":
    main()

