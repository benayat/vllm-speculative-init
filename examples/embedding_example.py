#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Using AutoVLLMEmbedding for efficient embedding generation with mean pooling

This example demonstrates how to use the AutoVLLMEmbedding client to generate embeddings
with automatic GPU configuration, including multi-GPU tensor parallelism support.

Features demonstrated:
- Automatic GPU memory configuration
- Mean pooling (default)
- L2 normalization
- Multi-GPU tensor parallelism (automatic when needed)
- Cosine similarity calculation

Usage:
    python embedding_example.py --model meta-llama/Llama-3.1-8B-Instruct
    python embedding_example.py --model nvidia/Llama-3.3-70B-Instruct-FP8 --max-model-len 256
"""

import argparse
import numpy as np
from typing import List

# Import the autoconfig embedding client
from vllm_autoconfig import AutoVLLMEmbedding, probe_all_gpus, GpuInfo


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two sets of L2-normalized vectors.
    
    Args:
        A: (N, D) normalized embeddings
        B: (M, D) normalized embeddings
    
    Returns:
        (N, M) matrix of cosine similarities
    """
    return A @ B.T


def demonstrate_embeddings(
    model: str,
    max_model_len: int = 512,
    auto_tensor_parallel: bool = True,
):
    """
    Demonstrate embedding generation with the AutoVLLMEmbedding client.
    """
    print(f"\n{'='*80}")
    print(f"AutoVLLMEmbedding Demo")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Max sequence length: {max_model_len}")
    print(f"Auto tensor parallelism: {auto_tensor_parallel}")
    
    # Show available GPUs
    print(f"\n{'='*80}")
    print(f"GPU Detection")
    print(f"{'='*80}")
    try:
        gpus = probe_all_gpus()
        print(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  {i}: {gpu.name} - {gpu.total_memory_bytes / (1024**3):.1f} GiB")
    except Exception as e:
        print(f"Could not probe GPUs: {e}")
    
    # Initialize the embedding client
    print(f"\n{'='*80}")
    print(f"Initializing AutoVLLMEmbedding...")
    print(f"{'='*80}")
    
    client = AutoVLLMEmbedding(
        model_name=model,
        max_model_len=max_model_len,
        pooling_type="MEAN",  # Mean pooling
        normalize=False,  # We'll normalize manually
        auto_tensor_parallel=auto_tensor_parallel,
        trust_remote_code=True,
        debug=True,
    )
    
    # Show configuration
    print(f"\n{'='*80}")
    print(f"Configuration Details")
    print(f"{'='*80}")
    print(f"Plan notes:")
    for key, value in client.plan.notes.items():
        if value is not None:
            print(f"  {key}: {value}")
    
    # Example texts to embed
    field_names = [
        "Artificial intelligence",
        "Archeology",
        "Literature",
        "Electrical engineering"
    ]
    
    positive_prompts = [
        "The leading academic discipline",
        "The finest field of study",
        "The top scholarly domain"
    ]
    
    negative_prompts = [
        "The most disappointing academic discipline",
        "The least worthwhile area of study",
        "The weakest scholarly field"
    ]
    
    # Generate embeddings
    print(f"\n{'='*80}")
    print(f"Generating Embeddings")
    print(f"{'='*80}")
    
    print(f"\nEmbedding {len(field_names)} field names...")
    field_embeddings = client.embed(field_names, normalize=True)
    print(f"Field embeddings shape: {field_embeddings.shape}")
    
    print(f"\nEmbedding {len(positive_prompts)} positive prompts...")
    positive_embeddings = client.embed(positive_prompts, normalize=True)
    print(f"Positive embeddings shape: {positive_embeddings.shape}")
    
    print(f"\nEmbedding {len(negative_prompts)} negative prompts...")
    negative_embeddings = client.embed(negative_prompts, normalize=True)
    print(f"Negative embeddings shape: {negative_embeddings.shape}")
    
    # Compute similarities
    print(f"\n{'='*80}")
    print(f"Computing Similarities")
    print(f"{'='*80}")
    
    # Positive prompts vs field names
    pos_similarities = cosine_similarity_matrix(positive_embeddings, field_embeddings)
    
    print(f"\nPositive prompts vs field names:")
    print(f"{'Prompt':<45} {'Top Field':<25} {'Similarity':>10}")
    print(f"{'-'*80}")
    for i, prompt in enumerate(positive_prompts):
        best_field_idx = np.argmax(pos_similarities[i])
        best_similarity = pos_similarities[i][best_field_idx]
        print(f"{prompt[:44]:<45} {field_names[best_field_idx]:<25} {best_similarity:>10.4f}")
    
    # Negative prompts vs field names
    neg_similarities = cosine_similarity_matrix(negative_embeddings, field_embeddings)
    
    print(f"\nNegative prompts vs field names:")
    print(f"{'Prompt':<45} {'Top Field':<25} {'Similarity':>10}")
    print(f"{'-'*80}")
    for i, prompt in enumerate(negative_prompts):
        best_field_idx = np.argmax(neg_similarities[i])
        best_similarity = neg_similarities[i][best_field_idx]
        print(f"{prompt[:44]:<45} {field_names[best_field_idx]:<25} {best_similarity:>10.4f}")
    
    # Mean similarities per field
    print(f"\n{'='*80}")
    print(f"Mean Similarities Across All Prompts")
    print(f"{'='*80}")
    
    all_prompts = positive_prompts + negative_prompts
    all_embeddings = np.vstack([positive_embeddings, negative_embeddings])
    all_similarities = cosine_similarity_matrix(all_embeddings, field_embeddings)
    
    mean_sims = all_similarities.mean(axis=0)
    
    print(f"\n{'Field':<30} {'Mean Similarity':>15}")
    print(f"{'-'*45}")
    field_ranking = sorted(zip(field_names, mean_sims), key=lambda x: x[1], reverse=True)
    for rank, (field, sim) in enumerate(field_ranking, 1):
        print(f"{rank}. {field:<27} {sim:>15.4f}")
    
    # Cleanup
    print(f"\n{'='*80}")
    print(f"Cleaning up...")
    print(f"{'='*80}")
    client.close()
    
    print(f"\n✓ Demo complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate AutoVLLMEmbedding with mean pooling and multi-GPU support"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for embeddings (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--no-auto-tensor-parallel",
        action="store_true",
        help="Disable automatic multi-GPU tensor parallelism"
    )
    
    args = parser.parse_args()
    
    demonstrate_embeddings(
        model=args.model,
        max_model_len=args.max_model_len,
        auto_tensor_parallel=not args.no_auto_tensor_parallel,
    )


if __name__ == "__main__":
    main()

