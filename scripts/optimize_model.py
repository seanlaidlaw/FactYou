#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def optimize_model(input_dir: str, output_dir: str):
    """
    Convert the model to an optimized format.

    Args:
        input_dir: Path to the original model directory
        output_dir: Path to save the optimized model
    """
    print(f"Loading model from {input_dir}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load original model
    model = AutoModel.from_pretrained(input_dir)

    # Load tokenizer from base model instead of saved model
    base_model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
    print(f"Loading tokenizer from base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load threshold
    with open(os.path.join(input_dir, "threshold.json"), "r") as f:
        threshold = json.load(f)["best_threshold"]

    # Save optimized model state dict
    print("Saving optimized model...")
    torch.save(
        model.state_dict(),
        os.path.join(output_dir, "pytorch_model.bin"),
        _use_new_zipfile_serialization=True,
    )

    # Save minimal config
    config = {
        "model_type": "deberta-v2",
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "intermediate_size": model.config.intermediate_size,
        "hidden_act": model.config.hidden_act,
        "hidden_dropout_prob": model.config.hidden_dropout_prob,
        "attention_probs_dropout_prob": model.config.attention_probs_dropout_prob,
        "max_position_embeddings": model.config.max_position_embeddings,
        "type_vocab_size": model.config.type_vocab_size,
        "initializer_range": model.config.initializer_range,
        "relative_attention": model.config.relative_attention,
        "max_relative_positions": model.config.max_relative_positions,
        "position_biased_input": model.config.position_biased_input,
        "pos_att_type": model.config.pos_att_type,
        "vocab_size": model.config.vocab_size,
        "layer_norm_eps": model.config.layer_norm_eps,
        "pooler_dropout": model.config.pooler_dropout,
        "pooler_hidden_act": model.config.pooler_hidden_act,
        "extra_numeric_features": 2,
        "cls_hidden_plus_flags": model.config.hidden_size + 2,
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save threshold
    with open(os.path.join(output_dir, "threshold.json"), "w") as f:
        json.dump({"best_threshold": threshold}, f)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"Optimized model saved to {output_dir}")
    print("Original model size:", get_dir_size(input_dir) / (1024 * 1024 * 1024), "GB")
    print(
        "Optimized model size:", get_dir_size(output_dir) / (1024 * 1024 * 1024), "GB"
    )


def get_dir_size(path):
    """Get the total size of a directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


if __name__ == "__main__":
    input_dir = "sentence_frag_chkpt/best_fragment_model"
    output_dir = "sentence_frag_chkpt/optimized_model"
    optimize_model(input_dir, output_dir)
