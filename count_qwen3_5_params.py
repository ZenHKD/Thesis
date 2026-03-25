"""
Count parameters for Qwen3.5-0.8B model components:
- Vision Encoder
- Text Encoder/Decoder (Language Model)
- Other components (embeddings, projections, etc.)

Output: prints to console AND writes to Qwen3.5-0.8B.txt
"""

import os
import torch
from transformers import AutoModelForImageTextToText, AutoConfig


def count_parameters(module):
    """Count total and trainable parameters for a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def format_params(num_params):
    """Format parameter count in human-readable form."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.4f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.4f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.4f}K"
    return str(num_params)


# Global file handle for dual output
_output_file = None

def log(msg=""):
    """Print to console and write to output file."""
    print(msg)
    if _output_file:
        _output_file.write(msg + "\n")

def main():
    global _output_file
    output_path = "Qwen3.5-0.8B.txt"
    _output_file = open(output_path, "w")

    model_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "qwen3.5-0.8b")

    log(f"Loading model: {model_name}")
    log("=" * 70)

    # Load config first to check structure
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    log(f"\nModel type: {config.model_type}")
    log(f"Architectures: {config.architectures}")

    # Load model (CPU)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    log(f"\n{'=' * 70}")
    log(f"MODEL ARCHITECTURE OVERVIEW")
    log(f"{'=' * 70}")

    # Print top-level modules
    log("\nTop-level modules:")
    for name, child in model.named_children():
        total, trainable = count_parameters(child)
        log(f"  {name}: {format_params(total)} total, {format_params(trainable)} trainable")

    log(f"\n{'=' * 70}")
    log(f"DETAILED PARAMETER BREAKDOWN")
    log(f"{'=' * 70}")

    # Categorize all parameters
    vision_params = 0
    text_params = 0
    embedding_params = 0
    other_params = 0

    for name, param in model.named_parameters():
        num = param.numel()

        if "visual" in name:
            vision_params += num
        elif "embed_tokens" in name:
            embedding_params += num
        elif "language_model" in name and "embed_tokens" not in name:
            text_params += num
        else:
            other_params += num

    total_all = sum(p.numel() for p in model.parameters())

    log(f"\n{'Component':<35} {'Parameters':>15} {'Formatted':>12} {'% of Total':>10}")
    log("-" * 75)
    log(f"{'Vision Encoder':<35} {vision_params:>15,} {format_params(vision_params):>12} {100*vision_params/total_all:>9.2f}%")
    log(f"{'Token Embeddings':<35} {embedding_params:>15,} {format_params(embedding_params):>12} {100*embedding_params/total_all:>9.2f}%")
    log(f"{'Text Decoder (LM Layers + Norm)':<35} {text_params:>15,} {format_params(text_params):>12} {100*text_params/total_all:>9.2f}%")
    log(f"{'Other':<35} {other_params:>15,} {format_params(other_params):>12} {100*other_params/total_all:>9.2f}%")
    log("-" * 75)
    log(f"{'TOTAL':<35} {total_all:>15,} {format_params(total_all):>12} {'100.00%':>10}")

    # Vision encoder breakdown
    log(f"\n{'=' * 70}")
    log(f"VISION ENCODER - DETAILED BREAKDOWN")
    log(f"{'=' * 70}")

    for name, child in model.named_modules():
        if ("visual" in name or "vision" in name) and name.count('.') <= 2:
            total, _ = count_parameters(child)
            if total > 0:
                log(f"  {name}: {format_params(total)}")

    # Print all named parameter groups for reference
    log(f"\n{'=' * 70}")
    log(f"ALL PARAMETER NAMES (for reference)")
    log(f"{'=' * 70}")
    for name, param in model.named_parameters():
        log(f"  {name}: {param.shape} -> {format_params(param.numel())}")

    _output_file.close()
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
