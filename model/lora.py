"""
Self-implemented LoRA (Low-Rank Adaptation).

Drop-in replacement for PEFT. Wraps target nn.Linear modules with
low-rank adapters (A, B matrices) while keeping base weights frozen.

    output = W @ x + (alpha / rank) * B @ A @ x

Usage:
    from model.lora import apply_lora, get_lora_state_dict, load_lora_state_dict

    # Apply LoRA to specific modules
    n_lora = apply_lora(model, target_suffixes=["attn.qkv", "mlp.gate_proj"],
                        rank=64, alpha=128.0, dropout=0.05)

    # Get only LoRA parameters (for saving)
    lora_sd = get_lora_state_dict(model)

    # Load LoRA weights
    load_lora_state_dict(model, lora_sd)
"""

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """nn.Linear wrapper with low-rank adaptation.

    Freezes the original weight and adds trainable A, B matrices:
        y = base_linear(x) + scaling * B(A(dropout(x)))

    A is initialized with Kaiming uniform, B with zeros,
    so the LoRA contribution starts at zero (no perturbation at init).
    """

    def __init__(self, base_linear, rank, alpha=None, dropout=0.05):
        super().__init__()
        assert isinstance(base_linear, nn.Linear), f"Expected nn.Linear, got {type(base_linear)}"

        d_out, d_in = base_linear.weight.shape
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank * 2)
        self.scaling = self.alpha / self.rank

        # Freeze the original linear
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # Low-rank adapters (trainable)
        self.lora_A = nn.Linear(d_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)

        # Dropout on input before LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Init: A = kaiming, B = zeros -> LoRA output starts at 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return base_out + lora_out * self.scaling

    def extra_repr(self):
        d_out, d_in = self.base.weight.shape
        return (f"in={d_in}, out={d_out}, rank={self.rank}, "
                f"alpha={self.alpha}, scaling={self.scaling:.4f}")


# =========================================================================
# Apply / Remove LoRA
# =========================================================================

def _set_module_by_name(model, name, new_module):
    """Replace a submodule by its dotted name path."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_lora(model, target_suffixes, rank, alpha=None, dropout=0.05,
               rank_overrides=None):
    """Apply LoRA adapters to all nn.Linear modules matching target suffixes.

    Args:
        model: nn.Module to modify in-place
        target_suffixes: list of str suffixes to match against module names
            e.g. ["attn.qkv", "attn.proj", "mlp.linear_fc1"]
        rank: default LoRA rank
        alpha: LoRA alpha (default: rank * 2)
        dropout: LoRA dropout rate
        rank_overrides: optional dict {suffix: rank} to override rank
            for specific module types

    Returns:
        n_lora_params: total number of LoRA parameters added
    """
    alpha = alpha if alpha is not None else float(rank * 2)
    rank_overrides = rank_overrides or {}
    replaced = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # Check if this module name ends with any target suffix
        matched_suffix = None
        for suffix in target_suffixes:
            if name.endswith(suffix):
                matched_suffix = suffix
                break
        if matched_suffix is None:
            continue

        # Determine rank (use override if specified)
        r = rank_overrides.get(matched_suffix, rank)
        a = float(r * 2) if matched_suffix in rank_overrides else alpha

        # Create LoRA wrapper and move to same device/dtype as original
        lora_module = LoRALinear(module, rank=r, alpha=a, dropout=dropout)
        lora_module = lora_module.to(
            device=module.weight.device,
            dtype=module.weight.dtype,
        )

        # Replace in model
        _set_module_by_name(model, name, lora_module)
        replaced.append((name, r))

    n_params = sum(
        p.numel() for n, m in model.named_modules()
        if isinstance(m, LoRALinear)
        for p in [m.lora_A.weight, m.lora_B.weight]
    )

    return n_params, replaced


# =========================================================================
# State Dict Utilities
# =========================================================================

def get_lora_state_dict(model):
    """Extract only LoRA adapter weights from model state_dict.

    Returns a dict with keys like 'model.visual.blocks.0.attn.qkv.lora_A.weight'.
    """
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = name
            sd[f"{prefix}.lora_A.weight"] = module.lora_A.weight.data.cpu()
            sd[f"{prefix}.lora_B.weight"] = module.lora_B.weight.data.cpu()
    return sd


def load_lora_state_dict(model, state_dict, strict=True):
    """Load LoRA adapter weights into model.

    Args:
        model: model with LoRALinear modules already applied
        state_dict: dict from get_lora_state_dict()
        strict: if True, raise error if keys don't match
    """
    model_lora_keys = set()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            model_lora_keys.add(f"{name}.lora_A.weight")
            model_lora_keys.add(f"{name}.lora_B.weight")

    if strict:
        missing = model_lora_keys - set(state_dict.keys())
        unexpected = set(state_dict.keys()) - model_lora_keys
        if missing:
            raise KeyError(f"Missing LoRA keys: {missing}")
        if unexpected:
            raise KeyError(f"Unexpected LoRA keys: {unexpected}")

    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A.weight"
            b_key = f"{name}.lora_B.weight"
            if a_key in state_dict:
                module.lora_A.weight.data.copy_(state_dict[a_key])
                loaded += 1
            if b_key in state_dict:
                module.lora_B.weight.data.copy_(state_dict[b_key])
                loaded += 1
    return loaded


def count_lora_params(model, group_by=None):
    """Count LoRA parameters, optionally grouped by name pattern.

    Args:
        model: model with LoRALinear modules
        group_by: optional dict {group_name: substring_to_match}
            e.g. {"vision": "visual", "backbone": "language_model"}

    Returns:
        total_params, groups_dict (if group_by) or just total_params
    """
    if group_by is None:
        total = sum(
            m.lora_A.weight.numel() + m.lora_B.weight.numel()
            for _, m in model.named_modules()
            if isinstance(m, LoRALinear)
        )
        return total

    groups = {name: 0 for name in group_by}
    for mod_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        n = module.lora_A.weight.numel() + module.lora_B.weight.numel()
        for group_name, pattern in group_by.items():
            if pattern in mod_name:
                groups[group_name] += n
                break

    return sum(groups.values()), groups


def print_lora_summary(model):
    """Print a summary of all LoRA modules in the model."""
    total = 0
    total_base = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            n_lora = module.lora_A.weight.numel() + module.lora_B.weight.numel()
            n_base = module.base.weight.numel()
            total += n_lora
            total_base += n_base

    all_params = sum(p.numel() for p in model.parameters())
    print(f"  LoRA params:    {total:>12,} ({total/1e6:.2f}M)")
    print(f"  Base params:    {total_base:>12,} ({total_base/1e6:.2f}M)")
    print(f"  Total params:   {all_params:>12,} ({all_params/1e6:.2f}M)")
    print(f"  Trainable%:     {total/all_params*100:.2f}%")
