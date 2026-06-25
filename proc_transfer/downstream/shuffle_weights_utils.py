import torch
from torch.nn import Module


def shuffle_attention_weights(model: Module, seed: int = 42) -> None:
    """Per-component shuffle of attention weights only (c_attn, c_proj).

    Flattens each tensor, randomly permutes its elements, then reshapes back.
    Preserves the per-tensor mean/std but destroys learned structure — useful
    as a control to check whether transfer benefit requires structured weights.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    attn_patterns = ["attn.c_attn", "attn.c_proj"]
    for name, param in model.named_parameters():
        if any(p in name for p in attn_patterns):
            flat = param.data.view(-1)
            idx = torch.randperm(flat.numel(), generator=rng)
            param.data = flat[idx].view(param.data.shape)


def shuffle_all_non_embedding_weights(model: Module, seed: int = 42) -> None:
    """Per-component shuffle of all non-embedding weights.

    Same as shuffle_attention_weights but applied to every parameter except
    token/position embeddings and the lm_head. Mutually exclusive with
    shuffle_attention_weights in practice.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    skip_patterns = ["wte", "wpe", "lm_head"]
    for name, param in model.named_parameters():
        if any(p in name for p in skip_patterns):
            continue
        flat = param.data.view(-1)
        idx = torch.randperm(flat.numel(), generator=rng)
        param.data = flat[idx].view(param.data.shape)
