"""
Depth-scaling probe for depo: train on a MIX of hop counts 1..N and track
per-hop answer accuracy over training. One run = one model (scratch or transfer).

HF Trainer accepts a dict of eval datasets, so we pass one fixed eval split per
hop k=1..N and get `eval_hop{k}_answer_acc` logged at every eval step. The full
per-hop learning curve is dumped to <out_json> for later plotting.

    python plotting/depo_depth_test.py --tag scratch --max_steps 100000
    python plotting/depo_depth_test.py --tag dyckcc --pretrained_path <ckpt>.pth --max_steps 100000
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402

import fire  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from torch.utils.data import IterableDataset  # noqa: E402
from transformers import GPT2Config, Trainer, TrainingArguments, set_seed  # noqa: E402

from downstream.synthetic_playground.depo.depo import (  # noqa: E402
    DIFFICULTY, build_vocab, make_example,
)
from downstream.synthetic_playground.common import (  # noqa: E402
    AnswerCollator, build_model, compute_metrics, preprocess_logits_for_metrics,
)


class DepoStream(IterableDataset):
    """Infinite stream; hops=None -> each query samples a hop uniformly in 1..num_hops."""

    def __init__(self, seed, params, vocab, hops=None):
        self.seed, self.params, self.vocab, self.hops = seed, params, vocab, hops

    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            yield make_example(rng, self.vocab, hops=self.hops, **self.params)


def _split(rng, n, params, vocab, hops):
    return Dataset.from_list(
        [make_example(rng, vocab, hops=hops, **params) for _ in range(n)])


def main(tag: str, max_hops: int = 5, num_entities: int = 8, num_queries: int = None,
         mini_vocab: int = None, ood_hops: int = 0,
         max_steps: int = 100000, bsz: int = 128, lr: float = 5e-4,
         eval_steps: int = 2000, n_eval: int = 500, seed: int = 0, data_seed: int = 1234,
         n_layer: int = 4, n_head: int = 4, n_embd: int = 512,
         pretrained_path: str = None, transfer: str = "attn,ffn,ln",
         shuffle_weights: bool = False, shuffle_seed: int = 42,
         report_to: str = "none", wandb_project: str = "depo-depth-v2-1kwarmup-again", wandb_name: str = None,
         out_json: str = None):
    set_seed(seed)
    # Start from the 'hard' preset, widen the chain to support max_hops dereferences.
    params = dict(DIFFICULTY["hard"])
    params["num_hops"] = max_hops
    params["num_entities"] = num_entities
    if mini_vocab is not None:
        params["mini_vocab"] = mini_vocab   # override preset when num_entities > 20
    # Default: query every entity once (balanced hop coverage). Pass num_queries to
    # match the standard depo presets (4).
    params["num_queries"] = num_queries if num_queries is not None else num_entities
    # Build vocab to cover all hops including OOD depths so eval tokens are valid.
    vocab = build_vocab(params["mini_vocab"], params["num_hops"] + ood_hops)

    # ID eval sets (hops 1..max_hops) + OOD eval sets (hops max_hops+1..max_hops+ood_hops).
    # OOD sets use a params copy with num_hops extended so make_example can build deeper chains.
    all_eval_hops = list(range(1, max_hops + 1 + ood_hops))
    ood_params = dict(params, num_hops=max_hops + ood_hops) if ood_hops > 0 else params
    eval_sets = {}
    for k in all_eval_hops:
        p = ood_params if k > max_hops else params
        eval_sets[f"hop{k}"] = _split(random.Random(data_seed + 100 + k), n_eval, p, vocab, hops=k)

    probe = _split(random.Random(data_seed + 2), 2000, params, vocab, hops=None)
    max_len = max(max(len(x) for x in probe["input_ids"]),
                  max(len(x) for ds in eval_sets.values() for x in ds["input_ids"]))
    n_positions = 1 << (max_len - 1).bit_length()

    train_ds = DepoStream(data_seed, params, vocab, hops=None)   # mixed 1..max_hops only
    config = GPT2Config(vocab_size=vocab["vocab_size"], n_positions=n_positions,
                        n_embd=n_embd, n_layer=n_layer, n_head=n_head)
    model, init = build_model(config, pretrained_path, transfer,
                              shuffle_weights=shuffle_weights, shuffle_seed=shuffle_seed)
    print(f"depo-depth[{tag}] init={init} max_hops={max_hops} num_entities={num_entities} "
          f"vocab={vocab['vocab_size']} n_positions={n_positions} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    if report_to == "wandb":
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_NAME"] = wandb_name or tag

    args = TrainingArguments(
        output_dir=f"output/depo_depth/{tag}",
        per_device_train_batch_size=bsz, per_device_eval_batch_size=bsz,
        max_steps=max_steps, learning_rate=lr, warmup_steps=1000, lr_scheduler_type="cosine",
        logging_steps=500, eval_strategy="steps", eval_steps=eval_steps, eval_on_start=True,
        save_strategy="no", report_to=report_to,
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=0, remove_unused_columns=False, seed=seed,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_sets,
        data_collator=AnswerCollator(), compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics)
    trainer.train()

    # Extract per-hop learning curves for both ID (1..max_hops) and OOD (max_hops+1..) hops.
    series = {f"hop{k}": [] for k in all_eval_hops}
    for rec in trainer.state.log_history:
        step = rec.get("step")
        if step is None:
            continue
        for k in all_eval_hops:
            key = f"eval_hop{k}_answer_acc"
            if key in rec:
                series[f"hop{k}"].append([step, rec[key]])
    final = {k: (series[k][-1][1] if series[k] else None) for k in series}
    id_hops  = [f"hop{k}" for k in range(1, max_hops + 1)]
    ood_keys = [f"hop{k}" for k in range(max_hops + 1, max_hops + 1 + ood_hops)]
    print(f"\nFINAL depo-depth[{tag}] ID  per-hop acc: " +
          "  ".join(f"{k}={final[k]:.1f}%" for k in id_hops if final[k] is not None))
    if ood_keys:
        print(f"FINAL depo-depth[{tag}] OOD per-hop acc: " +
              "  ".join(f"{k}={final[k]:.1f}%" for k in ood_keys if final[k] is not None))

    out_json = out_json or f"downstream/synthetic_playground/results/depo_depth_{tag}.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"tag": tag, "max_hops": max_hops, "ood_hops": ood_hops,
                   "num_entities": num_entities, "max_steps": max_steps,
                   "init": init, "pretrained_path": pretrained_path,
                   "series": series, "final": final}, f, indent=2)
    print(f"[json] wrote {out_json}")


if __name__ == "__main__":
    fire.Fire(main)
