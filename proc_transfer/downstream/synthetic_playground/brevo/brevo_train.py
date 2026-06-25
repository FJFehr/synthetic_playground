"""Train a GPT-2 (4×4×512) on brevo (topological sort).

    python -m downstream.synthetic_playground.brevo.brevo_train --difficulty easy
    python -m downstream.synthetic_playground.brevo.brevo_train --difficulty easy \
        --pretrained_path <ckpt>.pth --transfer attn,ffn,ln
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os  # noqa: E402
import random  # noqa: E402

import fire  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from torch.utils.data import IterableDataset  # noqa: E402
from transformers import (  # noqa: E402
    GPT2Config, Trainer, TrainerCallback, TrainingArguments, set_seed,
)

from downstream.synthetic_playground.brevo.brevo import (  # noqa: E402
    DIFFICULTY, build_vocab, make_example, validate,
)
from downstream.synthetic_playground.common import (  # noqa: E402
    AnswerCollator, append_csv_row, build_model, compute_metrics,
    preprocess_logits_for_metrics,
)


class BrevoStream(IterableDataset):
    def __init__(self, seed, params, vocab, max_seq_len):
        self.seed, self.params, self.vocab, self.max_seq_len = seed, params, vocab, max_seq_len

    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            ex = make_example(rng, self.vocab, **self.params)
            if len(ex["input_ids"]) <= self.max_seq_len:   # never exceed n_positions
                yield ex


class ValidityCallback(TrainerCallback):
    """Free-generate topo orders and report validity %. on_train_end sweeps all DAG
    sizes and writes an ID/OOD summary row to the results CSV."""

    def __init__(self, examples, vocab, n_positions, params, train_max,
                 csv_path=None, meta=None, n_eval=200, max_new=80):
        self.examples = examples[:n_eval]
        self.vocab, self.n_positions, self.max_new, self.params = vocab, n_positions, max_new, params
        self.train_max = train_max  # ID: n in [3, train_max]; OOD: (train_max, N]
        self.csv_path, self.meta = csv_path, (meta or {})

    def _gen_valid(self, model, ex):
        ANS, EOS = self.vocab["ans"], self.vocab["eos"]
        gen = list(ex["input_ids"][: ex["input_ids"].index(ANS) + 1])
        budget = min(self.max_new, self.n_positions - len(gen))
        for _ in range(max(budget, 0)):
            x = torch.tensor([gen], device=model.device)
            with torch.no_grad():
                nxt = model(input_ids=x).logits[0, -1].argmax().item()
            gen.append(nxt)
            if nxt == EOS:
                break
        return validate(gen, self.vocab)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        valid = sum(self._gen_valid(model, ex) for ex in self.examples)
        pct = 100.0 * valid / len(self.examples)
        print(f"--- brevo topo-validity (n=N) @ step {state.global_step}: {pct:.1f}%  "
              f"({valid}/{len(self.examples)}) ---")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        N = self.params["num_nodes"]
        print("--- brevo validity vs DAG size (final, 100 ex/size) ---")
        curve = {}
        for nn in range(3, N + 1):
            exs = [make_example(random.Random(990000 + nn * 1000 + i), self.vocab,
                                fixed_n=nn, **self.params) for i in range(100)]
            curve[nn] = sum(self._gen_valid(model, ex) for ex in exs)
            print(f"  n={nn:2d}: {curve[nn]}%")
        idv = [curve[n] for n in range(3, self.train_max + 1)]
        oodv = [curve[n] for n in range(self.train_max + 1, N + 1)]
        avg_id = sum(idv) / len(idv) if idv else 0.0
        avg_ood = (sum(oodv) / len(oodv)) if oodv else 0.0
        nN = curve.get(self.train_max, 0)
        print(f"BREVO_SUMMARY avg_ID={avg_id:.1f} n{self.train_max}={nN} avg_OOD={avg_ood:.1f}")
        if self.csv_path:
            m = self.meta
            curve_str = ";".join(f"{n}:{curve[n]}" for n in sorted(curve))
            append_csv_row(self.csv_path, [
                m.get("run", ""), m.get("source", "scratch"), m.get("transfer", "-"),
                m.get("seed", ""), m.get("data_seed", ""), m.get("difficulty", ""),
                self.train_max, N, f"{avg_id:.1f}", nN, f"{avg_ood:.1f}", curve_str])


def _build_split(rng, n, params, vocab, fixed_n=None):
    return Dataset.from_list(
        [make_example(rng, vocab, fixed_n=fixed_n, **params) for _ in range(n)])


def main(difficulty: str = "easy", n_eval: int = 500, n_embd: int = 512,
         n_layer: int = 4, n_head: int = 4, max_steps: int = 5000, bsz: int = 128,
         lr: float = 5e-4, eval_steps: int = 250, seed: int = 0, data_seed: int = 1234,
         pretrained_path: str = None, transfer: str = "attn,ffn,ln",
         shuffle_weights: bool = False, shuffle_seed: int = 42,
         vocab_n: int = None, train_max_n: int = None, results_csv: str = None,
         report_to: str = "none", wandb_project: str = "brevo", wandb_name: str = None,
         model_name: str = "scratch"):
    set_seed(seed)
    if report_to == "wandb":
        import wandb
        os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_name:
            os.environ["WANDB_NAME"] = wandb_name
        wandb.init(config={"task": "brevo", "model": model_name, "seed": seed})

    params = dict(DIFFICULTY[difficulty])
    if vocab_n:                         # widen vocab/label space (reserve sizes for OOD eval)
        params["num_nodes"] = vocab_n
    N = params["num_nodes"]             # vocab/label space + size-sweep upper bound
    train_max = train_max_n or N        # cap on DAG size seen during training
    train_params = {**params, "max_n": train_max}
    vocab = build_vocab(N)

    eval_ds = _build_split(random.Random(data_seed + 1), n_eval, params, vocab, fixed_n=train_max)
    probe = _build_split(random.Random(data_seed + 2), 2000, params, vocab, fixed_n=N)
    max_len = max(max(len(x) for x in probe["input_ids"]),
                  max(len(x) for x in eval_ds["input_ids"]))
    # Generous margin so streamed tail + free-generation never overrun the positions.
    n_positions = 1 << (max_len + 32 - 1).bit_length()
    train_ds = BrevoStream(data_seed, train_params, vocab, n_positions)
    eval_ds = eval_ds.filter(lambda x: len(x["input_ids"]) <= n_positions)

    config = GPT2Config(vocab_size=vocab["vocab_size"], n_positions=n_positions,
                        n_embd=n_embd, n_layer=n_layer, n_head=n_head)
    model, init = build_model(config, pretrained_path, transfer,
                              shuffle_weights=shuffle_weights, shuffle_seed=shuffle_seed)
    print(f"brevo[{difficulty}] init={init} vocab={vocab['vocab_size']} N={N} "
          f"train_max_n={train_max} n_positions={n_positions} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    args = TrainingArguments(
        output_dir=f"output/brevo/{difficulty}",
        per_device_train_batch_size=bsz, per_device_eval_batch_size=16,
        max_steps=max_steps, learning_rate=lr, warmup_steps=50, lr_scheduler_type="cosine",
        logging_steps=100, eval_strategy="steps", eval_steps=eval_steps, eval_on_start=True,
        save_strategy="no", report_to=report_to,
        bf16=torch.cuda.is_bf16_supported(), dataloader_drop_last=True,
        dataloader_num_workers=0,   # single worker keeps streamed data deterministic
        remove_unused_columns=False, seed=seed,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=AnswerCollator(), compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics)
    meta = {
        "run": wandb_name or "run",
        "source": pretrained_path if pretrained_path else "scratch",
        "transfer": (transfer if pretrained_path else "-"),
        "seed": seed, "data_seed": data_seed, "difficulty": difficulty,
    }
    trainer.add_callback(ValidityCallback(
        list(eval_ds), vocab, n_positions, params, train_max,
        csv_path=results_csv, meta=meta))
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
