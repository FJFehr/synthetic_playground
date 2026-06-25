"""Shared utilities for the synthetic-playground tasks (brevo / depo / mano).

Convention: {"input_ids", "labels"} where labels == 1 marks answer tokens; loss
and accuracy are computed on those tokens only.
"""

import torch
from transformers import GPT2LMHeadModel, TrainerCallback

from downstream.utils import initialize_model


class AnswerCollator:
    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attention, labels = [], [], []
        for f in features:
            ids, lab = f["input_ids"], f["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [0] * pad)
            attention.append([1] * len(ids) + [0] * pad)
            labels.append([(i if l == 1 else -100) for i, l in zip(ids, lab)] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention),
            "labels": torch.tensor(labels),
        }


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions[:, :-1]
    labels = labels[:, 1:]
    mask = labels != -100
    n = mask.sum()
    if n == 0:
        return {"answer_acc": 0.0}
    correct = ((preds == labels) & mask).sum()
    return {"answer_acc": 100.0 * float(correct) / float(n)}


class PeakAccCallback(TrainerCallback):
    def __init__(self):
        self.peak = 0.0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_answer_acc" in metrics:
            self.peak = max(self.peak, metrics["eval_answer_acc"])


def build_model(config, pretrained_path=None, transfer="attn,ffn,ln",
                shuffle_weights=False, shuffle_seed=42, gaussian_reinit=False):
    """Returns (model, init_description). Token embeddings are always reinitialised.

    shuffle_weights=True scrambles transferred weights per-component (preserves
    mean/std, destroys learned structure) — use as a statistics-vs-structure control.
    gaussian_reinit=True replaces each weight tensor with independent Gaussian draws
    matching the tensor's mean/std — stronger control than shuffling.
    """
    if not pretrained_path:
        return GPT2LMHeadModel(config), "scratch"
    weights = transfer.split(",") if isinstance(transfer, str) else list(transfer)
    model, _, _ = initialize_model(
        gpt2_config=config, pretrained_model_path=pretrained_path,
        weights_to_transfer=weights, weights_to_train=["everything"],
        embedding_init_strategy="average",
        shuffle_all_weights=shuffle_weights, shuffle_seed=shuffle_seed)
    tag = f"transfer[{transfer}]<-{pretrained_path.split('/')[-2]}"
    if shuffle_weights:
        return model, tag + f"+shuffled(seed{shuffle_seed})"
    if gaussian_reinit:
        from downstream.shuffle_weights_utils import gaussian_reinit_non_embedding_weights
        gaussian_reinit_non_embedding_weights(model, seed=shuffle_seed)
        return model, tag + f"+gaussian_reinit(seed{shuffle_seed})"
    return model, tag


def append_csv_row(path, row):
    with open(path, "a") as f:
        f.write(",".join(str(x) for x in row) + "\n")
    print(f"[csv] appended row to {path}")
