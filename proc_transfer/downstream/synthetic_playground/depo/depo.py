"""Multi-hop pointer-dereference generator ("depo").

Entities form a directed cycle; the model reads shuffled adjacency facts and answers
k-hop queries ("given X, what is k steps ahead?"). Token layout:
    0          BOS/PAD
    1..V       word interior tokens
    V+1..2V    word end-of-word tokens
    2V+1       SEP   (separates adjacency facts)
    2V+2       ANS   (separates query entity from answer)
    2V+2+k     QUERY marker encoding hop count k  (k=1..K)
    vocab_size = 2V + K + 3

labels==1 on answer tokens only; loss and scoring ignore all context.
"""

import json
import os
import random
from typing import Dict, List, Optional, Union

import fire
from datasets import Dataset

DIFFICULTY: Dict[str, Dict] = {
    "easy": dict(num_entities=8, num_hops=1, num_queries=4,
                 mini_vocab=20, min_tlen=1, max_tlen=1, separator=True),
    "medium": dict(num_entities=8, num_hops=2, num_queries=4,
                   mini_vocab=20, min_tlen=1, max_tlen=1, separator=True),
    "hard": dict(num_entities=8, num_hops=3, num_queries=4,
                 mini_vocab=20, min_tlen=1, max_tlen=1, separator=True),
}

_KNOBS = ("num_entities", "num_hops", "num_queries",
          "mini_vocab", "min_tlen", "max_tlen", "separator")


def build_vocab(mini_vocab: int, num_hops: int) -> Dict:
    end = 2 * mini_vocab
    return {
        "bos": 0,
        "pad": 0,
        "sep": end + 1,
        "ans": end + 2,
        "query_base": end + 2,          # query(k) = query_base + k
        "vocab_size": end + 2 + num_hops + 1,
    }


def _make_words(rng: random.Random, n: int, mini_vocab: int,
                min_tlen: int, max_tlen: int) -> List[List[int]]:
    capacity = sum(mini_vocab ** L for L in range(min_tlen, max_tlen + 1))
    if capacity < n:
        raise ValueError(
            f"Cannot build {n} distinct words from mini_vocab={mini_vocab}, "
            f"lengths {min_tlen}..{max_tlen} (capacity {capacity}). Increase mini_vocab/max_tlen."
        )
    words = set()
    while len(words) < n:
        length = rng.randint(min_tlen, max_tlen)
        toks = [rng.randint(1, mini_vocab) for _ in range(length)]
        toks[-1] += mini_vocab  # end-of-word marker
        words.add(tuple(toks))
    return [list(w) for w in words]


def _pick_hop(rng: random.Random, hops: Optional[Union[int, List[int]]],
              num_hops: int) -> int:
    if hops is None:
        return rng.randint(1, num_hops)
    if isinstance(hops, int):
        return hops
    return rng.choice(hops)


def make_example(rng: random.Random, vocab: Dict, *, num_entities: int, num_hops: int,
                 num_queries: int, mini_vocab: int, min_tlen: int, max_tlen: int,
                 separator: bool, hops: Optional[Union[int, List[int]]] = None) -> Dict:
    words = _make_words(rng, num_entities, mini_vocab, min_tlen, max_tlen)
    rng.shuffle(words)  # randomize which word sits where in the cycle

    text: List[int] = [vocab["bos"]]
    labels: List[int] = [0]

    for j in rng.sample(range(num_entities), num_entities):
        if separator:
            text.append(vocab["sep"])
            labels.append(0)
        succ = words[(j + 1) % num_entities]
        for tok in words[j] + succ:
            text.append(tok)
            labels.append(0)

    for idx in rng.sample(range(num_entities), min(num_entities, num_queries)):
        k = _pick_hop(rng, hops, num_hops)
        if not 1 <= k <= num_hops:
            raise ValueError(f"hop {k} outside 1..num_hops={num_hops}")
        ans_word = words[(idx + k) % num_entities]
        query = [vocab["query_base"] + k] + words[idx] + [vocab["ans"]] + ans_word
        text += query
        labels += [0] * (1 + len(words[idx]) + 1) + [1] * len(ans_word)

    assert len(text) == len(labels)
    return {"input_ids": text, "labels": labels}


def generate(out_dir: str, num_examples: int = 1000, difficulty: str = "easy",
             seed: int = 42, hops: Optional[Union[int, List[int]]] = None,
             **overrides) -> None:
    """Generate examples and save to disk. Override any knob via **overrides."""
    if difficulty not in DIFFICULTY:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Choose from {list(DIFFICULTY)}.")
    params = dict(DIFFICULTY[difficulty])
    for key, val in overrides.items():
        if key not in _KNOBS:
            raise ValueError(f"Unknown override '{key}'. Valid knobs: {_KNOBS}")
        params[key] = val

    rng = random.Random(seed)
    vocab = build_vocab(params["mini_vocab"], params["num_hops"])

    examples = [make_example(rng, vocab, hops=hops, **params) for _ in range(num_examples)]
    dataset = Dataset.from_list(examples)

    os.makedirs(out_dir, exist_ok=True)
    dataset.save_to_disk(out_dir)

    seq_lens = [len(e["input_ids"]) for e in examples]
    meta = {
        "difficulty": difficulty, "seed": seed, "num_examples": num_examples,
        "hops": hops, **params,
        "vocab_size": vocab["vocab_size"], "specials": vocab,
        "max_seq_len": max(seq_lens), "mean_seq_len": sum(seq_lens) / len(seq_lens),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {num_examples} examples to {out_dir}")
    print(f"  vocab_size={vocab['vocab_size']} max_seq_len={meta['max_seq_len']} "
          f"mean_seq_len={meta['mean_seq_len']:.0f}")
    print(f"  knobs: {params} hops={hops}")


if __name__ == "__main__":
    fire.Fire({"generate": generate})
