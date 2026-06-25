"""
Topological-sort synthetic data generator ("brevo").

Adapted from the topsort task (Zeyuan Allen-Zhu, Physics of Language Models
Part 4.1) which probes reasoning *breadth*. A DAG's edges are listed (parent,
child) in random order, then a query node is given; the model must output a
valid topological order of every node reachable from the query (its ancestors),
parents-before-children.

Token layout (standalone compact vocab; ids depend on num_nodes N):
    0                BOS / PAD
    1 .. N           node labels
    N+1              EDGE   (separates / precedes each edge pair)
    N+2              QUERY  (precedes the query node)
    N+3              ANS    (precedes the answer topo-order)
    N+4              EOS    (end of answer)
    vocab_size = N + 5

Each example is {"input_ids": [...], "labels": [...]} where labels are 1 on the
answer topo tokens (incl. EOS), so models train/score on the produced order only.
Because topo orders are NOT unique, exact-match scoring under-counts; use
validate() to check a produced order for correctness.

Usage:
    python -m downstream.synthetic_playground.brevo.brevo generate \
        --out_dir downstream/synthetic_playground/data/brevo_easy \
        --difficulty easy --num_examples 2000
"""

import json
import os
import random
from collections import deque
from typing import Dict, List, Optional

import fire
from datasets import Dataset

# Difficulty presets: num_nodes is the max DAG size; per example n ~ U[3, num_nodes].
DIFFICULTY: Dict[str, Dict] = {
    "easy": dict(num_nodes=10, max_parents=4, max_children=4),
    "medium": dict(num_nodes=15, max_parents=4, max_children=4),
    "hard":      dict(num_nodes=20, max_parents=4, max_children=4),
    "very_hard": dict(num_nodes=30, max_parents=4, max_children=4),
}

_KNOBS = ("num_nodes", "max_parents", "max_children")


def build_vocab(num_nodes: int) -> Dict:
    """Return special-token ids and vocab_size for an N-node graph."""
    n = num_nodes
    return {
        "bos": 0, "pad": 0,
        "edge": n + 1, "query": n + 2, "ans": n + 3, "eos": n + 4,
        "vocab_size": n + 5,
    }


def _generate_dag(rng: random.Random, n: int, max_parents: int, max_children: int):
    """DAG over struct ids 0..n-1; node i's parents come from earlier ids (leaves on left)."""
    dag = {i: [] for i in range(n)}      # child -> [parents]
    out_degree = {i: 0 for i in range(n)}
    leaves = rng.randint(1, (n - 1) // 4 + 1)
    for i in range(leaves, n):
        candidates = [j for j in range(i) if out_degree[j] < max_children]
        if not candidates:
            continue
        num_parents = rng.randint(1, min(len(candidates), max_parents))
        for parent in rng.sample(candidates, num_parents):
            dag[i].append(parent)
            out_degree[parent] += 1
    return dag


def _reachable(dag: Dict[int, List[int]], query: int):
    """Set of nodes reachable from query by following parents (its ancestors + itself)."""
    seen, stack = set(), [query]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for p in dag.get(node, []):
            if p not in seen:
                stack.append(p)
    sub = {node: [p for p in dag.get(node, []) if p in seen] for node in seen}
    return seen, sub


def _topo_sort(sub: Dict[int, List[int]], rng: random.Random) -> List[int]:
    """Random valid topological order (parents before children)."""
    indeg = {node: len(parents) for node, parents in sub.items()}
    children = {node: [] for node in sub}
    for node, parents in sub.items():
        for p in parents:
            children[p].append(node)
    ready = [node for node, d in indeg.items() if d == 0]
    order = []
    while ready:
        node = ready.pop(rng.randint(0, len(ready) - 1))
        order.append(node)
        for ch in children[node]:
            indeg[ch] -= 1
            if indeg[ch] == 0:
                ready.append(ch)
    return order


def make_example(rng: random.Random, vocab: Dict, *, num_nodes: int,
                 max_parents: int = 4, max_children: int = 4,
                 fixed_n: Optional[int] = None, max_n: Optional[int] = None) -> Dict:
    """Build one topsort example as {"input_ids", "labels"}.

    num_nodes sizes the vocab/label space; max_n caps the random DAG size for
    training (default num_nodes). Set max_n < num_nodes to train on small graphs
    while reserving larger sizes (with in-vocab node ids) for OOD evaluation.
    """
    cap = max_n if max_n is not None else num_nodes
    for _ in range(50):  # retry until we get a query with a non-trivial subtree
        n = fixed_n if fixed_n is not None else rng.randint(3, cap)
        dag = _generate_dag(rng, n, max_parents, max_children)
        # query: a node in the last quarter that has parents (non-trivial reachable set)
        tail = range(max(n * 3 // 4, n - 1), n)
        cands = [i for i in tail if dag[i]]
        if cands:
            query = rng.choice(cands)
            break
    else:
        query = max(range(n), key=lambda i: len(dag[i]))

    _, sub = _reachable(dag, query)
    topo = _topo_sort(sub, rng)

    # Random token labels (decorrelate id from structural position).
    labels_perm = rng.sample(range(1, num_nodes + 1), n)
    lab = {i: labels_perm[i] for i in range(n)}

    edges = [(p, c) for c, ps in dag.items() for p in ps]
    rng.shuffle(edges)

    ids: List[int] = [vocab["bos"]]
    mask: List[int] = [0]
    for p, c in edges:
        ids += [vocab["edge"], lab[p], lab[c]]
        mask += [0, 0, 0]
    ids += [vocab["query"], lab[query], vocab["ans"]]
    mask += [0, 0, 0]
    for node in topo:                      # the answer: a valid topo order
        ids.append(lab[node])
        mask.append(1)
    ids.append(vocab["eos"])
    mask.append(1)

    assert len(ids) == len(mask)
    return {"input_ids": ids, "labels": mask}


def validate(input_ids: List[int], vocab: Dict) -> bool:
    """True iff the answer region is a valid topo order of the query's reachable set."""
    EDGE, Q, ANS, EOS = vocab["edge"], vocab["query"], vocab["ans"], vocab["eos"]
    try:
        qi, ai = input_ids.index(Q), input_ids.index(ANS)
    except ValueError:
        return False
    edge_toks = [t for t in input_ids[1:qi] if t != EDGE]
    if len(edge_toks) % 2:
        return False
    dag: Dict[int, List[int]] = {}
    for i in range(0, len(edge_toks), 2):
        p, c = edge_toks[i], edge_toks[i + 1]
        dag.setdefault(c, []).append(p)
        dag.setdefault(p, [])
    query = input_ids[qi + 1]
    answer = input_ids[ai + 1:]
    if answer and answer[-1] == EOS:
        answer = answer[:-1]
    # reachable set from query
    reach, _ = _reachable(dag, query)
    if set(answer) != reach:
        return False
    seen = set()
    for node in answer:                    # parents must precede children
        if any(p not in seen for p in dag.get(node, [])):
            return False
        seen.add(node)
    return True


def generate(out_dir: str, num_examples: int = 1000, difficulty: str = "easy",
             seed: int = 42, **overrides) -> None:
    """Generate a topsort dataset and save it with datasets.save_to_disk."""
    if difficulty not in DIFFICULTY:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Choose from {list(DIFFICULTY)}.")
    params = dict(DIFFICULTY[difficulty])
    for key, val in overrides.items():
        if key not in _KNOBS:
            raise ValueError(f"Unknown override '{key}'. Valid knobs: {_KNOBS}")
        params[key] = val

    rng = random.Random(seed)
    vocab = build_vocab(params["num_nodes"])
    examples = [make_example(rng, vocab, **params) for _ in range(num_examples)]
    dataset = Dataset.from_list(examples)

    os.makedirs(out_dir, exist_ok=True)
    dataset.save_to_disk(out_dir)
    seq_lens = [len(e["input_ids"]) for e in examples]
    meta = {
        "difficulty": difficulty, "seed": seed, "num_examples": num_examples,
        **params, "vocab_size": vocab["vocab_size"], "specials": vocab,
        "max_seq_len": max(seq_lens), "mean_seq_len": sum(seq_lens) / len(seq_lens),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {num_examples} examples to {out_dir}")
    print(f"  vocab_size={vocab['vocab_size']} max_seq_len={meta['max_seq_len']} "
          f"mean_seq_len={meta['mean_seq_len']:.0f}")


if __name__ == "__main__":
    fire.Fire({"generate": generate})
