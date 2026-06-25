# synthetic_playground

Transfer learning probes for compositional reasoning, using GPT-2 on two synthetic tasks: **BREVO** (topological sort) and **DEPO** (multi-hop dereference). Given a pretrained checkpoint, these scripts fine-tune the model on a downstream task and measure in-distribution and out-of-distribution generalisation.

---

## Repo Structure

```
synthetic_playground/
├── pytorch_model_1_step2500.pth      # pretrained checkpoint — 2 500 training steps
├── pytorch_model_1_step10000.pth     # pretrained checkpoint — 10 000 training steps
└── proc_transfer/
    ├── run_brevo.sh                  # BREVO launcher (run from here)
    ├── run_depo.sh                   # DEPO launcher (run from here)
    ├── downstream/
    │   ├── utils.py                  # weight-transfer logic
    │   └── synthetic_playground/
    │       ├── common.py             # shared training utilities
    │       ├── brevo/                # BREVO data generation + training harness
    │       └── depo/                 # DEPO data generation
    └── plotting/
        └── depo_depth_test.py        # DEPO training harness with per-hop eval
```

---

## The Two Tasks

### BREVO — Topological Sort (tests *breadth* of reasoning)

Given shuffled edges of a DAG and a query node, produce a valid topological ordering of all ancestors. Evaluates whether the model can track transitive dependencies.

- **ID eval**: DAG sizes n ∈ [3, 30] (seen during training)
- **OOD eval**: DAG sizes n ∈ [31, 40] (never seen)
- **Output**: `results/brevo/brevo_results.csv` — one row per run with ID/OOD validity %

### DEPO — Multi-hop Dereference (tests *depth* of reasoning)

Entities form a directed cycle. Given shuffled adjacency facts and a k-hop query ("X is k steps ahead of?"), predict the answer. Evaluates pointer-chasing chains.

- **ID eval**: 1–4 hops
- **OOD eval**: 5+ hops (controlled by `--ood_hops`)
- **Output**: `results/depo/depo_<TAG>.json` — per-hop learning curves

---

## Checkpoints

Two pretrained checkpoints ship with the repo (excluded from git — see [Cluster Setup](#cluster-setup)):

| File | Steps | Notes |
|------|-------|-------|
| `pytorch_model_1_step2500.pth` | 2 500 | early checkpoint |
| `pytorch_model_1_step10000.pth` | 10 000 | later checkpoint |

Model architecture: GPT-2, 4 layers, 4 heads, d=512.

---

## Running Evaluations

> **GPU required.** These scripts are too slow on CPU for any practical use.

All scripts must be run from inside `proc_transfer/`:

```bash
cd proc_transfer
```

### BREVO

```bash
# Random-init baseline
bash run_brevo.sh scratch

# Transfer from a pretrained checkpoint
bash run_brevo.sh sort ../pytorch_model_1_step10000.pth
bash run_brevo.sh sort ../pytorch_model_1_step2500.pth
```

Training: 10 000 steps, eval every 1 000 steps, batch size 128.
Results appended to `results/brevo/brevo_results.csv`.

### DEPO

```bash
# Random-init baseline
bash run_depo.sh scratch

# Transfer from a pretrained checkpoint
bash run_depo.sh stack ../pytorch_model_1_step10000.pth
bash run_depo.sh stack ../pytorch_model_1_step2500.pth
```

Training: 100 000 steps, eval every 2 000 steps, batch size 128.
Results written to `results/depo/depo_<TAG>.json`.

---

## Transfer Options

Both scripts default to `--transfer attn,ffn,ln`. You can customise this in the scripts:

| Component | What gets transferred |
|-----------|----------------------|
| `attn` | Attention weights (c_attn, c_proj) |
| `ffn` | Feed-forward / MLP layers |
| `ln` | Layer normalisation (ln_1, ln_2, ln_f) |
| `embed` | Token & position embeddings |
| `everything` | All weights except embeddings |

---

## ID vs OOD

Both tasks train on small/shallow instances and evaluate on harder held-out instances:

- **ID** (in-distribution): sizes/depths seen during fine-tuning
- **OOD** (out-of-distribution): larger DAGs (BREVO n > 30) or deeper chains (DEPO hops > 4)

The gap between ID and OOD accuracy is the main quantity of interest.

---

## Cluster Setup

```bash
# 1. Clone
git clone git@github.com:fabiofehr96/synthetic_playground.git
cd synthetic_playground

# 2. Install dependencies (uv)
uv sync

# 3. Transfer checkpoints from your local machine
rsync -avz pytorch_model_1_step*.pth user@cluster:/path/to/synthetic_playground/

# 4. Run (from cluster, inside proc_transfer/)
cd proc_transfer
bash run_brevo.sh sort ../pytorch_model_1_step10000.pth
bash run_depo.sh stack ../pytorch_model_1_step10000.pth
```

---

## Dependencies

Managed with [uv](https://github.com/astral-sh/uv). See `pyproject.toml`.

Core packages: `torch`, `transformers`, `datasets`, `numpy`, `matplotlib`.
