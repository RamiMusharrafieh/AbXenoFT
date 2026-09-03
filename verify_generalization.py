#!/usr/bin/env python3
"""Masked-residue recovery on held-out V genes.

Rebuilds the same V-gene split the training run used, then scores whichever of
these models are present on germlines the V-gene model never trained on:

  - stock ESM-2                 (no antibody training at all)
  - the random-split model      (saw these genes in training, so optimistic)
  - the V-gene model            (never saw these genes, the honest number)

Models that have not been trained are skipped with a note rather than crashing,
so this is useful after training only some of them.

    python verify_generalization.py
"""

import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from train_mlm_esm2 import split_by_group  # noqa: E402

BASE = "facebook/esm2_t6_8M_UR50D"
CKPT = PROJECT / "checkpoints"
CACHE = PROJECT / "Data" / "frameworks_unique.tsv"
N_SEQS = 400
SEED = 42

MODELS = [
    ("stock ESM-2", BASE),
    ("random-split model", CKPT / "esm2_t6_8M_mouse_fwr" / "final"),
    ("V-gene early-stopped", CKPT / "esm2_t6_8M_mouse_fwr_vgene" / "final_earlystop_step60000"),
    ("V-gene full anneal", CKPT / "esm2_t6_8M_mouse_fwr_vgene" / "final"),
]


def evaluate(path, tok, seqs):
    model = AutoModelForMaskedLM.from_pretrained(str(path)).eval()
    g = torch.Generator().manual_seed(SEED)
    correct = total = 0
    nll = 0.0
    with torch.no_grad():
        for seq in seqs:
            enc = tok(seq, return_tensors="pt")
            ids = enc["input_ids"]
            n = ids.shape[1] - 2
            k = max(1, int(0.15 * n))
            pos = torch.randperm(n, generator=g)[:k] + 1
            truth = ids[0, pos].clone()
            masked = ids.clone()
            masked[0, pos] = tok.mask_token_id
            logits = model(input_ids=masked, attention_mask=enc["attention_mask"]).logits
            sel = logits[0, pos]
            correct += (sel.argmax(-1) == truth).sum().item()
            nll += torch.nn.functional.cross_entropy(sel, truth, reduction="sum").item()
            total += k
    return correct / total, nll / total


def main() -> None:
    if not CACHE.exists():
        raise SystemExit(f"{CACHE} not found. Run: python train_mlm_esm2.py --prep-only")

    raw = load_dataset("csv", data_files=str(CACHE), delimiter="\t")["train"]
    _, val = split_by_group(raw, 0.2, SEED, "v_gene")
    val = val.shuffle(seed=123).select(range(N_SEQS))
    seqs, genes = val["text"], val["v_gene"]
    print(f"scoring {N_SEQS} sequences from {len(set(genes))} held-out V genes\n")

    tok = AutoTokenizer.from_pretrained(BASE)
    print(f"{'model':24s} {'top-1 recovery':>15s} {'loss':>8s} {'perplexity':>12s}")
    print("-" * 62)
    for name, path in MODELS:
        if path != BASE and not Path(path).exists():
            print(f"{name:24s} {'not trained, skipped':>38s}")
            continue
        acc, loss = evaluate(path, tok, seqs)
        print(f"{name:24s} {acc:14.2%} {loss:8.4f} {torch.tensor(loss).exp():12.3f}")


if __name__ == "__main__":
    main()
