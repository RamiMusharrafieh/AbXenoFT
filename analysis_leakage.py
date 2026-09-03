"""Quantify train/validation leakage directly, for both species.

A random split over unique sequences cannot leak exact duplicates, because they
were already deduplicated. What it does leak is *near* duplicates: somatic-hypermutation
variants of the same germline differing by one or two residues, which land on
opposite sides of the split. Holding out whole V genes is meant to stop that.

This measures it rather than assuming it: for a sample of validation sequences,
find the Hamming distance to the nearest training sequence under each split
scheme. If a random split leaks, its validation sequences will sit 1-2 mutations
from something the model trained on; V-gene-held-out sequences should sit much
further away.

Also records the deduplication and QC funnel for each species.

Writes analysis/leakage_dedup.json
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "analysis"
CANON = 91
SEED = 42
N_VAL = 1500        # validation sequences probed
N_TRAIN = 150_000   # training sequences searched against
MAX_D = 20          # histogram ceiling

SPECIES = {
    "mouse": PROJECT / "Data" / "frameworks_unique.tsv",
    "macaque": PROJECT / "Data" / "frameworks_unique_macaque.tsv",
}

# raw -> QC-passed -> unique, from the extraction and prep logs
FUNNEL = {
    "mouse": {"raw": 17_933_653, "qc": 11_688_651, "unique": 2_767_822},
    "macaque": {"raw": 1_467_486, "qc": 1_419_444, "unique": 1_209_734},
}


def held_out_genes(genes, val_fraction=0.2, seed=SEED):
    counts = Counter(genes)
    groups = sorted(counts)
    random.Random(seed).shuffle(groups)
    target = val_fraction * len(genes)
    val, acc = set(), 0
    for g in groups:
        if acc >= target:
            break
        if acc + counts[g] > target * 1.15 and acc > target * 0.75:
            continue
        val.add(g)
        acc += counts[g]
    return val


def encode(seqs) -> np.ndarray:
    return np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), CANON)


def nearest_distances(val_arr: np.ndarray, train_arr: np.ndarray) -> np.ndarray:
    out = np.empty(len(val_arr), dtype=np.int16)
    for i, v in enumerate(val_arr):
        out[i] = (train_arr != v).sum(1).min()
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = {"funnel": FUNNEL, "n_val": N_VAL, "n_train": N_TRAIN, "leakage": {}}

    for sp, tsv in SPECIES.items():
        print(f"\n=== {sp} ===")
        df = pd.read_csv(tsv, sep="\t", dtype=str)
        val_genes = held_out_genes(df["v_gene"].tolist())          # same split training used
        df = df[df["text"].str.len() == CANON].reset_index(drop=True)
        seqs = df["text"].to_numpy()
        is_val_gene = df["v_gene"].isin(val_genes).to_numpy()
        print(f"  {len(df):,} canonical sequences, {is_val_gene.mean():.1%} in held-out genes")

        # random 80/20 over the same unique sequences
        perm = rng.permutation(len(df))
        n_val_rand = int(0.2 * len(df))
        rand_val_idx = perm[:n_val_rand]
        rand_train_idx = perm[n_val_rand:]

        schemes = {
            "random": (rand_val_idx, rand_train_idx),
            "v_gene": (np.flatnonzero(is_val_gene), np.flatnonzero(~is_val_gene)),
        }

        results["leakage"][sp] = {}
        for name, (vidx, tidx) in schemes.items():
            v_pick = rng.choice(vidx, size=min(N_VAL, len(vidx)), replace=False)
            t_pick = rng.choice(tidx, size=min(N_TRAIN, len(tidx)), replace=False)
            val_arr = encode(seqs[v_pick])
            train_arr = encode(seqs[t_pick])
            d = nearest_distances(val_arr, train_arr)

            hist = [int((d == k).sum()) for k in range(MAX_D + 1)]
            hist.append(int((d > MAX_D).sum()))
            entry = {
                "hist": hist,
                "median": float(np.median(d)),
                "mean": float(d.mean()),
                "le1": float((d <= 1).mean()),
                "le2": float((d <= 2).mean()),
                "le5": float((d <= 5).mean()),
            }
            results["leakage"][sp][name] = entry
            print(f"  {name:7s}: median nearest-neighbour distance {entry['median']:.0f} "
                  f"| within 1 mutation {entry['le1']:.1%} | within 2 {entry['le2']:.1%} "
                  f"| within 5 {entry['le5']:.1%}")

    path = OUT / "leakage_dedup.json"
    json.dump(results, open(path, "w"))
    print(f"\n[done] wrote {path}")


if __name__ == "__main__":
    main()
