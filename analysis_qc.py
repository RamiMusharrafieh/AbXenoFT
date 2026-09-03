"""Post-training QC for the V-gene-held-out ESM-2 framework model.

Produces four diagnostics, all scored on germlines the model never trained on:

  1. Consensus baseline  - a per-position most-frequent-residue lookup table built
     from the TRAINING germlines only. This is the control: framework positions are
     highly conserved, so if the model cannot beat a lookup table its recovery score
     means little.
  2. Stratified error    - accuracy by framework region, by position, by V family,
     and by how conserved each position is.
  3. Confusion matrix    - 20x20 true vs predicted residue, to see whether errors are
     conservative substitutions or random.
  4. Latent projection   - PCA of mean-pooled embeddings, stock vs fine-tuned,
     coloured by V family, with held-out genes marked.

Framework regions are fixed-length in this dataset (25/17/38/11 = 91), so positions
align exactly and regions can be sliced from the concatenated sequence.

Writes analysis/qc_results_<species>.json
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "analysis"
BASE_MODEL = "facebook/esm2_t6_8M_UR50D"

SPECIES = {
    "mouse": {
        "tsv": PROJECT / "Data" / "frameworks_unique.tsv",
        "model": PROJECT / "checkpoints/esm2_t6_8M_mouse_fwr_vgene/final",
    },
    "macaque": {
        "tsv": PROJECT / "Data" / "frameworks_unique_macaque.tsv",
        "model": PROJECT / "checkpoints/esm2_t6_8M_macaque_fwr_vgene/final",
    },
}

CANON = 91
REGIONS = [("FR1", 0, 25), ("FR2", 25, 42), ("FR3", 42, 80), ("FR4", 80, 91)]
FAMILY_RE = re.compile(r"(IGHV\d+)")
AAS = list("ACDEFGHIKLMNPQRSTVWY")

N_EVAL = 4000          # validation sequences scored for accuracy / confusion
N_EMBED = 1200         # sequences embedded for the latent projection
MASK_FRAC = 0.15
SEED = 42
BATCH = 128

device = "mps" if torch.backends.mps.is_available() else "cpu"


def held_out_genes(genes: list[str], val_fraction=0.2, seed=SEED) -> set[str]:
    """Reproduce the training split exactly (order-independent given the same rows)."""
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


def region_of(pos: int) -> str:
    for name, a, b in REGIONS:
        if a <= pos < b:
            return name
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", choices=sorted(SPECIES), default="mouse")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    TSV = SPECIES[args.species]["tsv"]
    TUNED_MODEL = str(SPECIES[args.species]["model"])
    out_path = args.out or (OUT / f"qc_results_{args.species}.json")
    if not Path(TUNED_MODEL).exists():
        raise SystemExit(f"model not found: {TUNED_MODEL}")
    print(f"[qc] species={args.species}  model={TUNED_MODEL}")
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("[load] reading cached sequences ...")
    df = pd.read_csv(TSV, sep="\t", dtype=str)
    # The split must be derived from every row the training run saw, before any
    # length filtering, otherwise the greedy assignment sees different group
    # sizes and holds out a different gene set than the model actually trained on.
    val_genes = held_out_genes(df["v_gene"].tolist())
    df = df[df["text"].str.len() == CANON]
    is_val = df["v_gene"].isin(val_genes).to_numpy()
    print(f"[load] {len(df):,} canonical-length sequences "
          f"({100 * is_val.mean():.1f}% held out, {len(val_genes)} genes)")

    train_seqs = df["text"].to_numpy()[~is_val]
    val_seqs = df["text"].to_numpy()[is_val]
    val_genes_col = df["v_gene"].to_numpy()[is_val]

    # ---------- 1. consensus table from TRAINING germlines only ----------
    print("[consensus] building per-position table from training germlines ...")
    arr = np.frombuffer("".join(train_seqs).encode(), dtype="S1").reshape(-1, CANON)
    consensus, conservation = [], []
    for p in range(CANON):
        col = Counter(arr[:, p].tobytes().decode())
        top, n = col.most_common(1)[0]
        consensus.append(top)
        conservation.append(n / len(train_seqs))
    print(f"[consensus] mean positional conservation: {np.mean(conservation):.3f}")

    # ---------- evaluation sample ----------
    idx = rng.choice(len(val_seqs), size=min(N_EVAL, len(val_seqs)), replace=False)
    seqs = val_seqs[idx]
    seq_genes = val_genes_col[idx]
    n_mask = max(1, int(MASK_FRAC * CANON))
    mask_pos = np.stack([rng.permutation(CANON)[:n_mask] for _ in range(len(seqs))])
    print(f"[eval] {len(seqs):,} sequences x {n_mask} masked positions "
          f"= {len(seqs) * n_mask:,} predictions")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def predict(model_path):
        model = AutoModelForMaskedLM.from_pretrained(model_path).eval().to(device)
        preds = np.empty((len(seqs), n_mask), dtype=object)
        with torch.no_grad():
            for s in range(0, len(seqs), BATCH):
                chunk = list(seqs[s:s + BATCH])
                enc = tok(chunk, return_tensors="pt")
                ids = enc["input_ids"].clone()
                mp = torch.tensor(mask_pos[s:s + BATCH]) + 1   # +1 for <cls>
                rows = torch.arange(ids.shape[0]).unsqueeze(1)
                ids[rows, mp] = tok.mask_token_id
                logits = model(input_ids=ids.to(device),
                               attention_mask=enc["attention_mask"].to(device)).logits
                got = logits[rows, mp].argmax(-1).cpu()
                for i in range(got.shape[0]):
                    preds[s + i] = [tok.convert_ids_to_tokens(int(t)) for t in got[i]]
        del model
        return preds

    print("[predict] fine-tuned model ...")
    pred_tuned = predict(TUNED_MODEL)
    print("[predict] stock ESM-2 ...")
    pred_base = predict(BASE_MODEL)

    truth = np.array([[s[p] for p in mp] for s, mp in zip(seqs, mask_pos)], dtype=object)
    pred_cons = np.array([[consensus[p] for p in mp] for mp in mask_pos], dtype=object)

    def acc(pred):
        return float((pred == truth).mean())

    results = {
        "n_sequences": int(len(seqs)),
        "n_predictions": int(truth.size),
        "n_val_genes": len(val_genes),
        "overall": {
            "consensus": acc(pred_cons),
            "stock": acc(pred_base),
            "tuned": acc(pred_tuned),
        },
        "mean_conservation": float(np.mean(conservation)),
    }
    print(f"\n  consensus lookup : {results['overall']['consensus']:.2%}")
    print(f"  stock ESM-2      : {results['overall']['stock']:.2%}")
    print(f"  fine-tuned       : {results['overall']['tuned']:.2%}\n")

    # ---------- 2. stratified error ----------
    flat_pos = mask_pos.ravel()
    ok_t = (pred_tuned == truth).ravel()
    ok_c = (pred_cons == truth).ravel()

    by_region = {}
    for name, a, b in REGIONS:
        m = (flat_pos >= a) & (flat_pos < b)
        by_region[name] = {"tuned": float(ok_t[m].mean()), "consensus": float(ok_c[m].mean()),
                           "n": int(m.sum())}
    results["by_region"] = by_region

    by_pos = []
    for p in range(CANON):
        m = flat_pos == p
        if m.sum() == 0:
            continue
        by_pos.append({"pos": p, "region": region_of(p), "conservation": conservation[p],
                       "tuned": float(ok_t[m].mean()), "consensus": float(ok_c[m].mean()),
                       "n": int(m.sum())})
    results["by_position"] = by_pos

    fam = np.array([FAMILY_RE.match(g).group(1) for g in seq_genes])
    fam_flat = np.repeat(fam, n_mask)
    by_family = []
    for f in sorted(set(fam)):
        m = fam_flat == f
        if m.sum() < 200:
            continue
        by_family.append({"family": f, "tuned": float(ok_t[m].mean()),
                          "consensus": float(ok_c[m].mean()), "n": int(m.sum()),
                          "n_seqs": int((fam == f).sum())})
    by_family.sort(key=lambda r: -r["n"])
    results["by_family"] = by_family

    # accuracy bucketed by how conserved the position is
    cons_flat = np.array([conservation[p] for p in flat_pos])
    edges = [0.0, 0.5, 0.7, 0.85, 0.95, 0.99, 1.001]
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (cons_flat >= lo) & (cons_flat < hi)
        if m.sum() == 0:
            continue
        buckets.append({"lo": lo, "hi": hi, "tuned": float(ok_t[m].mean()),
                        "consensus": float(ok_c[m].mean()), "n": int(m.sum())})
    results["by_conservation"] = buckets

    # ---------- 3. confusion matrix ----------
    conf = defaultdict(lambda: defaultdict(int))
    for t, p in zip(truth.ravel(), pred_tuned.ravel()):
        conf[t][p] += 1
    results["confusion"] = {t: {p: int(c) for p, c in row.items()} for t, row in conf.items()}
    results["aas"] = AAS

    # ---------- 4. latent projection ----------
    print("[embed] mean-pooled embeddings for PCA ...")
    eidx = rng.choice(len(df), size=min(N_EMBED, len(df)), replace=False)
    esq = df["text"].to_numpy()[eidx]
    egene = df["v_gene"].to_numpy()[eidx]
    eheld = is_val[eidx]

    def embed(model_path):
        model = AutoModelForMaskedLM.from_pretrained(model_path, output_hidden_states=True)
        model = model.eval().to(device)
        vecs = []
        with torch.no_grad():
            for s in range(0, len(esq), BATCH):
                enc = tok(list(esq[s:s + BATCH]), return_tensors="pt")
                out = model(input_ids=enc["input_ids"].to(device),
                            attention_mask=enc["attention_mask"].to(device))
                h = out.hidden_states[-1][:, 1:-1, :]     # drop <cls>/<eos>
                vecs.append(h.mean(1).cpu().numpy())
        del model
        return np.concatenate(vecs)

    def pca2(x):
        x = x - x.mean(0)
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        pcs = u[:, :2] * s[:2]
        var = (s ** 2 / (s ** 2).sum())[:2]
        return pcs, var

    proj = {}
    for tag, path in (("stock", BASE_MODEL), ("tuned", TUNED_MODEL)):
        pcs, var = pca2(embed(path))
        proj[tag] = {
            "var": [float(v) for v in var],
            "points": [
                {"x": round(float(a), 3), "y": round(float(b), 3),
                 "f": FAMILY_RE.match(g).group(1), "h": bool(hv)}
                for (a, b), g, hv in zip(pcs, egene, eheld)
            ],
        }
        print(f"[embed] {tag}: PC1 {var[0]:.1%}, PC2 {var[1]:.1%} of variance")
    results["projection"] = proj

    results["species"] = args.species
    json.dump(results, open(out_path, "w"))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
