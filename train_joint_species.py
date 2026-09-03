#!/usr/bin/env python3
"""
Species-conditioned framework model: one ESM-2 fine-tune that knows both mouse
and rhesus macaque heavy-chain framework space, and can be steered to either.

Every training example is prefixed with a single-residue species marker:

    B QVQLQQPGAELVKPGASV...     -> mouse
    O QVQLVQSGAEVKKPGASV...     -> macaque

'B' and 'O' are real entries in the ESM-2 vocabulary that never occur in antibody
framework sequences, so they can be repurposed as condition tokens. This matters
because the recipe freezes the input embeddings: a genuinely new token would keep
its random embedding forever, whereas these carry pretrained vectors that are
merely unused, and the trainable top layers learn to condition on them.

The marker is flagged as a special token during tokenisation so the MLM collator
never masks it: it is context to condition on, never a prediction target.

Validation holds out whole V genes within each species, so both conditions are
scored on germlines the model never trained on.

Usage:
    python train_joint_species.py --prep-only
    python train_joint_species.py
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    set_seed,
)

from train_mlm_esm2 import build_training_args, freeze_encoder_bottom

PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "Data"
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

SPECIES_TOKEN = {"mouse": "B", "macaque": "O"}
FWR_AA_COLS = ["fwr1_aa", "fwr2_aa", "fwr3_aa", "fwr4_aa"]

MOUSE_TSV = DATA / "frameworks_unique.tsv"                    # built by train_mlm_esm2.py
MACAQUE_CSV = DATA / "clean_rhesus_heavy_IGHM_frameworks.csv"
MACAQUE_TSV = DATA / "frameworks_unique_macaque.tsv"
JOINT_TSV = DATA / "frameworks_joint.tsv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT / "checkpoints" / "esm2_t6_8M_joint_fwr_vgene")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--prep-only", action="store_true")
    p.add_argument("--mouse-cap-ratio", type=float, default=3.0,
                   help="Keep at most this many mouse sequences per macaque sequence. "
                        "0 disables the cap.")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--max-eval-samples", type=int, default=10_000)

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument("--max-length", type=int, default=160)
    p.add_argument("--num-frozen-layers", type=int, default=4)
    p.add_argument("--keep-lm-head-tied", action="store_true")

    p.add_argument("--eval-steps", type=int, default=5000)
    p.add_argument("--logging-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--early-stopping-threshold", type=float, default=0.002)

    p.add_argument("--num-proc", type=int, default=1)
    p.add_argument("--dataloader-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume-from-checkpoint", default=None)
    return p.parse_args()


def build_macaque_cache(chunk_size: int = 200_000) -> None:
    if MACAQUE_TSV.exists():
        print(f"[data] {MACAQUE_TSV.name} already built")
        return
    if not MACAQUE_CSV.exists():
        raise FileNotFoundError(f"{MACAQUE_CSV} not found; run extract_frameworks.py first")

    print(f"[data] building {MACAQUE_TSV.name} from {MACAQUE_CSV.name} ...")
    seen: set[str] = set()
    n_read = n_kept = 0
    tmp = MACAQUE_TSV.with_suffix(".tmp")
    with tmp.open("w") as out:
        out.write("text\tv_gene\n")
        cols = FWR_AA_COLS + ["v_call"]
        for chunk in pd.read_csv(MACAQUE_CSV, usecols=cols, dtype=str, chunksize=chunk_size):
            n_read += len(chunk)
            chunk = chunk.dropna(subset=cols)
            seqs = (chunk["fwr1_aa"] + chunk["fwr2_aa"] + chunk["fwr3_aa"] + chunk["fwr4_aa"])
            genes = chunk["v_call"].str.split(",").str[0].str.split("*").str[0]
            for seq, gene in zip(seqs, genes):
                if seq in seen:
                    continue
                seen.add(seq)
                out.write(f"{seq}\t{gene}\n")
                n_kept += 1
            print(f"  read {n_read:,} -> unique {n_kept:,}", flush=True)
    tmp.rename(MACAQUE_TSV)
    print(f"[data] {n_kept:,} unique macaque frameworks from {n_read:,} rows")


def build_joint_cache(cap_ratio: float, seed: int) -> None:
    if JOINT_TSV.exists():
        print(f"[data] {JOINT_TSV.name} already built")
        return
    mouse = pd.read_csv(MOUSE_TSV, sep="\t", dtype=str).assign(species="mouse")
    mac = pd.read_csv(MACAQUE_TSV, sep="\t", dtype=str).assign(species="macaque")
    print(f"[data] mouse {len(mouse):,} | macaque {len(mac):,}")

    if cap_ratio > 0:
        cap = int(cap_ratio * len(mac))
        if len(mouse) > cap:
            mouse = mouse.sample(n=cap, random_state=seed)
            print(f"[data] mouse capped to {cap:,} ({cap_ratio:g}x macaque) so the "
                  f"macaque condition is not swamped")

    joint = pd.concat([mouse, mac], ignore_index=True)
    joint = joint.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    joint.to_csv(JOINT_TSV, sep="\t", index=False)
    print(f"[data] joint cache: {len(joint):,} sequences "
          f"({100 * (joint.species == 'mouse').mean():.1f}% mouse)")


def split_by_gene_per_species(ds, val_fraction: float, seed: int):
    """Hold out whole V genes independently within each species.

    Gene names are namespaced by species because mouse and macaque nomenclature
    can collide textually (both have IGHV1-* genes) while being different loci.
    """
    keys = [f"{s}:{g}" for s, g in zip(ds["species"], ds["v_gene"])]
    per_species: dict[str, list[str]] = {}
    for k in keys:
        per_species.setdefault(k.split(":", 1)[0], []).append(k)

    val_groups: set[str] = set()
    for sp, sp_keys in per_species.items():
        counts = Counter(sp_keys)
        groups = sorted(counts)
        random.Random(seed).shuffle(groups)
        target = val_fraction * len(sp_keys)
        acc = 0
        chosen = set()
        for g in groups:
            if acc >= target:
                break
            if acc + counts[g] > target * 1.15 and acc > target * 0.75:
                continue
            chosen.add(g)
            acc += counts[g]
        val_groups |= chosen
        print(f"[split] {sp}: holding out {len(chosen)}/{len(counts)} genes "
              f"-> {acc:,} sequences ({100 * acc / len(sp_keys):.1f}%)")

    val_idx = [i for i, k in enumerate(keys) if k in val_groups]
    train_idx = [i for i, k in enumerate(keys) if k not in val_groups]
    return ds.select(train_idx), ds.select(val_idx)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    build_macaque_cache()
    build_joint_cache(args.mouse_cap_ratio, args.seed)
    if args.prep_only:
        return

    raw = load_dataset("csv", data_files=str(JOINT_TSV), delimiter="\t")["train"]
    raw = raw.filter(lambda ex: ex["text"] is not None and len(ex["text"]) > 0)
    train_ds, eval_ds = split_by_gene_per_species(raw, args.val_fraction, args.seed)
    print(f"[split] train={len(train_ds):,}  validation={len(eval_ds):,}")

    # Keep a per-species copy of the validation split for the final report.
    eval_by_species = {
        sp: eval_ds.filter(lambda ex, s=sp: ex["species"] == s)
        for sp in SPECIES_TOKEN
    }
    for sp, d in eval_by_species.items():
        print(f"[split] validation {sp}: {len(d):,}")

    if args.max_eval_samples and len(eval_ds) > args.max_eval_samples:
        eval_ds = eval_ds.shuffle(seed=args.seed).select(range(args.max_eval_samples))
        print(f"[split] periodic eval on {len(eval_ds):,}-sequence subsample")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        texts = [SPECIES_TOKEN[s] + t for s, t in zip(batch["species"], batch["text"])]
        enc = tokenizer(texts, truncation=True, max_length=args.max_length)
        masks = []
        for ids in enc["input_ids"]:
            m = tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
            m[1] = 1   # species marker: condition, never a prediction target
            masks.append(m)
        enc["special_tokens_mask"] = masks
        return enc

    def prep(ds, desc):
        return ds.map(tokenize, batched=True, remove_columns=ds.column_names,
                      num_proc=args.num_proc, desc=desc)

    train_tok = prep(train_ds, "tokenizing train")
    eval_tok = prep(eval_ds, "tokenizing val")
    eval_species_tok = {sp: prep(d, f"tokenizing val/{sp}")
                        for sp, d in eval_by_species.items() if len(d)}

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )

    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    freeze_encoder_bottom(model, args.num_frozen_layers, args.keep_lm_head_tied)

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=build_training_args(args, len(train_tok)),
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        callbacks=callbacks,
    )
    print(f"[train] device={trainer.args.device}  batch={args.batch_size}  epochs={args.epochs}")

    base = trainer.evaluate()
    print(f"[eval] before: loss={base['eval_loss']:.4f} ppl={math.exp(base['eval_loss']):.3f}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    for sp, d in eval_species_tok.items():
        m = trainer.evaluate(eval_dataset=d, metric_key_prefix=f"eval_{sp}")
        loss = m[f"eval_{sp}_loss"]
        print(f"[eval] {sp:8s}: loss={loss:.4f} ppl={math.exp(loss):.3f}  (n={len(d):,})")

    final = args.output_dir / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"[save] {final}")


if __name__ == "__main__":
    main()
