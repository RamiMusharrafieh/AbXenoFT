#!/usr/bin/env python3
"""
Fine-tune facebook/esm2_t6_8M_UR50D with masked language modeling on
heavy-chain framework sequences produced by extract_frameworks.py.

Used for both species: point --data-csv and --cache-file at the mouse or the
macaque table.

Each training example is the concatenation FR1 + FR2 + FR3 + FR4 of a single
observed heavy chain (CDRs removed), so the model adapts to that species'
germline framework space rather than general UniRef protein space.

Only the top of the encoder is trained: the token/position embeddings and
encoder layers 0-3 are frozen, leaving encoder layers 4-5, the final encoder
LayerNorm, and the MLM head trainable.

Validation defaults to a grouped split: whole V genes are held out, so every
germline scaffold in the validation set is one the model never trained on. A
random split instead scores the model largely on memorised germlines, because
somatic-hypermutation variants of the same gene land on both sides.

Usage:
    python train_mlm_esm2.py                             # V-gene-held-out split
    python train_mlm_esm2.py --split-by v_family         # harsher: hold out families
    python train_mlm_esm2.py --split-by random           # original random split
    python train_mlm_esm2.py --max-sequences 5000 --epochs 1   # smoke test
"""

from __future__ import annotations

import argparse
import inspect
import math
import random
import re
import sys
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
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
FWR_AA_COLS = ["fwr1_aa", "fwr2_aa", "fwr3_aa", "fwr4_aa"]
PREP_COLS = FWR_AA_COLS + ["v_call"]
FAMILY_RE = re.compile(r"(IGHV\d+)")

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_DIR / "Data" / "clean_mouse_heavy_IGHM_frameworks.csv"
DEFAULT_CACHE_DIR = PROJECT_DIR / "Data"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "checkpoints" / "esm2_t6_8M_mouse_fwr"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--data-csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--cache-file", type=Path, default=None,
                   help="Plain-text sequence cache. Defaults to a name derived from the dedupe/limit settings.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-name", default=MODEL_NAME)

    p.add_argument("--no-dedupe", action="store_true",
                   help="Keep duplicate framework sequences (default is to train on unique sequences only).")
    p.add_argument("--max-sequences", type=int, default=None,
                   help="Cap sequences written during data prep. Use for smoke tests.")
    p.add_argument("--csv-chunk-size", type=int, default=500_000)
    p.add_argument("--prep-only", action="store_true",
                   help="Build the sequence cache and exit without training.")

    p.add_argument("--split-by", choices=["v_gene", "v_family", "random"], default="v_gene",
                   help="v_gene/v_family hold out whole germline groups (generalization test); "
                        "random splits sequences irrespective of germline.")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--max-eval-samples", type=int, default=20_000,
                   help="Subsample of the validation split used for periodic evaluation (0 = use all).")

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument("--max-length", type=int, default=160)
    p.add_argument("--num-frozen-layers", type=int, default=4)

    p.add_argument("--early-stopping-patience", type=int, default=4,
                   help="Stop after this many evaluations with no improvement (0 disables).")
    p.add_argument("--early-stopping-threshold", type=float, default=0.002,
                   help="Minimum eval-loss improvement that counts as progress.")
    p.add_argument("--eval-steps", type=int, default=2000,
                   help="Interval for both evaluation and checkpoint saving.")
    p.add_argument("--logging-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--keep-lm-head-tied", action="store_true",
                   help="Leave the MLM decoder tied to the frozen input embeddings (it will not train).")

    p.add_argument("--num-proc", type=int, default=1, help="Processes for tokenization.")
    p.add_argument("--dataloader-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if MPS/CUDA is available.")
    p.add_argument("--resume-from-checkpoint", default=None)

    return p.parse_args()


def default_cache_path(args: argparse.Namespace) -> Path:
    stem = "frameworks_all" if args.no_dedupe else "frameworks_unique"
    if args.max_sequences:
        stem += f"_max{args.max_sequences}"
    return DEFAULT_CACHE_DIR / f"{stem}.tsv"


def prepare_sequences(args: argparse.Namespace, cache_path: Path) -> int:
    if cache_path.exists():
        n = sum(1 for _ in cache_path.open())
        print(f"[data] using cached {cache_path.name} ({n:,} sequences)")
        return n

    if not args.data_csv.exists():
        raise FileNotFoundError(
            f"{args.data_csv} not found. Run extract_frameworks.py first."
        )

    print(f"[data] building sequence cache from {args.data_csv.name} ...")
    dedupe = not args.no_dedupe
    seen: set[str] = set()
    n_read = n_written = 0
    min_len, max_len = 10**9, 0
    done = False

    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("w") as out:
        out.write("text\tv_gene\n")
        reader = pd.read_csv(
            args.data_csv, usecols=PREP_COLS, dtype=str, chunksize=args.csv_chunk_size
        )
        for chunk in reader:
            n_read += len(chunk)
            chunk = chunk.dropna(subset=PREP_COLS)
            seqs = (
                chunk["fwr1_aa"] + chunk["fwr2_aa"] + chunk["fwr3_aa"] + chunk["fwr4_aa"]
            )
            # v_call is "IGHV1-72*01"; drop the allele so the split groups by gene.
            genes = chunk["v_call"].str.split(",").str[0].str.split("*").str[0]
            for seq, gene in zip(seqs, genes):
                if dedupe:
                    if seq in seen:
                        continue
                    seen.add(seq)
                out.write(f"{seq}\t{gene}\n")
                n_written += 1
                min_len = min(min_len, len(seq))
                max_len = max(max_len, len(seq))
                if args.max_sequences and n_written >= args.max_sequences:
                    done = True
                    break
            print(f"  read {n_read:,} rows -> kept {n_written:,} sequences", flush=True)
            if done:
                break

    tmp_path.rename(cache_path)
    print(f"[data] wrote {n_written:,} sequences (from {n_read:,} rows) to {cache_path.name}")
    print(f"[data] sequence length: min={min_len} max={max_len}")
    if max_len > args.max_length:
        print(f"[data] WARNING: longest sequence ({max_len}) exceeds --max-length {args.max_length}; it will be truncated.")
    return n_written


def group_key(gene: str, level: str) -> str:
    if level == "v_family":
        m = FAMILY_RE.match(gene)
        return m.group(1) if m else gene
    return gene


def split_by_group(dataset, val_fraction: float, seed: int, level: str):
    """Hold out whole V genes (or families), so no germline is shared across splits."""
    keys = [group_key(g, level) for g in dataset["v_gene"]]
    counts = Counter(keys)
    groups = sorted(counts)
    random.Random(seed).shuffle(groups)

    target = val_fraction * len(keys)
    val_groups: set[str] = set()
    acc = 0
    for g in groups:
        if acc >= target:
            break
        # Skip a group that would badly overshoot once we are already close;
        # gene usage is skewed enough that one group can carry several percent.
        if acc + counts[g] > target * 1.15 and acc > target * 0.75:
            continue
        val_groups.add(g)
        acc += counts[g]

    val_idx = [i for i, k in enumerate(keys) if k in val_groups]
    train_idx = [i for i, k in enumerate(keys) if k not in val_groups]
    n_groups = len(counts)

    print(f"[split] holding out whole {level} groups: "
          f"{len(val_groups)}/{n_groups} groups -> {len(val_idx):,} sequences "
          f"({100 * len(val_idx) / len(keys):.1f}%)")
    print(f"[split] train groups: {n_groups - len(val_groups)}  |  "
          f"train sequences: {len(train_idx):,}")
    sample = sorted(val_groups)[:8]
    print(f"[split] held-out examples: {', '.join(sample)}"
          f"{' ...' if len(val_groups) > len(sample) else ''}")
    return dataset.select(train_idx), dataset.select(val_idx)


def freeze_encoder_bottom(model, num_frozen_layers: int, keep_tied: bool) -> None:
    esm = model.esm
    n_layers = len(esm.encoder.layer)
    if num_frozen_layers > n_layers:
        raise ValueError(f"cannot freeze {num_frozen_layers} of {n_layers} layers")

    # The MLM decoder is weight-tied to the input embeddings in ESM-2, so
    # freezing the embeddings would also freeze the head's output projection.
    # Untie first (unless asked not to) so the head can actually train.
    tied = model.lm_head.decoder.weight is esm.embeddings.word_embeddings.weight
    if tied and not keep_tied:
        model.lm_head.decoder.weight = torch.nn.Parameter(
            esm.embeddings.word_embeddings.weight.detach().clone()
        )
        model.config.tie_word_embeddings = False
        print("[freeze] untied MLM decoder from input embeddings so the head stays trainable")
    elif tied:
        print("[freeze] MLM decoder left tied to frozen embeddings; only its dense/LayerNorm/bias will train")

    for p in esm.embeddings.parameters():
        p.requires_grad = False
    for layer in esm.encoder.layer[:num_frozen_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    # Contact head is unused by the MLM objective; leaving it trainable would
    # produce parameters that never receive gradients.
    if getattr(esm, "contact_head", None) is not None:
        for p in esm.contact_head.parameters():
            p.requires_grad = False

    def count(module):
        t = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return t, sum(p.numel() for p in module.parameters())

    print(f"[freeze] frozen: embeddings + encoder layers 0-{num_frozen_layers - 1}")
    for i, layer in enumerate(esm.encoder.layer):
        tr, tot = count(layer)
        print(f"  encoder.layer[{i}]: {tr:,}/{tot:,} trainable")
    tr, tot = count(model.lm_head)
    print(f"  lm_head:           {tr:,}/{tot:,} trainable")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[freeze] total trainable: {trainable:,}/{total:,} ({100 * trainable / total:.1f}%)")


def build_training_args(args: argparse.Namespace, num_train_examples: int) -> TrainingArguments:
    kwargs = dict(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_workers,
        seed=args.seed,
        report_to="none",
        # Progress bars are unreadable once stdout is redirected to a log file.
        disable_tqdm=not sys.stdout.isatty(),
        # fp16 autocast is a CUDA path; MPS/CPU train in fp32 here.
        fp16=torch.cuda.is_available() and not args.cpu,
        use_cpu=args.cpu,
    )
    # transformers renamed evaluation_strategy -> eval_strategy in 4.46 and
    # dropped warmup_ratio in favour of warmup_steps in 5.0.
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    if "warmup_ratio" in params:
        kwargs["warmup_ratio"] = args.warmup_ratio
    else:
        steps_per_epoch = math.ceil(num_train_examples / args.batch_size)
        kwargs["warmup_steps"] = int(steps_per_epoch * args.epochs * args.warmup_ratio)
    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cache_path = args.cache_file or default_cache_path(args)
    prepare_sequences(args, cache_path)
    if args.prep_only:
        return

    raw = load_dataset("csv", data_files=str(cache_path), delimiter="\t")["train"]
    raw = raw.filter(lambda ex: ex["text"] is not None and len(ex["text"]) > 0)

    if args.split_by == "random":
        split = raw.train_test_split(test_size=args.val_fraction, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = split_by_group(raw, args.val_fraction, args.seed, args.split_by)
    print(f"[split] mode={args.split_by}  train={len(train_ds):,}  validation={len(eval_ds):,}")

    if args.max_eval_samples and len(eval_ds) > args.max_eval_samples:
        eval_ds = eval_ds.shuffle(seed=args.seed).select(range(args.max_eval_samples))
        print(f"[split] evaluating on a {len(eval_ds):,}-sequence subsample of the validation split")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names,
                            num_proc=args.num_proc, desc="tokenizing train")
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names,
                          num_proc=args.num_proc, desc="tokenizing val")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )

    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    freeze_encoder_bottom(model, args.num_frozen_layers, args.keep_lm_head_tied)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ))
        print(f"[train] early stopping: patience={args.early_stopping_patience} evals "
              f"({args.early_stopping_patience * args.eval_steps:,} steps) "
              f"threshold={args.early_stopping_threshold}")
        print("[train] note: the LR schedule still targets the full run, so an early stop "
              "ends before the anneal completes.")

    trainer = Trainer(
        model=model,
        args=build_training_args(args, len(train_ds)),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    device = trainer.args.device
    print(f"[train] device={device}  batch_size={args.batch_size}  epochs={args.epochs}")

    baseline = trainer.evaluate()
    print(f"[eval] before fine-tuning: loss={baseline['eval_loss']:.4f} "
          f"perplexity={math.exp(baseline['eval_loss']):.3f}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final = trainer.evaluate()
    print(f"[eval] after fine-tuning:  loss={final['eval_loss']:.4f} "
          f"perplexity={math.exp(final['eval_loss']):.3f}")

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[save] best model + tokenizer written to {final_dir}")


if __name__ == "__main__":
    main()
