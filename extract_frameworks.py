#!/usr/bin/env python3
"""
Extract clean heavy-chain (IGHM) framework region sequences from OAS
(Observed Antibody Space) bulk CSV.gz downloads.

Species-agnostic: select the input files with --glob and optionally assert the
species recorded in their metadata with --expect-species.

Input files (OAS "unpaired" bulk format):
    - Line 1: JSON metadata (run, species, subject, isotype, ...)
    - Line 2: CSV header
    - Line 3+: one row per observed sequence, IgBLAST/ANARCI-annotated

Output: a single clean CSV with one row per valid sequence, containing
    - provenance columns (run_id, subject, species, v_call, j_call)
    - nucleotide framework regions: fwr1, fwr2, fwr3, fwr4
    - amino-acid framework regions: fwr1_aa, fwr2_aa, fwr3_aa, fwr4_aa

A sequence is kept only if it passes all of:
    - locus == 'H'                  (heavy chain)
    - productive == 'T'             (productive rearrangement)
    - stop_codon == 'F'             (no stop codon anywhere in the sequence)
    - vj_in_frame == 'T'            (V and J segments are in frame)
    - v_frameshift == 'F'           (no frameshift in the V segment)
    - complete_vdj == 'T'           (full V(D)J span present, so FR1-FR4 all exist)
    - all four fwr*/fwr*_aa fields are non-empty
    - none of the fwr*_aa fields contain '*' (stop) or 'X' (untranslatable/ambiguous codon)

Files are streamed in chunks so peak memory stays low regardless of file size.
"""

import argparse
import csv
import gzip
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "Data"
MOUSE_GLOB = "ERR17596*_Heavy_IGHM.csv.gz"
OUTPUT_FILE = DATA_DIR / "clean_mouse_heavy_IGHM_frameworks.csv"

CHUNK_SIZE = 200_000

USE_COLS = [
    "locus",
    "stop_codon",
    "vj_in_frame",
    "v_frameshift",
    "productive",
    "complete_vdj",
    "v_call",
    "j_call",
    "fwr1", "fwr1_aa",
    "fwr2", "fwr2_aa",
    "fwr3", "fwr3_aa",
    "fwr4", "fwr4_aa",
]

FWR_NT_COLS = ["fwr1", "fwr2", "fwr3", "fwr4"]
FWR_AA_COLS = ["fwr1_aa", "fwr2_aa", "fwr3_aa", "fwr4_aa"]

OUTPUT_COLS = [
    "run_id", "subject", "species",
    "v_call", "j_call",
    "fwr1", "fwr1_aa",
    "fwr2", "fwr2_aa",
    "fwr3", "fwr3_aa",
    "fwr4", "fwr4_aa",
]


def read_metadata(path: Path) -> dict:
    # OAS bulk downloads store the metadata line as a CSV row containing a
    # single quoted field with a JSON blob inside (quotes doubled per CSV
    # escaping rules), so it must be unescaped via csv before json.loads.
    with gzip.open(path, "rt", newline="") as fh:
        header_line = fh.readline()
    (field,) = next(csv.reader([header_line]))
    return json.loads(field)


def filter_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    df = chunk

    # QC / productivity filters -> drop non-productive, out-of-frame,
    # frameshifted, stop-codon-containing or incomplete VDJ sequences.
    mask = (
        (df["locus"] == "H")
        & (df["productive"] == "T")
        & (df["stop_codon"] == "F")
        & (df["vj_in_frame"] == "T")
        & (df["v_frameshift"] == "F")
        & (df["complete_vdj"] == "T")
    )
    df = df.loc[mask]

    # Drop rows with missing/empty framework region data.
    fwr_cols = FWR_NT_COLS + FWR_AA_COLS
    df = df.dropna(subset=fwr_cols)
    for col in fwr_cols:
        df = df.loc[df[col].str.strip() != ""]

    # Guard against premature stop codons ('*') or untranslatable/ambiguous
    # residues ('X') slipping through within the framework regions themselves.
    for col in FWR_AA_COLS:
        df = df.loc[~df[col].str.contains(r"[\*X]", regex=True, na=True)]

    return df


def process_file(path: Path, wrote_header: bool) -> tuple[int, int, bool]:
    meta = read_metadata(path)
    run_id = meta.get("Run", path.stem)
    subject = meta.get("Subject", "")
    species = meta.get("Species", "")

    total_rows = 0
    kept_rows = 0

    reader = pd.read_csv(
        path,
        compression="gzip",
        skiprows=1,          # skip the JSON metadata line
        usecols=USE_COLS,
        dtype=str,
        keep_default_na=True,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        total_rows += len(chunk)
        clean = filter_chunk(chunk)
        kept_rows += len(clean)

        if clean.empty:
            continue

        clean = clean.assign(run_id=run_id, subject=subject, species=species)
        clean = clean[OUTPUT_COLS]

        clean.to_csv(
            OUTPUT_FILE,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True

    return total_rows, kept_rows, wrote_header


def main() -> None:
    global OUTPUT_FILE
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default=MOUSE_GLOB,
                   help="Filename glob for the OAS .csv.gz files to process.")
    p.add_argument("--output", type=Path, default=OUTPUT_FILE)
    p.add_argument("--expect-species", default=None,
                   help="Abort if a file's metadata Species does not contain this string.")
    args = p.parse_args()
    OUTPUT_FILE = args.output

    inputs = sorted(DATA_DIR.glob(args.glob))
    if not inputs:
        raise SystemExit(f"no files matched {args.glob!r} in {DATA_DIR}")
    print(f"Matched {len(inputs)} file(s) for {args.glob!r}")

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    wrote_header = False
    grand_total = 0
    grand_kept = 0

    for path in inputs:
        name = path.name
        if args.expect_species:
            got = read_metadata(path).get("Species", "")
            if args.expect_species.lower() not in got.lower():
                raise SystemExit(f"{name}: expected species {args.expect_species!r}, got {got!r}")

        print(f"Processing {name} ...", flush=True)
        total_rows, kept_rows, wrote_header = process_file(path, wrote_header)

        grand_total += total_rows
        grand_kept += kept_rows
        pct = (kept_rows / total_rows * 100) if total_rows else 0.0
        print(f"  {name}: {kept_rows:,} / {total_rows:,} sequences kept ({pct:.2f}%)", flush=True)

    pct = (grand_kept / grand_total * 100) if grand_total else 0.0
    print(f"\nTOTAL: {grand_kept:,} / {grand_total:,} sequences kept ({pct:.2f}%)")
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
