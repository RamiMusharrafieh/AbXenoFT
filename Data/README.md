# Data

Everything in this directory is gitignored. The raw downloads and derived
tables total roughly 15 GB. This file explains how to obtain and regenerate all of it.

## 1. Source repertoires (OAS)

Bulk unpaired heavy-chain IgM reads from the
[Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/) database.
Download the `.csv.gz` files into this directory.

| Species | Study | Runs | Cells | Raw sequences |
|---|---|---|---|---|
| Mouse (C57BL/6) | Greiff et al. 2017 | `ERR1759659`, `ERR1759661`, `ERR1759668`, `ERR1759677` | spleen, naive B | 17,933,653 |
| Rhesus macaque | same series | `ERR1759737`, `ERR1759738`, `ERR1759739`, `ERR1759741`, `ERR1759744` | PBMC, unsorted B | 1,467,486 |

Expected filenames: `ERR17596*_Heavy_IGHM.csv.gz` and
`rhesus_ERR*_Heavy_IGHM.csv.gz`.

Each OAS file has a JSON metadata line, then a CSV header, then one row per
sequence. The extractor handles that layout.

## 2. Approved-antibody table

`TABS_exported_Antibodies*.csv`, an export from
[The Antibody Society's therapeutic antibody database](https://www.antibodysociety.org/antibody-therapeutics-product-data/).
Not redistributed here; export it yourself and drop it in the repo root.

Only aggregate counts derived from it are tracked, in
`analysis/approvals_by_technology.json`.

## 3. Regenerating the derived tables

```bash
# QC-filtered framework tables (from the repo root)
python extract_frameworks.py                                    # -> clean_mouse_heavy_IGHM_frameworks.csv   (4.7 GB)
python extract_frameworks.py --glob "rhesus_*_Heavy_IGHM.csv.gz" \
    --output Data/clean_rhesus_heavy_IGHM_frameworks.csv \
    --expect-species rhesus                                     # -> 564 MB

# Deduplicated training caches
python train_mlm_esm2.py --prep-only                            # -> frameworks_unique.tsv          (266 MB)
python train_mlm_esm2.py --prep-only \
    --data-csv Data/clean_rhesus_heavy_IGHM_frameworks.csv \
    --cache-file Data/frameworks_unique_macaque.tsv             # -> frameworks_unique_macaque.tsv (116 MB)
```

## 4. What ends up here

| File | Rows | Size |
|---|---|---|
| `clean_mouse_heavy_IGHM_frameworks.csv` | 11,688,651 | 4.7 GB |
| `clean_rhesus_heavy_IGHM_frameworks.csv` | 1,419,444 | 564 MB |
| `frameworks_unique.tsv` (mouse, deduped) | 2,767,822 | 266 MB |
| `frameworks_unique_macaque.tsv` | 1,209,734 | 116 MB |

The two `frameworks_unique*.tsv` files are the actual training inputs: one
concatenated FR1+FR2+FR3+FR4 sequence per line with its V gene.
