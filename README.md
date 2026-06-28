# Biological-process observability in cancer transcriptomic systems

Reproducibility code for the manuscript:

**Biological-process observability reveals discriminative and endpoint-invariant information regimes in cancer transcriptomic systems**

Author: **Sin Guan Kong**

## Repository scope

This repository provides portable code for the main manuscript-facing analyses and retains historical development scripts in a clearly separated `legacy/` directory.

The primary reproducibility workflows are:

1. **Core endpoint-discriminability pipeline**
   - Mann–Whitney endpoint association
   - discriminability score, `D = -log10(P)`
   - orientation-corrected AUC
   - class-stratified bootstrap stability
   - leakage-controlled observation-scale analysis

2. **Scoring-formulation sensitivity analysis**
   - mean-z
   - GSVA
   - ssGSEA
   - repeated stratified cross-validation
   - cross-method D-rank correlations
   - top-K overlap and Jaccard concordance

## Repository layout

```text
.
├── src/aido_d_xobs/              # portable core Python package
├── scripts/                      # command-line analysis scripts
├── configs/                      # portable example manifest
├── data/                         # local data location; data are not committed
├── results/                      # generated outputs; ignored by Git
├── tests/                        # synthetic smoke test
├── docs/SCRIPT_INVENTORY.md      # status and provenance of every script
└── legacy/                       # historical/internal development scripts
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the portable core:

```bash
pip install -e .
```

For GSVA/ssGSEA and the historical scripts:

```bash
pip install -e ".[full]"
```

Alternatively:

```bash
pip install -r requirements.txt
```

## Core analysis

The BP activity matrix must contain one sample-ID column and one column per biological-process observable. The endpoint file must contain the same sample IDs and a binary label column.

```bash
python scripts/run_core_analysis.py \
  --bp-matrix data/example/bp_activity.tsv \
  --endpoint-file data/example/endpoint.tsv \
  --output-dir results/core_stage \
  --sample-column sample_id \
  --label-column stage_group \
  --positive-label late \
  --n-bootstrap 100 \
  --bootstrap-top-k 50 \
  --bootstrap-ranking D_score \
  --k-values 1,3,5,10,15,20,30,40,50 \
  --n-splits 5 \
  --n-repeats 20 \
  --random-seed 42
```

The installed command can also be used:

```bash
aido-d-xobs --help
```

### Core outputs

- `process_discriminability.csv`
- `bootstrap_stability.csv`
- `observation_scale_summary.csv`
- `observation_scale_fold_results.csv`
- `run_config.json`

## Scoring-formulation sensitivity

Copy and edit the example manifest:

```bash
cp configs/example_manifest.json configs/local_manifest.json
```

Then run:

```bash
python scripts/scoring_sensitivity_analysis.py \
  --manifest configs/local_manifest.json \
  --output-dir results \
  --methods mean_z gsva ssgsea \
  --k-values 1 3 5 10 15 20 30 40 50 \
  --cv-splits 5 \
  --cv-repeats 20 \
  --threads 4
```

For a quick installation check, use `--methods mean_z --k-values 1 10 50 --cv-splits 3 --cv-repeats 1`.

## Reproducibility notes

- Feature ranking is repeated within each cross-validation training partition.
- Repeated cross-validation and bootstrap analysis quantify within-cohort robustness; they are not external validation.
- The core script supports both `D_score` and the earlier internal `abs_cohen_d` bootstrap ranking. The selected mode is written to `run_config.json` and should be reported explicitly.
- Raw datasets and controlled clinical files are not included.
- Update paths and column names in the manifest before running the scoring-sensitivity workflow.

## Testing

```bash
pip install pytest
pytest -q
```

The test uses synthetic data and does not reproduce manuscript results.

## Historical scripts

The `legacy/` directory contains the original development scripts, grouped by purpose. Some include hard-coded local paths, earlier internal terminology, or superseded code. They are retained for provenance. See [`docs/SCRIPT_INVENTORY.md`](docs/SCRIPT_INVENTORY.md).

## Citation

Use `CITATION.cff` after replacing the repository URL and adding the final journal citation when available.

## License

No open-source license has been selected yet. See `LICENSE.md` before public release.
