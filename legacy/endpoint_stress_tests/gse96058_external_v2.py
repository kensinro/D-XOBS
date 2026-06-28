# -*- coding: utf-8 -*-
r"""
D-PHY internal experiment V5

Public manuscript direction:
Assessing biological-process observability and interpretation readiness
in partially observable cancer systems

Main update in V5:
- Supports your UCSC_XENA folder naming style.
- Recognizes GE.tsv / GE.txt as gene-expression input.
- Fixes duplicated TCGA sample IDs after aliquot/barcode truncation by aggregating duplicated samples.
- Outputs all results to D:/AIDO-Temp/

Internal modules:
1. Input data
2. Biological-process observable construction
3. D layer: observability / discriminability
4. Statistical reliability layer
5. h layer: biological anchoring
6. PPI / network support layer
7. Clinical concordance layer
8. Interpretation readiness assessment
"""

import os
import re
import gc
import glob
import json
import time
import warnings
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

try:
    import networkx as nx
except ImportError:
    nx = None


# ============================================================
# 0. USER CONFIG
# ============================================================

ROOT_UCSC_XENA = Path("D:/AIDO-Data/UCSC_XENA")
BASE_OUTPUT_DIR = Path("D:/AIDO-Temp")

BIOMARKER_PATHS = [
    Path("D:/AIDO-Data/Biomarkers/CIViC/01-May-2026-AcceptedClinicalEvidenceSummaries.tsv"),
    Path("D:/AIDO-Data/Biomarkers/CIViC/01-May-2026-FeatureSummaries.tsv"),
    Path("D:/AIDO-Data/Biomarkers/COSMIC/Census_allThu May 28 05_04_17 2026.tsv"),
    Path("D:/AIDO-Data/Biomarkers/OncoKB/cancerGeneList.tsv"),
]

PPI_PATHS = {
    "biogrid": Path("D:/AIDO-Data/PPI/BioGrid/BIOGRID-ORGANISM-Homo_sapiens-5.0.258.tab3"),
    "omnipath": Path("D:/AIDO-Data/PPI/OmniPath/omnipath_interactions.tsv"),
    "string_info": Path("D:/AIDO-Data/PPI/STRING-PPI/9606.protein.info.v12.0.txt"),
    "string_links": Path("D:/AIDO-Data/PPI/STRING-PPI/9606.protein.links.v12.0.txt"),
    "string_physical_links": Path("D:/AIDO-Data/PPI/STRING-PPI/9606.protein.physical.links.v12.0.txt"),
    "string_aliases": Path("D:/AIDO-Data/PPI/STRING-PPI/9606.protein.aliases.v12.0.txt"),
}

# Default: first run BRCA only.
# To run all cancer folders, set TARGET_CANCERS = None
TARGET_CANCERS = ["Breast Cancer (BRCA)"]
# TARGET_CANCERS = None

# Gene set GMT. If this does not exist, script searches D:/AIDO-Data/**/*.gmt
GENESET_GMT = Path("D:/AIDO-Data/GeneSets/c5.go.bp.v2025.1.Hs.symbols.gmt")

MIN_GENES_PER_BP = 8
MAX_GENES_PER_BP = 500
MAX_BP_TERMS = None

N_BOOTSTRAP = 100
N_PERMUTATION = 200
RANDOM_BASELINE_N = 200
RANDOM_SEED = 42

FDR_THRESHOLD = 0.05
BOOTSTRAP_STABILITY_THRESHOLD = 0.60
H_FDR_THRESHOLD = 0.10
PPI_MIN_EDGES = 3
PPI_MIN_LCC_SIZE = 3

EARLY_STAGE_LABEL = "early"
LATE_STAGE_LABEL = "late"

np.random.seed(RANDOM_SEED)


# ============================================================
# 1. BASIC UTILITIES
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^\w\-.()]+", "_", x)
    return x[:160]


def read_table_auto(path, nrows=None):
    """
    Robust table reader for UCSC/Xena and locally saved files.

    Handles:
    - utf-8 / utf-8-sig
    - utf-16 Phenotype.tsv exported from some tools
    - files with no extension, e.g. TCGA.BRCA(1).sampleMap_BRCA_clinicalMatrix
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    lower = path.name.lower()

    if lower.endswith(".csv"):
        seps = [","]
    else:
        seps = ["\t", ","]

    encodings = ["utf-8-sig", "utf-8", "utf-16", "utf-16le", "latin1"]

    last_error = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    low_memory=False,
                    nrows=nrows,
                    encoding=enc
                )

                # Avoid accepting a wrong delimiter that collapses everything into 1 column
                if df.shape[1] >= 2:
                    return df

            except Exception as e:
                last_error = e

    # Last fallback: delimiter sniffing.
    # low_memory is not supported with engine="python".
    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                nrows=nrows,
                encoding=enc
            )
        except Exception as e:
            last_error = e

    raise last_error


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.ones_like(pvals, dtype=float)
    mask = np.isfinite(pvals)

    if mask.sum() > 0:
        out[mask] = multipletests(pvals[mask], method="fdr_bh")[1]

    return out


def standardize_gene_symbol(x):
    if pd.isna(x):
        return None

    x = str(x).strip()
    if x == "":
        return None

    if "|" in x:
        x = x.split("|")[0]

    if x.upper().startswith("ENSG") and "." in x:
        x = x.split(".")[0]

    x = x.upper().strip()
    x = re.sub(r"\s+", "", x)

    return x if x else None


def standardize_sample_id(x):
    if pd.isna(x):
        return None

    s = str(x).strip()
    s = s.replace(".", "-")
    s = s.upper()

    if s.startswith("TCGA") and len(s) >= 12:
        return s[:12]

    return s


def collapse_duplicate_samples(expr):
    """
    TCGA expression matrices often contain multiple aliquots/sample columns.
    After truncating TCGA barcodes to patient-level IDs, duplicate sample IDs can appear.
    This function averages duplicated rows so X and y have consistent sample counts.
    """
    expr = pd.DataFrame(expr)
    expr.index = [standardize_sample_id(x) for x in expr.index]
    expr = expr[expr.index.notna()]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.groupby(expr.index).mean()
    return expr


def infer_cancer_code(folder_name):
    m = re.search(r"\(([A-Za-z0-9]+)\)", str(folder_name))
    if m:
        return m.group(1).upper()
    return safe_name(folder_name).upper()


# ============================================================
# 2. DISCOVER FILES
# ============================================================

def list_cancer_folders(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"UCSC_XENA root not found: {root}")

    folders = [p for p in root.iterdir() if p.is_dir()]
    folders = sorted(folders, key=lambda x: x.name.lower())
    return folders


def find_candidate_files(cancer_folder):
    """
    V4 updated for your current local UCSC_XENA structure.

    Key support:
    - GE.tsv / GE.txt = expression candidate
    - files with no extension are also scanned, because Xena clinicalMatrix files often have no extension
    - TCGA.BRCA(1).sampleMap_BRCA_clinicalMatrix is prioritized as clinical file
    - Phenotype.tsv is kept as clinical candidate but often has survival only, not stage
    """
    cancer_folder = Path(cancer_folder)

    all_files = []

    # Include normal tabular extensions
    for ext in ["*.tsv", "*.txt", "*.csv", "*.tab", "*.tab3", "*.tsv.gz", "*.txt.gz", "*.csv.gz"]:
        all_files.extend(glob.glob(str(cancer_folder / "**" / ext), recursive=True))

    # Include no-extension files and other files directly under subfolders
    for f in glob.glob(str(cancer_folder / "**" / "*"), recursive=True):
        if Path(f).is_file():
            all_files.append(f)

    all_files = sorted(set(all_files))

    expression_candidates = []
    clinical_candidates = []

    for f in all_files:
        base = os.path.basename(f).lower()

        # Direct support for your UCSC_XENA naming style
        if base in ["ge.tsv", "ge.txt", "gene_expression.tsv", "gene_expression.txt"]:
            expression_candidates.append(f)
            continue

        # Clinical matrix can have no extension
        if "clinicalmatrix" in base or "samplemap_brca_clinicalmatrix" in base:
            clinical_candidates.append(f)
            continue

        if base in ["phenotype.tsv", "phenotype.txt", "clinical.tsv", "clinical.txt"]:
            clinical_candidates.append(f)
            continue

        if "stage_groups" in base or "survival" in base:
            clinical_candidates.append(f)
            continue

        # Skip non-GE omics layers for this first D-PHY run
        if base in ["cn.tsv", "mu.tsv", "mu_fixed.tsv", "rppa.tsv"]:
            continue

        if any(k in base for k in ["mirna", "methyl", "mutation", "maf", "copy", "cnv", "seg", "protein", "rppa"]):
            continue

        expr_keywords = [
            "gene", "expression", "rnaseq", "rna_seq", "rnaseqv2", "hiseq",
            "fpkm", "tpm", "rsem", "star", "counts", "xena", "ge"
        ]

        clin_keywords = [
            "phenotype", "clinical", "survival", "sample", "patient", "phen", "stage", "clinicalmatrix"
        ]

        if any(k in base for k in expr_keywords):
            expression_candidates.append(f)

        if any(k in base for k in clin_keywords):
            clinical_candidates.append(f)

    def expr_score(f):
        b = os.path.basename(f).lower()
        score = 0

        if b in ["ge.tsv", "ge.txt"]:
            score += 1000

        for k in ["hiseq", "rnaseq", "rsem", "tpm", "fpkm", "expression", "gene", "ge"]:
            if k in b:
                score += 2

        if "samplemap" in b or "clinicalmatrix" in b:
            score -= 20
        if "clinical" in b or "phenotype" in b:
            score -= 20

        return score

    def clin_score(f):
        b = os.path.basename(f).lower()
        score = 0

        # Highest priority: full Xena clinical matrix, because it contains pathologic_stage
        if "clinicalmatrix" in b:
            score += 1000
        if "samplemap_brca_clinicalmatrix" in b:
            score += 1000

        # Useful if manually created
        if b == "brca_stage_groups_from_survival.tsv":
            score += 500

        # Phenotype.tsv often has survival columns only, so lower than clinicalMatrix
        if b == "phenotype.tsv":
            score += 100

        for k in ["pathologic", "ajcc", "stage", "clinical", "patient", "sample", "phenotype", "survival"]:
            if k in b:
                score += 3

        return score

    expression_candidates = sorted(set(expression_candidates), key=expr_score, reverse=True)
    clinical_candidates = sorted(set(clinical_candidates), key=clin_score, reverse=True)

    return expression_candidates, clinical_candidates, all_files


# ============================================================
# 3. LOAD EXPRESSION AND CLINICAL DATA
# ============================================================

def load_expression_matrix(path):
    log(f"Loading expression: {path}")

    df = read_table_auto(path)

    if df.shape[0] < 5 or df.shape[1] < 5:
        raise ValueError("Expression file too small.")

    first_col = df.columns[0]

    first_values = df[first_col].astype(str).head(20).tolist()
    first_col_gene_like = sum([
        bool(re.match(r"^[A-Za-z0-9\-_.|]+$", x))
        for x in first_values
    ]) > 5

    col_tcga_count = sum([
        str(c).upper().replace(".", "-").startswith("TCGA")
        for c in df.columns[1:50]
    ])

    # Common format: rows = genes, columns = samples
    if first_col_gene_like and col_tcga_count >= 3:
        genes = df[first_col].map(standardize_gene_symbol)
        mat = df.drop(columns=[first_col])
        mat.columns = [standardize_sample_id(c) for c in mat.columns]
        mat.index = genes
        mat = mat[mat.index.notna()]
        mat = mat.apply(pd.to_numeric, errors="coerce")
        mat = mat.groupby(mat.index).mean()

        expr = mat.T
        expr = collapse_duplicate_samples(expr)
        expr = expr.loc[:, expr.notna().mean(axis=0) > 0.80]

        return expr

    # Alternative: rows = samples, columns = genes
    first_tcga_count = sum([
        str(x).upper().replace(".", "-").startswith("TCGA")
        for x in df[first_col].head(50)
    ])

    if first_tcga_count >= 3:
        samples = df[first_col].map(standardize_sample_id)
        mat = df.drop(columns=[first_col])
        mat.columns = [standardize_gene_symbol(c) for c in mat.columns]
        mat.index = samples
        mat = mat.loc[mat.index.notna(), :]
        mat = mat.loc[:, pd.Series(mat.columns).notna().values]
        mat = mat.apply(pd.to_numeric, errors="coerce")
        mat = mat.groupby(mat.index).mean()
        mat = collapse_duplicate_samples(mat)
        mat = mat.T.groupby(level=0).mean().T
        mat = mat.loc[:, mat.notna().mean(axis=0) > 0.80]

        return mat

    # Last attempt: treat first column as gene name
    genes = df[first_col].map(standardize_gene_symbol)
    mat = df.drop(columns=[first_col])
    mat.columns = [standardize_sample_id(c) for c in mat.columns]
    mat.index = genes
    mat = mat[mat.index.notna()]
    mat = mat.apply(pd.to_numeric, errors="coerce")
    mat = mat.groupby(mat.index).mean()

    expr = mat.T
    expr = collapse_duplicate_samples(expr)
    expr = expr.loc[:, expr.notna().mean(axis=0) > 0.80]

    if expr.shape[0] < 10 or expr.shape[1] < 100:
        raise ValueError("Could not confidently parse expression orientation.")

    return expr


def parse_stage_value(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().upper()
    s = s.replace("STAGE", "").strip()
    s = s.replace(" ", "")
    s = s.replace("_", "")
    s = s.replace("-", "")

    if s in ["", "NAN", "NA", "NOTREPORTED", "UNKNOWN", "[NOTAVAILABLE]", "[NOTAPPLICABLE]"]:
        return np.nan

    if s.startswith("IV"):
        return LATE_STAGE_LABEL
    if s.startswith("III"):
        return LATE_STAGE_LABEL
    if s.startswith("II"):
        return EARLY_STAGE_LABEL
    if s.startswith("I"):
        return EARLY_STAGE_LABEL

    # In case already numeric
    if s in ["1", "2"]:
        return EARLY_STAGE_LABEL
    if s in ["3", "4"]:
        return LATE_STAGE_LABEL

    # In case already grouped
    if s in ["EARLY", "EARLYSTAGE", "STAGEEARLY"]:
        return EARLY_STAGE_LABEL
    if s in ["LATE", "LATESTAGE", "STAGELATE"]:
        return LATE_STAGE_LABEL

    return np.nan


def get_stage_candidate_columns(clinical_df):
    """
    Return possible stage/group columns. V4 uses all candidates and then
    chooses the one that yields the most early/late labels.
    """
    candidates = []

    # Strong/direct names first
    strong_patterns = [
        "pathologic_stage",
        "ajcc_stage",
        "converted_stage",
        "stage_group",
        "stagegroup",
        "early_late",
        "earlylate",
        "tumor_stage"
    ]

    for c in clinical_df.columns:
        cl = str(c).lower()
        if any(p in cl for p in strong_patterns):
            candidates.append(c)

    # General stage columns
    for c in clinical_df.columns:
        cl = str(c).lower()
        if "stage" in cl and c not in candidates:
            candidates.append(c)

    # Put likely best columns first
    candidates = sorted(
        candidates,
        key=lambda c: (
            "pathologic_stage" in str(c).lower(),
            "ajcc_stage" in str(c).lower(),
            "converted_stage" in str(c).lower(),
            "stage_group" in str(c).lower(),
            "stage" in str(c).lower()
        ),
        reverse=True
    )

    return candidates


def find_stage_column(clinical_df):
    candidates = get_stage_candidate_columns(clinical_df)
    return candidates[0] if candidates else None


def find_sample_column(clinical_df):
    possible = []

    for c in clinical_df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["sampleid", "sample", "patient", "submitter", "barcode", "id", "_patient"]):
            possible.append(c)

    scored = []

    for c in possible + list(clinical_df.columns[:5]):
        try:
            vals = clinical_df[c].astype(str).head(300).tolist()
            tcga_full = sum([
                v.upper().replace(".", "-").startswith("TCGA") and len(v) >= 12
                for v in vals
            ])
            # Prefer sampleID / sample over genomic UUID fields
            name_bonus = 0
            cl = str(c).lower()
            if cl in ["sampleid", "sample", "sample_id"]:
                name_bonus += 100
            if cl in ["_patient", "patient", "bcr_patient_barcode"]:
                name_bonus += 50
            if "genomic" in cl or "uuid" in cl:
                name_bonus -= 50

            scored.append((tcga_full + name_bonus, c))
        except Exception:
            pass

    scored = sorted(scored, reverse=True)

    if scored and scored[0][0] >= 3:
        return scored[0][1]

    return clinical_df.columns[0]


def load_clinical_labels(paths):
    """
    V4:
    - Reads clinicalMatrix with utf-8-sig and Phenotype.tsv with utf-16.
    - Tries every stage-like column.
    - Chooses the file + column with the largest usable early/late count.
    """
    best = None
    errors = []
    diagnostics = []

    for path in paths:
        try:
            df = read_table_auto(path)

            if df.shape[0] < 5 or df.shape[1] < 2:
                diagnostics.append({
                    "path": str(path),
                    "status": "skipped_too_small",
                    "shape": str(df.shape)
                })
                continue

            sample_col = find_sample_column(df)
            stage_candidates = get_stage_candidate_columns(df)

            if not stage_candidates:
                diagnostics.append({
                    "path": str(path),
                    "status": "no_stage_like_columns",
                    "shape": str(df.shape),
                    "sample_col": sample_col,
                    "columns_first_30": ";".join(map(str, df.columns[:30]))
                })
                continue

            for stage_col in stage_candidates:
                out = pd.DataFrame()
                out["sample_id"] = df[sample_col].map(standardize_sample_id)
                out["raw_stage"] = df[stage_col]
                out["stage_group"] = df[stage_col].map(parse_stage_value)

                out = out.dropna(subset=["sample_id", "stage_group"])
                out = out.drop_duplicates("sample_id")

                n_early = int((out["stage_group"] == EARLY_STAGE_LABEL).sum())
                n_late = int((out["stage_group"] == LATE_STAGE_LABEL).sum())
                score = n_early + n_late

                diagnostics.append({
                    "path": str(path),
                    "status": "tested",
                    "shape": str(df.shape),
                    "sample_col": sample_col,
                    "stage_col": stage_col,
                    "n_early": n_early,
                    "n_late": n_late,
                    "score": score
                })

                if n_early >= 10 and n_late >= 10:
                    # Prefer pathologic_stage when counts are similar
                    priority = 0
                    scl = str(stage_col).lower()
                    if "pathologic_stage" in scl:
                        priority += 1000
                    if "ajcc" in scl:
                        priority += 100
                    if "converted_stage" in scl:
                        priority += 50

                    total_score = score + priority

                    if best is None or total_score > best["total_score"]:
                        best = {
                            "path": path,
                            "df": out,
                            "stage_col": stage_col,
                            "sample_col": sample_col,
                            "score": score,
                            "total_score": total_score,
                            "n_early": n_early,
                            "n_late": n_late,
                            "diagnostics": diagnostics
                        }

        except Exception as e:
            errors.append({"path": str(path), "error": str(e)})
            diagnostics.append({
                "path": str(path),
                "status": "read_failed",
                "error": str(e)
            })

    if best is None:
        diag_text = pd.DataFrame(diagnostics).to_string(index=False) if diagnostics else "No diagnostics."
        raise ValueError(
            "No usable early/late stage labels found. "
            "Clinical diagnostics:\n" + diag_text
        )

    log(f"Clinical selected: {best['path']}")
    log(f"Sample column: {best['sample_col']}")
    log(f"Stage column: {best['stage_col']} | early={best['n_early']} late={best['n_late']}")

    # Attach diagnostics for saving later
    best["diagnostics_df"] = pd.DataFrame(diagnostics)

    return best["df"], best


# ============================================================
# 4. GENE SETS / BP OBSERVABLES
# ============================================================

def auto_find_gmt():
    if GENESET_GMT.exists():
        return GENESET_GMT

    log("Configured GMT not found. Searching D:/AIDO-Data for *.gmt ...")

    candidates = glob.glob("D:/AIDO-Data/**/*.gmt", recursive=True)

    if not candidates:
        return None

    def score(f):
        b = os.path.basename(f).lower()
        s = 0

        if "go" in b:
            s += 3
        if "bp" in b or "biological" in b:
            s += 5
        if "c5" in b:
            s += 2
        if "symbols" in b:
            s += 2
        if "hallmark" in b:
            s += 1

        return s

    candidates = sorted(candidates, key=score, reverse=True)
    return Path(candidates[0])


def load_gmt(path, expression_genes=None):
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            "No GMT file found. Please provide a GMT file, e.g. MSigDB GO BP GMT, "
            "or update GENESET_GMT in the code."
        )

    path = Path(path)
    log(f"Loading gene sets GMT: {path}")

    gene_sets = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 3:
                continue

            term = parts[0]
            genes = [standardize_gene_symbol(g) for g in parts[2:]]
            genes = set([g for g in genes if g])

            if expression_genes is not None:
                genes = genes.intersection(expression_genes)

            if MIN_GENES_PER_BP <= len(genes) <= MAX_GENES_PER_BP:
                gene_sets[term] = genes

    if MAX_BP_TERMS is not None:
        gene_sets = dict(list(gene_sets.items())[:MAX_BP_TERMS])

    log(f"Usable BP/gene sets: {len(gene_sets)}")

    if len(gene_sets) == 0:
        raise ValueError("No usable gene sets after expression-gene overlap.")

    return gene_sets


def construct_bp_observable_matrix(expr_samples_by_genes, gene_sets):
    log("Constructing BP-level observable matrix ...")

    expr = collapse_duplicate_samples(expr_samples_by_genes.copy())
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.loc[:, expr.notna().mean(axis=0) > 0.80]
    expr = expr.fillna(expr.median(axis=0))

    vals = expr.values

    if np.nanpercentile(vals, 99) > 100:
        expr = np.log2(expr + 1)

    scaler = StandardScaler(with_mean=True, with_std=True)

    z = pd.DataFrame(
        scaler.fit_transform(expr.values),
        index=expr.index,
        columns=expr.columns
    )

    bp_scores = {}

    for term, genes in gene_sets.items():
        genes2 = [g for g in genes if g in z.columns]

        if len(genes2) >= MIN_GENES_PER_BP:
            bp_scores[term] = z[genes2].mean(axis=1)

    bp_df = pd.DataFrame(bp_scores, index=z.index)

    log(f"BP observable matrix shape: {bp_df.shape}")

    return bp_df


# ============================================================
# 5. D LAYER AND STATISTICAL RELIABILITY
# ============================================================

def cohen_d(x1, x2):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    x1 = x1[np.isfinite(x1)]
    x2 = x2[np.isfinite(x2)]

    if len(x1) < 2 or len(x2) < 2:
        return np.nan

    n1, n2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(n1 + n2 - 2, 1))

    if sp == 0:
        return 0.0

    return (np.mean(x2) - np.mean(x1)) / sp


def auc_from_scores(y_binary, scores):
    try:
        return roc_auc_score(y_binary, scores)
    except Exception:
        return np.nan


def compute_d_layer(bp_df, labels):
    log("Computing D layer: early vs late BP discriminability ...")

    bp_df = collapse_duplicate_samples(bp_df)
    labels = labels.groupby(labels.index).first()

    common = bp_df.index.intersection(labels.index)

    X = bp_df.loc[common]
    y = labels.loc[common]

    early_samples = y[y == EARLY_STAGE_LABEL].index
    late_samples = y[y == LATE_STAGE_LABEL].index

    rows = []
    y_bin = (y == LATE_STAGE_LABEL).astype(int).values

    for term in X.columns:
        early_vals = X.loc[early_samples, term].values
        late_vals = X.loc[late_samples, term].values

        d = cohen_d(early_vals, late_vals)

        try:
            t_p = stats.ttest_ind(late_vals, early_vals, equal_var=False, nan_policy="omit").pvalue
        except Exception:
            t_p = np.nan

        try:
            mw_p = stats.mannwhitneyu(late_vals, early_vals, alternative="two-sided").pvalue
        except Exception:
            mw_p = np.nan

        auc = auc_from_scores(y_bin, X[term].values)

        direction = "late_up" if np.nanmean(late_vals) > np.nanmean(early_vals) else "early_up"

        rows.append({
            "BP_term": term,
            "n_early": len(early_vals),
            "n_late": len(late_vals),
            "mean_early": np.nanmean(early_vals),
            "mean_late": np.nanmean(late_vals),
            "direction": direction,
            "cohen_d_late_minus_early": d,
            "abs_cohen_d": abs(d) if np.isfinite(d) else np.nan,
            "auc_late_vs_early": auc,
            "auc_distance": abs(auc - 0.5) if np.isfinite(auc) else np.nan,
            "welch_p": t_p,
            "mannwhitney_p": mw_p,
        })

    res = pd.DataFrame(rows)

    res["welch_fdr"] = bh_fdr(res["welch_p"].values)
    res["mannwhitney_fdr"] = bh_fdr(res["mannwhitney_p"].values)

    res["D_score"] = (
        res["abs_cohen_d"].fillna(0)
        * (1.0 + 2.0 * res["auc_distance"].fillna(0))
        * (-np.log10(res["welch_fdr"].clip(lower=1e-300)))
    )

    res = res.sort_values(["D_score", "abs_cohen_d"], ascending=False).reset_index(drop=True)

    return res


def bootstrap_stability(bp_df, labels, d_results, n_boot=N_BOOTSTRAP, top_k=50):
    log(f"Bootstrap stability: n_boot={n_boot}, top_k={top_k} ...")

    bp_df = collapse_duplicate_samples(bp_df)
    labels = labels.groupby(labels.index).first()

    common = bp_df.index.intersection(labels.index)

    X = bp_df.loc[common]
    y = labels.loc[common]

    early = y[y == EARLY_STAGE_LABEL].index.values
    late = y[y == LATE_STAGE_LABEL].index.values

    terms = list(X.columns)

    top_counts = Counter()
    direction_counts = Counter()

    original_direction = dict(zip(d_results["BP_term"], d_results["direction"]))

    rng = np.random.default_rng(RANDOM_SEED)

    for _ in range(n_boot):
        bs_early = rng.choice(early, size=len(early), replace=True)
        bs_late = rng.choice(late, size=len(late), replace=True)

        scores = []
        dirs = {}

        for term in terms:
            ev = X.loc[bs_early, term].values
            lv = X.loc[bs_late, term].values

            d = cohen_d(ev, lv)
            direction = "late_up" if np.nanmean(lv) > np.nanmean(ev) else "early_up"

            scores.append((term, abs(d) if np.isfinite(d) else 0))
            dirs[term] = direction

        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

        for term, _score in scores:
            top_counts[term] += 1

            if dirs.get(term) == original_direction.get(term):
                direction_counts[term] += 1

    rows = []

    for term in terms:
        top_stability = top_counts[term] / n_boot
        direction_stability = direction_counts[term] / max(top_counts[term], 1)

        rows.append({
            "BP_term": term,
            "bootstrap_topk_stability": top_stability,
            "bootstrap_direction_stability": direction_stability
        })

    return pd.DataFrame(rows)


def permutation_label_test(bp_df, labels, d_results, n_perm=N_PERMUTATION):
    log(f"Permutation label test: n_perm={n_perm} ...")

    bp_df = collapse_duplicate_samples(bp_df)
    labels = labels.groupby(labels.index).first()

    common = bp_df.index.intersection(labels.index)

    X = bp_df.loc[common]
    y = labels.loc[common].values

    terms = list(X.columns)
    observed = dict(zip(d_results["BP_term"], d_results["abs_cohen_d"]))

    rng = np.random.default_rng(RANDOM_SEED + 1)
    perm_counts = Counter()

    for _ in range(n_perm):
        yp = rng.permutation(y)

        early_idx = np.where(yp == EARLY_STAGE_LABEL)[0]
        late_idx = np.where(yp == LATE_STAGE_LABEL)[0]

        for term in terms:
            vals = X[term].values
            d = abs(cohen_d(vals[early_idx], vals[late_idx]))

            if np.isfinite(d) and d >= observed.get(term, np.inf):
                perm_counts[term] += 1

    rows = []

    for term in terms:
        p = (perm_counts[term] + 1) / (n_perm + 1)
        rows.append({
            "BP_term": term,
            "permutation_p": p
        })

    out = pd.DataFrame(rows)
    out["permutation_fdr"] = bh_fdr(out["permutation_p"].values)

    return out


def random_gene_set_baseline(expr_df, labels, gene_sets, n_random=RANDOM_BASELINE_N):
    log(f"Random gene-set baseline: n_random={n_random} ...")

    genes = np.array(list(expr_df.columns))
    sizes = [len(v) for v in gene_sets.values()]

    if len(sizes) == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(RANDOM_SEED + 2)

    random_sets = {}

    for i in range(n_random):
        size = int(rng.choice(sizes))
        size = min(size, len(genes))

        random_sets[f"RANDOM_SET_{i+1:04d}_N{size}"] = set(
            rng.choice(genes, size=size, replace=False)
        )

    random_bp = construct_bp_observable_matrix(expr_df, random_sets)
    random_d = compute_d_layer(random_bp, labels)
    random_d["baseline_type"] = "random_gene_set"

    return random_d


# ============================================================
# 6. h LAYER
# ============================================================

def extract_gene_columns(df):
    gene_like_cols = []

    for c in df.columns:
        cl = str(c).lower()

        if any(k in cl for k in [
            "gene", "symbol", "feature", "entrez", "hugo", "gene_symbol",
            "genesymbol", "gene name", "gene_name"
        ]):
            gene_like_cols.append(c)

    return gene_like_cols


def load_biomarker_sets(paths):
    log("Loading biomarker / oncogene / tumor suppressor gene references ...")

    all_genes = set()
    oncogenes = set()
    tsg = set()
    source_rows = []

    for path in paths:
        path = Path(path)

        if not path.exists():
            log(f"Missing biomarker file: {path}")
            continue

        try:
            df = read_table_auto(path)
            cols = extract_gene_columns(df)

            for c in cols:
                vals = df[c].dropna().astype(str).tolist()

                for v in vals:
                    parts = re.split(r"[;,/| ]+", v)

                    for p in parts:
                        g = standardize_gene_symbol(p)

                        if g and re.match(r"^[A-Z0-9\-]+$", g) and len(g) <= 20:
                            all_genes.add(g)

            role_cols = [c for c in df.columns if "role" in str(c).lower()]
            gene_cols = cols

            for _, row in df.iterrows():
                genes_here = []

                for gc in gene_cols:
                    g = standardize_gene_symbol(row.get(gc))

                    if g:
                        genes_here.append(g)

                role_text = " ".join([str(row.get(rc, "")) for rc in role_cols]).lower()

                for g in genes_here:
                    if "oncogene" in role_text:
                        oncogenes.add(g)

                    if "tumour suppressor" in role_text or "tumor suppressor" in role_text or "tsg" in role_text:
                        tsg.add(g)

                for c in df.columns:
                    cl = str(c).lower()
                    val = str(row.get(c, "")).lower()

                    if "oncogene" in cl and val in ["yes", "true", "1"]:
                        for g in genes_here:
                            oncogenes.add(g)

                    if ("tumorsuppressor" in cl or "tumour" in cl or "suppressor" in cl) and val in ["yes", "true", "1"]:
                        for g in genes_here:
                            tsg.add(g)

            source_rows.append({
                "path": str(path),
                "rows": df.shape[0],
                "cols": df.shape[1],
                "gene_columns": ";".join(map(str, cols)),
            })

        except Exception as e:
            log(f"Failed biomarker file: {path} | {e}")

    log(f"Biomarker/cancer genes: {len(all_genes)} | oncogenes: {len(oncogenes)} | TSG: {len(tsg)}")

    return {
        "cancer_genes": all_genes,
        "oncogenes": oncogenes,
        "tumor_suppressor_genes": tsg,
        "source_summary": pd.DataFrame(source_rows)
    }


def overlap_stats(gene_set, reference_set, universe):
    gene_set = set(gene_set).intersection(universe)
    reference_set = set(reference_set).intersection(universe)

    a = len(gene_set.intersection(reference_set))
    b = len(gene_set - reference_set)
    c = len(reference_set - gene_set)
    d = len(universe - gene_set - reference_set)

    try:
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    except Exception:
        odds, p = np.nan, np.nan

    return a, odds, p


def compute_h_layer(gene_sets, biomarker_sets, expression_genes):
    log("Computing h layer: biological anchoring ...")

    universe = set(expression_genes)

    refs = {
        "cancer_gene": biomarker_sets["cancer_genes"],
        "oncogene": biomarker_sets["oncogenes"],
        "tumor_suppressor": biomarker_sets["tumor_suppressor_genes"],
    }

    rows = []

    for term, genes in gene_sets.items():
        row = {
            "BP_term": term,
            "bp_gene_count": len(set(genes).intersection(universe))
        }

        for ref_name, ref_genes in refs.items():
            a, odds, p = overlap_stats(genes, ref_genes, universe)

            row[f"{ref_name}_overlap_n"] = a
            row[f"{ref_name}_fisher_odds"] = odds
            row[f"{ref_name}_fisher_p"] = p

            overlap_genes = sorted(set(genes).intersection(ref_genes).intersection(universe))
            row[f"{ref_name}_overlap_genes"] = ";".join(overlap_genes[:100])

        rows.append(row)

    out = pd.DataFrame(rows)

    for ref_name in refs.keys():
        out[f"{ref_name}_fisher_fdr"] = bh_fdr(out[f"{ref_name}_fisher_p"].values)

    out["h_overlap_total"] = (
        out["cancer_gene_overlap_n"].fillna(0)
        + out["oncogene_overlap_n"].fillna(0)
        + out["tumor_suppressor_overlap_n"].fillna(0)
    )

    out["h_best_fdr"] = out[[
        "cancer_gene_fisher_fdr",
        "oncogene_fisher_fdr",
        "tumor_suppressor_fisher_fdr"
    ]].min(axis=1)

    out["h_score"] = (
        out["h_overlap_total"].fillna(0)
        * (-np.log10(out["h_best_fdr"].clip(lower=1e-300)))
    )

    return out.sort_values("h_score", ascending=False).reset_index(drop=True)


# ============================================================
# 7. PPI / NETWORK SUPPORT LAYER
# ============================================================

def load_ppi_edges_fallback(ppi_paths, expression_genes=None):
    edges = []
    universe = set(expression_genes) if expression_genes is not None else None

    omni = ppi_paths.get("omnipath")

    if omni and Path(omni).exists():
        try:
            log(f"Loading OmniPath fallback: {omni}")

            df = read_table_auto(omni)
            source_col = None
            target_col = None

            for c in df.columns:
                cl = str(c).lower()

                if cl in ["source", "source_genesymbol", "genesymbol_source"]:
                    source_col = c

                if cl in ["target", "target_genesymbol", "genesymbol_target"]:
                    target_col = c

            if source_col and target_col:
                for a, b in zip(df[source_col], df[target_col]):
                    g1 = standardize_gene_symbol(a)
                    g2 = standardize_gene_symbol(b)

                    if g1 and g2 and g1 != g2:
                        if universe is None or (g1 in universe and g2 in universe):
                            edges.append(tuple(sorted((g1, g2))))

        except Exception as e:
            log(f"Failed OmniPath fallback: {e}")

    bg = ppi_paths.get("biogrid")

    if bg and Path(bg).exists() and len(edges) == 0:
        try:
            log(f"Loading BioGRID fallback: {bg}")

            df = read_table_auto(bg)
            cols = list(df.columns)

            gene_cols = [
                c for c in cols
                if "Official Symbol" in str(c) or "Systematic Name" in str(c)
            ]

            if len(gene_cols) >= 2:
                c1, c2 = gene_cols[:2]
            else:
                c1, c2 = cols[0], cols[1]

            for a, b in zip(df[c1], df[c2]):
                g1 = standardize_gene_symbol(a)
                g2 = standardize_gene_symbol(b)

                if g1 and g2 and g1 != g2:
                    if universe is None or (g1 in universe and g2 in universe):
                        edges.append(tuple(sorted((g1, g2))))

        except Exception as e:
            log(f"Failed BioGRID fallback: {e}")

    edges = list(set(edges))

    log(f"Loaded fallback PPI edges: {len(edges)}")

    return edges


def load_string_mapping_and_edges(ppi_paths, expression_genes=None, min_score=700, physical=True):
    if nx is None:
        log("networkx not installed. PPI layer will be skipped.")
        return []

    aliases_path = ppi_paths.get("string_aliases")
    info_path = ppi_paths.get("string_info")

    if physical and ppi_paths.get("string_physical_links") and Path(ppi_paths["string_physical_links"]).exists():
        links_path = ppi_paths["string_physical_links"]
    else:
        links_path = ppi_paths.get("string_links")

    if not links_path or not Path(links_path).exists():
        log("STRING links not found. Trying fallback PPI sources.")
        return load_ppi_edges_fallback(ppi_paths, expression_genes)

    log(f"Loading STRING PPI links: {links_path}")

    protein_to_symbol = {}

    if info_path and Path(info_path).exists():
        try:
            info = read_table_auto(info_path)
            cols_lower = {str(c).lower(): c for c in info.columns}

            protein_col = (
                cols_lower.get("#string_protein_id")
                or cols_lower.get("string_protein_id")
                or info.columns[0]
            )

            symbol_col = (
                cols_lower.get("preferred_name")
                or cols_lower.get("protein_external_id")
                or info.columns[1]
            )

            for _, row in info.iterrows():
                pid = str(row[protein_col]).strip()
                sym = standardize_gene_symbol(row[symbol_col])

                if pid and sym:
                    protein_to_symbol[pid] = sym

        except Exception as e:
            log(f"Failed STRING info parse: {e}")

    if len(protein_to_symbol) == 0 and aliases_path and Path(aliases_path).exists():
        try:
            alias = read_table_auto(aliases_path)

            protein_col = alias.columns[0]
            alias_col = alias.columns[1]

            for _, row in alias.iterrows():
                pid = str(row[protein_col]).strip()
                sym = standardize_gene_symbol(row[alias_col])

                if pid and sym:
                    protein_to_symbol[pid] = sym

        except Exception as e:
            log(f"Failed STRING alias parse: {e}")

    if len(protein_to_symbol) == 0:
        log("No STRING protein-to-symbol mapping found. Trying fallback PPI.")
        return load_ppi_edges_fallback(ppi_paths, expression_genes)

    edges = []
    chunksize = 500000

    try:
        for chunk in pd.read_csv(links_path, sep=" ", chunksize=chunksize):
            cols = list(chunk.columns)

            p1_col = "protein1" if "protein1" in cols else cols[0]
            p2_col = "protein2" if "protein2" in cols else cols[1]
            score_col = "combined_score" if "combined_score" in cols else cols[-1]

            chunk[score_col] = pd.to_numeric(chunk[score_col], errors="coerce")
            chunk = chunk[chunk[score_col] >= min_score]

            for p1, p2 in zip(chunk[p1_col], chunk[p2_col]):
                g1 = protein_to_symbol.get(str(p1).strip())
                g2 = protein_to_symbol.get(str(p2).strip())

                if not g1 or not g2 or g1 == g2:
                    continue

                if expression_genes is not None:
                    if g1 not in expression_genes or g2 not in expression_genes:
                        continue

                edges.append(tuple(sorted((g1, g2))))

    except Exception as e:
        log(f"Failed STRING link parse: {e}")
        return load_ppi_edges_fallback(ppi_paths, expression_genes)

    edges = list(set(edges))

    log(f"Loaded STRING edges: {len(edges)}")

    return edges


def compute_ppi_layer(gene_sets, ppi_edges, biomarker_sets):
    log("Computing PPI / network support layer ...")

    if nx is None:
        return pd.DataFrame({"BP_term": list(gene_sets.keys())})

    G = nx.Graph()
    G.add_edges_from(ppi_edges)

    cancer_genes = set(biomarker_sets["cancer_genes"])
    oncogenes = set(biomarker_sets["oncogenes"])
    tsg = set(biomarker_sets["tumor_suppressor_genes"])

    rows = []

    for term, genes in gene_sets.items():
        genes = set(genes)
        present = genes.intersection(G.nodes())

        sub = G.subgraph(present).copy()

        n_nodes = sub.number_of_nodes()
        n_edges = sub.number_of_edges()
        density = nx.density(sub) if n_nodes > 1 else 0

        if n_nodes > 0:
            comps = sorted(nx.connected_components(sub), key=len, reverse=True)
            lcc = comps[0] if comps else set()
            lcc_size = len(lcc)
        else:
            lcc_size = 0

        degrees = dict(sub.degree())
        hub_genes = sorted(degrees, key=degrees.get, reverse=True)[:10]

        hub_cancer = sorted(set(hub_genes).intersection(cancer_genes))
        hub_onco = sorted(set(hub_genes).intersection(oncogenes))
        hub_tsg = sorted(set(hub_genes).intersection(tsg))

        rows.append({
            "BP_term": term,
            "ppi_nodes_present": n_nodes,
            "ppi_edges_within_bp": n_edges,
            "ppi_density": density,
            "ppi_largest_component_size": lcc_size,
            "ppi_largest_component_fraction": lcc_size / n_nodes if n_nodes else 0,
            "ppi_top_hub_genes": ";".join(hub_genes),
            "ppi_hub_cancer_gene_overlap": ";".join(hub_cancer),
            "ppi_hub_oncogene_overlap": ";".join(hub_onco),
            "ppi_hub_tsg_overlap": ";".join(hub_tsg),
            "ppi_hub_cancer_gene_n": len(hub_cancer),
        })

    out = pd.DataFrame(rows)

    out["ppi_score"] = (
        np.log1p(out["ppi_edges_within_bp"].fillna(0))
        + np.log1p(out["ppi_largest_component_size"].fillna(0))
        + out["ppi_hub_cancer_gene_n"].fillna(0)
    )

    return out.sort_values("ppi_score", ascending=False).reset_index(drop=True)


# ============================================================
# 8. CLINICAL CONCORDANCE
# ============================================================

def clinical_concordance_layer(bp_df, labels, d_results, top_n=50):
    log("Computing clinical concordance layer: D-PHY vs D-Clinical ...")

    bp_df = collapse_duplicate_samples(bp_df)
    labels = labels.groupby(labels.index).first()

    common = bp_df.index.intersection(labels.index)

    X_all = bp_df.loc[common].copy()
    y = labels.loc[common].copy()

    if X_all.shape[0] != len(y):
        raise ValueError(
            f"Clinical concordance alignment failed after duplicate collapse: "
            f"X={X_all.shape[0]}, y={len(y)}"
        )

    y_bin = (y == LATE_STAGE_LABEL).astype(int).values

    top_terms = d_results.head(min(top_n, len(d_results)))["BP_term"].tolist()
    top_terms = [t for t in top_terms if t in X_all.columns]

    X_top = X_all[top_terms]

    def cv_model_eval(X, y_bin, label):
        if X.shape[1] < 1 or len(np.unique(y_bin)) < 2:
            return {
                "feature_set": label,
                "n_features": X.shape[1],
                "cv_auc": np.nan,
                "cv_balanced_accuracy": np.nan,
                "cv_accuracy": np.nan
            }

        n_splits = min(5, np.bincount(y_bin).min())

        if n_splits < 2:
            return {
                "feature_set": label,
                "n_features": X.shape[1],
                "cv_auc": np.nan,
                "cv_balanced_accuracy": np.nan,
                "cv_accuracy": np.nan
            }

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                random_state=RANDOM_SEED
            ))
        ])

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

        prob = cross_val_predict(clf, X.values, y_bin, cv=cv, method="predict_proba")[:, 1]
        pred = (prob >= 0.5).astype(int)

        return {
            "feature_set": label,
            "n_features": X.shape[1],
            "cv_auc": roc_auc_score(y_bin, prob),
            "cv_balanced_accuracy": balanced_accuracy_score(y_bin, pred),
            "cv_accuracy": accuracy_score(y_bin, pred)
        }

    rows = []
    rows.append(cv_model_eval(X_all, y_bin, "D-Clinical_all_BP_space"))
    rows.append(cv_model_eval(X_top, y_bin, f"D-PHY_top_{len(top_terms)}_BP_space"))

    out = pd.DataFrame(rows)

    endpoint_summary = {
        "endpoint": "early_vs_late_stage",
        "n_samples": len(y),
        "n_early": int((y == EARLY_STAGE_LABEL).sum()),
        "n_late": int((y == LATE_STAGE_LABEL).sum()),
        "top_DPHY_terms_used": len(top_terms),
        "top_DPHY_median_abs_cohen_d": float(d_results.head(len(top_terms))["abs_cohen_d"].median()) if top_terms else np.nan,
        "top_DPHY_median_FDR": float(d_results.head(len(top_terms))["welch_fdr"].median()) if top_terms else np.nan,
    }

    return out, pd.DataFrame([endpoint_summary])


# ============================================================
# 9. INTERPRETATION READINESS
# ============================================================

def build_final_audit_table(d_results, stability_df, perm_df, h_df, ppi_df):
    log("Building interpretation-readiness profile ...")

    final = d_results.copy()

    for df in [stability_df, perm_df, h_df, ppi_df]:
        if df is not None and len(df) > 0 and "BP_term" in df.columns:
            final = final.merge(df, on="BP_term", how="left")

    final["flag_D_significant"] = final["welch_fdr"] <= FDR_THRESHOLD
    final["flag_effect_moderate"] = final["abs_cohen_d"] >= 0.50
    final["flag_bootstrap_stable"] = final["bootstrap_topk_stability"].fillna(0) >= BOOTSTRAP_STABILITY_THRESHOLD
    final["flag_permutation_supported"] = final["permutation_fdr"].fillna(1) <= FDR_THRESHOLD

    final["flag_biologically_anchored"] = (
        (final["h_best_fdr"].fillna(1) <= H_FDR_THRESHOLD)
        | (final["h_overlap_total"].fillna(0) >= 3)
    )

    if "ppi_edges_within_bp" in final.columns:
        final["flag_network_supported"] = (
            (final["ppi_edges_within_bp"].fillna(0) >= PPI_MIN_EDGES)
            | (final["ppi_largest_component_size"].fillna(0) >= PPI_MIN_LCC_SIZE)
        )
    else:
        final["flag_network_supported"] = False

    bool_cols = [
        "flag_D_significant",
        "flag_effect_moderate",
        "flag_bootstrap_stable",
        "flag_permutation_supported",
        "flag_biologically_anchored",
        "flag_network_supported"
    ]

    final["readiness_support_count"] = final[bool_cols].sum(axis=1)

    def classify(row):
        d_sig = row["flag_D_significant"]
        eff = row["flag_effect_moderate"]
        stat_ok = row["flag_bootstrap_stable"] or row["flag_permutation_supported"]
        bio = row["flag_biologically_anchored"]
        net = row["flag_network_supported"]

        if d_sig and eff and stat_ok and bio and net:
            return "interpretation_ready_strong"

        if d_sig and eff and stat_ok and (bio or net):
            return "interpretation_ready_moderate"

        if d_sig and eff and not stat_ok:
            return "discriminative_but_statistically_unstable"

        if d_sig and stat_ok and not (bio or net):
            return "statistically_supported_but_weakly_anchored"

        if bio and net and not d_sig:
            return "biologically_anchored_but_endpoint_weak"

        return "exploratory_or_weak"

    final["interpretation_readiness_class"] = final.apply(classify, axis=1)

    sort_cols = [
        "readiness_support_count",
        "D_score",
        "h_score",
        "ppi_score"
    ]

    sort_cols = [c for c in sort_cols if c in final.columns]

    final = final.sort_values(sort_cols, ascending=False, na_position="last").reset_index(drop=True)

    return final


# ============================================================
# 10. ONE-CANCER PIPELINE
# ============================================================

def run_one_cancer(cancer_folder, output_root, global_biomarker_sets, global_ppi_edges_cache):
    cancer_folder = Path(cancer_folder)
    cancer_name = cancer_folder.name
    cancer_code = infer_cancer_code(cancer_name)

    log("=" * 80)
    log(f"Running cancer: {cancer_name} | code={cancer_code}")

    out_dir = Path(output_root) / f"D_PHY_internal_{safe_name(cancer_code)}_{safe_name(cancer_name)}"
    ensure_dir(out_dir)

    expr_candidates, clin_candidates, all_files = find_candidate_files(cancer_folder)

    pd.DataFrame({"expression_candidates": expr_candidates}).to_csv(out_dir / "00_expression_candidates.csv", index=False)
    pd.DataFrame({"clinical_candidates": clin_candidates}).to_csv(out_dir / "00_clinical_candidates.csv", index=False)
    pd.DataFrame({"all_files": all_files}).to_csv(out_dir / "00_all_files_detected.csv", index=False)

    if len(expr_candidates) == 0:
        raise ValueError(f"No expression candidates found for {cancer_name}")

    if len(clin_candidates) == 0:
        raise ValueError(f"No clinical candidates found for {cancer_name}")

    expr = None
    expr_path = None
    expr_errors = []

    for f in expr_candidates[:10]:
        try:
            expr = load_expression_matrix(f)

            if expr.shape[0] >= 20 and expr.shape[1] >= 100:
                expr_path = f
                break

        except Exception as e:
            expr_errors.append({"path": f, "error": str(e)})

    pd.DataFrame(expr_errors).to_csv(out_dir / "00_expression_load_errors.csv", index=False)

    if expr is None:
        raise ValueError(f"Could not load a valid expression matrix for {cancer_name}")

    # Collapse duplicated patient/sample IDs before label alignment.
    # This fixes TCGA aliquot-level expression columns that map to the same TCGA patient/sample ID.
    expr_raw_n = expr.shape[0]
    expr = collapse_duplicate_samples(expr)
    expr_collapsed_n = expr.shape[0]

    clinical, clinical_info = load_clinical_labels(clin_candidates)

    if "diagnostics_df" in clinical_info:
        clinical_info["diagnostics_df"].to_csv(out_dir / "00_clinical_label_diagnostics.csv", index=False)

    labels = clinical.set_index("sample_id")["stage_group"]
    labels = labels.groupby(labels.index).first()

    common = expr.index.intersection(labels.index)

    expr = expr.loc[common]
    labels = labels.loc[common]

    if (labels == EARLY_STAGE_LABEL).sum() < 10 or (labels == LATE_STAGE_LABEL).sum() < 10:
        raise ValueError(f"Insufficient early/late samples after alignment for {cancer_name}")

    log(
        f"Aligned expression: {expr.shape}; "
        f"early={(labels == EARLY_STAGE_LABEL).sum()} "
        f"late={(labels == LATE_STAGE_LABEL).sum()}"
    )

    pd.DataFrame({
        "sample_id": labels.index,
        "stage_group": labels.values
    }).to_csv(out_dir / "01_input_endpoint_definitions.csv", index=False)

    pd.DataFrame({
        "cancer_name": [cancer_name],
        "cancer_code": [cancer_code],
        "expression_path": [expr_path],
        "clinical_path": [clinical_info["path"]],
        "stage_column": [clinical_info["stage_col"]],
        "n_expression_rows_before_duplicate_collapse": [expr_raw_n],
        "n_expression_rows_after_duplicate_collapse": [expr_collapsed_n],
        "n_samples_aligned": [len(labels)],
        "n_early": [int((labels == EARLY_STAGE_LABEL).sum())],
        "n_late": [int((labels == LATE_STAGE_LABEL).sum())],
        "n_genes": [expr.shape[1]]
    }).to_csv(out_dir / "00_run_metadata.csv", index=False)

    gmt_path = auto_find_gmt()
    gene_sets = load_gmt(gmt_path, expression_genes=set(expr.columns))

    bp = construct_bp_observable_matrix(expr, gene_sets)
    bp.to_csv(out_dir / "02_BP_observable_matrix.csv")

    d_res = compute_d_layer(bp, labels)
    d_res.to_csv(out_dir / "03_D_layer_ranked_BP_signals.csv", index=False)

    stability = bootstrap_stability(bp, labels, d_res, n_boot=N_BOOTSTRAP, top_k=min(50, len(d_res)))
    stability.to_csv(out_dir / "04A_bootstrap_stability.csv", index=False)

    perm = permutation_label_test(bp, labels, d_res, n_perm=N_PERMUTATION)
    perm.to_csv(out_dir / "04B_permutation_test.csv", index=False)

    random_baseline = random_gene_set_baseline(expr, labels, gene_sets, n_random=RANDOM_BASELINE_N)
    random_baseline.to_csv(out_dir / "04C_random_gene_set_baseline.csv", index=False)

    h_res = compute_h_layer(gene_sets, global_biomarker_sets, expression_genes=set(expr.columns))
    h_res.to_csv(out_dir / "05_h_layer_biological_anchoring.csv", index=False)

    if global_ppi_edges_cache.get("edges") is None:
        global_ppi_edges_cache["edges"] = load_string_mapping_and_edges(
            PPI_PATHS,
            expression_genes=None,
            min_score=700,
            physical=True
        )

    expr_gene_set = set(expr.columns)

    ppi_edges = [
        e for e in global_ppi_edges_cache["edges"]
        if e[0] in expr_gene_set and e[1] in expr_gene_set
    ]

    ppi_res = compute_ppi_layer(gene_sets, ppi_edges, global_biomarker_sets)
    ppi_res.to_csv(out_dir / "06_PPI_network_support_layer.csv", index=False)

    clinical_concordance, endpoint_summary = clinical_concordance_layer(bp, labels, d_res, top_n=50)
    clinical_concordance.to_csv(out_dir / "07A_DPHY_vs_DClinical_concordance.csv", index=False)
    endpoint_summary.to_csv(out_dir / "07B_endpoint_summary.csv", index=False)

    final = build_final_audit_table(d_res, stability, perm, h_res, ppi_res)
    final.to_csv(out_dir / "08_interpretation_readiness_profile.csv", index=False)

    top_cols = [
        "BP_term",
        "interpretation_readiness_class",
        "readiness_support_count",
        "direction",
        "D_score",
        "abs_cohen_d",
        "auc_late_vs_early",
        "welch_fdr",
        "bootstrap_topk_stability",
        "permutation_fdr",
        "h_overlap_total",
        "h_best_fdr",
        "ppi_edges_within_bp",
        "ppi_largest_component_size",
        "ppi_top_hub_genes",
        "cancer_gene_overlap_genes",
        "oncogene_overlap_genes",
        "tumor_suppressor_overlap_genes",
    ]

    existing_top_cols = [c for c in top_cols if c in final.columns]

    final[existing_top_cols].head(100).to_csv(
        out_dir / "TOP100_interpretation_ready_BP_signals.csv",
        index=False
    )

    dclinical_auc = np.nan
    dphy_auc = np.nan

    try:
        dclinical_auc = float(
            clinical_concordance.loc[
                clinical_concordance["feature_set"] == "D-Clinical_all_BP_space",
                "cv_auc"
            ].iloc[0]
        )
    except Exception:
        pass

    try:
        dphy_auc = float(
            clinical_concordance.loc[
                clinical_concordance["feature_set"].str.startswith("D-PHY_top_"),
                "cv_auc"
            ].iloc[0]
        )
    except Exception:
        pass

    summary = {
        "cancer_name": cancer_name,
        "cancer_code": cancer_code,
        "n_samples": int(len(labels)),
        "n_early": int((labels == EARLY_STAGE_LABEL).sum()),
        "n_late": int((labels == LATE_STAGE_LABEL).sum()),
        "n_genes": int(expr.shape[1]),
        "n_bp_terms": int(bp.shape[1]),
        "n_D_significant_FDR05": int((d_res["welch_fdr"] <= FDR_THRESHOLD).sum()),
        "n_interpretation_ready_strong": int((final["interpretation_readiness_class"] == "interpretation_ready_strong").sum()),
        "n_interpretation_ready_moderate": int((final["interpretation_readiness_class"] == "interpretation_ready_moderate").sum()),
        "DClinical_all_BP_auc": dclinical_auc,
        "DPHY_top_BP_auc": dphy_auc,
        "output_dir": str(out_dir)
    }

    with open(out_dir / "SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    excel_path = out_dir / f"D_PHY_internal_{cancer_code}_interpretation_readiness_results.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
        d_res.head(500).to_excel(writer, sheet_name="D_layer_top500", index=False)
        final[existing_top_cols].head(500).to_excel(writer, sheet_name="readiness_top500", index=False)
        h_res.head(500).to_excel(writer, sheet_name="h_layer_top500", index=False)
        ppi_res.head(500).to_excel(writer, sheet_name="ppi_top500", index=False)
        clinical_concordance.to_excel(writer, sheet_name="clinical_concordance", index=False)
        endpoint_summary.to_excel(writer, sheet_name="endpoint_summary", index=False)

    log(f"Completed {cancer_name}. Output: {out_dir}")

    del expr, bp
    gc.collect()

    return summary


# ============================================================
# 11. MAIN
# ============================================================

def main():
    ensure_dir(BASE_OUTPUT_DIR)

    run_root = BASE_OUTPUT_DIR / f"D_PHY_BioSystems_V5_Run_{time.strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(run_root)

    log("=" * 80)
    log("D-PHY internal BioSystems upgrade experiment V5")
    log(f"UCSC XENA root: {ROOT_UCSC_XENA}")
    log(f"Output root: {run_root}")
    log("=" * 80)

    config = {
        "ROOT_UCSC_XENA": str(ROOT_UCSC_XENA),
        "OUTPUT_ROOT": str(run_root),
        "TARGET_CANCERS": TARGET_CANCERS,
        "GENESET_GMT": str(GENESET_GMT),
        "MIN_GENES_PER_BP": MIN_GENES_PER_BP,
        "MAX_GENES_PER_BP": MAX_GENES_PER_BP,
        "MAX_BP_TERMS": MAX_BP_TERMS,
        "N_BOOTSTRAP": N_BOOTSTRAP,
        "N_PERMUTATION": N_PERMUTATION,
        "RANDOM_BASELINE_N": RANDOM_BASELINE_N,
        "FDR_THRESHOLD": FDR_THRESHOLD,
        "BOOTSTRAP_STABILITY_THRESHOLD": BOOTSTRAP_STABILITY_THRESHOLD,
        "H_FDR_THRESHOLD": H_FDR_THRESHOLD,
        "PPI_MIN_EDGES": PPI_MIN_EDGES,
        "PPI_MIN_LCC_SIZE": PPI_MIN_LCC_SIZE,
        "BIOMARKER_PATHS": [str(p) for p in BIOMARKER_PATHS],
        "PPI_PATHS": {k: str(v) for k, v in PPI_PATHS.items()},
    }

    with open(run_root / "CONFIG.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    biomarker_sets = load_biomarker_sets(BIOMARKER_PATHS)

    biomarker_sets["source_summary"].to_csv(
        run_root / "00_biomarker_source_summary.csv",
        index=False
    )

    pd.DataFrame({
        "set_name": ["cancer_genes", "oncogenes", "tumor_suppressor_genes"],
        "n_genes": [
            len(biomarker_sets["cancer_genes"]),
            len(biomarker_sets["oncogenes"]),
            len(biomarker_sets["tumor_suppressor_genes"]),
        ]
    }).to_csv(
        run_root / "00_biomarker_set_sizes.csv",
        index=False
    )

    folders = list_cancer_folders(ROOT_UCSC_XENA)

    if TARGET_CANCERS is not None:
        target_set = set(TARGET_CANCERS)
        folders = [f for f in folders if f.name in target_set]

    if len(folders) == 0:
        raise ValueError("No cancer folders selected/found. Check TARGET_CANCERS and ROOT_UCSC_XENA.")

    pd.DataFrame({
        "selected_cancer_folder": [f.name for f in folders]
    }).to_csv(
        run_root / "00_selected_cancers.csv",
        index=False
    )

    summaries = []
    failures = []
    ppi_cache = {"edges": None}

    for folder in folders:
        try:
            summary = run_one_cancer(folder, run_root, biomarker_sets, ppi_cache)
            summaries.append(summary)

        except Exception as e:
            log(f"FAILED: {folder.name} | {e}")
            failures.append({
                "cancer_folder": folder.name,
                "error": str(e)
            })

    summary_df = pd.DataFrame(summaries)
    failure_df = pd.DataFrame(failures)

    summary_df.to_csv(run_root / "ALL_CANCERS_SUMMARY.csv", index=False)
    failure_df.to_csv(run_root / "ALL_CANCERS_FAILURES.csv", index=False)

    combined_xlsx = run_root / "D_PHY_BioSystems_ALL_SUMMARY.xlsx"

    with pd.ExcelWriter(combined_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="success_summary", index=False)
        failure_df.to_excel(writer, sheet_name="failures", index=False)
        pd.DataFrame([config]).to_excel(writer, sheet_name="config", index=False)

    readme = f"""
D-PHY internal BioSystems upgrade experiment V5

Public-facing title direction:
Assessing biological-process observability and interpretation readiness
in partially observable cancer systems

Main modules:
1. Input data
2. Biological-process observable construction
3. D layer: observability / discriminability
4. Statistical reliability layer
5. h layer: biological anchoring
6. PPI / network support layer
7. Clinical concordance layer
8. Interpretation readiness assessment

Main output:
{run_root}

Key output files:
- ALL_CANCERS_SUMMARY.csv
- ALL_CANCERS_FAILURES.csv
- D_PHY_BioSystems_ALL_SUMMARY.xlsx

Per-cancer output folder includes:
- 00_expression_candidates.csv
- 00_clinical_candidates.csv
- 00_clinical_label_diagnostics.csv
- 00_run_metadata.csv
- 01_input_endpoint_definitions.csv
- 02_BP_observable_matrix.csv
- 03_D_layer_ranked_BP_signals.csv
- 04A_bootstrap_stability.csv
- 04B_permutation_test.csv
- 04C_random_gene_set_baseline.csv
- 05_h_layer_biological_anchoring.csv
- 06_PPI_network_support_layer.csv
- 07A_DPHY_vs_DClinical_concordance.csv
- 07B_endpoint_summary.csv
- 08_interpretation_readiness_profile.csv
- TOP100_interpretation_ready_BP_signals.csv

Interpretation:
- D layer: process-level early-vs-late discriminability.
- Statistical reliability: FDR, bootstrap stability, permutation support, random baseline.
- h layer: biomarker, oncogene, tumor suppressor, and cancer-gene anchoring.
- PPI layer: network coherence and hub-gene support.
- Clinical concordance: D-PHY top BP space vs full BP clinical observability.
- Final profile: interpretation-ready / moderate / unstable / weakly anchored / exploratory.

Important:
- D-PHY-I is internal naming only.
- Public manuscript title should avoid D-PHY-I.
- Audit is a benefit of the framework, not necessarily the title-level main claim.
"""

    with open(run_root / "README_DPHY_RUN.txt", "w", encoding="utf-8") as f:
        f.write(readme)

    log("=" * 80)
    log("RUN COMPLETE")
    log(f"Output: {run_root}")
    log(f"Successful cancers: {len(summary_df)}")
    log(f"Failed cancers: {len(failure_df)}")
    log("=" * 80)



# ============================================================
# 12. EXTERNAL DATA: GSE96058 BRCA cohort
# ============================================================

GSE96058_ROOT = Path("D:/AIDO-Data/External/GSE96058")

# Default expression priority:
# transformed gene-expression file first, then transcript-expression file.
PREFER_GSE96058_TRANSFORMED = True

# Endpoint priority:
# 1. stage-like, if available
# 2. lymph-node positive vs negative
# 3. grade high vs low
# 4. ER positive vs negative
GSE96058_ENDPOINT_MODE = "auto"


def read_gse_table(path, nrows=None):
    """
    Robust CSV/TXT reader for GSE96058 large expression and series matrix files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    lower = path.name.lower()

    if lower.endswith(".csv"):
        sep = ","
    else:
        sep = "\t"

    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False, nrows=nrows, encoding=enc)
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            last_error = e

    for enc in encodings:
        try:
            return pd.read_csv(path, sep=None, engine="python", nrows=nrows, encoding=enc)
        except Exception as e:
            last_error = e

    raise last_error


def find_gse96058_files(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"GSE96058 root not found: {root}")

    all_files = [p for p in root.rglob("*") if p.is_file()]
    expression_candidates = []
    clinical_candidates = []
    annotation_candidates = []

    for p in all_files:
        b = p.name.lower()

        if "gene_expression" in b or "transcript_expression" in b:
            expression_candidates.append(p)
        elif "series_matrix" in b:
            clinical_candidates.append(p)

        if b.endswith(".gtf") or "knowngenes" in b:
            annotation_candidates.append(p)

    def expr_score(p):
        b = p.name.lower()
        score = 0

        if PREFER_GSE96058_TRANSFORMED and "gene_expression" in b and "transformed" in b:
            score += 1000
        if "gene_expression" in b:
            score += 500
        if "transformed" in b:
            score += 100
        if "transcript_expression" in b:
            score += 50

        return score

    def clin_score(p):
        b = p.name.lower()
        score = 0

        if "gse96058" in b:
            score += 100
        if "gpl11154" in b:
            score += 50
        if "series_matrix" in b:
            score += 500

        return score

    expression_candidates = sorted(set(expression_candidates), key=expr_score, reverse=True)
    clinical_candidates = sorted(set(clinical_candidates), key=clin_score, reverse=True)
    annotation_candidates = sorted(set(annotation_candidates))

    return expression_candidates, clinical_candidates, annotation_candidates, all_files


def load_gse96058_expression(path):
    """
    Load GSE96058 expression matrix.

    Supported broad forms:
    A. rows = genes/transcripts, columns = samples
       first column: gene symbol / gene id / transcript id
    B. rows = samples, columns = genes

    Returns:
    expr: rows = samples, columns = gene symbols if possible.
    """
    path = Path(path)
    log(f"Loading GSE96058 expression: {path}")

    df = read_gse_table(path)

    if df.shape[0] < 50 or df.shape[1] < 50:
        raise ValueError(f"GSE96058 expression file too small: {df.shape}")

    first_col = df.columns[0]

    # Identify sample-like columns/values.
    col_names = [str(c) for c in df.columns]
    first_values = df[first_col].astype(str).head(100).tolist()

    # GSE/GSM/SCAN-B sample IDs are often GSM..., GSE..., or numeric/sample identifiers.
    gsm_cols = sum([c.upper().startswith("GSM") for c in col_names[1:200]])
    mb_cols = sum([c.upper().startswith("MB-") for c in col_names[1:200]])
    sample_cols_like = gsm_cols + mb_cols

    # More general: many columns after metadata should be numeric sample columns.
    # If the first column looks like gene names and there are thousands of rows, assume genes x samples.
    first_gene_like = sum([
        bool(re.match(r"^[A-Za-z0-9_.:\-|]+$", v))
        for v in first_values
    ]) > 20

    # Case A: genes/transcripts rows, samples columns.
    if first_gene_like and df.shape[0] > df.shape[1] / 2:
        # Try to find a gene symbol column.
        cols_lower = {str(c).lower(): c for c in df.columns}

        gene_col = (
            cols_lower.get("hugo_symbol")
            or cols_lower.get("hugo symbol")
            or cols_lower.get("gene_symbol")
            or cols_lower.get("gene symbol")
            or cols_lower.get("gene")
            or cols_lower.get("symbol")
            or cols_lower.get("external_gene_name")
            or first_col
        )

        metadata_cols = set()
        for c in df.columns:
            cl = str(c).lower()
            if any(k in cl for k in [
                "gene", "symbol", "ensembl", "entrez", "transcript", "chr",
                "start", "end", "strand", "description", "name"
            ]):
                metadata_cols.add(c)

        sample_cols = [c for c in df.columns if c not in metadata_cols]
        if len(sample_cols) < 20:
            # Fallback: only first column as metadata
            sample_cols = [c for c in df.columns if c != gene_col]

        genes = df[gene_col].map(standardize_gene_symbol)
        mat = df[sample_cols].copy()
        mat.index = genes
        mat = mat[mat.index.notna()]
        mat = mat.apply(pd.to_numeric, errors="coerce")
        mat = mat.groupby(mat.index).mean()

        expr = mat.T
        expr.index = [str(x).strip() for x in expr.index]
        expr = expr[expr.index.notna()]
        expr = expr.loc[:, expr.notna().mean(axis=0) > 0.80]

        if expr.shape[0] < 100 or expr.shape[1] < 1000:
            raise ValueError(f"Parsed GSE96058 expression looks too small as genes x samples: {expr.shape}")

        log(f"GSE96058 expression parsed as genes x samples: samples={expr.shape[0]}, genes={expr.shape[1]}")
        return expr

    # Case B: samples rows, genes columns.
    first_sample_like = (
        sum([v.upper().startswith("GSM") for v in first_values]) > 10
        or sum([v.upper().startswith("MB-") for v in first_values]) > 10
    )

    if first_sample_like or df.shape[0] > 1000:
        samples = df[first_col].astype(str).str.strip()
        mat = df.drop(columns=[first_col])
        mat.columns = [standardize_gene_symbol(c) for c in mat.columns]
        mat.index = samples
        mat = mat.loc[mat.index.notna(), :]
        mat = mat.loc[:, pd.Series(mat.columns).notna().values]
        mat = mat.apply(pd.to_numeric, errors="coerce")
        mat = collapse_duplicate_samples(mat)
        mat = mat.T.groupby(level=0).mean().T
        mat = mat.loc[:, mat.notna().mean(axis=0) > 0.80]

        if mat.shape[0] < 100 or mat.shape[1] < 1000:
            raise ValueError(f"Parsed GSE96058 expression looks too small as samples x genes: {mat.shape}")

        log(f"GSE96058 expression parsed as samples x genes: samples={mat.shape[0]}, genes={mat.shape[1]}")
        return mat

    raise ValueError(f"Could not infer GSE96058 expression orientation: {df.shape}")


def parse_geo_series_matrix(path):
    """
    Parse GEO series_matrix file into a sample-level clinical dataframe.

    It extracts:
    - !Sample_geo_accession
    - !Sample_title
    - !Sample_characteristics_ch1 rows, parsed as key: value
    - other !Sample_* rows as generic columns
    """
    path = Path(path)
    log(f"Parsing GEO series matrix: {path}")

    sample_accessions = None
    rows = []
    characteristics_rows = []

    with open(path, "r", encoding="latin1", errors="ignore") as f:
        for line in f:
            if not line.startswith("!Sample_"):
                continue

            parts = line.rstrip("\n").split("\t")
            key = parts[0].replace("!Sample_", "").strip()
            values = [v.strip().strip('"') for v in parts[1:]]

            if key == "geo_accession":
                sample_accessions = values

            if key.startswith("characteristics_ch1"):
                characteristics_rows.append(values)
            else:
                rows.append((key, values))

    if sample_accessions is None:
        raise ValueError(f"No !Sample_geo_accession found in {path}")

    clinical = pd.DataFrame({"sample_id": sample_accessions})

    # Add generic !Sample_* rows
    for key, values in rows:
        if len(values) == len(sample_accessions):
            col = key.lower()
            col = re.sub(r"[^a-zA-Z0-9_]+", "_", col)
            clinical[col] = values

    # Parse characteristics: key: value
    for values in characteristics_rows:
        if len(values) != len(sample_accessions):
            continue

        parsed_keys = []
        parsed_vals = []

        for v in values:
            if ":" in v:
                k, val = v.split(":", 1)
                k = k.strip().lower()
                val = val.strip()
            else:
                k = "characteristic"
                val = v.strip()

            k = re.sub(r"[^a-zA-Z0-9_]+", "_", k)
            parsed_keys.append(k)
            parsed_vals.append(val)

        # Usually one characteristic row has one same key across samples.
        common_key = Counter(parsed_keys).most_common(1)[0][0]
        if common_key in clinical.columns:
            # Avoid overwriting duplicate characteristic names.
            suffix = 2
            new_key = f"{common_key}_{suffix}"
            while new_key in clinical.columns:
                suffix += 1
                new_key = f"{common_key}_{suffix}"
            common_key = new_key

        clinical[common_key] = parsed_vals

    log(f"GEO clinical parsed: samples={clinical.shape[0]}, columns={clinical.shape[1]}")
    return clinical


def parse_gse_endpoint_value(x, endpoint_type):
    """
    Map clinical values to early/late labels for GSE96058.

    endpoint_type options:
    - stage
    - node
    - grade
    - er
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().upper()
    if s == "" or s in ["NA", "N/A", "NAN", "UNKNOWN", "NOT AVAILABLE", "NULL", "NONE"]:
        return np.nan

    if endpoint_type == "stage":
        return parse_stage_value(s)

    if endpoint_type == "node":
        # Early-like = node negative; late-like = node positive
        if any(k in s for k in ["NEGATIVE", "N0", "NO", "FALSE", "0"]):
            if not any(k in s for k in ["POSITIVE", "N1", "N2", "N3"]):
                return EARLY_STAGE_LABEL
        if any(k in s for k in ["POSITIVE", "N1", "N2", "N3", "YES", "TRUE"]):
            return LATE_STAGE_LABEL

        # Numeric count of positive nodes
        m = re.search(r"[-+]?\d+", s)
        if m:
            val = int(m.group(0))
            return EARLY_STAGE_LABEL if val == 0 else LATE_STAGE_LABEL

    if endpoint_type == "grade":
        # Low/intermediate = grade 1/2; high = grade 3
        m = re.search(r"([1-3])(?:\.0+)?", s)
        if m:
            val = int(m.group(1))
            if val in [1, 2]:
                return EARLY_STAGE_LABEL
            if val == 3:
                return LATE_STAGE_LABEL

        if "LOW" in s or "INTERMEDIATE" in s:
            return EARLY_STAGE_LABEL
        if "HIGH" in s or "POOR" in s:
            return LATE_STAGE_LABEL

    if endpoint_type == "er":
        # This is not progression, but can serve as a strong contrast.
        if "NEG" in s or s in ["0", "FALSE", "NO"]:
            return EARLY_STAGE_LABEL
        if "POS" in s or s in ["1", "TRUE", "YES"]:
            return LATE_STAGE_LABEL

    return np.nan


def choose_gse96058_endpoint(clinical_df):
    """
    Automatically choose a usable endpoint from GSE96058 metadata.

    Priority:
    1. stage
    2. node status
    3. grade
    4. ER status

    Returns:
    labels_df, info dict
    """
    df = clinical_df.copy()
    diagnostics = []

    candidate_specs = []

    for c in df.columns:
        cl = str(c).lower()

        # Stage-like
        if "stage" in cl:
            candidate_specs.append(("stage", c, 1000))

        # Node-like
        if any(k in cl for k in ["node", "lymph"]):
            candidate_specs.append(("node", c, 800))

        # Grade-like
        if "grade" in cl:
            candidate_specs.append(("grade", c, 600))

        # ER-like
        if cl in ["er", "er_status", "estrogen_receptor", "estrogen_receptor_status"] or ("estrogen" in cl and "receptor" in cl):
            candidate_specs.append(("er", c, 300))
        elif cl in ["er_status", "er_status_ihc", "er_ihc"]:
            candidate_specs.append(("er", c, 300))

    # Remove duplicates while keeping max priority.
    best_priority = {}
    for endpoint_type, c, p in candidate_specs:
        key = (endpoint_type, c)
        best_priority[key] = max(best_priority.get(key, 0), p)

    candidate_specs = [(k[0], k[1], p) for k, p in best_priority.items()]
    candidate_specs = sorted(candidate_specs, key=lambda x: x[2], reverse=True)

    best = None

    for endpoint_type, c, priority in candidate_specs:
        out = pd.DataFrame()
        out["sample_id"] = df["sample_id"].astype(str).str.strip()
        out["raw_endpoint"] = df[c]
        out["stage_group"] = df[c].map(lambda x: parse_gse_endpoint_value(x, endpoint_type))
        out = out.dropna(subset=["sample_id", "stage_group"])
        out = out.drop_duplicates("sample_id")

        n_early = int((out["stage_group"] == EARLY_STAGE_LABEL).sum())
        n_late = int((out["stage_group"] == LATE_STAGE_LABEL).sum())
        score = n_early + n_late

        preview = ";".join(df[c].dropna().astype(str).value_counts().head(10).index.tolist())

        diagnostics.append({
            "endpoint_type": endpoint_type,
            "column": c,
            "priority": priority,
            "value_preview": preview,
            "n_early_like": n_early,
            "n_late_like": n_late,
            "score": score
        })

        # Require at least 30 in each group for GSE external.
        if n_early >= 30 and n_late >= 30:
            total_score = priority + score
            if best is None or total_score > best["total_score"]:
                best = {
                    "df": out,
                    "endpoint_type": endpoint_type,
                    "column": c,
                    "n_early": n_early,
                    "n_late": n_late,
                    "score": score,
                    "total_score": total_score,
                    "diagnostics_df": pd.DataFrame(diagnostics)
                }

    if best is None:
        diag_text = pd.DataFrame(diagnostics).to_string(index=False) if diagnostics else "No endpoint candidates."
        raise ValueError(
            "No usable GSE96058 endpoint found. Endpoint diagnostics:\n" + diag_text
        )

    log(
        f"GSE96058 endpoint selected: type={best['endpoint_type']} | "
        f"column={best['column']} | early_like={best['n_early']} late_like={best['n_late']}"
    )

    return best["df"], best



def normalize_external_sample_id(x):
    """
    Normalize external cohort sample IDs for matching.

    Handles common cases:
    - quotes and spaces
    - GEO GSM IDs
    - Excel/Pandas-added X prefix before numeric IDs
    - dots vs hyphens
    """
    if pd.isna(x):
        return None

    s = str(x).strip().strip('"').strip("'")
    if s == "" or s.upper() in ["NA", "NAN", "NONE", "NULL"]:
        return None

    s = s.replace(" ", "")
    s = s.replace(".", "-")
    s = s.upper()

    # Pandas/R may prefix numeric column names with X.
    # Keep GSM/TCGA/MB untouched.
    if s.startswith("X") and len(s) > 1 and re.match(r"^[0-9]", s[1:]):
        s2 = s[1:]
        # Store without X for matching numeric-like sample IDs.
        s = s2

    return s


def normalize_expr_index_for_external(expr):
    expr = pd.DataFrame(expr).copy()
    expr.index = [normalize_external_sample_id(x) for x in expr.index]
    expr = expr[expr.index.notna()]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.groupby(expr.index).mean()
    return expr


def make_gse96058_label_alignment(expr, labels_df, clinical_df, out_dir):
    """
    Find the best ID column in parsed GEO metadata to align expression sample IDs
    with endpoint labels.

    Why needed:
    GSE96058 expression columns may use array/sample IDs, while GEO series_matrix
    labels use GSM accessions. This function tries sample_id, title, source_name,
    and all other metadata columns, then chooses the column with maximum overlap.
    """
    expr = normalize_expr_index_for_external(expr)
    expr_ids = set(expr.index)

    endpoint_by_gsm = labels_df.copy()
    endpoint_by_gsm["sample_id_norm"] = endpoint_by_gsm["sample_id"].map(normalize_external_sample_id)
    endpoint_map = endpoint_by_gsm.set_index("sample_id_norm")["stage_group"].to_dict()

    clinical = clinical_df.copy()

    # Make sure clinical has normalized version of every possible ID column.
    candidate_cols = []

    preferred_terms = [
        "sample_id", "geo_accession", "title", "source_name", "description",
        "scan_b", "scanb", "id", "sample", "patient"
    ]

    for c in clinical.columns:
        cl = str(c).lower()
        if any(t in cl for t in preferred_terms):
            candidate_cols.append(c)

    # Also try all columns because some metadata rows contain the expression sample IDs.
    for c in clinical.columns:
        if c not in candidate_cols:
            candidate_cols.append(c)

    diagnostics = []

    best = None

    for c in candidate_cols:
        try:
            temp = clinical[["sample_id", c]].copy()
            temp["sample_id_norm"] = temp["sample_id"].map(normalize_external_sample_id)
            temp["candidate_norm"] = temp[c].map(normalize_external_sample_id)
            temp["stage_group"] = temp["sample_id_norm"].map(endpoint_map)
            temp = temp.dropna(subset=["candidate_norm", "stage_group"])
            temp = temp.drop_duplicates("candidate_norm")

            overlap = sorted(set(temp["candidate_norm"]).intersection(expr_ids))
            n_overlap = len(overlap)

            n_early = int((temp[temp["candidate_norm"].isin(overlap)]["stage_group"] == EARLY_STAGE_LABEL).sum())
            n_late = int((temp[temp["candidate_norm"].isin(overlap)]["stage_group"] == LATE_STAGE_LABEL).sum())

            diagnostics.append({
                "candidate_column": c,
                "n_overlap_with_expression": n_overlap,
                "n_early_after_alignment": n_early,
                "n_late_after_alignment": n_late,
                "preview_values": ";".join(temp[c].dropna().astype(str).head(5).tolist())
            })

            if n_overlap > 0:
                score = n_overlap + min(n_early, n_late) * 10
                if best is None or score > best["score"]:
                    labels_aligned = temp[temp["candidate_norm"].isin(expr_ids)].copy()
                    labels_series = labels_aligned.set_index("candidate_norm")["stage_group"]
                    labels_series = labels_series.groupby(labels_series.index).first()

                    best = {
                        "column": c,
                        "score": score,
                        "n_overlap": n_overlap,
                        "n_early": n_early,
                        "n_late": n_late,
                        "labels": labels_series
                    }

        except Exception as e:
            diagnostics.append({
                "candidate_column": c,
                "error": str(e)
            })

    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(Path(out_dir) / "00_expression_label_alignment_diagnostics.csv", index=False)

    if best is None:
        raise ValueError(
            "Could not align GSE96058 expression sample IDs to GEO clinical metadata. "
            "See 00_expression_label_alignment_diagnostics.csv."
        )

    log(
        f"GSE96058 expression-label alignment selected column: {best['column']} | "
        f"overlap={best['n_overlap']} early_like={best['n_early']} late_like={best['n_late']}"
    )

    return expr, best["labels"], best



def run_gse96058_external(output_root, global_biomarker_sets, global_ppi_edges_cache):
    cohort_name = "External_GSE96058_BRCA"
    cohort_code = "GSE96058_BRCA"

    log("=" * 80)
    log(f"Running external cohort: {cohort_name}")

    out_dir = Path(output_root) / f"D_PHY_external_{cohort_code}"
    ensure_dir(out_dir)

    expr_candidates, clin_candidates, annotation_candidates, all_files = find_gse96058_files(GSE96058_ROOT)

    pd.DataFrame({"expression_candidates": [str(x) for x in expr_candidates]}).to_csv(out_dir / "00_expression_candidates.csv", index=False)
    pd.DataFrame({"clinical_candidates": [str(x) for x in clin_candidates]}).to_csv(out_dir / "00_clinical_candidates.csv", index=False)
    pd.DataFrame({"annotation_candidates": [str(x) for x in annotation_candidates]}).to_csv(out_dir / "00_annotation_candidates.csv", index=False)
    pd.DataFrame({"all_files": [str(x) for x in all_files]}).to_csv(out_dir / "00_all_files_detected.csv", index=False)

    if len(expr_candidates) == 0:
        raise ValueError(f"No GSE96058 expression candidates found in {GSE96058_ROOT}")

    if len(clin_candidates) == 0:
        raise ValueError(f"No GSE96058 series matrix / clinical candidates found in {GSE96058_ROOT}")

    expr = None
    expr_path = None
    expr_errors = []

    for f in expr_candidates[:5]:
        try:
            expr = load_gse96058_expression(f)
            if expr.shape[0] >= 100 and expr.shape[1] >= 1000:
                expr_path = f
                break
        except Exception as e:
            expr_errors.append({"path": str(f), "error": str(e)})

    pd.DataFrame(expr_errors).to_csv(out_dir / "00_expression_load_errors.csv", index=False)

    if expr is None:
        raise ValueError("Could not load a valid GSE96058 expression matrix.")

    expr_raw_n = expr.shape[0]
    expr = collapse_duplicate_samples(expr)
    expr_collapsed_n = expr.shape[0]

    # Parse all series matrices and combine clinical columns by sample_id.
    clinical_tables = []
    clinical_errors = []

    for f in clin_candidates:
        try:
            cdf = parse_geo_series_matrix(f)
            cdf["source_series_matrix"] = str(f)
            clinical_tables.append(cdf)
        except Exception as e:
            clinical_errors.append({"path": str(f), "error": str(e)})

    pd.DataFrame(clinical_errors).to_csv(out_dir / "00_clinical_parse_errors.csv", index=False)

    if not clinical_tables:
        raise ValueError("No GSE96058 clinical metadata could be parsed from series matrix files.")

    # Prefer the largest clinical table.
    clinical_df = sorted(clinical_tables, key=lambda x: x.shape[0] * x.shape[1], reverse=True)[0]
    clinical_df.to_csv(out_dir / "00_parsed_geo_clinical_metadata.csv", index=False)

    labels_df, endpoint_info = choose_gse96058_endpoint(clinical_df)

    if "diagnostics_df" in endpoint_info:
        endpoint_info["diagnostics_df"].to_csv(out_dir / "00_endpoint_label_diagnostics.csv", index=False)

    # Align expression IDs to GEO clinical metadata.
    # This is necessary because GSE96058 expression columns may use array/sample IDs,
    # while GEO endpoint labels are indexed by GSM IDs.
    expr, labels, alignment_info = make_gse96058_label_alignment(
        expr=expr,
        labels_df=labels_df,
        clinical_df=clinical_df,
        out_dir=out_dir
    )

    common = expr.index.intersection(labels.index)
    expr = expr.loc[common]
    labels = labels.loc[common]

    if (labels == EARLY_STAGE_LABEL).sum() < 30 or (labels == LATE_STAGE_LABEL).sum() < 30:
        raise ValueError(
            f"Insufficient GSE96058 endpoint samples after expression-label alignment: "
            f"early_like={(labels == EARLY_STAGE_LABEL).sum()}, "
            f"late_like={(labels == LATE_STAGE_LABEL).sum()}, common={len(common)}. "
            f"See 00_expression_label_alignment_diagnostics.csv."
        )

    log(
        f"Aligned GSE96058 expression: {expr.shape}; "
        f"early_like={(labels == EARLY_STAGE_LABEL).sum()} "
        f"late_like={(labels == LATE_STAGE_LABEL).sum()}"
    )

    pd.DataFrame({
        "sample_id": labels.index,
        "stage_group": labels.values
    }).to_csv(out_dir / "01_input_endpoint_definitions.csv", index=False)

    pd.DataFrame({
        "cohort_name": [cohort_name],
        "cohort_code": [cohort_code],
        "endpoint_type": [endpoint_info["endpoint_type"]],
        "endpoint_column": [endpoint_info["column"]],
        "alignment_column": [alignment_info["column"]],
        "expression_path": [str(expr_path)],
        "clinical_source": [str(clinical_df["source_series_matrix"].iloc[0]) if "source_series_matrix" in clinical_df.columns else ""],
        "n_expression_rows_before_duplicate_collapse": [expr_raw_n],
        "n_expression_rows_after_duplicate_collapse": [expr_collapsed_n],
        "n_samples_aligned": [len(labels)],
        "n_early_like": [int((labels == EARLY_STAGE_LABEL).sum())],
        "n_late_like": [int((labels == LATE_STAGE_LABEL).sum())],
        "n_genes": [expr.shape[1]]
    }).to_csv(out_dir / "00_run_metadata.csv", index=False)

    # Gene sets
    gmt_path = auto_find_gmt()
    gene_sets = load_gmt(gmt_path, expression_genes=set(expr.columns))

    # BP observable construction
    bp = construct_bp_observable_matrix(expr, gene_sets)
    bp.to_csv(out_dir / "02_BP_observable_matrix.csv")

    # D layer
    d_res = compute_d_layer(bp, labels)
    d_res.to_csv(out_dir / "03_D_layer_ranked_BP_signals.csv", index=False)

    # Statistics
    stability = bootstrap_stability(bp, labels, d_res, n_boot=N_BOOTSTRAP, top_k=min(50, len(d_res)))
    stability.to_csv(out_dir / "04A_bootstrap_stability.csv", index=False)

    perm = permutation_label_test(bp, labels, d_res, n_perm=N_PERMUTATION)
    perm.to_csv(out_dir / "04B_permutation_test.csv", index=False)

    random_baseline = random_gene_set_baseline(expr, labels, gene_sets, n_random=RANDOM_BASELINE_N)
    random_baseline.to_csv(out_dir / "04C_random_gene_set_baseline.csv", index=False)

    # h layer
    h_res = compute_h_layer(gene_sets, global_biomarker_sets, expression_genes=set(expr.columns))
    h_res.to_csv(out_dir / "05_h_layer_biological_anchoring.csv", index=False)

    # PPI
    if global_ppi_edges_cache.get("edges") is None:
        global_ppi_edges_cache["edges"] = load_string_mapping_and_edges(
            PPI_PATHS,
            expression_genes=None,
            min_score=700,
            physical=True
        )

    expr_gene_set = set(expr.columns)

    ppi_edges = [
        e for e in global_ppi_edges_cache["edges"]
        if e[0] in expr_gene_set and e[1] in expr_gene_set
    ]

    ppi_res = compute_ppi_layer(gene_sets, ppi_edges, global_biomarker_sets)
    ppi_res.to_csv(out_dir / "06_PPI_network_support_layer.csv", index=False)

    # Clinical concordance
    clinical_concordance, endpoint_summary = clinical_concordance_layer(bp, labels, d_res, top_n=50)
    clinical_concordance.to_csv(out_dir / "07A_DPHY_vs_DClinical_concordance.csv", index=False)
    endpoint_summary.to_csv(out_dir / "07B_endpoint_summary.csv", index=False)

    # Final readiness profile
    final = build_final_audit_table(d_res, stability, perm, h_res, ppi_res)
    final.to_csv(out_dir / "08_interpretation_readiness_profile.csv", index=False)

    top_cols = [
        "BP_term",
        "interpretation_readiness_class",
        "readiness_support_count",
        "direction",
        "D_score",
        "abs_cohen_d",
        "auc_late_vs_early",
        "welch_fdr",
        "bootstrap_topk_stability",
        "permutation_fdr",
        "h_overlap_total",
        "h_best_fdr",
        "ppi_edges_within_bp",
        "ppi_largest_component_size",
        "ppi_top_hub_genes",
        "cancer_gene_overlap_genes",
        "oncogene_overlap_genes",
        "tumor_suppressor_overlap_genes",
    ]

    existing_top_cols = [c for c in top_cols if c in final.columns]

    final[existing_top_cols].head(100).to_csv(
        out_dir / "TOP100_interpretation_ready_BP_signals.csv",
        index=False
    )

    dclinical_auc = np.nan
    dphy_auc = np.nan

    try:
        dclinical_auc = float(
            clinical_concordance.loc[
                clinical_concordance["feature_set"] == "D-Clinical_all_BP_space",
                "cv_auc"
            ].iloc[0]
        )
    except Exception:
        pass

    try:
        dphy_auc = float(
            clinical_concordance.loc[
                clinical_concordance["feature_set"].str.startswith("D-PHY_top_"),
                "cv_auc"
            ].iloc[0]
        )
    except Exception:
        pass

    summary = {
        "cohort_name": cohort_name,
        "cohort_code": cohort_code,
        "endpoint_type": endpoint_info["endpoint_type"],
        "endpoint_column": endpoint_info["column"],
        "alignment_column": alignment_info["column"],
        "n_samples": int(len(labels)),
        "n_early_like": int((labels == EARLY_STAGE_LABEL).sum()),
        "n_late_like": int((labels == LATE_STAGE_LABEL).sum()),
        "n_genes": int(expr.shape[1]),
        "n_bp_terms": int(bp.shape[1]),
        "n_D_significant_FDR05": int((d_res["welch_fdr"] <= FDR_THRESHOLD).sum()),
        "n_interpretation_ready_strong": int((final["interpretation_readiness_class"] == "interpretation_ready_strong").sum()),
        "n_interpretation_ready_moderate": int((final["interpretation_readiness_class"] == "interpretation_ready_moderate").sum()),
        "DClinical_all_BP_auc": dclinical_auc,
        "DPHY_top_BP_auc": dphy_auc,
        "output_dir": str(out_dir)
    }

    with open(out_dir / "SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    excel_path = out_dir / f"D_PHY_external_{cohort_code}_interpretation_readiness_results.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
        d_res.head(500).to_excel(writer, sheet_name="D_layer_top500", index=False)
        final[existing_top_cols].head(500).to_excel(writer, sheet_name="readiness_top500", index=False)
        h_res.head(500).to_excel(writer, sheet_name="h_layer_top500", index=False)
        ppi_res.head(500).to_excel(writer, sheet_name="ppi_top500", index=False)
        clinical_concordance.to_excel(writer, sheet_name="clinical_concordance", index=False)
        endpoint_summary.to_excel(writer, sheet_name="endpoint_summary", index=False)

    log(f"Completed GSE96058 external cohort. Output: {out_dir}")

    del expr, bp
    gc.collect()

    return summary


def main_gse96058_external():
    ensure_dir(BASE_OUTPUT_DIR)

    run_root = BASE_OUTPUT_DIR / f"D_PHY_GSE96058_External_V2_Run_{time.strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(run_root)

    log("=" * 80)
    log("D-PHY external validation experiment: GSE96058 BRCA V2")
    log(f"GSE96058 root: {GSE96058_ROOT}")
    log(f"Output root: {run_root}")
    log("=" * 80)

    config = {
        "GSE96058_ROOT": str(GSE96058_ROOT),
        "OUTPUT_ROOT": str(run_root),
        "PREFER_GSE96058_TRANSFORMED": PREFER_GSE96058_TRANSFORMED,
        "GSE96058_ENDPOINT_MODE": GSE96058_ENDPOINT_MODE,
        "GENESET_GMT": str(GENESET_GMT),
        "MIN_GENES_PER_BP": MIN_GENES_PER_BP,
        "MAX_GENES_PER_BP": MAX_GENES_PER_BP,
        "MAX_BP_TERMS": MAX_BP_TERMS,
        "N_BOOTSTRAP": N_BOOTSTRAP,
        "N_PERMUTATION": N_PERMUTATION,
        "RANDOM_BASELINE_N": RANDOM_BASELINE_N,
        "FDR_THRESHOLD": FDR_THRESHOLD,
        "BOOTSTRAP_STABILITY_THRESHOLD": BOOTSTRAP_STABILITY_THRESHOLD,
        "H_FDR_THRESHOLD": H_FDR_THRESHOLD,
        "PPI_MIN_EDGES": PPI_MIN_EDGES,
        "PPI_MIN_LCC_SIZE": PPI_MIN_LCC_SIZE,
        "BIOMARKER_PATHS": [str(p) for p in BIOMARKER_PATHS],
        "PPI_PATHS": {k: str(v) for k, v in PPI_PATHS.items()},
    }

    with open(run_root / "CONFIG.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    biomarker_sets = load_biomarker_sets(BIOMARKER_PATHS)

    biomarker_sets["source_summary"].to_csv(
        run_root / "00_biomarker_source_summary.csv",
        index=False
    )

    pd.DataFrame({
        "set_name": ["cancer_genes", "oncogenes", "tumor_suppressor_genes"],
        "n_genes": [
            len(biomarker_sets["cancer_genes"]),
            len(biomarker_sets["oncogenes"]),
            len(biomarker_sets["tumor_suppressor_genes"]),
        ]
    }).to_csv(
        run_root / "00_biomarker_set_sizes.csv",
        index=False
    )

    summaries = []
    failures = []
    ppi_cache = {"edges": None}

    try:
        summary = run_gse96058_external(run_root, biomarker_sets, ppi_cache)
        summaries.append(summary)
    except Exception as e:
        log(f"FAILED GSE96058 external cohort | {e}")
        failures.append({
            "cohort": "GSE96058_BRCA",
            "error": str(e)
        })

    summary_df = pd.DataFrame(summaries)
    failure_df = pd.DataFrame(failures)

    summary_df.to_csv(run_root / "ALL_EXTERNAL_SUMMARY.csv", index=False)
    failure_df.to_csv(run_root / "ALL_EXTERNAL_FAILURES.csv", index=False)

    combined_xlsx = run_root / "D_PHY_GSE96058_EXTERNAL_SUMMARY.xlsx"

    with pd.ExcelWriter(combined_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="success_summary", index=False)
        failure_df.to_excel(writer, sheet_name="failures", index=False)
        pd.DataFrame([config]).to_excel(writer, sheet_name="config", index=False)

    readme = f"""
D-PHY external validation experiment: GSE96058 BRCA V2

Public-facing title direction:
Assessing biological-process observability and interpretation readiness
in partially observable cancer systems

External dataset root:
{GSE96058_ROOT}

Main output:
{run_root}

Key files:
- ALL_EXTERNAL_SUMMARY.csv
- ALL_EXTERNAL_FAILURES.csv
- D_PHY_GSE96058_EXTERNAL_SUMMARY.xlsx

Per-cohort folder includes:
- 00_expression_candidates.csv
- 00_expression_load_errors.csv
- 00_clinical_candidates.csv
- 00_clinical_parse_errors.csv
- 00_endpoint_label_diagnostics.csv
- 00_expression_label_alignment_diagnostics.csv
- 00_parsed_geo_clinical_metadata.csv
- 00_run_metadata.csv
- 01_input_endpoint_definitions.csv
- 02_BP_observable_matrix.csv
- 03_D_layer_ranked_BP_signals.csv
- 04A_bootstrap_stability.csv
- 04B_permutation_test.csv
- 04C_random_gene_set_baseline.csv
- 05_h_layer_biological_anchoring.csv
- 06_PPI_network_support_layer.csv
- 07A_DPHY_vs_DClinical_concordance.csv
- 07B_endpoint_summary.csv
- 08_interpretation_readiness_profile.csv
- TOP100_interpretation_ready_BP_signals.csv

Important:
GSE96058 may not contain AJCC stage in the same way as TCGA/METABRIC.
This script automatically chooses the best available clinical contrast:
stage, lymph-node status, grade, or ER status.
The selected contrast is recorded in 00_run_metadata.csv and ALL_EXTERNAL_SUMMARY.csv.
"""

    with open(run_root / "README_GSE96058_EXTERNAL_RUN.txt", "w", encoding="utf-8") as f:
        f.write(readme)

    log("=" * 80)
    log("GSE96058 EXTERNAL RUN COMPLETE")
    log(f"Output: {run_root}")
    log(f"Successful external cohorts: {len(summary_df)}")
    log(f"Failed external cohorts: {len(failure_df)}")
    log("=" * 80)


if __name__ == "__main__":
    main_gse96058_external()
