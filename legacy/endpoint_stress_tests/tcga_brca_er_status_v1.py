# -*- coding: utf-8 -*-
r"""
D-PHY TCGA-BRCA ER-status endpoint test V1
==========================================

Purpose
-------
Run the same D-PHY / interpretation-readiness pipeline used for the stage
analysis, but replace the endpoint with clinical estrogen-receptor status:
ER-negative versus ER-positive primary breast tumors.

This is an exploratory endpoint-swap test. It preserves the data paths,
biological-process construction, audit layers, and output structure of the
uploaded D-PHY V5 script.

Data sources
------------
- Expression: D:/AIDO-Data/UCSC_XENA/Breast Cancer (BRCA)/GE.tsv, or another
  expression candidate auto-detected in the BRCA folder.
- Clinical endpoint: a UCSC Xena TCGA-BRCA clinical matrix in the same folder.
- GO BP GMT, CIViC, COSMIC, OncoKB, and STRING/PPI: same configured paths as V5.

Endpoint handling
-----------------
Positive and negative ER-status values are mapped conservatively. Equivocal,
borderline, indeterminate, unknown, not-reported, and missing values are
excluded. Candidate clinical columns are evaluated by name specificity and
usable matched sample count.

Outputs
-------
D:/AIDO-Temp/D_PHY_TCGA_BRCA_ER_Status_Test_V1_<timestamp>/
with the same per-cancer output files as the uploaded V5 pipeline.

Note
----
The internal variable names EARLY_STAGE_LABEL and LATE_STAGE_LABEL are retained
only to minimize changes to validated V5 calculation functions. Here they map
to ER_negative and ER_positive, respectively.
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

ENDPOINT_NAME = "ER_status_negative_vs_positive"
ENDPOINT_DISPLAY = "ER-negative versus ER-positive"
EARLY_STAGE_LABEL = "ER_negative"   # internal class 0 alias
LATE_STAGE_LABEL = "ER_positive"    # internal class 1 alias

# Optional direct-marker sensitivity analysis. Keep False for the first run.
EXCLUDE_ESR1_FROM_BP_SCORING = False

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

    # Last fallback: delimiter sniffing
    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                low_memory=False,
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


def parse_er_status_value(x):
    """Map heterogeneous clinical ER-status values to ER_negative/ER_positive."""
    if pd.isna(x):
        return np.nan

    s = str(x).strip().upper()
    if not s:
        return np.nan

    s = s.replace("ESTROGEN RECEPTOR", "ER")
    s = s.replace("RECEPTOR STATUS", "STATUS")
    s = re.sub(r"[\[\]{}()]", " ", s)
    s = re.sub(r"[_/\\]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if any(token in s for token in [
        "EQUIVOCAL", "INDETERMINATE", "BORDERLINE", "UNKNOWN",
        "NOT REPORTED", "NOT AVAILABLE", "NOT APPLICABLE",
        "NOT TESTED", "NOT ASSESSED", "UNDETERMINED", "PENDING"
    ]):
        return np.nan

    if s in {"", "NA", "N A", "NAN", "NONE"}:
        return np.nan

    if re.search(r"\bER\s*\+", s) or re.search(r"\bER\s*POS", s):
        return LATE_STAGE_LABEL
    if re.search(r"\bER\s*[-−]", s) or re.search(r"\bER\s*NEG", s):
        return EARLY_STAGE_LABEL

    positive_values = {
        "POSITIVE", "POS", "PRESENT", "DETECTED", "YES", "Y", "TRUE",
        "1", "1.0", "ER POSITIVE", "ER POS", "ER+", "RECEPTOR POSITIVE"
    }
    negative_values = {
        "NEGATIVE", "NEG", "ABSENT", "NOT DETECTED", "NO", "N", "FALSE",
        "0", "0.0", "ER NEGATIVE", "ER NEG", "ER-", "RECEPTOR NEGATIVE"
    }

    if s in positive_values:
        return LATE_STAGE_LABEL
    if s in negative_values:
        return EARLY_STAGE_LABEL
    if "POSITIVE" in s and "NEGATIVE" not in s:
        return LATE_STAGE_LABEL
    if "NEGATIVE" in s and "POSITIVE" not in s:
        return EARLY_STAGE_LABEL
    return np.nan


def get_er_candidate_columns(clinical_df):
    """Return likely estrogen-receptor status columns, most specific first."""
    candidates = []
    patterns = [
        "breast_carcinoma_estrogen_receptor_status",
        "estrogen_receptor_status",
        "estrogen receptor status",
        "er_status_by_ihc",
        "er_ihc_status",
        "er_status",
        "estrogen_receptor",
        "estrogen receptor"
    ]

    for c in clinical_df.columns:
        cl = str(c).lower().strip()
        if any(p in cl for p in patterns) and "her2" not in cl and "erbb2" not in cl:
            candidates.append(c)

    for c in clinical_df.columns:
        compact = re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")
        if compact in {"er", "er_status", "er_ihc", "er_result", "er_status_ihc"}:
            candidates.append(c)

    def priority(c):
        cl = str(c).lower()
        score = 0
        if "breast_carcinoma_estrogen_receptor_status" in cl:
            score += 1000
        if "estrogen_receptor_status" in cl or "estrogen receptor status" in cl:
            score += 800
        if "ihc" in cl:
            score += 100
        if "status" in cl:
            score += 50
        if "her2" in cl or "erbb2" in cl:
            score -= 2000
        return score

    return sorted(set(candidates), key=priority, reverse=True)


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
            tcga_full = sum(v.upper().replace(".", "-").startswith("TCGA") and len(v) >= 12 for v in vals)
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
    """Select the clinical file/column with the best usable ER-status endpoint."""
    best = None
    diagnostics = []

    for path in paths:
        try:
            df = read_table_auto(path)
            if df.shape[0] < 5 or df.shape[1] < 2:
                diagnostics.append({"path": str(path), "status": "skipped_too_small", "shape": str(df.shape)})
                continue

            sample_col = find_sample_column(df)
            er_candidates = get_er_candidate_columns(df)
            if not er_candidates:
                diagnostics.append({
                    "path": str(path), "status": "no_er_status_columns",
                    "shape": str(df.shape), "sample_col": sample_col,
                    "columns_first_40": ";".join(map(str, df.columns[:40]))
                })
                continue

            for er_col in er_candidates:
                endpoint = pd.DataFrame({
                    "sample_id": df[sample_col].map(standardize_sample_id),
                    "raw_er_status": df[er_col],
                    "endpoint_group": df[er_col].map(parse_er_status_value)
                })
                endpoint = endpoint.dropna(subset=["sample_id", "endpoint_group"]).drop_duplicates("sample_id")

                n_neg = int((endpoint["endpoint_group"] == EARLY_STAGE_LABEL).sum())
                n_pos = int((endpoint["endpoint_group"] == LATE_STAGE_LABEL).sum())
                usable = n_neg + n_pos

                ecl = str(er_col).lower()
                name_priority = 0
                if "breast_carcinoma_estrogen_receptor_status" in ecl:
                    name_priority += 1000
                if "estrogen_receptor_status" in ecl or "estrogen receptor status" in ecl:
                    name_priority += 800
                if "ihc" in ecl:
                    name_priority += 100

                diagnostics.append({
                    "path": str(path), "status": "tested", "shape": str(df.shape),
                    "sample_col": sample_col, "er_status_col": er_col,
                    "n_er_negative": n_neg, "n_er_positive": n_pos,
                    "usable_total": usable, "name_priority": name_priority
                })

                if n_neg >= 10 and n_pos >= 10:
                    score = usable + name_priority
                    if best is None or score > best["score"]:
                        best = {
                            "path": path, "df": endpoint, "er_col": er_col,
                            "sample_col": sample_col, "n_negative": n_neg,
                            "n_positive": n_pos, "score": score
                        }
        except Exception as exc:
            diagnostics.append({"path": str(path), "status": "read_failed", "error": str(exc)})

    if best is None:
        diag_text = pd.DataFrame(diagnostics).to_string(index=False) if diagnostics else "No diagnostics."
        raise ValueError(
            "No usable ER-positive/ER-negative clinical endpoint was found. "
            "Inspect the clinical diagnostics and source columns.\n" + diag_text
        )

    log(f"Clinical selected: {best['path']}")
    log(f"Sample column: {best['sample_col']}")
    log(f"ER-status column: {best['er_col']} | ER-={best['n_negative']} ER+={best['n_positive']}")

    info = dict(best)
    info["diagnostics_df"] = pd.DataFrame(diagnostics)
    return best["df"], info


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
    log("Computing D layer: ER-negative versus ER-positive BP discriminability ...")

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

        direction = "ER_positive_up" if np.nanmean(late_vals) > np.nanmean(early_vals) else "ER_negative_up"

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

    # ER-specific aliases; legacy early/late columns remain for compatibility.
    res["n_er_negative"] = res["n_early"]
    res["n_er_positive"] = res["n_late"]
    res["mean_er_negative"] = res["mean_early"]
    res["mean_er_positive"] = res["mean_late"]
    res["cohen_d_er_positive_minus_negative"] = res["cohen_d_late_minus_early"]
    res["auc_er_positive_vs_negative_raw"] = res["auc_late_vs_early"]
    res["auc_er_positive_vs_negative_oriented"] = res["auc_late_vs_early"].map(
        lambda a: max(a, 1.0 - a) if np.isfinite(a) else np.nan
    )

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
            direction = "ER_positive_up" if np.nanmean(lv) > np.nanmean(ev) else "ER_negative_up"

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
        "endpoint": ENDPOINT_NAME,
        "n_samples": len(y),
        "n_er_negative": int((y == EARLY_STAGE_LABEL).sum()),
        "n_er_positive": int((y == LATE_STAGE_LABEL).sum()),
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

    out_dir = Path(output_root) / f"D_PHY_ER_Status_{safe_name(cancer_code)}_{safe_name(cancer_name)}"
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

    labels = clinical.set_index("sample_id")["endpoint_group"]
    labels = labels.groupby(labels.index).first()

    common = expr.index.intersection(labels.index)

    expr = expr.loc[common]
    labels = labels.loc[common]

    if (labels == EARLY_STAGE_LABEL).sum() < 10 or (labels == LATE_STAGE_LABEL).sum() < 10:
        raise ValueError(f"Insufficient ER-negative/ER-positive samples after alignment for {cancer_name}")

    log(
        f"Aligned expression: {expr.shape}; "
        f"ER-negative={(labels == EARLY_STAGE_LABEL).sum()} "
        f"ER-positive={(labels == LATE_STAGE_LABEL).sum()}"
    )

    pd.DataFrame({
        "sample_id": labels.index,
        "endpoint_name": ENDPOINT_NAME,
        "endpoint_group": labels.values
    }).to_csv(out_dir / "01_input_endpoint_definitions.csv", index=False)

    pd.DataFrame({
        "cancer_name": [cancer_name],
        "cancer_code": [cancer_code],
        "expression_path": [expr_path],
        "clinical_path": [clinical_info["path"]],
        "er_status_column": [clinical_info["er_col"]],
        "n_expression_rows_before_duplicate_collapse": [expr_raw_n],
        "n_expression_rows_after_duplicate_collapse": [expr_collapsed_n],
        "n_samples_aligned": [len(labels)],
        "n_er_negative": [int((labels == EARLY_STAGE_LABEL).sum())],
        "n_er_positive": [int((labels == LATE_STAGE_LABEL).sum())],
        "n_genes": [expr.shape[1]]
    }).to_csv(out_dir / "00_run_metadata.csv", index=False)

    gmt_path = auto_find_gmt()
    gene_sets = load_gmt(gmt_path, expression_genes=set(expr.columns))

    if EXCLUDE_ESR1_FROM_BP_SCORING:
        gene_sets = {
            term: [g for g in genes if g != "ESR1"]
            for term, genes in gene_sets.items()
        }
        gene_sets = {
            term: genes for term, genes in gene_sets.items()
            if MIN_GENES_PER_BP <= len(genes) <= MAX_GENES_PER_BP
        }
        log("ESR1-excluded sensitivity mode enabled.")

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
        "auc_er_positive_vs_negative_oriented",
        "mean_er_negative",
        "mean_er_positive",
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
        "endpoint": ENDPOINT_NAME,
        "class0_label": EARLY_STAGE_LABEL,
        "class1_label": LATE_STAGE_LABEL,
        "n_er_negative": int((labels == EARLY_STAGE_LABEL).sum()),
        "n_er_positive": int((labels == LATE_STAGE_LABEL).sum()),
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

    excel_path = out_dir / f"D_PHY_ER_Status_{cancer_code}_interpretation_readiness_results.xlsx"

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

    run_root = BASE_OUTPUT_DIR / f"D_PHY_TCGA_BRCA_ER_Status_Test_V1_{time.strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(run_root)

    log("=" * 80)
    log("D-PHY TCGA-BRCA ER-status endpoint test V1")
    log(f"UCSC XENA root: {ROOT_UCSC_XENA}")
    log(f"Output root: {run_root}")
    log("=" * 80)

    config = {
        "ROOT_UCSC_XENA": str(ROOT_UCSC_XENA),
        "OUTPUT_ROOT": str(run_root),
        "TARGET_CANCERS": TARGET_CANCERS,
        "ENDPOINT_NAME": ENDPOINT_NAME,
        "CLASS0_LABEL": EARLY_STAGE_LABEL,
        "CLASS1_LABEL": LATE_STAGE_LABEL,
        "EXCLUDE_ESR1_FROM_BP_SCORING": EXCLUDE_ESR1_FROM_BP_SCORING,
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

    combined_xlsx = run_root / "D_PHY_ER_Status_Test_ALL_SUMMARY.xlsx"

    with pd.ExcelWriter(combined_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="success_summary", index=False)
        failure_df.to_excel(writer, sheet_name="failures", index=False)
        pd.DataFrame([config]).to_excel(writer, sheet_name="config", index=False)

    readme = f"""
D-PHY TCGA-BRCA ER-status endpoint test V1

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
- D_PHY_ER_Status_Test_ALL_SUMMARY.xlsx

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
- D layer: process-level ER-negative versus ER-positive discriminability.
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

    with open(run_root / "README_ER_STATUS_TEST.txt", "w", encoding="utf-8") as f:
        f.write(readme)

    log("=" * 80)
    log("RUN COMPLETE")
    log(f"Output: {run_root}")
    log(f"Successful cancers: {len(summary_df)}")
    log(f"Failed cancers: {len(failure_df)}")
    log("=" * 80)


if __name__ == "__main__":
    main()
