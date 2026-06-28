# -*- coding: utf-8 -*-
"""
AIDO D-PHY + D-Clinical BP-Enrichment State Reconstruction V1.0
===============================================================

This is the D-PHY/D-Clinical-only pipeline.

No TTU.
No drug target analysis.
No candidate drug scoring.

Purpose
-------
Build the BP-Enrichment layer from D-PHY + D-Clinical, then test whether the
reconstructed BP-state modules contain clinically useful information.

Core question
-------------
After D-PHY and D-Clinical select BP observables, can BP-Enrichment reconstruct
coherent biological-process state modules that:

    1. distinguish clinically relevant cancer states,
    2. carry local BP deviation evidence,
    3. carry global clinical-state contribution evidence,
    4. are biologically anchored by oncogenes and tumor suppressor genes,
    5. support mechanistic cancer interpretation?

Pipeline
--------
Raw gene expression
    ↓
BP activity matrix
    ↓
D-PHY: local single-BP state deviation
    ↓
D-Clinical: global clinical-state contribution
    ↓
Dual-D BP candidate pool
    ↓
BP-Enrichment / BP-state module reconstruction
    ↓
Reconstructed state activity matrix
    ↓
State reconstruction discriminability audit
    ↓
Oncogene / tumor-suppressor mechanism audit
    ↓
Interpretation-readiness table

Default target
--------------
The default CONFIG is TCGA tumor-vs-normal only as a positive-control.
For the actual paper task, set TARGET_MODE = "custom_binary" and provide
stage/node/survival-risk labels.

Important:
    tumor-vs-normal = positive control
    patient-internal endpoint = main D-PHY task

Outputs
-------
OUT_DIR:
    00_config.json
    01_sample_labels.csv
    02_BP_activity_matrix.csv.gz
    03_DPHY_ranked_BP_signals.csv
    04_DClinical_BP_coefficients.csv
    05_h_layer_BP_oncogene_TSG_anchoring.csv
    06_DualD_BP_candidates.csv
    07_BPenrichment_module_members.csv
    08_BPenrichment_module_summary.csv
    09_reconstructed_state_module_activity.csv.gz
    10_reconstructed_state_discriminability.csv
    11_module_oncogene_TSG_mechanism_audit.csv
    12_interpretation_readiness_audit.csv
    13_RUN_SUMMARY.json
    14_DPHY_DClinical_BPEnrichment_summary.xlsx
    figures/*.png
    README_DPHY_DClinical_BPEnrichment_StateReconstruction.txt
    ZIP package
"""

# =============================================================================
# CONFIG
# =============================================================================

# Expression input
GE_FILE = r"D:/AIDO-Data/UCSC_XENA/Breast Cancer (BRCA)/GE.tsv"

# GO BP / MSigDB BP gene-set file.
# If None, the script searches common D:/AIDO-Data folders.
BP_GENESET_FILE = None

# Output
OUT_DIR = r"D:/AIDO-Temp/DPHY_DClinical_BPEnrichment_StateReconstruction_V1"

# Target mode:
#   "tcga_tumor_vs_normal" = positive-control only
#   "custom_binary"        = main paper mode, e.g. stage early/late, node neg/pos
TARGET_MODE = "tcga_tumor_vs_normal"

# Required for TARGET_MODE = "custom_binary"
LABEL_FILE = None
SAMPLE_COL = None
LABEL_COL = None

# Optional: keep only cancer samples for custom labels if expression file includes normal samples.
# For patient-internal endpoints, this should usually be True.
CUSTOM_MODE_KEEP_TCGA_PRIMARY_TUMOR_ONLY = True

# Biomarker / oncogene / TSG files
CIVIC_ACCEPTED_FILE = r"D:/AIDO-Data/Biomarkers/CIViC/01-May-2026-AcceptedClinicalEvidenceSummaries.tsv"
CIVIC_FEATURE_FILE = r"D:/AIDO-Data/Biomarkers/CIViC/01-May-2026-FeatureSummaries.tsv"
COSMIC_CENSUS_FILE = r"D:/AIDO-Data/Biomarkers/COSMIC/Census_allThu May 28 05_04_17 2026.tsv"
ONCOKB_CANCER_GENE_FILE = r"D:/AIDO-Data/Biomarkers/OncoKB/cancerGeneList.tsv"

# BP gene-set filtering
MIN_GENES_PER_BP = 5
MAX_GENES_PER_BP = 500
MAX_BP_TERMS = None

# D-PHY
TOP_K_DPHY = 300
DPHY_FDR_CUTOFF = 0.05
DPHY_ABS_D_CUTOFF = 0.50
BOOTSTRAP_N = 50
BOOTSTRAP_TOP_K = 100

# D-Clinical
TOP_K_DCLINICAL = 100
CV_SPLITS = 5
CV_REPEATS = 10
LOGISTIC_C = 0.25
MAX_ITER = 3000
RANDOM_STATE = 42

# Dual-D pool for BP-Enrichment
#   "strict"     = Tier1 + top D-PHY + top D-Clinical; recommended
#   "union_all"  = all D-PHY-selected ∪ D-Clinical-selected; can be very large
DUALD_POOL_MODE = "strict"
STRICT_TOP_K_DPHY_FOR_MODULES = 300
STRICT_TOP_K_DCLINICAL_FOR_MODULES = 100
INCLUDE_TIER1_ALWAYS = True

# BP-Enrichment / module reconstruction
MIN_GENE_JACCARD = 0.20
MIN_LEXICAL_SIMILARITY = 0.35
MIN_MODULE_SIZE = 2

# Reconstructed state analysis
MODULE_ACTIVITY_METHOD = "mean_bp_activity"   # current supported: mean_bp_activity
STATE_CLASSIFIER_C = 0.5

# Output controls
SAVE_BP_MATRIX = True
SAVE_EXPRESSION_MATRIX = False
RUN_NOW = True


# =============================================================================
# Imports
# =============================================================================


import json
import math
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import fisher_exact

try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None

from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# =============================================================================
# General utilities
# =============================================================================

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_gene_symbol(x: str) -> str:
    return str(x).strip().upper()


def normalize_bp_name(x: str) -> str:
    return str(x).strip().replace(" ", "_")


def safe_read_table(path: Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, **kwargs)

    if suffix == ".gz":
        if name.endswith(".tsv.gz") or name.endswith(".txt.gz"):
            return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False, **kwargs)
        return pd.read_csv(path, compression="gzip", low_memory=False, **kwargs)

    if suffix in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)

    try:
        df = pd.read_csv(path, low_memory=False, **kwargs)
        if df.shape[1] == 1:
            df2 = pd.read_csv(path, sep="\t", low_memory=False, **kwargs)
            if df2.shape[1] > df.shape[1]:
                return df2
        return df
    except Exception:
        return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = pd.Series(pvals).fillna(1.0).clip(lower=1e-300, upper=1.0).values
    if multipletests is not None:
        return multipletests(p, method="fdr_bh")[1]

    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def cap_neglog10_p(p: float, cap: float = 50.0) -> float:
    try:
        p = float(p)
    except Exception:
        return np.nan
    if not np.isfinite(p) or p <= 0:
        return cap
    return min(-math.log10(max(p, 1e-300)), cap)


def q_rank_score(values: pd.Series, larger_better: bool = True) -> pd.Series:
    """
    0-1 percentile score.
    larger_better=True means larger value receives larger percentile score.
    """
    x = pd.to_numeric(values, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(0.0, index=values.index)
    return x.rank(pct=True, ascending=larger_better).fillna(0.0)


def infer_gene_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "gene", "Gene", "genes", "Genes", "gene_symbol", "Gene Symbol",
        "GENE_SYMBOL", "symbol", "Symbol", "feature", "Feature",
        "Feature Name", "feature_name", "name", "Name", "Hugo Symbol",
        "HUGO_SYMBOL", "Hugo_Symbol"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def zip_dir(src_dir: Path, zip_path: Path) -> Path:
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in src_dir.rglob("*"):
            if f.is_file() and f.resolve() != zip_path.resolve():
                zf.write(f, f.relative_to(src_dir))
    return zip_path


# =============================================================================
# BP gene sets
# =============================================================================

def auto_find_bp_gmt() -> Optional[Path]:
    roots = [
        Path("D:/AIDO-Data"),
        Path("D:/AIDO-Data/MSigDB"),
        Path("D:/AIDO-Data/GeneSets"),
        Path("D:/AIDO-Data/Pathways"),
        Path("D:/AIDO-Data/UCSC_XENA"),
    ]
    patterns = [
        "*c5*go*bp*symbols*.gmt",
        "*C5*GO*BP*symbols*.gmt",
        "*go*bp*.gmt",
        "*GOBP*.gmt",
        "*biological*process*.gmt",
    ]
    hits = []
    for root in roots:
        if root.exists():
            for pat in patterns:
                hits.extend(root.rglob(pat))
    hits = sorted(set(hits))
    if hits:
        print("[INFO] Auto-detected BP gene-set file:", hits[0])
        return hits[0]
    return None


def read_gmt(path: Path) -> Dict[str, Set[str]]:
    gene_sets = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                name = normalize_bp_name(parts[0])
                genes = {normalize_gene_symbol(g) for g in parts[2:] if str(g).strip()}
                if genes:
                    gene_sets[name] = genes
    return gene_sets


def read_gene_set_table(path: Path) -> Dict[str, Set[str]]:
    df = safe_read_table(path)
    lower = {c.lower(): c for c in df.columns}

    bp_col = None
    for c in ["bp_term", "bp", "pathway", "term", "geneset", "gene_set", "name"]:
        if c in lower:
            bp_col = lower[c]
            break
    if bp_col is None:
        bp_col = df.columns[0]

    gene_col = infer_gene_column(df)
    if gene_col is None:
        gene_col = df.columns[1] if len(df.columns) >= 2 else None
    if gene_col is None:
        raise ValueError("Cannot infer gene column from BP gene-set table.")

    out = {}
    for bp, sub in df.groupby(bp_col):
        genes = set()
        for val in sub[gene_col].dropna().astype(str):
            for g in re.split(r"[,;|\s]+", val):
                g = normalize_gene_symbol(g)
                if g:
                    genes.add(g)
        if genes:
            out[normalize_bp_name(bp)] = genes
    return out


def load_bp_gene_sets(path: Optional[str]) -> Tuple[Dict[str, Set[str]], str]:
    if path is None:
        p = auto_find_bp_gmt()
    else:
        p = Path(path)

    if p is None or not p.exists():
        raise FileNotFoundError(
            "Cannot find BP gene-set file. Please set BP_GENESET_FILE to a GO BP GMT file."
        )

    if p.suffix.lower() in [".gmt", ".gmx"]:
        gene_sets = read_gmt(p)
    else:
        gene_sets = read_gene_set_table(p)

    print(f"[INFO] Loaded BP gene sets: {len(gene_sets)}")
    return gene_sets, str(p)


def filter_bp_gene_sets(
    gene_sets: Dict[str, Set[str]],
    genes_available: Set[str],
    min_genes: int,
    max_genes: int,
    max_terms: Optional[int]
) -> Dict[str, List[str]]:
    out = {}
    for bp, genes in gene_sets.items():
        mapped = sorted(set(genes) & genes_available)
        if min_genes <= len(mapped) <= max_genes:
            out[bp] = mapped

    if max_terms is not None and len(out) > max_terms:
        items = sorted(out.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:max_terms]
        out = dict(items)

    print(f"[INFO] BP terms after mapping/filtering: {len(out)}")
    return out


# =============================================================================
# Expression loading and labels
# =============================================================================

def load_expression_matrix(ge_file: str) -> pd.DataFrame:
    path = Path(ge_file)
    print("[INFO] Reading expression file:", path)
    df = safe_read_table(path)

    first = df.columns[0]
    first_values = df[first].astype(str).head(50).tolist()
    col_values = list(map(str, df.columns[1:20]))

    n_first_tcga = sum(v.startswith("TCGA-") for v in first_values)
    n_col_tcga = sum(v.startswith("TCGA-") for v in col_values)

    if n_col_tcga >= max(2, n_first_tcga):
        # genes × samples
        df = df.rename(columns={first: "gene"})
        df["gene"] = df["gene"].astype(str).map(lambda x: x.split("|")[0]).map(normalize_gene_symbol)
        df = df[df["gene"].astype(str).str.len() > 0]
        for c in df.columns:
            if c != "gene":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.groupby("gene", as_index=True).mean(numeric_only=True)
        expr = df.T
        expr.index = expr.index.astype(str)
        expr.columns = [normalize_gene_symbol(c) for c in expr.columns]
    else:
        # samples × genes
        df = df.rename(columns={first: "sample_id"})
        df["sample_id"] = df["sample_id"].astype(str)
        df = df.set_index("sample_id")
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        expr = df
        expr.columns = [normalize_gene_symbol(str(c).split("|")[0]) for c in expr.columns]
        expr = expr.groupby(expr.columns, axis=1).mean(numeric_only=True)

    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=0, how="all").dropna(axis=1, how="all")
    print(f"[INFO] Expression matrix loaded: samples={expr.shape[0]}, genes={expr.shape[1]}")
    return expr


def zscore_genes(expr: pd.DataFrame) -> pd.DataFrame:
    mu = expr.mean(axis=0, skipna=True)
    sd = expr.std(axis=0, skipna=True).replace(0, np.nan)
    return ((expr - mu) / sd).replace([np.inf, -np.inf], np.nan)


def parse_tcga_sample_type(sample_id: str) -> Optional[int]:
    parts = str(sample_id).split("-")
    if len(parts) >= 4:
        code = parts[3][:2]
        if code.isdigit():
            return int(code)
    return None


def make_tcga_tumor_normal_labels(samples: Sequence[str]) -> pd.Series:
    labels = {}
    for sid in samples:
        st = parse_tcga_sample_type(sid)
        if st == 11:
            labels[str(sid)] = 0
        elif st == 1:
            labels[str(sid)] = 1
    y = pd.Series(labels, name="label")
    if len(y) < 20:
        raise ValueError("Too few tumor/normal labels inferred from TCGA barcodes.")
    return y.astype(int)


def keep_tcga_primary_tumor_only(expr: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for sid in expr.index:
        st = parse_tcga_sample_type(sid)
        if st == 1:
            keep.append(sid)
    if len(keep) >= 20:
        print(f"[INFO] Keeping TCGA primary tumor samples only: {len(keep)}")
        return expr.loc[keep].copy()
    print("[WARN] Could not identify enough primary tumor samples. Keeping all samples.")
    return expr


def load_custom_labels(path: str, sample_col: Optional[str], label_col: Optional[str]) -> pd.Series:
    df = safe_read_table(Path(path))

    if sample_col is None:
        for c in ["sample_id", "sample", "Sample", "SAMPLE", "patient_id", "Patient", "id", "ID", "barcode"]:
            if c in df.columns:
                sample_col = c
                break
    if sample_col is None:
        sample_col = df.columns[0]

    if label_col is None:
        for c in ["label", "class", "target", "y", "phenotype", "status", "group", "endpoint", "stage_group", "node_status"]:
            if c in df.columns:
                label_col = c
                break
    if label_col is None:
        for c in df.columns:
            if c == sample_col:
                continue
            vals = df[c].dropna().astype(str).unique()
            if 2 <= len(vals) <= 8:
                label_col = c
                break
    if label_col is None:
        raise ValueError("Cannot infer label column. Set LABEL_COL manually.")

    tmp = df[[sample_col, label_col]].dropna().copy()
    tmp[sample_col] = tmp[sample_col].astype(str)
    raw = tmp[label_col]

    if pd.api.types.is_numeric_dtype(raw):
        ynum = pd.to_numeric(raw, errors="coerce")
        uniq = sorted(ynum.dropna().unique())
        if len(uniq) != 2:
            raise ValueError(f"Numeric label column must have exactly 2 classes. Found: {uniq[:10]}")
        y01 = ynum.map({uniq[0]: 0, uniq[1]: 1})
    else:
        s = raw.astype(str).str.strip().str.lower()
        pos = {"1", "tumor", "tumour", "late", "late_like", "high", "positive", "node_positive",
               "node-positive", "poor", "case", "cancer", "yes", "true", "pcr", "bad",
               "stage_iii_iv", "stage iii/iv", "iii_iv", "advanced"}
        neg = {"0", "normal", "early", "early_like", "low", "negative", "node_negative",
               "node-negative", "good", "control", "no", "false", "rd",
               "stage_i_ii", "stage i/ii", "i_ii", "localized"}
        vals = []
        for v in s:
            if v in pos:
                vals.append(1)
            elif v in neg:
                vals.append(0)
            else:
                vals.append(np.nan)
        y01 = pd.Series(vals, index=tmp.index)

        if y01.notna().sum() < 10:
            uniq = sorted(s.dropna().unique())
            if len(uniq) != 2:
                raise ValueError(f"Cannot convert labels to binary. Unique labels: {uniq[:20]}")
            y01 = s.map({uniq[0]: 0, uniq[1]: 1})

    return pd.Series(y01.values, index=tmp[sample_col].values, name="label").dropna().astype(int)


def align_expr_labels(expr: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    expr = expr.copy()
    y = y.copy()
    expr.index = expr.index.astype(str)
    y.index = y.index.astype(str)

    common = sorted(set(expr.index) & set(y.index))
    if len(common) >= 20:
        return expr.loc[common], y.loc[common].astype(int)

    expr2 = expr.copy()
    y2 = y.copy()
    expr2.index = expr2.index.astype(str).str[:15]
    y2.index = y2.index.astype(str).str[:15]
    common = sorted(set(expr2.index) & set(y2.index))
    if len(common) >= 20:
        return expr2.loc[common], y2.loc[common].astype(int)

    raise ValueError(f"Too few matched expression-label samples: {len(common)}")


# =============================================================================
# BP activity
# =============================================================================

def compute_bp_activity_matrix(zexpr: pd.DataFrame, bp_genes: Dict[str, List[str]]) -> pd.DataFrame:
    data = {}
    total = len(bp_genes)
    for i, (bp, genes) in enumerate(bp_genes.items(), start=1):
        if i % 500 == 0:
            print(f"[INFO] Computing BP activity: {i}/{total}")
        data[bp] = zexpr[genes].mean(axis=1, skipna=True)
    bp_mat = pd.DataFrame(data, index=zexpr.index)
    bp_mat = bp_mat.dropna(axis=1, how="all")
    print(f"[INFO] BP activity matrix: samples={bp_mat.shape[0]}, BP={bp_mat.shape[1]}")
    return bp_mat


# =============================================================================
# D-PHY
# =============================================================================

def cohen_d(x0: np.ndarray, x1: np.ndarray) -> float:
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x0 = x0[np.isfinite(x0)]
    x1 = x1[np.isfinite(x1)]
    n0, n1 = len(x0), len(x1)
    if n0 < 2 or n1 < 2:
        return np.nan
    s0 = np.var(x0, ddof=1)
    s1 = np.var(x1, ddof=1)
    sp = math.sqrt(((n0 - 1) * s0 + (n1 - 1) * s1) / max(n0 + n1 - 2, 1))
    if sp == 0:
        return np.nan
    return (np.mean(x1) - np.mean(x0)) / sp


def safe_auc(y: pd.Series, score: pd.Series) -> float:
    try:
        return roc_auc_score(y, score)
    except Exception:
        return np.nan


def compute_dphy(bp_mat: pd.DataFrame, y: pd.Series, bp_genes: Dict[str, List[str]]) -> pd.DataFrame:
    y = y.loc[bp_mat.index].astype(int)
    group0 = y[y == 0].index
    group1 = y[y == 1].index

    rows = []
    for i, bp in enumerate(bp_mat.columns, start=1):
        if i % 500 == 0:
            print(f"[INFO] Computing D-PHY: {i}/{bp_mat.shape[1]}")

        x0 = pd.to_numeric(bp_mat.loc[group0, bp], errors="coerce")
        x1 = pd.to_numeric(bp_mat.loc[group1, bp], errors="coerce")

        mean0 = float(np.nanmean(x0))
        mean1 = float(np.nanmean(x1))
        delta = mean1 - mean0

        d = cohen_d(x0.values, x1.values)
        try:
            p = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit").pvalue
        except Exception:
            p = np.nan

        auc = safe_auc(y, bp_mat[bp])
        direction = "positive_state_up" if delta > 0 else "negative_or_good_state_up"

        rows.append({
            "BP_term": bp,
            "bp_gene_count": len(bp_genes.get(bp, [])),
            "mean_negative_or_good_state": mean0,
            "mean_positive_or_bad_state": mean1,
            "delta_positive_minus_negative": delta,
            "cohen_d": d,
            "abs_cohen_d": abs(d) if pd.notna(d) else np.nan,
            "auc_positive_vs_negative": auc,
            "auc_distance": abs(auc - 0.5) if pd.notna(auc) else np.nan,
            "welch_p": p,
            "direction": direction,
        })

    df = pd.DataFrame(rows)
    df["welch_fdr"] = bh_fdr(df["welch_p"])
    df["neglog10_fdr_capped50"] = df["welch_fdr"].map(lambda p: cap_neglog10_p(p, cap=50.0))
    df["D_score_capped50"] = (
        df["abs_cohen_d"].fillna(0)
        * (1 + 2 * df["auc_distance"].fillna(0))
        * df["neglog10_fdr_capped50"].fillna(0)
    )

    df["DPHY_selection_score"] = pd.concat([
        q_rank_score(df["D_score_capped50"], True),
        q_rank_score(df["abs_cohen_d"], True),
        q_rank_score(df["auc_distance"], True),
        q_rank_score(df["neglog10_fdr_capped50"], True),
    ], axis=1).mean(axis=1)

    df = df.sort_values("DPHY_selection_score", ascending=False).reset_index(drop=True)
    df["DPHY_rank"] = np.arange(1, len(df) + 1)
    return df


def bootstrap_dphy_stability(bp_mat: pd.DataFrame, y: pd.Series, dphy: pd.DataFrame,
                             n_boot: int, top_k: int, random_state: int) -> pd.DataFrame:
    if n_boot <= 0:
        dphy["bootstrap_topk_stability"] = np.nan
        return dphy

    rng = np.random.default_rng(random_state)
    counts = pd.Series(0.0, index=dphy["BP_term"].astype(str).values)
    idx0 = np.array(y[y == 0].index)
    idx1 = np.array(y[y == 1].index)

    for b in range(n_boot):
        if (b + 1) % 10 == 0:
            print(f"[INFO] Bootstrap D-PHY stability: {b+1}/{n_boot}")

        s0 = rng.choice(idx0, size=len(idx0), replace=True)
        s1 = rng.choice(idx1, size=len(idx1), replace=True)

        scores = []
        for bp in bp_mat.columns:
            x0 = pd.to_numeric(bp_mat.loc[s0, bp], errors="coerce")
            x1 = pd.to_numeric(bp_mat.loc[s1, bp], errors="coerce")
            d = cohen_d(x0.values, x1.values)
            try:
                p = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit").pvalue
            except Exception:
                p = 1.0
            scores.append((bp, abs(d) if pd.notna(d) else 0.0, p if pd.notna(p) else 1.0))

        tmp = pd.DataFrame(scores, columns=["BP_term", "abs_d", "p"])
        tmp["fdr"] = bh_fdr(tmp["p"])
        tmp["score"] = tmp["abs_d"] * tmp["fdr"].map(lambda p: cap_neglog10_p(p, 50.0))
        top = tmp.sort_values("score", ascending=False).head(top_k)["BP_term"]
        counts.loc[top] += 1

    dphy = dphy.copy()
    dphy["bootstrap_topk_stability"] = dphy["BP_term"].map((counts / n_boot).to_dict()).fillna(0.0)
    return dphy


def select_dphy_bp(dphy: pd.DataFrame) -> Set[str]:
    selected = set(dphy.head(TOP_K_DPHY)["BP_term"].astype(str))
    mask = (dphy["welch_fdr"] <= DPHY_FDR_CUTOFF) & (dphy["abs_cohen_d"] >= DPHY_ABS_D_CUTOFF)
    selected |= set(dphy.loc[mask, "BP_term"].astype(str))
    return selected


# =============================================================================
# D-Clinical
# =============================================================================

def compute_dclinical(bp_mat: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, Dict]:
    X = bp_mat.copy()
    y = y.loc[X.index].astype(int)

    nunique = X.nunique(dropna=True)
    X = X[nunique[nunique > 1].index.tolist()]

    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE
    )

    coef_records = []
    aucs = []
    fold_id = 0

    for train_idx, test_idx in cv.split(X, y):
        fold_id += 1
        if fold_id % 10 == 0:
            print(f"[INFO] D-Clinical CV fold: {fold_id}/{CV_SPLITS * CV_REPEATS}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=LOGISTIC_C,
                solver="liblinear",
                class_weight="balanced",
                max_iter=MAX_ITER,
                random_state=RANDOM_STATE + fold_id,
            ))
        ])

        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, prob)
        except Exception:
            auc = np.nan
        aucs.append(auc)

        coef = pipe.named_steps["clf"].coef_.ravel()
        coef_records.append(pd.Series(coef, index=X.columns, name=f"fold_{fold_id}"))

    coef_df = pd.DataFrame(coef_records)
    mean_coef = coef_df.mean(axis=0)
    mean_abs_coef = coef_df.abs().mean(axis=0)
    sd_coef = coef_df.std(axis=0)
    selection_freq = (coef_df.abs() > 1e-8).mean(axis=0)
    sign_consistency = coef_df.apply(lambda col: max((col > 0).mean(), (col < 0).mean()), axis=0)

    out = pd.DataFrame({
        "BP_term": X.columns,
        "DClinical_mean_coef": mean_coef.values,
        "DClinical_mean_abs_coef": mean_abs_coef.values,
        "DClinical_sd_coef": sd_coef.values,
        "DClinical_selection_frequency": selection_freq.values,
        "DClinical_sign_consistency": sign_consistency.values,
    })

    out["DClinical_selection_score"] = (
        q_rank_score(out["DClinical_mean_abs_coef"], True) * 0.55
        + q_rank_score(out["DClinical_selection_frequency"], True) * 0.25
        + q_rank_score(out["DClinical_sign_consistency"], True) * 0.20
    )

    out = out.sort_values("DClinical_selection_score", ascending=False).reset_index(drop=True)
    out["DClinical_rank"] = np.arange(1, len(out) + 1)
    out["DClinical_topK"] = out["DClinical_rank"] <= TOP_K_DCLINICAL

    info = {
        "n_samples": int(X.shape[0]),
        "n_bp_features": int(X.shape[1]),
        "n_positive_or_bad_state": int(y.sum()),
        "n_negative_or_good_state": int((1 - y).sum()),
        "cv_auc_mean": float(np.nanmean(aucs)),
        "cv_auc_sd": float(np.nanstd(aucs)),
        "cv_auc_median": float(np.nanmedian(aucs)),
    }

    return out, info


def select_dclinical_bp(dclinical: pd.DataFrame) -> Set[str]:
    return set(dclinical.head(TOP_K_DCLINICAL)["BP_term"].astype(str))


# =============================================================================
# h-layer oncogene / tumor suppressor anchoring
# =============================================================================

def read_genes_from_any_table(path: Optional[str]) -> Set[str]:
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        print("[WARN] Biomarker file not found:", p)
        return set()

    try:
        df = safe_read_table(p)
    except Exception as e:
        print(f"[WARN] Cannot read biomarker file {p}: {e}")
        return set()

    cols = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["gene", "feature", "symbol", "name"]):
            cols.append(c)
    if not cols:
        cols = list(df.columns[:3])

    genes = set()
    for c in cols:
        for val in df[c].dropna().astype(str):
            for g in re.split(r"[,;|/\s]+", val):
                g = normalize_gene_symbol(g)
                if re.match(r"^[A-Z0-9\-\.]{2,20}$", g):
                    genes.add(g)
    return genes


def load_oncokb_genes(path: Optional[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    if path is None or not Path(path).exists():
        return set(), set(), set()

    df = safe_read_table(Path(path))
    gene_col = infer_gene_column(df) or df.columns[0]
    cancer = {normalize_gene_symbol(x) for x in df[gene_col].dropna().astype(str)}
    onc = set()
    tsg = set()

    for _, row in df.iterrows():
        gene = normalize_gene_symbol(row.get(gene_col, ""))
        if not gene:
            continue
        text = " ".join([str(v).lower() for v in row.values])
        if "oncogene" in text or re.search(r"\bonc\b", text):
            onc.add(gene)
        if "tumor suppressor" in text or "tumour suppressor" in text or "tsg" in text:
            tsg.add(gene)
    return cancer, onc, tsg


def load_cosmic_genes(path: Optional[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    if path is None or not Path(path).exists():
        return set(), set(), set()

    df = safe_read_table(Path(path))
    gene_col = None
    for c in ["Gene Symbol", "GeneSymbol", "gene_symbol", "Gene", "GENE_SYMBOL", "Hugo Symbol", "Hugo_Symbol"]:
        if c in df.columns:
            gene_col = c
            break
    if gene_col is None:
        gene_col = infer_gene_column(df) or df.columns[0]

    cancer = {normalize_gene_symbol(x) for x in df[gene_col].dropna().astype(str)}
    onc = set()
    tsg = set()

    for _, row in df.iterrows():
        gene = normalize_gene_symbol(row.get(gene_col, ""))
        if not gene:
            continue
        text = " ".join([str(v).lower() for v in row.values])
        if "oncogene" in text or re.search(r"\bonc\b", text):
            onc.add(gene)
        if "tsg" in text or "tumor suppressor" in text or "tumour suppressor" in text:
            tsg.add(gene)
    return cancer, onc, tsg


def build_biomarker_reference() -> Tuple[Set[str], Set[str], Set[str]]:
    civic1 = read_genes_from_any_table(CIVIC_ACCEPTED_FILE)
    civic2 = read_genes_from_any_table(CIVIC_FEATURE_FILE)
    cosmic_cancer, cosmic_onc, cosmic_tsg = load_cosmic_genes(COSMIC_CENSUS_FILE)
    oncokb_cancer, oncokb_onc, oncokb_tsg = load_oncokb_genes(ONCOKB_CANCER_GENE_FILE)

    cancer = set().union(civic1, civic2, cosmic_cancer, oncokb_cancer)
    onc = set().union(cosmic_onc, oncokb_onc)
    tsg = set().union(cosmic_tsg, oncokb_tsg)

    print(f"[INFO] Cancer genes: {len(cancer)} | oncogenes: {len(onc)} | TSG: {len(tsg)}")
    return cancer, onc, tsg


def fisher_overlap(bp_genes: Set[str], ref_genes: Set[str], universe: Set[str]) -> Tuple[int, float, float, str]:
    a = len(bp_genes & ref_genes)
    b = len(bp_genes - ref_genes)
    c = len(ref_genes - bp_genes)
    d = max(len(universe - (bp_genes | ref_genes)), 0)

    if a == 0:
        return 0, np.nan, 1.0, ""

    try:
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    except Exception:
        odds, p = np.nan, 1.0

    return a, odds, p, ";".join(sorted(bp_genes & ref_genes))


def compute_h_layer(bp_genes: Dict[str, List[str]], universe: Set[str],
                    cancer: Set[str], onc: Set[str], tsg: Set[str]) -> pd.DataFrame:
    rows = []
    for bp, genes_list in bp_genes.items():
        g = set(genes_list)
        c_n, c_odds, c_p, c_genes = fisher_overlap(g, cancer, universe)
        o_n, o_odds, o_p, o_genes = fisher_overlap(g, onc, universe)
        t_n, t_odds, t_p, t_genes = fisher_overlap(g, tsg, universe)

        rows.append({
            "BP_term": bp,
            "bp_gene_count": len(g),
            "cancer_gene_overlap_n": c_n,
            "cancer_gene_fisher_odds": c_odds,
            "cancer_gene_fisher_p": c_p,
            "cancer_gene_overlap_genes": c_genes,
            "oncogene_overlap_n": o_n,
            "oncogene_fisher_odds": o_odds,
            "oncogene_fisher_p": o_p,
            "oncogene_overlap_genes": o_genes,
            "tumor_suppressor_overlap_n": t_n,
            "tumor_suppressor_fisher_odds": t_odds,
            "tumor_suppressor_fisher_p": t_p,
            "tumor_suppressor_overlap_genes": t_genes,
        })

    h = pd.DataFrame(rows)
    for prefix in ["cancer_gene", "oncogene", "tumor_suppressor"]:
        h[f"{prefix}_fisher_fdr"] = bh_fdr(h[f"{prefix}_fisher_p"])

    h["h_overlap_total"] = (
        h["cancer_gene_overlap_n"].fillna(0)
        + h["oncogene_overlap_n"].fillna(0)
        + h["tumor_suppressor_overlap_n"].fillna(0)
    )
    h["h_best_fdr"] = h[[
        "cancer_gene_fisher_fdr",
        "oncogene_fisher_fdr",
        "tumor_suppressor_fisher_fdr"
    ]].min(axis=1)
    h["h_score"] = h["h_overlap_total"].fillna(0) * h["h_best_fdr"].map(lambda p: cap_neglog10_p(p, 50.0))
    h["flag_biologically_anchored"] = (h["h_overlap_total"] > 0) & (h["h_best_fdr"] <= 0.05)

    return h.sort_values(["h_score", "h_overlap_total"], ascending=False).reset_index(drop=True)


# =============================================================================
# Dual-D
# =============================================================================

def build_dual_d_table(dphy: pd.DataFrame, dclinical: pd.DataFrame, h: pd.DataFrame,
                       dphy_selected: Set[str], dclinical_selected: Set[str]) -> pd.DataFrame:
    all_bp = sorted(set(dphy["BP_term"]) | set(dclinical["BP_term"]))
    base = pd.DataFrame({"BP_term": all_bp})

    dphy_cols = [
        "BP_term", "DPHY_rank", "DPHY_selection_score", "D_score_capped50",
        "abs_cohen_d", "cohen_d", "auc_positive_vs_negative", "auc_distance",
        "welch_p", "welch_fdr", "direction", "mean_negative_or_good_state",
        "mean_positive_or_bad_state", "delta_positive_minus_negative",
        "bootstrap_topk_stability"
    ]
    dphy_cols = [c for c in dphy_cols if c in dphy.columns]
    base = base.merge(dphy[dphy_cols].drop_duplicates("BP_term"), on="BP_term", how="left")

    dc_cols = [
        "BP_term", "DClinical_rank", "DClinical_selection_score",
        "DClinical_mean_coef", "DClinical_mean_abs_coef",
        "DClinical_sd_coef", "DClinical_selection_frequency",
        "DClinical_sign_consistency"
    ]
    base = base.merge(dclinical[dc_cols].drop_duplicates("BP_term"), on="BP_term", how="left")

    h_cols = [
        "BP_term", "h_overlap_total", "h_best_fdr", "h_score",
        "flag_biologically_anchored",
        "cancer_gene_overlap_n", "oncogene_overlap_n", "tumor_suppressor_overlap_n",
        "cancer_gene_overlap_genes", "oncogene_overlap_genes", "tumor_suppressor_overlap_genes",
        "cancer_gene_fisher_fdr", "oncogene_fisher_fdr", "tumor_suppressor_fisher_fdr"
    ]
    h_cols = [c for c in h_cols if c in h.columns]
    base = base.merge(h[h_cols].drop_duplicates("BP_term"), on="BP_term", how="left")

    base["selected_by_DPHY"] = base["BP_term"].isin(dphy_selected)
    base["selected_by_DClinical"] = base["BP_term"].isin(dclinical_selected)
    base["selected_by_dualD_union"] = base["selected_by_DPHY"] | base["selected_by_DClinical"]
    base["selected_by_dualD_intersection"] = base["selected_by_DPHY"] & base["selected_by_DClinical"]

    def tier(row):
        if row["selected_by_DPHY"] and row["selected_by_DClinical"]:
            return "Tier1_core_dual_D"
        if row["selected_by_DPHY"] and not row["selected_by_DClinical"]:
            return "Tier2_DPHY_only_direct_BP_deviation"
        if (not row["selected_by_DPHY"]) and row["selected_by_DClinical"]:
            return "Tier3_DClinical_only_distributed_state_contributor"
        return "Tier4_weak_or_unselected"

    base["DualD_tier"] = base.apply(tier, axis=1)

    def state_class(row):
        direction = str(row.get("direction", "")).lower()
        if "positive_state_up" in direction:
            return "positive_or_bad_state_up"
        if "negative_or_good_state_up" in direction:
            return "negative_or_good_state_up"
        coef = row.get("DClinical_mean_coef", np.nan)
        if pd.notna(coef):
            return "positive_state_contributor" if coef > 0 else "negative_state_contributor"
        return "unknown"

    base["BP_state_direction_class"] = base.apply(state_class, axis=1)

    base["DualD_preliminary_priority_score"] = (
        base["DPHY_selection_score"].fillna(0) * 0.40
        + base["DClinical_selection_score"].fillna(0) * 0.40
        + q_rank_score(base["h_score"], True).fillna(0) * 0.20
    )

    tier_order = {
        "Tier1_core_dual_D": 1,
        "Tier2_DPHY_only_direct_BP_deviation": 2,
        "Tier3_DClinical_only_distributed_state_contributor": 3,
        "Tier4_weak_or_unselected": 4,
    }
    base["tier_order"] = base["DualD_tier"].map(tier_order)
    base = base.sort_values(["tier_order", "DualD_preliminary_priority_score"], ascending=[True, False])
    base = base.drop(columns=["tier_order"]).reset_index(drop=True)
    return base


def get_module_candidate_pool(dual: pd.DataFrame) -> Set[str]:
    if DUALD_POOL_MODE == "union_all":
        return set(dual.loc[dual["selected_by_dualD_union"], "BP_term"].astype(str))

    if DUALD_POOL_MODE != "strict":
        raise ValueError("DUALD_POOL_MODE must be 'strict' or 'union_all'.")

    pool = set()

    if INCLUDE_TIER1_ALWAYS:
        pool |= set(dual.loc[dual["DualD_tier"] == "Tier1_core_dual_D", "BP_term"].astype(str))

    pool |= set(
        dual.sort_values("DPHY_selection_score", ascending=False)
        .head(STRICT_TOP_K_DPHY_FOR_MODULES)["BP_term"]
        .astype(str)
    )

    pool |= set(
        dual.sort_values("DClinical_selection_score", ascending=False)
        .head(STRICT_TOP_K_DCLINICAL_FOR_MODULES)["BP_term"]
        .astype(str)
    )

    return pool


# =============================================================================
# BP-Enrichment / module reconstruction
# =============================================================================

def lexical_tokens(bp: str) -> Set[str]:
    bp = str(bp).upper().replace("GOBP_", "")
    toks = re.split(r"[_\W]+", bp)
    stop = {
        "PROCESS", "REGULATION", "POSITIVE", "NEGATIVE", "BIOLOGICAL",
        "CELLULAR", "CELL", "OF", "TO", "IN", "BY", "AND", "OR",
        "RESPONSE", "INVOLVED"
    }
    return {t for t in toks if len(t) >= 4 and t not in stop}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lexical_similarity(a: str, b: str) -> float:
    return jaccard(lexical_tokens(a), lexical_tokens(b))


def build_similarity_edges(candidates: List[str], bp_genes: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(candidates):
        ga = set(bp_genes.get(a, []))
        for b in candidates[i + 1:]:
            gb = set(bp_genes.get(b, []))
            if ga and gb:
                sim = jaccard(ga, gb)
                sim_type = "gene_jaccard"
                keep = sim >= MIN_GENE_JACCARD
            else:
                sim = lexical_similarity(a, b)
                sim_type = "lexical_fallback"
                keep = sim >= MIN_LEXICAL_SIMILARITY

            if keep:
                rows.append({"BP1": a, "BP2": b, "similarity": sim, "similarity_type": sim_type})
    return pd.DataFrame(rows)


def connected_components(nodes: List[str], edges: pd.DataFrame) -> Dict[str, int]:
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if not edges.empty:
        for _, r in edges.iterrows():
            union(r["BP1"], r["BP2"])

    root_to_id = {}
    comp = {}
    next_id = 1
    for n in nodes:
        r = find(n)
        if r not in root_to_id:
            root_to_id[r] = next_id
            next_id += 1
        comp[n] = root_to_id[r]
    return comp


def module_name_from_bps(bps: List[str]) -> str:
    toks = []
    for bp in bps:
        toks.extend(list(lexical_tokens(bp)))
    if not toks:
        return "miscellaneous_BP_state_module"
    counts = pd.Series(toks).value_counts()
    return "_".join([t.lower() for t in counts.head(4).index]) + "_module"


def reconstruct_modules(dual: pd.DataFrame, bp_genes: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = sorted(get_module_candidate_pool(dual))
    if not candidates:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"[INFO] BP-Enrichment candidate pool size: {len(candidates)} using mode={DUALD_POOL_MODE}")

    edges = build_similarity_edges(candidates, bp_genes)
    comp = connected_components(candidates, edges)

    members = dual[dual["BP_term"].isin(candidates)].copy()
    members["BP_module_id"] = members["BP_term"].map(lambda x: f"M{comp[x]:03d}")

    name_map = {}
    for mid, sub in members.groupby("BP_module_id"):
        top_bps = sub.sort_values("DualD_preliminary_priority_score", ascending=False)["BP_term"].head(8).tolist()
        name_map[mid] = module_name_from_bps(top_bps)

    members["BP_module_name"] = members["BP_module_id"].map(name_map)
    sizes = members["BP_module_id"].value_counts()
    members["BP_module_size"] = members["BP_module_id"].map(sizes)
    members["BP_module_status"] = np.where(
        members["BP_module_size"] >= MIN_MODULE_SIZE,
        "module",
        "singleton_candidate"
    )

    rows = []
    for mid, sub in members.groupby("BP_module_id"):
        tiers = sub["DualD_tier"].value_counts().to_dict()
        dirs = sub["BP_state_direction_class"].value_counts().to_dict()

        module_genes = set()
        for bp in sub["BP_term"]:
            module_genes |= set(bp_genes.get(bp, []))

        row = {
            "BP_module_id": mid,
            "BP_module_name": name_map[mid],
            "BP_module_size": int(len(sub)),
            "module_gene_count": int(len(module_genes)),
            "module_genes": ";".join(sorted(module_genes)),
            "n_Tier1_core_dual_D": int(tiers.get("Tier1_core_dual_D", 0)),
            "n_Tier2_DPHY_only": int(tiers.get("Tier2_DPHY_only_direct_BP_deviation", 0)),
            "n_Tier3_DClinical_only": int(tiers.get("Tier3_DClinical_only_distributed_state_contributor", 0)),
            "mean_DualD_priority_score": float(sub["DualD_preliminary_priority_score"].mean()),
            "max_DualD_priority_score": float(sub["DualD_preliminary_priority_score"].max()),
            "mean_abs_cohen_d": float(pd.to_numeric(sub["abs_cohen_d"], errors="coerce").mean()),
            "mean_DClinical_abs_coef": float(pd.to_numeric(sub["DClinical_mean_abs_coef"], errors="coerce").mean()),
            "dominant_direction_class": max(dirs.items(), key=lambda kv: kv[1])[0] if dirs else "unknown",
            "top_BP_terms": ";".join(
                sub.sort_values("DualD_preliminary_priority_score", ascending=False)["BP_term"].head(10).tolist()
            ),
        }

        if "h_score" in sub.columns:
            row["mean_h_score"] = float(pd.to_numeric(sub["h_score"], errors="coerce").mean())
        if "h_overlap_total" in sub.columns:
            row["n_h_anchored_BP"] = int(pd.to_numeric(sub["h_overlap_total"], errors="coerce").fillna(0).gt(0).sum())
        if "oncogene_overlap_n" in sub.columns:
            row["module_oncogene_BP_count"] = int(pd.to_numeric(sub["oncogene_overlap_n"], errors="coerce").fillna(0).gt(0).sum())
        if "tumor_suppressor_overlap_n" in sub.columns:
            row["module_TSG_BP_count"] = int(pd.to_numeric(sub["tumor_suppressor_overlap_n"], errors="coerce").fillna(0).gt(0).sum())

        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["n_Tier1_core_dual_D", "mean_DualD_priority_score", "BP_module_size"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    return members, summary, edges


# =============================================================================
# Reconstructed state analysis
# =============================================================================

def compute_module_activity(bp_mat: pd.DataFrame, module_members: pd.DataFrame) -> pd.DataFrame:
    if module_members is None or module_members.empty:
        return pd.DataFrame(index=bp_mat.index)

    data = {}
    for mid, sub in module_members.groupby("BP_module_id"):
        bps = [bp for bp in sub["BP_term"].astype(str).tolist() if bp in bp_mat.columns]
        if not bps:
            continue
        data[mid] = bp_mat[bps].mean(axis=1, skipna=True)

    mod_mat = pd.DataFrame(data, index=bp_mat.index)
    return mod_mat.dropna(axis=1, how="all")


def compute_state_discriminability(module_activity: pd.DataFrame, y: pd.Series,
                                   module_summary: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    if module_activity.empty:
        return pd.DataFrame(), {}

    y = y.loc[module_activity.index].astype(int)
    rows = []

    idx0 = y[y == 0].index
    idx1 = y[y == 1].index

    for mid in module_activity.columns:
        x0 = pd.to_numeric(module_activity.loc[idx0, mid], errors="coerce")
        x1 = pd.to_numeric(module_activity.loc[idx1, mid], errors="coerce")
        d = cohen_d(x0.values, x1.values)
        delta = float(np.nanmean(x1) - np.nanmean(x0))
        try:
            p = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit").pvalue
        except Exception:
            p = np.nan
        auc = safe_auc(y, module_activity[mid])

        rows.append({
            "BP_module_id": mid,
            "module_mean_negative_or_good_state": float(np.nanmean(x0)),
            "module_mean_positive_or_bad_state": float(np.nanmean(x1)),
            "module_delta_positive_minus_negative": delta,
            "module_cohen_d": d,
            "module_abs_cohen_d": abs(d) if pd.notna(d) else np.nan,
            "module_auc_positive_vs_negative": auc,
            "module_auc_distance": abs(auc - 0.5) if pd.notna(auc) else np.nan,
            "module_welch_p": p,
            "module_direction": "positive_state_up" if delta > 0 else "negative_or_good_state_up",
        })

    disc = pd.DataFrame(rows)
    disc["module_welch_fdr"] = bh_fdr(disc["module_welch_p"])
    disc["module_neglog10_fdr_capped50"] = disc["module_welch_fdr"].map(lambda p: cap_neglog10_p(p, 50.0))
    disc["module_state_information_score"] = (
        q_rank_score(disc["module_abs_cohen_d"], True) * 0.35
        + q_rank_score(disc["module_auc_distance"], True) * 0.35
        + q_rank_score(disc["module_neglog10_fdr_capped50"], True) * 0.30
    )

    if module_summary is not None and not module_summary.empty:
        keep_cols = [
            "BP_module_id", "BP_module_name", "BP_module_size", "module_gene_count",
            "n_Tier1_core_dual_D", "n_Tier2_DPHY_only", "n_Tier3_DClinical_only",
            "mean_DualD_priority_score", "mean_h_score", "dominant_direction_class",
            "top_BP_terms", "module_genes"
        ]
        keep_cols = [c for c in keep_cols if c in module_summary.columns]
        disc = disc.merge(module_summary[keep_cols], on="BP_module_id", how="left")

    disc = disc.sort_values("module_state_information_score", ascending=False).reset_index(drop=True)

    # Multivariate reconstructed state classifier using module activities
    info = {}
    X = module_activity.copy()
    nunique = X.nunique(dropna=True)
    X = X[nunique[nunique > 1].index.tolist()]

    if X.shape[1] >= 1 and len(np.unique(y)) == 2 and min(y.value_counts()) >= CV_SPLITS:
        cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
        aucs = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l2",
                    C=STATE_CLASSIFIER_C,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=MAX_ITER,
                    random_state=RANDOM_STATE
                ))
            ])
            pipe.fit(X_train, y_train)
            prob = pipe.predict_proba(X_test)[:, 1]
            try:
                aucs.append(roc_auc_score(y_test, prob))
            except Exception:
                aucs.append(np.nan)

        info = {
            "n_reconstructed_modules": int(X.shape[1]),
            "reconstructed_state_cv_auc_mean": float(np.nanmean(aucs)),
            "reconstructed_state_cv_auc_sd": float(np.nanstd(aucs)),
            "reconstructed_state_cv_auc_median": float(np.nanmedian(aucs)),
        }

    return disc, info


# =============================================================================
# Module mechanism audit
# =============================================================================

def collect_genes_from_semicolon(series: pd.Series) -> Set[str]:
    genes = set()
    for val in series.dropna().astype(str):
        for g in val.split(";"):
            g = normalize_gene_symbol(g)
            if g:
                genes.add(g)
    return genes


def build_module_mechanism_audit(module_members: pd.DataFrame, module_summary: pd.DataFrame) -> pd.DataFrame:
    if module_members is None or module_members.empty:
        return pd.DataFrame()

    rows = []
    for mid, sub in module_members.groupby("BP_module_id"):
        oncogenes = collect_genes_from_semicolon(sub.get("oncogene_overlap_genes", pd.Series(dtype=str)))
        tsg = collect_genes_from_semicolon(sub.get("tumor_suppressor_overlap_genes", pd.Series(dtype=str)))
        cancer_genes = collect_genes_from_semicolon(sub.get("cancer_gene_overlap_genes", pd.Series(dtype=str)))

        direction_counts = sub["BP_state_direction_class"].value_counts().to_dict()
        dominant_direction = max(direction_counts.items(), key=lambda kv: kv[1])[0] if direction_counts else "unknown"

        if dominant_direction in ["positive_or_bad_state_up", "positive_state_contributor"]:
            mechanism_hint = "candidate activation/gain-of-cancer-state mechanism"
        elif dominant_direction in ["negative_or_good_state_up", "negative_state_contributor"]:
            mechanism_hint = "candidate loss-of-good-state or suppression mechanism"
        else:
            mechanism_hint = "direction unclear; requires cautious interpretation"

        if len(oncogenes) > 0 and len(tsg) > 0:
            onc_tsg_pattern = "mixed oncogene and tumor-suppressor anchoring"
        elif len(oncogenes) > 0:
            onc_tsg_pattern = "oncogene-anchored"
        elif len(tsg) > 0:
            onc_tsg_pattern = "tumor-suppressor-anchored"
        else:
            onc_tsg_pattern = "no direct oncogene/TSG overlap detected"

        rows.append({
            "BP_module_id": mid,
            "dominant_direction_class_from_members": dominant_direction,
            "mechanism_hint": mechanism_hint,
            "oncogene_TSG_pattern": onc_tsg_pattern,
            "module_cancer_gene_count": len(cancer_genes),
            "module_oncogene_count": len(oncogenes),
            "module_TSG_count": len(tsg),
            "module_cancer_genes": ";".join(sorted(cancer_genes)),
            "module_oncogenes": ";".join(sorted(oncogenes)),
            "module_tumor_suppressor_genes": ";".join(sorted(tsg)),
        })

    audit = pd.DataFrame(rows)

    if module_summary is not None and not module_summary.empty:
        keep_cols = [
            "BP_module_id", "BP_module_name", "BP_module_size", "module_gene_count",
            "n_Tier1_core_dual_D", "mean_DualD_priority_score",
            "dominant_direction_class", "top_BP_terms"
        ]
        keep_cols = [c for c in keep_cols if c in module_summary.columns]
        audit = audit.merge(module_summary[keep_cols], on="BP_module_id", how="left")

    audit["mechanism_anchor_score"] = (
        q_rank_score(audit["module_cancer_gene_count"], True) * 0.30
        + q_rank_score(audit["module_oncogene_count"], True) * 0.35
        + q_rank_score(audit["module_TSG_count"], True) * 0.35
    )
    audit = audit.sort_values("mechanism_anchor_score", ascending=False).reset_index(drop=True)
    return audit


def build_interpretation_readiness(module_summary: pd.DataFrame,
                                   state_disc: pd.DataFrame,
                                   mechanism_audit: pd.DataFrame) -> pd.DataFrame:
    if module_summary is None or module_summary.empty:
        return pd.DataFrame()

    out = module_summary.copy()

    if state_disc is not None and not state_disc.empty:
        cols = [
            "BP_module_id", "module_abs_cohen_d", "module_auc_positive_vs_negative",
            "module_auc_distance", "module_welch_fdr", "module_state_information_score",
            "module_direction"
        ]
        cols = [c for c in cols if c in state_disc.columns]
        out = out.merge(state_disc[cols], on="BP_module_id", how="left")

    if mechanism_audit is not None and not mechanism_audit.empty:
        cols = [
            "BP_module_id", "mechanism_hint", "oncogene_TSG_pattern",
            "module_cancer_gene_count", "module_oncogene_count", "module_TSG_count",
            "module_cancer_genes", "module_oncogenes", "module_tumor_suppressor_genes",
            "mechanism_anchor_score"
        ]
        cols = [c for c in cols if c in mechanism_audit.columns]
        out = out.merge(mechanism_audit[cols], on="BP_module_id", how="left")

    out["readiness_score"] = (
        q_rank_score(out.get("mean_DualD_priority_score", pd.Series(0, index=out.index)), True).fillna(0) * 0.25
        + q_rank_score(out.get("module_state_information_score", pd.Series(0, index=out.index)), True).fillna(0) * 0.35
        + q_rank_score(out.get("mechanism_anchor_score", pd.Series(0, index=out.index)), True).fillna(0) * 0.25
        + q_rank_score(out.get("n_Tier1_core_dual_D", pd.Series(0, index=out.index)), True).fillna(0) * 0.15
    )

    def readiness_class(row):
        if (
            row.get("readiness_score", 0) >= 0.75
            and row.get("module_state_information_score", 0) >= 0.65
            and row.get("mechanism_anchor_score", 0) >= 0.50
        ):
            return "interpretation_ready_high"
        if row.get("readiness_score", 0) >= 0.55 and row.get("module_state_information_score", 0) >= 0.45:
            return "interpretation_ready_moderate"
        if row.get("module_state_information_score", 0) >= 0.50:
            return "state_informative_but_mechanism_weak"
        return "exploratory_or_weak"

    out["interpretation_readiness_class"] = out.apply(readiness_class, axis=1)
    out = out.sort_values("readiness_score", ascending=False).reset_index(drop=True)
    return out


# =============================================================================
# Figures
# =============================================================================

def make_figures(out_dir: Path, dphy: pd.DataFrame, dclinical: pd.DataFrame, dual: pd.DataFrame,
                 module_summary: pd.DataFrame, state_disc: pd.DataFrame,
                 mechanism_audit: pd.DataFrame, dclinical_info: Dict,
                 state_info: Dict) -> None:
    if not HAS_MPL:
        return

    fig_dir = ensure_dir(out_dir / "figures")

    # Top D-PHY
    top = dphy.head(20).copy()
    if not top.empty:
        labels = top["BP_term"].str.replace("GOBP_", "", regex=False).str.replace("_", " ").str[:55]
        plt.figure(figsize=(10, max(5, 0.34 * len(top))))
        yv = np.arange(len(top))
        plt.barh(yv, top["abs_cohen_d"])
        plt.yticks(yv, labels)
        plt.xlabel("|Cohen d|")
        plt.title("Top D-PHY BP deviations")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(fig_dir / "FIG01_top_DPHY_BP_deviations.png", dpi=300)
        plt.close()

    # Top DClinical
    top = dclinical.head(20).copy()
    if not top.empty:
        labels = top["BP_term"].str.replace("GOBP_", "", regex=False).str.replace("_", " ").str[:55]
        plt.figure(figsize=(10, max(5, 0.34 * len(top))))
        yv = np.arange(len(top))
        plt.barh(yv, top["DClinical_mean_abs_coef"])
        plt.yticks(yv, labels)
        plt.xlabel("Mean absolute standardized coefficient")
        plt.title("Top D-Clinical BP contributors")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(fig_dir / "FIG02_top_DClinical_BP_contributors.png", dpi=300)
        plt.close()

    # Dual-D evidence space
    plt.figure(figsize=(7, 6))
    for tier, sub in dual.groupby("DualD_tier"):
        plt.scatter(sub["DPHY_selection_score"], sub["DClinical_selection_score"], s=18, alpha=0.65, label=tier.replace("_", " "))
    plt.xlabel("D-PHY selection score")
    plt.ylabel("D-Clinical selection score")
    plt.title("Dual-D BP evidence space")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "FIG03_DualD_BP_evidence_space.png", dpi=300)
    plt.close()

    # Module state information
    if state_disc is not None and not state_disc.empty:
        top = state_disc.head(20)
        labels = top["BP_module_name"].fillna(top["BP_module_id"]).astype(str)
        plt.figure(figsize=(10, max(5, 0.34 * len(top))))
        yv = np.arange(len(top))
        plt.barh(yv, top["module_state_information_score"])
        plt.yticks(yv, labels)
        plt.xlabel("State information score")
        plt.title("Top reconstructed BP-state modules")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(fig_dir / "FIG04_top_reconstructed_state_modules.png", dpi=300)
        plt.close()

    # Oncogene/TSG mechanism audit
    if mechanism_audit is not None and not mechanism_audit.empty:
        top = mechanism_audit.head(20)
        labels = top["BP_module_name"].fillna(top["BP_module_id"]).astype(str)
        plt.figure(figsize=(10, max(5, 0.34 * len(top))))
        yv = np.arange(len(top))
        plt.barh(yv, top["mechanism_anchor_score"])
        plt.yticks(yv, labels)
        plt.xlabel("Oncogene/TSG mechanism anchor score")
        plt.title("Top mechanism-anchored BP modules")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(fig_dir / "FIG05_top_oncogene_TSG_anchored_modules.png", dpi=300)
        plt.close()

    # AUC comparison
    vals = []
    labs = []
    if dclinical_info:
        vals.append(dclinical_info.get("cv_auc_mean", np.nan))
        labs.append("D-Clinical BP-space")
    if state_info:
        vals.append(state_info.get("reconstructed_state_cv_auc_mean", np.nan))
        labs.append("Reconstructed BP-state")
    if vals:
        plt.figure(figsize=(7, 4))
        plt.bar(labs, vals)
        plt.ylim(0.45, 1.05)
        plt.ylabel("Mean CV AUC")
        plt.title("Clinical discriminability comparison")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "FIG06_DClinical_vs_reconstructed_state_AUC.png", dpi=300)
        plt.close()


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline() -> Path:
    out_dir = ensure_dir(Path(OUT_DIR))

    bp_gene_sets_raw, bp_gene_set_path = load_bp_gene_sets(BP_GENESET_FILE)
    config = {
        "GE_FILE": GE_FILE,
        "BP_GENESET_FILE": bp_gene_set_path,
        "OUT_DIR": OUT_DIR,
        "TARGET_MODE": TARGET_MODE,
        "LABEL_FILE": LABEL_FILE,
        "CUSTOM_MODE_KEEP_TCGA_PRIMARY_TUMOR_ONLY": CUSTOM_MODE_KEEP_TCGA_PRIMARY_TUMOR_ONLY,
        "DUALD_POOL_MODE": DUALD_POOL_MODE,
        "TOP_K_DPHY": TOP_K_DPHY,
        "TOP_K_DCLINICAL": TOP_K_DCLINICAL,
        "STRICT_TOP_K_DPHY_FOR_MODULES": STRICT_TOP_K_DPHY_FOR_MODULES,
        "STRICT_TOP_K_DCLINICAL_FOR_MODULES": STRICT_TOP_K_DCLINICAL_FOR_MODULES,
        "MIN_GENE_JACCARD": MIN_GENE_JACCARD,
        "MIN_LEXICAL_SIMILARITY": MIN_LEXICAL_SIMILARITY,
        "BOOTSTRAP_N": BOOTSTRAP_N,
        "CV_SPLITS": CV_SPLITS,
        "CV_REPEATS": CV_REPEATS,
        "note": "No TTU/drug task. This pipeline is D-PHY + D-Clinical + BP-Enrichment + state reconstruction + oncogene/TSG mechanism audit only."
    }
    (out_dir / "00_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Expression
    expr = load_expression_matrix(GE_FILE)

    # Labels
    if TARGET_MODE == "tcga_tumor_vs_normal":
        y = make_tcga_tumor_normal_labels(expr.index)
    elif TARGET_MODE == "custom_binary":
        if LABEL_FILE is None:
            raise ValueError("LABEL_FILE is required when TARGET_MODE='custom_binary'.")
        if CUSTOM_MODE_KEEP_TCGA_PRIMARY_TUMOR_ONLY:
            expr = keep_tcga_primary_tumor_only(expr)
        y = load_custom_labels(LABEL_FILE, SAMPLE_COL, LABEL_COL)
    else:
        raise ValueError(f"Unknown TARGET_MODE: {TARGET_MODE}")

    expr, y = align_expr_labels(expr, y)
    print(f"[INFO] After alignment: samples={expr.shape[0]}, genes={expr.shape[1]}")
    print(f"[INFO] Label counts: negative/good={int((y == 0).sum())}, positive/bad={int((y == 1).sum())}")

    labels_out = pd.DataFrame({"sample_id": y.index, "label": y.values})
    labels_out["label_name"] = np.where(labels_out["label"] == 1, "positive_or_bad_state", "negative_or_good_state")
    labels_out.to_csv(out_dir / "01_sample_labels.csv", index=False)

    if SAVE_EXPRESSION_MATRIX:
        expr.to_csv(out_dir / "01b_expression_sample_gene_matrix.csv.gz", compression="gzip")

    # BP activity
    zexpr = zscore_genes(expr)
    bp_genes = filter_bp_gene_sets(
        bp_gene_sets_raw,
        genes_available=set(zexpr.columns),
        min_genes=MIN_GENES_PER_BP,
        max_genes=MAX_GENES_PER_BP,
        max_terms=MAX_BP_TERMS
    )
    bp_mat = compute_bp_activity_matrix(zexpr, bp_genes)
    if SAVE_BP_MATRIX:
        bp_mat.to_csv(out_dir / "02_BP_activity_matrix.csv.gz", compression="gzip")

    # D-PHY
    dphy = compute_dphy(bp_mat, y, bp_genes)
    dphy = bootstrap_dphy_stability(bp_mat, y, dphy, BOOTSTRAP_N, BOOTSTRAP_TOP_K, RANDOM_STATE)
    dphy.to_csv(out_dir / "03_DPHY_ranked_BP_signals.csv", index=False)
    dphy_selected = select_dphy_bp(dphy)

    # D-Clinical
    dclinical, dclinical_info = compute_dclinical(bp_mat, y)
    dclinical.to_csv(out_dir / "04_DClinical_BP_coefficients.csv", index=False)
    dclinical_selected = select_dclinical_bp(dclinical)

    # h-layer
    cancer_genes, oncogenes, tsg = build_biomarker_reference()
    h = compute_h_layer(bp_genes, set(zexpr.columns), cancer_genes, oncogenes, tsg)
    h.to_csv(out_dir / "05_h_layer_BP_oncogene_TSG_anchoring.csv", index=False)

    # Dual-D
    dual = build_dual_d_table(dphy, dclinical, h, dphy_selected, dclinical_selected)
    dual.to_csv(out_dir / "06_DualD_BP_candidates.csv", index=False)

    # BP-Enrichment
    module_members, module_summary, module_edges = reconstruct_modules(dual, bp_genes)
    module_members.to_csv(out_dir / "07_BPenrichment_module_members.csv", index=False)
    module_summary.to_csv(out_dir / "08_BPenrichment_module_summary.csv", index=False)
    module_edges.to_csv(out_dir / "08b_BPenrichment_BP_similarity_edges.csv", index=False)

    # Reconstructed state matrix and discriminability
    module_activity = compute_module_activity(bp_mat, module_members)
    module_activity.to_csv(out_dir / "09_reconstructed_state_module_activity.csv.gz", compression="gzip")
    state_disc, state_info = compute_state_discriminability(module_activity, y, module_summary)
    state_disc.to_csv(out_dir / "10_reconstructed_state_discriminability.csv", index=False)

    # Mechanism audit
    mechanism_audit = build_module_mechanism_audit(module_members, module_summary)
    mechanism_audit.to_csv(out_dir / "11_module_oncogene_TSG_mechanism_audit.csv", index=False)

    # Interpretation readiness
    readiness = build_interpretation_readiness(module_summary, state_disc, mechanism_audit)
    readiness.to_csv(out_dir / "12_interpretation_readiness_audit.csv", index=False)

    # Figures
    make_figures(out_dir, dphy, dclinical, dual, module_summary, state_disc, mechanism_audit, dclinical_info, state_info)

    # Summary
    tier_counts = dual.loc[dual["selected_by_dualD_union"], "DualD_tier"].value_counts().to_dict()
    readiness_counts = readiness["interpretation_readiness_class"].value_counts().to_dict() if not readiness.empty else {}
    run_summary = {
        "run_timestamp": now_stamp(),
        "target_mode": TARGET_MODE,
        "n_samples": int(bp_mat.shape[0]),
        "n_negative_or_good": int((y == 0).sum()),
        "n_positive_or_bad": int((y == 1).sum()),
        "n_genes": int(expr.shape[1]),
        "n_bp_terms": int(bp_mat.shape[1]),
        "n_DPHY_selected": int(len(dphy_selected)),
        "n_DClinical_selected": int(len(dclinical_selected)),
        "n_dualD_union": int(dual["selected_by_dualD_union"].sum()),
        "n_dualD_intersection_core": int(dual["selected_by_dualD_intersection"].sum()),
        "dualD_tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "BP_enrichment": {
            "pool_mode": DUALD_POOL_MODE,
            "n_module_candidate_BP": int(module_members["BP_term"].nunique()) if not module_members.empty else 0,
            "n_BP_modules": int(module_summary.shape[0]) if module_summary is not None else 0,
            "n_similarity_edges": int(module_edges.shape[0]) if module_edges is not None else 0,
        },
        "DClinical": dclinical_info,
        "reconstructed_state": state_info,
        "oncogene_TSG_audit": {
            "n_mechanism_audited_modules": int(mechanism_audit.shape[0]) if mechanism_audit is not None else 0,
            "n_oncogene_anchored_modules": int((mechanism_audit["module_oncogene_count"] > 0).sum()) if not mechanism_audit.empty else 0,
            "n_TSG_anchored_modules": int((mechanism_audit["module_TSG_count"] > 0).sum()) if not mechanism_audit.empty else 0,
        },
        "interpretation_readiness_counts": {str(k): int(v) for k, v in readiness_counts.items()},
        "interpretation": (
            "This run excludes TTU/drug tasks. It tests whether D-PHY + D-Clinical selected BP-Enrichment "
            "modules reconstruct informative biological-process states, and whether oncogene/tumor-suppressor "
            "anchoring can support mechanism interpretation."
        )
    }
    (out_dir / "13_RUN_SUMMARY.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    # Excel
    try:
        with pd.ExcelWriter(out_dir / "14_DPHY_DClinical_BPEnrichment_summary.xlsx", engine="openpyxl") as writer:
            pd.DataFrame([run_summary]).to_excel(writer, sheet_name="run_summary", index=False)
            dphy.head(500).to_excel(writer, sheet_name="DPHY_top500", index=False)
            dclinical.head(500).to_excel(writer, sheet_name="DClinical_top500", index=False)
            dual.to_excel(writer, sheet_name="DualD_candidates", index=False)
            h.head(1000).to_excel(writer, sheet_name="h_layer_top1000", index=False)
            module_summary.to_excel(writer, sheet_name="BP_modules", index=False)
            module_members.to_excel(writer, sheet_name="module_members", index=False)
            state_disc.to_excel(writer, sheet_name="state_discriminability", index=False)
            mechanism_audit.to_excel(writer, sheet_name="oncogene_TSG_audit", index=False)
            readiness.to_excel(writer, sheet_name="readiness_audit", index=False)
    except Exception as e:
        print("[WARN] Excel export failed, CSV outputs are still available:", e)

    readme = f"""AIDO D-PHY + D-Clinical BP-Enrichment State Reconstruction V1.0
=================================================================

This pipeline intentionally excludes TTU/drug analysis.

Main purpose
------------
D-PHY + D-Clinical selected BP observables are reconstructed into BP-Enrichment
modules. The reconstructed BP-state modules are then tested for clinical-state
information and oncogene/tumor-suppressor mechanism anchoring.

Target mode: {TARGET_MODE}

Main counts
-----------
Samples: {bp_mat.shape[0]}
Negative/good state: {(y == 0).sum()}
Positive/bad state: {(y == 1).sum()}
Genes: {expr.shape[1]}
BP terms: {bp_mat.shape[1]}

D-PHY selected: {len(dphy_selected)}
D-Clinical selected: {len(dclinical_selected)}
Dual-D union: {int(dual['selected_by_dualD_union'].sum())}
Dual-D core/intersection: {int(dual['selected_by_dualD_intersection'].sum())}

BP-Enrichment
-------------
Pool mode: {DUALD_POOL_MODE}
Module candidate BP: {module_members['BP_term'].nunique() if not module_members.empty else 0}
BP modules: {module_summary.shape[0] if module_summary is not None else 0}

Reconstructed state
-------------------
{json.dumps(state_info, indent=2)}

Mechanism audit
---------------
Oncogene / tumor-suppressor anchoring is used to ask whether reconstructed BP-state
modules may support cancer-mechanism interpretation.

No TTU
------
Drug target loading, drug-module overlap, TTU scoring, and side-effect modules
are intentionally not included here.
"""
    (out_dir / "README_DPHY_DClinical_BPEnrichment_StateReconstruction.txt").write_text(readme, encoding="utf-8")

    zip_path = out_dir.parent / f"{out_dir.name}.zip"
    zip_dir(out_dir, zip_path)

    print("\n[DONE] D-PHY + D-Clinical BP-Enrichment State Reconstruction completed.")
    print("[DONE] Output folder:", out_dir)
    print("[DONE] ZIP package:", zip_path)
    print("\n[SUMMARY]")
    print(json.dumps(run_summary, indent=2))

    return zip_path


# =============================================================================
# Execute
# =============================================================================

if RUN_NOW:
    ZIP_OUTPUT = run_pipeline()
