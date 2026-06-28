# -*- coding: utf-8 -*-
"""
DUAL-D FROM SCRATCH V1.0
========================

重新開始版：從 raw gene-expression 重新計算整條 DUAL-D pipeline。

Core pipeline
-------------
Raw GE matrix
    ↓
Gene symbol × sample expression matrix
    ↓
BP activity matrix, samples × BP
    ↓
D-PHY: local BP-state deviation
    ↓
D-Clinical: global clinical/cancer-state contribution
    ↓
DUAL-D integration: union + intersection + tier classification
    ↓
BP-Enrichment / BP-module reconstruction
    ↓
Optional h-layer: CIViC / COSMIC / OncoKB biomarker-oncogene-TSG anchoring
    ↓
Tables + figures + ZIP package

Designed for Jupyter / Spyder:
    1. Edit CONFIG section
    2. Press Run

Default first run:
    TCGA-BRCA tumor vs normal
    tumor = TCGA sample type 01
    normal = TCGA sample type 11

Important input files
---------------------
Expression file:
    GE_FILE = r"D:/AIDO-Data/UCSC_XENA/Breast Cancer (BRCA)/GE.tsv"

BP gene-set file:
    BP_GENESET_FILE = r".../c5.go.bp.v202x.x.Hs.symbols.gmt"

If BP_GENESET_FILE is None, the script searches common D:/AIDO-Data folders.

Optional biomarker files:
    CIVIC_ACCEPTED_FILE
    CIVIC_FEATURE_FILE
    COSMIC_CENSUS_FILE
    ONCOKB_CANCER_GENE_FILE

Outputs
-------
OUT_DIR:
    00_config_and_input_summary.json
    01_expression_sample_gene_matrix.csv.gz
    02_sample_labels.csv
    03_BP_activity_matrix.csv.gz
    04_DPHY_ranked_BP_signals.csv
    05_DClinical_BP_coefficients.csv
    06_DualD_BP_candidates.csv
    07_h_layer_biological_anchoring.csv
    08_BPenrichment_module_members.csv
    09_BPenrichment_module_summary.csv
    10_DUALD_RUN_SUMMARY.json
    11_DUALD_summary.xlsx
    figures/*.png
    ZIP package
"""

# =============================================================================
# CONFIG: edit here
# =============================================================================

# Main data
GE_FILE = r"D:/AIDO-Data/UCSC_XENA/Breast Cancer (BRCA)/GE.tsv"

# If None, the script will search common locations under D:/AIDO-Data
BP_GENESET_FILE = None

# Output
OUT_DIR = r"D:/AIDO-Temp/DUAL_D_FROM_SCRATCH_TCGA_TumorVsNormal_V1"

# Target mode
# Currently implemented:
#   "tcga_tumor_vs_normal"
TARGET_MODE = "tcga_tumor_vs_normal"

# Optional label file for future non-TCGA targets.
# Leave None for TCGA tumor-vs-normal because labels are inferred from barcode.
LABEL_FILE = None
SAMPLE_COL = None
LABEL_COL = None

# Biomarker / oncogene / TSG files
CIVIC_ACCEPTED_FILE = r"D:/AIDO-Data/Biomarkers/CIViC/01-May-2026-AcceptedClinicalEvidenceSummaries.tsv"
CIVIC_FEATURE_FILE = r"D:/AIDO-Data/Biomarkers/CIViC/01-May-2026-FeatureSummaries.tsv"
COSMIC_CENSUS_FILE = r"D:/AIDO-Data/Biomarkers/COSMIC/Census_allThu May 28 05_04_17 2026.tsv"
ONCOKB_CANCER_GENE_FILE = r"D:/AIDO-Data/Biomarkers/OncoKB/cancerGeneList.tsv"

# Gene-set filtering
MIN_GENES_PER_BP = 5
MAX_GENES_PER_BP = 500
MAX_BP_TERMS = None     # None = all; set e.g. 6000 if needed

# D-PHY
DPHY_FDR_CUTOFF = 0.05
DPHY_ABS_D_CUTOFF = 0.30
TOP_K_DPHY = 50
BOOTSTRAP_N = 100
BOOTSTRAP_TOP_K = 50

# D-Clinical
TOP_K_DCLINICAL = 50
CV_SPLITS = 5
CV_REPEATS = 10
LOGISTIC_C = 0.25
MAX_ITER = 3000
RANDOM_STATE = 42

# BP-Enrichment / module reconstruction
MIN_GENE_JACCARD = 0.12
MIN_LEXICAL_SIMILARITY = 0.20
MIN_MODULE_SIZE = 2

# Performance / output
SAVE_FULL_EXPRESSION_MATRIX = False   # raw expression can be huge; keep False by default
SAVE_BP_MATRIX = True
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
from statsmodels.stats.multitest import multipletests

from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
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
# Utility
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
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, **kwargs)
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


def q_rank_score(values: pd.Series, larger_better: bool = True) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(0.0, index=values.index)
    return x.rank(pct=True, ascending=not larger_better).fillna(0.0)


def cap_neglog10_p(p: float, cap: float = 50.0) -> float:
    try:
        p = float(p)
    except Exception:
        return np.nan
    if not np.isfinite(p) or p <= 0:
        return cap
    return min(-math.log10(max(p, 1e-300)), cap)


def infer_table_gene_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "gene", "Gene", "genes", "Genes", "gene_symbol", "Gene Symbol",
        "GENE_SYMBOL", "symbol", "Symbol", "feature", "Feature",
        "Feature Name", "feature_name", "name", "Name"
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
# Input discovery
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
        print("[INFO] Auto-detected BP GMT:", hits[0])
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
    cols_lower = {c.lower(): c for c in df.columns}

    bp_col = None
    for c in ["bp_term", "bp", "pathway", "term", "geneset", "gene_set", "name"]:
        if c in cols_lower:
            bp_col = cols_lower[c]
            break
    if bp_col is None:
        bp_col = df.columns[0]

    gene_col = infer_table_gene_column(df)
    if gene_col is None:
        if len(df.columns) >= 2:
            gene_col = df.columns[1]
        else:
            raise ValueError("Cannot infer gene column from gene-set table.")

    gene_sets = {}
    for bp, sub in df.groupby(bp_col):
        genes = set()
        for val in sub[gene_col].dropna().astype(str):
            for g in re.split(r"[,;|\s]+", val):
                g = normalize_gene_symbol(g)
                if g:
                    genes.add(g)
        if genes:
            gene_sets[normalize_bp_name(bp)] = genes
    return gene_sets


def load_bp_gene_sets(path: Optional[str]) -> Dict[str, Set[str]]:
    if path is None:
        p = auto_find_bp_gmt()
    else:
        p = Path(path)

    if p is None or not p.exists():
        raise FileNotFoundError(
            "Cannot find BP gene-set file. Please set BP_GENESET_FILE to a GO BP GMT file."
        )

    if p.suffix.lower() in [".gmt", ".gmx"]:
        gs = read_gmt(p)
    else:
        gs = read_gene_set_table(p)

    print(f"[INFO] Loaded BP gene sets: {len(gs)} from {p}")
    return gs


# =============================================================================
# Expression loading
# =============================================================================

def load_expression_matrix(ge_file: str) -> pd.DataFrame:
    """
    Return sample × gene matrix with numeric expression.

    Handles common formats:
        genes × samples with first column gene
        samples × genes with first column sample
    """
    path = Path(ge_file)
    print("[INFO] Reading expression:", path)
    df = safe_read_table(path)

    first = df.columns[0]
    first_lower = str(first).lower()

    # Common UCSC/Xena gene × sample format: first col gene symbol
    if first_lower in ["gene", "genes", "sample", "sample_id", "id", "identifier"]:
        id_col = first
    else:
        id_col = first

    # Decide orientation.
    # If first column values look like genes and many remaining columns look like TCGA samples -> genes × samples.
    first_values = df[id_col].astype(str).head(50).tolist()
    col_values = list(map(str, df.columns[1:20]))

    n_first_tcga = sum(v.startswith("TCGA-") for v in first_values)
    n_col_tcga = sum(v.startswith("TCGA-") for v in col_values)

    if n_col_tcga >= max(2, n_first_tcga):
        # genes × samples
        df = df.rename(columns={id_col: "gene"})
        df["gene"] = df["gene"].astype(str).map(lambda x: x.split("|")[0]).map(normalize_gene_symbol)
        df = df.dropna(subset=["gene"])
        df = df[df["gene"] != ""]
        df = df.groupby("gene", as_index=True).mean(numeric_only=True)
        expr = df.T
        expr.index = expr.index.astype(str)
        expr.columns = [normalize_gene_symbol(c) for c in expr.columns]
    else:
        # samples × genes
        df = df.rename(columns={id_col: "sample_id"})
        df["sample_id"] = df["sample_id"].astype(str)
        df = df.set_index("sample_id")
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        expr = df
        expr.columns = [normalize_gene_symbol(c.split("|")[0]) for c in expr.columns]
        expr = expr.groupby(expr.columns, axis=1).mean(numeric_only=True)

    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=0, how="all").dropna(axis=1, how="all")
    print(f"[INFO] Expression matrix: samples={expr.shape[0]}, genes={expr.shape[1]}")
    return expr


def zscore_genes(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Cohort-wise z-score per gene.
    """
    mu = expr.mean(axis=0, skipna=True)
    sd = expr.std(axis=0, skipna=True).replace(0, np.nan)
    z = (expr - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


# =============================================================================
# Labels
# =============================================================================

def parse_tcga_sample_type(sample_id: str) -> Optional[int]:
    s = str(sample_id)
    parts = s.split("-")
    if len(parts) >= 4:
        code = parts[3][:2]
        if code.isdigit():
            return int(code)
    return None


def make_tcga_tumor_normal_labels(samples: Sequence[str]) -> pd.Series:
    labels = {}
    for sid in samples:
        st = parse_tcga_sample_type(str(sid))
        if st == 11:
            labels[str(sid)] = 0    # good/normal
        elif st == 1:
            labels[str(sid)] = 1    # cancerized/tumor
    y = pd.Series(labels, name="label")
    if len(y) < 20:
        raise ValueError("Too few TCGA tumor/normal labels inferred from barcodes.")
    return y.astype(int)


def load_external_labels(label_file: str, sample_col: Optional[str], label_col: Optional[str]) -> pd.Series:
    df = safe_read_table(Path(label_file))

    if sample_col is None:
        for c in ["sample_id", "sample", "Sample", "SAMPLE", "patient_id", "Patient", "id", "ID"]:
            if c in df.columns:
                sample_col = c
                break
    if sample_col is None:
        sample_col = df.columns[0]

    if label_col is None:
        for c in ["label", "class", "target", "y", "phenotype", "status", "group", "endpoint"]:
            if c in df.columns:
                label_col = c
                break
    if label_col is None:
        for c in df.columns:
            if c == sample_col:
                continue
            vals = df[c].dropna().astype(str).unique()
            if 2 <= len(vals) <= 5:
                label_col = c
                break
    if label_col is None:
        raise ValueError("Cannot infer label_col. Please set LABEL_COL.")

    tmp = df[[sample_col, label_col]].dropna().copy()
    tmp[sample_col] = tmp[sample_col].astype(str)
    raw = tmp[label_col]

    if pd.api.types.is_numeric_dtype(raw):
        y = pd.to_numeric(raw, errors="coerce")
        uniq = sorted(y.dropna().unique())
        if len(uniq) != 2:
            raise ValueError(f"Numeric label column must have exactly 2 classes. Found: {uniq[:10]}")
        y01 = y.map({uniq[0]: 0, uniq[1]: 1})
    else:
        s = raw.astype(str).str.strip().str.lower()
        pos = {"1", "tumor", "tumour", "late", "late_like", "high", "positive", "node_positive",
               "node-positive", "poor", "case", "cancer", "yes", "true", "pcr"}
        neg = {"0", "normal", "early", "early_like", "low", "negative", "node_negative",
               "node-negative", "good", "control", "no", "false", "rd"}
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

    # TCGA fallback first 15 chars
    expr2 = expr.copy()
    y2 = y.copy()
    expr2.index = expr2.index.astype(str).str[:15]
    y2.index = y2.index.astype(str).str[:15]
    common = sorted(set(expr2.index) & set(y2.index))
    if len(common) >= 20:
        return expr2.loc[common], y2.loc[common].astype(int)

    raise ValueError(f"Too few matched expression-label samples: {len(common)}")


# =============================================================================
# BP activity matrix
# =============================================================================

def filter_bp_gene_sets(gene_sets: Dict[str, Set[str]], genes_available: Set[str],
                        min_genes: int, max_genes: int, max_bp_terms: Optional[int]) -> Dict[str, List[str]]:
    out = {}
    for bp, genes in gene_sets.items():
        mapped = sorted(set(genes) & genes_available)
        if min_genes <= len(mapped) <= max_genes:
            out[bp] = mapped

    # Stable sort: larger gene sets not necessarily better; keep alphabetical if no cap.
    if max_bp_terms is not None and len(out) > max_bp_terms:
        items = sorted(out.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:max_bp_terms]
        out = dict(items)

    print(f"[INFO] BP terms after gene mapping/filtering: {len(out)}")
    return out


def compute_bp_activity_matrix(zexpr: pd.DataFrame, bp_genes: Dict[str, List[str]]) -> pd.DataFrame:
    """
    BP activity = mean z-scored expression of mapped genes.
    Output: samples × BP.
    """
    data = {}
    for i, (bp, genes) in enumerate(bp_genes.items(), start=1):
        if i % 500 == 0:
            print(f"[INFO] Computing BP activity: {i}/{len(bp_genes)}")
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
    rows = []

    group0 = y[y == 0].index
    group1 = y[y == 1].index

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
            t_p = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit").pvalue
        except Exception:
            t_p = np.nan

        auc = safe_auc(y, bp_mat[bp])
        direction = "cancerized_up" if delta > 0 else "good_state_up_or_cancerized_suppressed"

        rows.append({
            "BP_term": bp,
            "bp_gene_count": len(bp_genes.get(bp, [])),
            "mean_good_state": mean0,
            "mean_cancerized_state": mean1,
            "delta_cancerized_minus_good": delta,
            "cohen_d": d,
            "abs_cohen_d": abs(d) if pd.notna(d) else np.nan,
            "auc_cancerized_vs_good": auc,
            "auc_distance": abs(auc - 0.5) if pd.notna(auc) else np.nan,
            "welch_p": t_p,
            "direction": direction,
        })

    df = pd.DataFrame(rows)
    pvals = df["welch_p"].fillna(1.0).clip(lower=1e-300, upper=1.0)
    df["welch_fdr"] = multipletests(pvals, method="fdr_bh")[1]
    df["neglog10_fdr_capped50"] = df["welch_fdr"].map(lambda p: cap_neglog10_p(p, cap=50.0))
    df["D_score_capped50"] = (
        df["abs_cohen_d"].fillna(0)
        * (1 + 2 * df["auc_distance"].fillna(0))
        * df["neglog10_fdr_capped50"].fillna(0)
    )

    parts = [
        q_rank_score(df["D_score_capped50"], True),
        q_rank_score(df["abs_cohen_d"], True),
        q_rank_score(df["auc_distance"], True),
        q_rank_score(df["neglog10_fdr_capped50"], True),
    ]
    df["DPHY_selection_score"] = pd.concat(parts, axis=1).mean(axis=1)
    df = df.sort_values("DPHY_selection_score", ascending=False).reset_index(drop=True)
    df["DPHY_rank"] = np.arange(1, len(df) + 1)
    return df


def bootstrap_dphy_stability(bp_mat: pd.DataFrame, y: pd.Series, dphy: pd.DataFrame,
                             n_boot: int, top_k: int, random_state: int) -> pd.DataFrame:
    """
    Simple bootstrap top-k stability for D-PHY.
    """
    rng = np.random.default_rng(random_state)
    bp_list = dphy["BP_term"].tolist()
    counts = pd.Series(0, index=bp_list, dtype=float)

    idx0 = np.array(y[y == 0].index)
    idx1 = np.array(y[y == 1].index)

    if n_boot <= 0:
        dphy["bootstrap_topk_stability"] = np.nan
        return dphy

    for b in range(n_boot):
        if (b + 1) % 20 == 0:
            print(f"[INFO] Bootstrap D-PHY stability: {b+1}/{n_boot}")

        s0 = rng.choice(idx0, size=len(idx0), replace=True)
        s1 = rng.choice(idx1, size=len(idx1), replace=True)
        sample_ids = list(s0) + list(s1)
        yb = pd.Series([0] * len(s0) + [1] * len(s1), index=sample_ids)

        rows = []
        for bp in bp_mat.columns:
            x0 = pd.to_numeric(bp_mat.loc[s0, bp], errors="coerce")
            x1 = pd.to_numeric(bp_mat.loc[s1, bp], errors="coerce")
            d = cohen_d(x0.values, x1.values)
            try:
                p = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit").pvalue
            except Exception:
                p = np.nan
            rows.append((bp, abs(d) if pd.notna(d) else 0, p if pd.notna(p) else 1.0))

        tmp = pd.DataFrame(rows, columns=["BP_term", "abs_d", "p"])
        tmp["fdr"] = multipletests(tmp["p"].clip(lower=1e-300, upper=1.0), method="fdr_bh")[1]
        tmp["score"] = tmp["abs_d"] * tmp["fdr"].map(lambda p: cap_neglog10_p(p, 50.0))
        top = tmp.sort_values("score", ascending=False).head(top_k)["BP_term"]
        counts.loc[top] += 1

    dphy = dphy.copy()
    dphy["bootstrap_topk_stability"] = dphy["BP_term"].map((counts / n_boot).to_dict()).fillna(0.0)
    return dphy


def select_dphy_bp(dphy: pd.DataFrame, top_k: int, fdr_cutoff: float, abs_d_cutoff: float) -> Set[str]:
    selected = set(dphy.head(top_k)["BP_term"].astype(str))
    mask = (dphy["welch_fdr"] <= fdr_cutoff) & (dphy["abs_cohen_d"] >= abs_d_cutoff)
    selected |= set(dphy.loc[mask, "BP_term"].astype(str))
    return selected


# =============================================================================
# D-Clinical
# =============================================================================

def compute_dclinical(bp_mat: pd.DataFrame, y: pd.Series, n_splits: int, n_repeats: int,
                      random_state: int, C: float, max_iter: int, top_k: int) -> Tuple[pd.DataFrame, Dict]:
    X = bp_mat.copy()
    y = y.loc[X.index].astype(int)

    nunique = X.nunique(dropna=True)
    X = X[nunique[nunique > 1].index.tolist()]

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    coefs = []
    aucs = []
    fold_id = 0

    for train_idx, test_idx in cv.split(X, y):
        fold_id += 1
        if fold_id % 10 == 0:
            print(f"[INFO] D-Clinical CV fold: {fold_id}/{n_splits*n_repeats}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=C,
                solver="liblinear",
                class_weight="balanced",
                max_iter=max_iter,
                random_state=random_state + fold_id,
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
        coefs.append(pd.Series(coef, index=X.columns, name=f"fold_{fold_id}"))

    coef_df = pd.DataFrame(coefs)
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
    out["DClinical_topK"] = out["DClinical_rank"] <= top_k

    info = {
        "n_samples": int(X.shape[0]),
        "n_bp_features": int(X.shape[1]),
        "n_positive_or_cancerized": int(y.sum()),
        "n_negative_or_good": int((1 - y).sum()),
        "cv_auc_mean": float(np.nanmean(aucs)),
        "cv_auc_sd": float(np.nanstd(aucs)),
        "cv_auc_median": float(np.nanmedian(aucs)),
    }

    return out, info


def select_dclinical_bp(dclinical: pd.DataFrame, top_k: int) -> Set[str]:
    return set(dclinical.head(top_k)["BP_term"].astype(str))


# =============================================================================
# h-layer biomarker anchoring
# =============================================================================

def read_gene_set_from_any_file(path: Optional[str]) -> Set[str]:
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Biomarker file not found: {p}")
        return set()

    try:
        df = safe_read_table(p)
    except Exception as e:
        print(f"[WARN] Cannot read biomarker file {p}: {e}")
        return set()

    genes = set()

    # Collect from likely gene columns
    likely_cols = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["gene", "feature", "symbol", "name"]):
            likely_cols.append(c)

    if not likely_cols:
        likely_cols = list(df.columns[:3])

    for c in likely_cols:
        for val in df[c].dropna().astype(str):
            # split common separators
            for g in re.split(r"[,;|/\s]+", val):
                g = normalize_gene_symbol(g)
                if re.match(r"^[A-Z0-9\-\.]{2,20}$", g):
                    genes.add(g)

    return genes


def load_oncokb_genes(path: Optional[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Return cancer_genes, oncogenes, tsg from OncoKB if columns available.
    """
    if path is None or not Path(path).exists():
        return set(), set(), set()

    df = safe_read_table(Path(path))
    gene_col = infer_table_gene_column(df)
    if gene_col is None:
        gene_col = df.columns[0]

    cancer = {normalize_gene_symbol(x) for x in df[gene_col].dropna().astype(str)}
    onc = set()
    tsg = set()

    for _, row in df.iterrows():
        gene = normalize_gene_symbol(row.get(gene_col, ""))
        if not gene:
            continue
        row_text = " ".join([str(v).lower() for v in row.values])
        if "oncogene" in row_text or re.search(r"\bonc\b", row_text):
            onc.add(gene)
        if "tumor suppressor" in row_text or "tumour suppressor" in row_text or "tsg" in row_text:
            tsg.add(gene)

    return cancer, onc, tsg


def load_cosmic_genes(path: Optional[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    if path is None or not Path(path).exists():
        return set(), set(), set()

    df = safe_read_table(Path(path))
    gene_col = None
    for c in ["Gene Symbol", "GeneSymbol", "gene_symbol", "Gene", "GENE_SYMBOL"]:
        if c in df.columns:
            gene_col = c
            break
    if gene_col is None:
        gene_col = infer_table_gene_column(df) or df.columns[0]

    cancer = {normalize_gene_symbol(x) for x in df[gene_col].dropna().astype(str)}
    onc = set()
    tsg = set()

    for _, row in df.iterrows():
        gene = normalize_gene_symbol(row.get(gene_col, ""))
        if not gene:
            continue
        row_text = " ".join([str(v).lower() for v in row.values])
        if "oncogene" in row_text or re.search(r"\bonc\b", row_text):
            onc.add(gene)
        if "tsg" in row_text or "tumour suppressor" in row_text or "tumor suppressor" in row_text:
            tsg.add(gene)

    return cancer, onc, tsg


def build_biomarker_reference() -> Tuple[Set[str], Set[str], Set[str]]:
    civic1 = read_gene_set_from_any_file(CIVIC_ACCEPTED_FILE)
    civic2 = read_gene_set_from_any_file(CIVIC_FEATURE_FILE)
    cosmic_cancer, cosmic_onc, cosmic_tsg = load_cosmic_genes(COSMIC_CENSUS_FILE)
    oncokb_cancer, oncokb_onc, oncokb_tsg = load_oncokb_genes(ONCOKB_CANCER_GENE_FILE)

    cancer_genes = set().union(civic1, civic2, cosmic_cancer, oncokb_cancer)
    oncogenes = set().union(cosmic_onc, oncokb_onc)
    tsg = set().union(cosmic_tsg, oncokb_tsg)

    print(f"[INFO] Biomarker/cancer genes: {len(cancer_genes)} | oncogenes: {len(oncogenes)} | TSG: {len(tsg)}")
    return cancer_genes, oncogenes, tsg


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

    genes = ";".join(sorted(bp_genes & ref_genes))
    return a, odds, p, genes


def compute_h_layer(bp_genes: Dict[str, List[str]], universe: Set[str],
                    cancer_genes: Set[str], oncogenes: Set[str], tsg: Set[str]) -> pd.DataFrame:
    rows = []

    for bp, genes_list in bp_genes.items():
        g = set(genes_list)
        c_n, c_odds, c_p, c_genes = fisher_overlap(g, cancer_genes, universe)
        o_n, o_odds, o_p, o_genes = fisher_overlap(g, oncogenes, universe)
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
        pcol = f"{prefix}_fisher_p"
        fcol = f"{prefix}_fisher_fdr"
        h[fcol] = multipletests(h[pcol].fillna(1.0).clip(lower=1e-300), method="fdr_bh")[1]

    h["h_overlap_total"] = (
        h["cancer_gene_overlap_n"].fillna(0)
        + h["oncogene_overlap_n"].fillna(0)
        + h["tumor_suppressor_overlap_n"].fillna(0)
    )
    h["h_best_fdr"] = h[
        ["cancer_gene_fisher_fdr", "oncogene_fisher_fdr", "tumor_suppressor_fisher_fdr"]
    ].min(axis=1)
    h["h_score"] = h["h_overlap_total"].fillna(0) * h["h_best_fdr"].map(lambda p: cap_neglog10_p(p, 50.0))
    h["flag_biologically_anchored"] = (h["h_overlap_total"] > 0) & (h["h_best_fdr"] <= 0.05)

    return h.sort_values(["h_score", "h_overlap_total"], ascending=False).reset_index(drop=True)


# =============================================================================
# DUAL-D integration
# =============================================================================

def build_dual_d_table(dphy: pd.DataFrame, dclinical: pd.DataFrame, h: pd.DataFrame,
                       dphy_selected: Set[str], dclinical_selected: Set[str]) -> pd.DataFrame:
    all_bp = sorted(set(dphy["BP_term"]) | set(dclinical["BP_term"]))
    base = pd.DataFrame({"BP_term": all_bp})

    dphy_cols = [
        "BP_term", "DPHY_rank", "DPHY_selection_score", "D_score_capped50",
        "abs_cohen_d", "cohen_d", "auc_cancerized_vs_good", "auc_distance",
        "welch_p", "welch_fdr", "direction", "mean_good_state",
        "mean_cancerized_state", "delta_cancerized_minus_good",
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

    if h is not None and not h.empty:
        h_cols = [
            "BP_term", "h_overlap_total", "h_best_fdr", "h_score",
            "flag_biologically_anchored",
            "cancer_gene_overlap_n", "oncogene_overlap_n", "tumor_suppressor_overlap_n",
            "cancer_gene_overlap_genes", "oncogene_overlap_genes", "tumor_suppressor_overlap_genes",
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
        if "cancerized_up" in direction:
            return "aberrantly_activated_in_cancerized_state"
        if "good_state_up" in direction or "suppressed" in direction:
            return "missing_good_state_or_cancerized_suppressed_BP"
        coef = row.get("DClinical_mean_coef", np.nan)
        if pd.notna(coef):
            return "clinical_positive_contributor" if coef > 0 else "clinical_negative_contributor"
        return "unknown"

    base["BP_state_direction_class"] = base.apply(state_class, axis=1)

    score = 0
    score = score + base["DPHY_selection_score"].fillna(0) * 0.40
    score = score + base["DClinical_selection_score"].fillna(0) * 0.40
    if "h_score" in base.columns:
        score = score + q_rank_score(base["h_score"], True).fillna(0) * 0.20
    base["DualD_preliminary_priority_score"] = score

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


# =============================================================================
# BP module reconstruction
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
        for b in candidates[i+1:]:
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
    candidates = sorted(set(dual.loc[dual["selected_by_dualD_union"], "BP_term"].astype(str)))
    if not candidates:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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
    members["BP_module_status"] = np.where(members["BP_module_size"] >= MIN_MODULE_SIZE, "module", "singleton_candidate")

    rows = []
    for mid, sub in members.groupby("BP_module_id"):
        tiers = sub["DualD_tier"].value_counts().to_dict()
        dirs = sub["BP_state_direction_class"].value_counts().to_dict()

        row = {
            "BP_module_id": mid,
            "BP_module_name": name_map[mid],
            "BP_module_size": int(len(sub)),
            "n_Tier1_core_dual_D": int(tiers.get("Tier1_core_dual_D", 0)),
            "n_Tier2_DPHY_only": int(tiers.get("Tier2_DPHY_only_direct_BP_deviation", 0)),
            "n_Tier3_DClinical_only": int(tiers.get("Tier3_DClinical_only_distributed_state_contributor", 0)),
            "mean_DualD_priority_score": float(sub["DualD_preliminary_priority_score"].mean()),
            "max_DualD_priority_score": float(sub["DualD_preliminary_priority_score"].max()),
            "dominant_direction_class": max(dirs.items(), key=lambda kv: kv[1])[0] if dirs else "unknown",
            "top_BP_terms": ";".join(
                sub.sort_values("DualD_preliminary_priority_score", ascending=False)["BP_term"].head(10).tolist()
            )
        }
        if "h_score" in sub.columns:
            row["mean_h_score"] = float(pd.to_numeric(sub["h_score"], errors="coerce").mean())
        if "h_overlap_total" in sub.columns:
            row["n_h_anchored"] = int(pd.to_numeric(sub["h_overlap_total"], errors="coerce").fillna(0).gt(0).sum())
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["n_Tier1_core_dual_D", "mean_DualD_priority_score", "BP_module_size"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    return members, summary, edges


# =============================================================================
# Figures
# =============================================================================

def make_figures(out_dir: Path, dphy: pd.DataFrame, dclinical: pd.DataFrame,
                 dual: pd.DataFrame, module_summary: pd.DataFrame, dclinical_info: Dict) -> None:
    if not HAS_MPL:
        return

    fig_dir = ensure_dir(out_dir / "figures")

    # Fig 1 D-PHY top effects
    top = dphy.head(20).copy()
    if not top.empty:
        labels = top["BP_term"].str.replace("GOBP_", "", regex=False).str.replace("_", " ").str[:50]
        plt.figure(figsize=(10, max(5, 0.35 * len(top))))
        y = np.arange(len(top))
        plt.barh(y, top["abs_cohen_d"])
        plt.yticks(y, labels)
        plt.xlabel("|Cohen d|")
        plt.title("Top D-PHY BP state deviations")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(fig_dir / "FIG01_top_DPHY_BP_effects.png", dpi=300)
        plt.close()

    # Fig 2 D-Clinical top coefficients
    top = dclinical.head(20).copy()
    if not top.empty:
        labels = top["BP_term"].str.replace("GOBP_", "", regex=False).str.replace("_", " ").str[:50]
        plt.figure(figsize=(10, max(5, 0.35 * len(top))))
        y = np.arange(len(top))
        plt.barh(y, top["DClinical_mean_abs_coef"])
        plt.yticks(y, labels)
        plt.xlabel("Mean absolute standardized coefficient")
        plt.title("Top D-Clinical BP contributors")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(fig_dir / "FIG02_top_DClinical_BP_coefficients.png", dpi=300)
        plt.close()

    # Fig 3 Dual-D space
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

    # Fig 4 tier counts
    counts = dual.loc[dual["selected_by_dualD_union"], "DualD_tier"].value_counts()
    if not counts.empty:
        plt.figure(figsize=(9, 5))
        plt.bar(counts.index.astype(str), counts.values)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("BP count")
        plt.title("Dual-D selected BP tiers")
        plt.tight_layout()
        plt.savefig(fig_dir / "FIG04_DualD_tier_counts.png", dpi=300)
        plt.close()

    # Fig 5 modules
    if module_summary is not None and not module_summary.empty:
        top = module_summary.head(20)
        plt.figure(figsize=(10, max(5, 0.3 * len(top) + 2)))
        y = np.arange(len(top))
        plt.barh(y, top["BP_module_size"])
        plt.yticks(y, top["BP_module_name"])
        plt.xlabel("BP count")
        plt.title("Top Dual-D BP-enrichment modules")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(fig_dir / "FIG05_top_BP_modules.png", dpi=300)
        plt.close()

    # Fig 6 D-Clinical AUC
    plt.figure(figsize=(6, 4))
    vals = [dclinical_info.get("cv_auc_mean", np.nan), dclinical_info.get("cv_auc_median", np.nan)]
    plt.bar(["Mean CV AUC", "Median CV AUC"], vals)
    plt.ylim(0.45, 1.05)
    plt.ylabel("AUC")
    plt.title("D-Clinical full BP-space performance")
    plt.tight_layout()
    plt.savefig(fig_dir / "FIG06_DClinical_CV_AUC.png", dpi=300)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def run_pipeline() -> Path:
    out_dir = ensure_dir(Path(OUT_DIR))

    # Input summary
    bp_file = BP_GENESET_FILE or auto_find_bp_gmt()
    if bp_file is None:
        raise FileNotFoundError("Cannot auto-find BP gene-set GMT. Set BP_GENESET_FILE manually.")

    config = {
        "GE_FILE": GE_FILE,
        "BP_GENESET_FILE": str(bp_file),
        "OUT_DIR": OUT_DIR,
        "TARGET_MODE": TARGET_MODE,
        "MIN_GENES_PER_BP": MIN_GENES_PER_BP,
        "MAX_GENES_PER_BP": MAX_GENES_PER_BP,
        "TOP_K_DPHY": TOP_K_DPHY,
        "TOP_K_DCLINICAL": TOP_K_DCLINICAL,
        "BOOTSTRAP_N": BOOTSTRAP_N,
        "CV_SPLITS": CV_SPLITS,
        "CV_REPEATS": CV_REPEATS,
    }
    (out_dir / "00_config_and_input_summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # 1 expression
    expr = load_expression_matrix(GE_FILE)

    # 2 labels
    if TARGET_MODE == "tcga_tumor_vs_normal":
        y = make_tcga_tumor_normal_labels(expr.index)
    else:
        if LABEL_FILE is None:
            raise ValueError("LABEL_FILE is required for non-TCGA target modes.")
        y = load_external_labels(LABEL_FILE, SAMPLE_COL, LABEL_COL)

    expr, y = align_expr_labels(expr, y)
    print(f"[INFO] After alignment: samples={expr.shape[0]}, genes={expr.shape[1]}")
    print(f"[INFO] Label counts: good/negative={int((y==0).sum())}, cancerized/positive={int((y==1).sum())}")

    y_out = pd.DataFrame({"sample_id": y.index, "label": y.values})
    y_out["label_name"] = np.where(y_out["label"] == 1, "cancerized_or_positive", "good_or_negative")
    y_out.to_csv(out_dir / "02_sample_labels.csv", index=False)

    if SAVE_FULL_EXPRESSION_MATRIX:
        expr.to_csv(out_dir / "01_expression_sample_gene_matrix.csv.gz", compression="gzip")

    # 3 BP gene sets
    gene_sets_raw = load_bp_gene_sets(str(bp_file))
    zexpr = zscore_genes(expr)
    bp_genes = filter_bp_gene_sets(
        gene_sets_raw,
        genes_available=set(zexpr.columns),
        min_genes=MIN_GENES_PER_BP,
        max_genes=MAX_GENES_PER_BP,
        max_bp_terms=MAX_BP_TERMS
    )

    # 4 BP activity
    bp_mat = compute_bp_activity_matrix(zexpr, bp_genes)
    if SAVE_BP_MATRIX:
        bp_mat.to_csv(out_dir / "03_BP_activity_matrix.csv.gz", compression="gzip")

    # 5 D-PHY
    dphy = compute_dphy(bp_mat, y, bp_genes)
    dphy = bootstrap_dphy_stability(bp_mat, y, dphy, BOOTSTRAP_N, BOOTSTRAP_TOP_K, RANDOM_STATE)
    dphy.to_csv(out_dir / "04_DPHY_ranked_BP_signals.csv", index=False)
    dphy_selected = select_dphy_bp(dphy, TOP_K_DPHY, DPHY_FDR_CUTOFF, DPHY_ABS_D_CUTOFF)

    # 6 D-Clinical
    dclinical, dclinical_info = compute_dclinical(
        bp_mat, y,
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
        C=LOGISTIC_C,
        max_iter=MAX_ITER,
        top_k=TOP_K_DCLINICAL
    )
    dclinical.to_csv(out_dir / "05_DClinical_BP_coefficients.csv", index=False)
    dclinical_selected = select_dclinical_bp(dclinical, TOP_K_DCLINICAL)

    # 7 h-layer
    cancer_genes, oncogenes, tsg = build_biomarker_reference()
    universe = set(zexpr.columns)
    h = compute_h_layer(bp_genes, universe, cancer_genes, oncogenes, tsg)
    h.to_csv(out_dir / "07_h_layer_biological_anchoring.csv", index=False)

    # 8 Dual-D
    dual = build_dual_d_table(dphy, dclinical, h, dphy_selected, dclinical_selected)
    dual.to_csv(out_dir / "06_DualD_BP_candidates.csv", index=False)

    # 9 BP enrichment / modules
    module_members, module_summary, module_edges = reconstruct_modules(dual, bp_genes)
    module_members.to_csv(out_dir / "08_BPenrichment_module_members.csv", index=False)
    module_summary.to_csv(out_dir / "09_BPenrichment_module_summary.csv", index=False)
    module_edges.to_csv(out_dir / "09b_BPenrichment_BP_similarity_edges.csv", index=False)

    # 10 figures
    make_figures(out_dir, dphy, dclinical, dual, module_summary, dclinical_info)

    # 11 summary
    tier_counts = dual.loc[dual["selected_by_dualD_union"], "DualD_tier"].value_counts().to_dict()
    run_summary = {
        "run_timestamp": now_stamp(),
        "target_mode": TARGET_MODE,
        "n_samples": int(bp_mat.shape[0]),
        "n_good_or_negative": int((y == 0).sum()),
        "n_cancerized_or_positive": int((y == 1).sum()),
        "n_genes": int(expr.shape[1]),
        "n_bp_terms": int(bp_mat.shape[1]),
        "n_DPHY_selected": int(len(dphy_selected)),
        "n_DClinical_selected": int(len(dclinical_selected)),
        "n_dualD_union": int(dual["selected_by_dualD_union"].sum()),
        "n_dualD_intersection_core": int(dual["selected_by_dualD_intersection"].sum()),
        "dualD_tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "DClinical": dclinical_info,
        "n_h_biologically_anchored": int(h["flag_biologically_anchored"].sum()),
        "n_BP_modules": int(module_summary.shape[0]),
        "interpretation": (
            "D-PHY captures local BP-state deviation; D-Clinical captures global clinical/cancer-state "
            "contribution. The union selected BP pool is passed to BP-enrichment / module reconstruction. "
            "Tier1 core dual-D BPs are highest-priority BP-state candidates."
        ),
    }
    (out_dir / "10_DUALD_RUN_SUMMARY.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    # Excel
    with pd.ExcelWriter(out_dir / "11_DUALD_summary.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([run_summary]).to_excel(writer, sheet_name="run_summary", index=False)
        dphy.head(500).to_excel(writer, sheet_name="DPHY_top500", index=False)
        dclinical.head(500).to_excel(writer, sheet_name="DClinical_top500", index=False)
        dual.to_excel(writer, sheet_name="DualD_candidates", index=False)
        h.head(1000).to_excel(writer, sheet_name="h_layer_top1000", index=False)
        module_summary.to_excel(writer, sheet_name="BP_modules", index=False)
        module_members.to_excel(writer, sheet_name="module_members", index=False)

    # README
    readme = f"""DUAL-D FROM SCRATCH V1.0
==========================

Target mode: {TARGET_MODE}

Main counts
-----------
Samples: {bp_mat.shape[0]}
Good/negative: {(y == 0).sum()}
Cancerized/positive: {(y == 1).sum()}
Genes: {expr.shape[1]}
BP terms: {bp_mat.shape[1]}

D-PHY selected: {len(dphy_selected)}
D-Clinical selected: {len(dclinical_selected)}
Dual-D union: {int(dual['selected_by_dualD_union'].sum())}
Dual-D core/intersection: {int(dual['selected_by_dualD_intersection'].sum())}

D-Clinical CV AUC mean: {dclinical_info.get('cv_auc_mean', np.nan):.4f}
D-Clinical CV AUC SD: {dclinical_info.get('cv_auc_sd', np.nan):.4f}

Concept
-------
D-module = D-PHY + D-Clinical.
D-PHY captures local BP-state deviation.
D-Clinical captures global clinical/cancer-state contribution.
D-selected BP candidates are passed to BP-enrichment / BP-module reconstruction.

Tier interpretation
-------------------
Tier1_core_dual_D:
    BP has both local deviation and clinical-state contribution.

Tier2_DPHY_only_direct_BP_deviation:
    BP has direct state deviation but weaker clinical-model contribution.

Tier3_DClinical_only_distributed_state_contributor:
    BP contributes to multivariate clinical/cancer-state observability but is weaker as a single-BP signal.

Tier4_weak_or_unselected:
    Not selected by either D layer.
"""
    (out_dir / "README_DUALD_FROM_SCRATCH.txt").write_text(readme, encoding="utf-8")

    # ZIP
    zip_path = out_dir.parent / f"{out_dir.name}.zip"
    zip_dir(out_dir, zip_path)

    print("\n[DONE] DUAL-D from scratch completed.")
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
