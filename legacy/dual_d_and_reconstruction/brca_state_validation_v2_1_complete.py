# -*- coding: utf-8 -*-
"""
AIDO D-PHY/D-Clinical BRCA Endpoint-State Reconstruction Completion V2.1
====================================================================

Purpose
-------
This is the upgraded pipeline that is deliberately separated from the already-submitted
old IMU Post-D/BPState manuscript.

Old IMU line:
    Patient-level BP-state reconstruction from task-discriminative BP observables.

This V2 line:
    D-PHY/D-Clinical-audited BP-Enrichment state reconstruction with endpoint
    discriminability, random-control validation, reconstructed-state profiling,
    and oncogene/tumor-suppressor interpretation-readiness audit.

NO TTU.
NO drug target analysis.
NO drug ranking.

What is new versus the old IMU version?
---------------------------------------
1. D-PHY layer:
   Local single-BP endpoint deviation / local BP observability.

2. D-Clinical layer:
   Multivariate endpoint contribution of each BP in BP-space.

3. Dual-D evidence stratification:
   Tier1 = D-PHY + D-Clinical core
   Tier2 = D-PHY-only
   Tier3 = D-Clinical-only
   Tier4 = weak/unselected

4. BP-Enrichment:
   Dual-D selected BP terms are reconstructed into BP-state modules.

5. Reconstructed-state information audit:
   sample × BP-state module activity matrix
   module-level discriminability
   multivariate reconstructed-state AUC

6. Old-version-style validation, but now applied to D-PHY/D-Clinical selected states:
   centroid similarity
   random BP-module baseline
   endpoint-label shuffling
   patient scrambling
   bootstrap resampling
   leave-one-module sensitivity

7. Mechanism audit:
   oncogene / tumor-suppressor / cancer-gene anchoring
   interpretation-readiness classes

Cancer support
--------------
This BRCA completion version is designed to finish the unfinished BRCA D-PHY/D-Clinical comparison first. Multi-cancer switching remains possible later:

    CANCER_TYPE = "BRCA"
    CANCER_TYPE = "BRCA"
    CANCER_TYPE = "KIRP"
    CANCER_TYPE = "LUAD"
    etc.

For each cancer, set GE_FILE and CLINICAL_FILE paths.

Recommended distinction from IMU manuscript
-------------------------------------------
Use this script for KIRC/KIRP or another cancer type if you want strong separation
from the old submitted BRCA IMU manuscript.

Example:
    CANCER_TYPE = "BRCA"
    ENDPOINT_MODE = "stage_early_late"

Outputs
-------
OUT_DIR:
    00_config.json
    01_endpoint_labels.csv
    02_BP_activity_matrix.csv.gz
    03_DPHY_ranked_BP_signals.csv
    04_DClinical_BP_coefficients.csv
    05_h_layer_BP_oncogene_TSG_anchoring.csv
    06_DualD_BP_candidates.csv
    07_BPenrichment_module_members.csv
    08_BPenrichment_module_summary.csv
    08b_BPenrichment_BP_similarity_edges.csv
    09_reconstructed_state_module_activity.csv.gz
    10_reconstructed_state_discriminability.csv
    11_patient_level_state_profile.csv
    12_centroid_state_metrics.csv
    13_random_BP_module_baseline.csv
    14_endpoint_label_shuffle_control.csv
    15_patient_scramble_control.csv
    16_bootstrap_state_stability.csv
    17_leave_one_module_sensitivity.csv
    18_module_oncogene_TSG_mechanism_audit.csv
    19_interpretation_readiness_audit.csv
    20_endpoint_run_summary.json
    21_AIDO_DPHY_DClinical_StateValidation_summary.xlsx
    figures/*.png
    ZIP package
"""

# =============================================================================
# CONFIG
# =============================================================================

# Choose cancer type. Used mainly for labels/output naming.
# Examples: "BRCA", "KIRC", "KIRP", "LUAD", "LUSC", "COAD", "PRAD"
CANCER_TYPE = "BRCA"

# Expression input.
# Supports either:
#   genes × samples with first column gene and samples as columns
#   samples × genes with first column sample_id
#
# Change these paths for your local data.
GE_FILE = r"D:/AIDO-Data/UCSC_XENA/Breast Cancer (BRCA)/GE.tsv"

# Clinical / phenotype table.
CLINICAL_FILE = r"D:/AIDO-Data/UCSC_XENA/Breast Cancer (BRCA)/TCGA.BRCA.sampleMap_BRCA_clinicalMatrix"

# GO BP / MSigDB BP gene-set file.
# If None, script searches common D:/AIDO-Data folders.
BP_GENESET_FILE = None

# Endpoint modes:
#   "stage_early_late"      stage I/II vs stage III/IV
#   "node_negative_positive" N0 vs N1/N2/N3 if available
#   "survival_short_long"   binary survival/risk endpoint
#   "tumor_vs_normal"       positive control only
#   "custom_binary"         user-provided labels
ENDPOINT_MODE = "stage_early_late"

# Output directory.
OUT_DIR = r"D:/AIDO-Temp/AIDO_DPHY_DClinical_StateValidation_BRCA_Stage_V2_1_COMPLETE"

# Custom binary endpoint settings
CUSTOM_LABEL_FILE = None
CUSTOM_SAMPLE_COL = None
CUSTOM_LABEL_COL = None

# For patient-internal endpoints, keep TCGA primary tumor only.
KEEP_TCGA_PRIMARY_TUMOR_ONLY = True

# Stage grouping
STAGE_EARLY_PATTERNS = ["stage i", "stage ia", "stage ib", "stage ii", "stage iia", "stage iib"]
STAGE_LATE_PATTERNS = ["stage iii", "stage iiia", "stage iiib", "stage iiic", "stage iv"]

# Survival endpoint
SURVIVAL_LOWER_QUANTILE = 0.33
SURVIVAL_UPPER_QUANTILE = 0.67

# Optional secondary clinical context.
# For non-BRCA cancers, PAM50 usually does not exist. The script will skip if not found.
RUN_SECONDARY_CONTEXT_SCAN = True
SECONDARY_CONTEXT_KEYWORDS = [
    ["pam50"],
    ["subtype"],
    ["histological", "type"],
    ["grade"],
    ["pathologic", "t"],
    ["pathologic", "n"],
    ["pathologic", "m"],
    ["vital", "status"],
]

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
DPHY_ABS_D_CUTOFF = 0.30
BOOTSTRAP_DPHY_N = 30
BOOTSTRAP_DPHY_TOP_K = 100

# D-Clinical
TOP_K_DCLINICAL = 100
CV_SPLITS = 5
CV_REPEATS = 10
LOGISTIC_C = 0.25
STATE_CLASSIFIER_C = 0.50
MAX_ITER = 3000
RANDOM_STATE = 42

# BP-Enrichment
DUALD_POOL_MODE = "strict"  # "strict" or "union_all"
STRICT_TOP_K_DPHY_FOR_MODULES = 300
STRICT_TOP_K_DCLINICAL_FOR_MODULES = 100
INCLUDE_TIER1_ALWAYS = True
MIN_GENE_JACCARD = 0.20
MIN_LEXICAL_SIMILARITY = 0.35
MIN_MODULE_SIZE = 2

# Validation controls
N_RANDOM_MODULE_BASELINES = 200
N_LABEL_SHUFFLES = 200
N_PATIENT_SCRAMBLES = 200
N_BOOTSTRAPS_STATE = 200
BOOTSTRAP_FRACTION = 0.80

# Performance settings
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


def tcga_patient_barcode(x: str) -> str:
    s = str(x).strip()
    parts = s.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return s[:12]


def parse_tcga_sample_type(sample_id: str) -> Optional[int]:
    parts = str(sample_id).split("-")
    if len(parts) >= 4:
        code = parts[3][:2]
        if code.isdigit():
            return int(code)
    return None


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

    if suffix in [".tsv", ".txt"] or "clinicalmatrix" in name.lower():
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


def empirical_p_value(observed: float, null_values: Sequence[float], direction: str = "greater") -> float:
    vals = np.asarray(null_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0 or not np.isfinite(observed):
        return np.nan
    if direction == "greater":
        return float((1 + np.sum(vals >= observed)) / (len(vals) + 1))
    if direction == "less":
        return float((1 + np.sum(vals <= observed)) / (len(vals) + 1))
    if direction == "abs_greater":
        return float((1 + np.sum(np.abs(vals) >= abs(observed))) / (len(vals) + 1))
    raise ValueError("direction must be greater, less, or abs_greater")


# =============================================================================
# Expression loading
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


def keep_tcga_primary_tumor_only(expr: pd.DataFrame) -> pd.DataFrame:
    keep = [sid for sid in expr.index if parse_tcga_sample_type(sid) == 1]
    if len(keep) >= 20:
        print(f"[INFO] Keeping TCGA primary tumor samples only: {len(keep)}")
        return expr.loc[keep].copy()
    print("[WARN] Could not identify enough primary tumor samples. Keeping all samples.")
    return expr


def zscore_genes(expr: pd.DataFrame) -> pd.DataFrame:
    mu = expr.mean(axis=0, skipna=True)
    sd = expr.std(axis=0, skipna=True).replace(0, np.nan)
    return ((expr - mu) / sd).replace([np.inf, -np.inf], np.nan)


# =============================================================================
# Clinical endpoint labels
# =============================================================================

def infer_sample_column(df: pd.DataFrame) -> str:
    candidates = [
        "sample_id", "sample", "Sample", "SAMPLE", "barcode", "bcr_patient_barcode",
        "patient_id", "Patient", "ID", "id", "sampleID", "sample_id.samples"
    ]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns[:8]:
        vals = df[c].dropna().astype(str).head(20)
        if len(vals) > 0 and vals.str.contains("TCGA-", regex=False).mean() > 0.5:
            return c
    return df.columns[0]


def clinical_to_patient_index(df: pd.DataFrame, sample_col: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    if sample_col is None:
        sample_col = infer_sample_column(df)
    df[sample_col] = df[sample_col].astype(str)
    df["patient_barcode"] = df[sample_col].map(tcga_patient_barcode)
    df = df.drop_duplicates("patient_barcode")
    df = df.set_index("patient_barcode", drop=False)
    return df


def find_column_by_keywords(df: pd.DataFrame, keyword_groups: List[List[str]]) -> Optional[str]:
    cols = list(df.columns)
    lower = {c: str(c).lower() for c in cols}
    for group in keyword_groups:
        for c in cols:
            s = lower[c]
            if all(k.lower() in s for k in group):
                return c
    return None


def read_clinical_table(path: str) -> pd.DataFrame:
    if path is None:
        raise ValueError("CLINICAL_FILE is required.")
    df = safe_read_table(Path(path))
    print(f"[INFO] Clinical table loaded: rows={df.shape[0]}, cols={df.shape[1]}")
    return df


def make_tumor_normal_labels(samples: Sequence[str]) -> pd.Series:
    labels = {}
    for sid in samples:
        st = parse_tcga_sample_type(sid)
        if st == 11:
            labels[str(sid)] = 0
        elif st == 1:
            labels[str(sid)] = 1
    y = pd.Series(labels, name="label")
    if len(y) < 20:
        raise ValueError("Too few tumor/normal labels inferred from TCGA barcode.")
    return y.astype(int)


def make_stage_labels(clinical: pd.DataFrame) -> pd.Series:
    cdf = clinical_to_patient_index(clinical)
    col = find_column_by_keywords(cdf, [
        ["pathologic", "stage"],
        ["pathologic_stage"],
        ["ajcc", "stage"],
        ["clinical", "stage"],
        ["stage"]
    ])
    if col is None:
        raise ValueError("Cannot find stage column in clinical table.")

    print("[INFO] Stage column:", col)
    s = cdf[col].astype(str).str.lower().str.strip()

    labels = {}
    for pid, val in s.items():
        if val in ["nan", "none", "", "not reported", "not available", "na", "unknown"]:
            continue
        # Stage IV before Stage I to avoid stage i matching stage iv.
        if any(pat in val for pat in STAGE_LATE_PATTERNS):
            labels[pid] = 1
        elif any(pat in val for pat in STAGE_EARLY_PATTERNS):
            labels[pid] = 0

    y = pd.Series(labels, name="label").astype(int)
    if y.nunique() != 2 or len(y) < 30:
        raise ValueError(f"Stage label failed/too small. n={len(y)}, counts={y.value_counts().to_dict()}")
    print("[INFO] Stage label counts:", y.value_counts().to_dict())
    return y


def make_node_labels(clinical: pd.DataFrame) -> pd.Series:
    cdf = clinical_to_patient_index(clinical)
    col = find_column_by_keywords(cdf, [
        ["pathologic", "n"],
        ["ajcc", "n"],
        ["lymph", "node"],
        ["node"],
        ["n_stage"]
    ])
    if col is None:
        raise ValueError("Cannot find node/N-stage column in clinical table.")

    print("[INFO] Node column:", col)
    s = cdf[col].astype(str).str.lower().str.strip()

    labels = {}
    for pid, val in s.items():
        if val in ["nan", "none", "", "not reported", "not available", "na", "nx"]:
            continue
        if re.search(r"\bn0\b", val) or "node negative" in val or val in ["negative", "0"]:
            labels[pid] = 0
        elif re.search(r"\bn1\b", val) or re.search(r"\bn2\b", val) or re.search(r"\bn3\b", val):
            labels[pid] = 1
        elif "positive" in val:
            labels[pid] = 1

    y = pd.Series(labels, name="label").astype(int)
    if y.nunique() != 2 or len(y) < 30:
        raise ValueError(f"Node label failed/too small. n={len(y)}, counts={y.value_counts().to_dict()}")
    print("[INFO] Node label counts:", y.value_counts().to_dict())
    return y


def make_survival_labels(clinical: pd.DataFrame) -> pd.Series:
    cdf = clinical_to_patient_index(clinical)

    # Prefer OS fields if available.
    time_col = find_column_by_keywords(cdf, [
        ["os", "time"],
        ["survival", "time"],
        ["days", "death"],
        ["days", "last", "follow"],
        ["days_to_death"],
        ["days_to_last_followup"]
    ])
    status_col = find_column_by_keywords(cdf, [
        ["os", "event"],
        ["vital", "status"],
        ["death", "event"],
        ["status"]
    ])

    if time_col is None:
        raise ValueError("Cannot find survival time column.")
    print("[INFO] Survival time column:", time_col)
    print("[INFO] Survival status column:", status_col)

    time = pd.to_numeric(cdf[time_col], errors="coerce")

    # Try alternative if current time has too few values.
    if time.notna().sum() < 30:
        alt = find_column_by_keywords(cdf, [["days", "last", "follow"], ["followup"], ["follow_up"]])
        if alt is not None:
            time = pd.to_numeric(cdf[alt], errors="coerce")
            time_col = alt
            print("[INFO] Switched survival time column to:", time_col)

    if status_col is not None:
        status_raw = cdf[status_col].astype(str).str.lower().str.strip()
        event = status_raw.map(lambda x: 1 if ("dead" in x or x in ["1", "deceased", "death", "true"]) else 0)
    else:
        event = pd.Series(0, index=cdf.index)

    valid = time.notna()
    time = time[valid]
    event = event.loc[time.index]

    low_q = time.quantile(SURVIVAL_LOWER_QUANTILE)
    high_q = time.quantile(SURVIVAL_UPPER_QUANTILE)

    labels = {}
    for pid in time.index:
        t = time.loc[pid]
        e = event.loc[pid]
        if e == 1 or t <= low_q:
            labels[pid] = 1
        elif t >= high_q and e == 0:
            labels[pid] = 0

    y = pd.Series(labels, name="label").astype(int)
    if y.nunique() != 2 or len(y) < 30:
        raise ValueError(f"Survival label failed/too small. n={len(y)}, counts={y.value_counts().to_dict()}")
    print("[INFO] Survival label counts:", y.value_counts().to_dict())
    return y


def load_custom_binary_labels(path: str, sample_col: Optional[str], label_col: Optional[str]) -> pd.Series:
    df = safe_read_table(Path(path))
    if sample_col is None:
        sample_col = infer_sample_column(df)

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
        raise ValueError("Cannot infer label column. Set CUSTOM_LABEL_COL.")

    tmp = df[[sample_col, label_col]].dropna().copy()
    tmp["patient_barcode"] = tmp[sample_col].astype(str).map(tcga_patient_barcode)

    raw = tmp[label_col]
    if pd.api.types.is_numeric_dtype(raw):
        ynum = pd.to_numeric(raw, errors="coerce")
        uniq = sorted(ynum.dropna().unique())
        if len(uniq) != 2:
            raise ValueError(f"Numeric custom label must have exactly two classes. Found: {uniq[:10]}")
        y01 = ynum.map({uniq[0]: 0, uniq[1]: 1})
    else:
        s = raw.astype(str).str.lower().str.strip()
        pos = {"1", "late", "high", "positive", "node_positive", "poor", "bad", "advanced", "case", "yes", "true"}
        neg = {"0", "early", "low", "negative", "node_negative", "good", "localized", "control", "no", "false"}

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
                raise ValueError(f"Cannot convert custom labels to binary. Unique labels: {uniq[:20]}")
            y01 = s.map({uniq[0]: 0, uniq[1]: 1})

    y = pd.Series(y01.values, index=tmp["patient_barcode"].values, name="label").dropna().astype(int)
    y = y[~y.index.duplicated(keep="first")]
    print("[INFO] Custom label counts:", y.value_counts().to_dict())
    return y


def build_endpoint_labels(expr: pd.DataFrame) -> pd.Series:
    if ENDPOINT_MODE == "tumor_vs_normal":
        return make_tumor_normal_labels(expr.index)

    if ENDPOINT_MODE == "custom_binary":
        if CUSTOM_LABEL_FILE is None:
            raise ValueError("CUSTOM_LABEL_FILE is required for custom_binary.")
        return load_custom_binary_labels(CUSTOM_LABEL_FILE, CUSTOM_SAMPLE_COL, CUSTOM_LABEL_COL)

    clinical = read_clinical_table(CLINICAL_FILE)

    if ENDPOINT_MODE == "stage_early_late":
        return make_stage_labels(clinical)

    if ENDPOINT_MODE == "node_negative_positive":
        return make_node_labels(clinical)

    if ENDPOINT_MODE == "survival_short_long":
        return make_survival_labels(clinical)

    raise ValueError(f"Unknown ENDPOINT_MODE: {ENDPOINT_MODE}")


def align_expression_endpoint(expr: pd.DataFrame, y_patient: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    expr = expr.copy()
    y_patient = y_patient.copy()
    y_patient.index = y_patient.index.astype(str)

    if ENDPOINT_MODE == "tumor_vs_normal":
        y = y_patient.copy()
        y.index = y.index.astype(str)
        common = sorted(set(expr.index.astype(str)) & set(y.index))
        if len(common) >= 20:
            return expr.loc[common], y.loc[common].astype(int)

    expr["__patient_barcode__"] = [tcga_patient_barcode(x) for x in expr.index]
    expr = expr[expr["__patient_barcode__"].isin(y_patient.index)].copy()
    if expr.shape[0] < 20:
        raise ValueError("Too few expression samples matched endpoint labels.")

    y = pd.Series(expr["__patient_barcode__"].map(y_patient).values, index=expr.index, name="label").astype(int)
    expr = expr.drop(columns=["__patient_barcode__"])

    if y.nunique() != 2:
        raise ValueError("Endpoint labels after alignment do not contain two classes.")

    return expr, y


# =============================================================================
# BP gene sets and activity
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
        raise FileNotFoundError("Cannot find BP gene-set file. Set BP_GENESET_FILE manually.")

    if p.suffix.lower() in [".gmt", ".gmx"]:
        gene_sets = read_gmt(p)
    else:
        gene_sets = read_gene_set_table(p)

    print(f"[INFO] Loaded BP gene sets: {len(gene_sets)}")
    return gene_sets, str(p)


def filter_bp_gene_sets(gene_sets: Dict[str, Set[str]],
                        genes_available: Set[str],
                        min_genes: int,
                        max_genes: int,
                        max_terms: Optional[int]) -> Dict[str, List[str]]:
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
# D-PHY and D-Clinical
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
    idx0 = y[y == 0].index
    idx1 = y[y == 1].index

    rows = []
    for i, bp in enumerate(bp_mat.columns, start=1):
        if i % 500 == 0:
            print(f"[INFO] Computing D-PHY: {i}/{bp_mat.shape[1]}")

        x0 = pd.to_numeric(bp_mat.loc[idx0, bp], errors="coerce")
        x1 = pd.to_numeric(bp_mat.loc[idx1, bp], errors="coerce")

        mean0 = float(np.nanmean(x0))
        mean1 = float(np.nanmean(x1))
        delta = mean1 - mean0

        d = cohen_d(x0.values, x1.values)
        try:
            p = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit").pvalue
        except Exception:
            p = np.nan

        auc = safe_auc(y, bp_mat[bp])
        direction = "endpoint_positive_state_up" if delta > 0 else "endpoint_negative_state_up"

        rows.append({
            "BP_term": bp,
            "bp_gene_count": len(bp_genes.get(bp, [])),
            "mean_endpoint_negative_state": mean0,
            "mean_endpoint_positive_state": mean1,
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
    df["neglog10_fdr_capped50"] = df["welch_fdr"].map(lambda p: cap_neglog10_p(p, 50.0))
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


def compute_dclinical(bp_mat: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, Dict]:
    X = bp_mat.copy()
    y = y.loc[X.index].astype(int)
    nunique = X.nunique(dropna=True)
    X = X[nunique[nunique > 1].index.tolist()]

    min_class = int(y.value_counts().min())
    n_splits = min(CV_SPLITS, min_class)
    if n_splits < 2:
        raise ValueError("Not enough samples in the smaller class for CV.")

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    coef_records = []
    aucs = []
    fold_id = 0

    for train_idx, test_idx in cv.split(X, y):
        fold_id += 1
        if fold_id % 10 == 0:
            print(f"[INFO] D-Clinical CV fold: {fold_id}/{n_splits * CV_REPEATS}")

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
            aucs.append(roc_auc_score(y_test, prob))
        except Exception:
            aucs.append(np.nan)

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
        "n_endpoint_positive": int(y.sum()),
        "n_endpoint_negative": int((1 - y).sum()),
        "cv_splits_used": int(n_splits),
        "cv_auc_mean": float(np.nanmean(aucs)),
        "cv_auc_sd": float(np.nanstd(aucs)),
        "cv_auc_median": float(np.nanmedian(aucs)),
    }
    return out, info


def select_dclinical_bp(dclinical: pd.DataFrame) -> Set[str]:
    return set(dclinical.head(TOP_K_DCLINICAL)["BP_term"].astype(str))


# =============================================================================
# h-layer
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
# Dual-D / BP-Enrichment
# =============================================================================

def build_dual_d_table(dphy: pd.DataFrame, dclinical: pd.DataFrame, h: pd.DataFrame,
                       dphy_selected: Set[str], dclinical_selected: Set[str]) -> pd.DataFrame:
    all_bp = sorted(set(dphy["BP_term"]) | set(dclinical["BP_term"]))
    base = pd.DataFrame({"BP_term": all_bp})

    dphy_cols = [
        "BP_term", "DPHY_rank", "DPHY_selection_score", "D_score_capped50",
        "abs_cohen_d", "cohen_d", "auc_positive_vs_negative", "auc_distance",
        "welch_p", "welch_fdr", "direction", "mean_endpoint_negative_state",
        "mean_endpoint_positive_state", "delta_positive_minus_negative",
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
        if "endpoint_positive_state_up" in direction:
            return "endpoint_positive_state_up"
        if "endpoint_negative_state_up" in direction:
            return "endpoint_negative_state_up"
        coef = row.get("DClinical_mean_coef", np.nan)
        if pd.notna(coef):
            return "endpoint_positive_contributor" if coef > 0 else "endpoint_negative_contributor"
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
        raise ValueError("DUALD_POOL_MODE must be strict or union_all")

    pool = set()
    if INCLUDE_TIER1_ALWAYS:
        pool |= set(dual.loc[dual["DualD_tier"] == "Tier1_core_dual_D", "BP_term"].astype(str))

    pool |= set(
        dual.sort_values("DPHY_selection_score", ascending=False)
        .head(STRICT_TOP_K_DPHY_FOR_MODULES)["BP_term"].astype(str)
    )
    pool |= set(
        dual.sort_values("DClinical_selection_score", ascending=False)
        .head(STRICT_TOP_K_DCLINICAL_FOR_MODULES)["BP_term"].astype(str)
    )
    return pool


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
    print(f"[INFO] BP-Enrichment candidate pool size: {len(candidates)} using mode={DUALD_POOL_MODE}")
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
    members["BP_module_status"] = np.where(members["BP_module_size"] >= MIN_MODULE_SIZE, "module", "singleton_component")

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
# Reconstructed state profile / validation
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
    mod_mat = pd.DataFrame(data, index=bp_mat.index).dropna(axis=1, how="all")
    # Standardize module scores across patients for centroid metrics.
    mu = mod_mat.mean(axis=0)
    sd = mod_mat.std(axis=0).replace(0, np.nan)
    mod_z = ((mod_mat - mu) / sd).replace([np.inf, -np.inf], np.nan)
    return mod_z


def compute_state_discriminability(module_activity: pd.DataFrame, y: pd.Series,
                                   module_summary: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    if module_activity.empty:
        return pd.DataFrame(), {}

    y = y.loc[module_activity.index].astype(int)
    idx0 = y[y == 0].index
    idx1 = y[y == 1].index

    rows = []
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
            "module_mean_endpoint_negative": float(np.nanmean(x0)),
            "module_mean_endpoint_positive": float(np.nanmean(x1)),
            "module_delta_positive_minus_negative": delta,
            "module_cohen_d": d,
            "module_abs_cohen_d": abs(d) if pd.notna(d) else np.nan,
            "module_auc_positive_vs_negative": auc,
            "module_auc_distance": abs(auc - 0.5) if pd.notna(auc) else np.nan,
            "module_welch_p": p,
            "module_direction": "endpoint_positive_state_up" if delta > 0 else "endpoint_negative_state_up",
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

    info = compute_module_classifier_auc(module_activity, y)
    return disc, info


def compute_module_classifier_auc(module_activity: pd.DataFrame, y: pd.Series) -> Dict:
    if module_activity.empty:
        return {}
    y = y.loc[module_activity.index].astype(int)
    X = module_activity.copy()
    nunique = X.nunique(dropna=True)
    X = X[nunique[nunique > 1].index.tolist()]
    min_class = int(y.value_counts().min())
    n_splits = min(CV_SPLITS, min_class)
    if X.shape[1] < 1 or n_splits < 2:
        return {}

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
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

    return {
        "n_reconstructed_modules": int(X.shape[1]),
        "cv_splits_used": int(n_splits),
        "reconstructed_state_cv_auc_mean": float(np.nanmean(aucs)),
        "reconstructed_state_cv_auc_sd": float(np.nanstd(aucs)),
        "reconstructed_state_cv_auc_median": float(np.nanmedian(aucs)),
    }


def cosine_similarity_matrix_to_vector(X: pd.DataFrame, centroid: np.ndarray) -> pd.Series:
    arr = X.values.astype(float)
    c = np.asarray(centroid, dtype=float)
    num = np.nansum(arr * c, axis=1)
    den = np.sqrt(np.nansum(arr ** 2, axis=1)) * np.sqrt(np.nansum(c ** 2))
    out = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den != 0)
    return pd.Series(out, index=X.index)


def euclidean_distance_to_vector(X: pd.DataFrame, centroid: np.ndarray) -> pd.Series:
    arr = X.values.astype(float)
    c = np.asarray(centroid, dtype=float)
    d = np.sqrt(np.nansum((arr - c) ** 2, axis=1))
    return pd.Series(d, index=X.index)


def compute_patient_state_profile(module_activity: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, Dict]:
    if module_activity.empty:
        return pd.DataFrame(), {}

    X = module_activity.copy().apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(axis=0))
    y = y.loc[X.index].astype(int)

    idx0 = y[y == 0].index
    idx1 = y[y == 1].index

    c0 = X.loc[idx0].mean(axis=0).values
    c1 = X.loc[idx1].mean(axis=0).values

    profile = pd.DataFrame(index=X.index)
    profile["label"] = y
    profile["endpoint_negative_centroid_similarity"] = cosine_similarity_matrix_to_vector(X, c0)
    profile["endpoint_positive_centroid_similarity"] = cosine_similarity_matrix_to_vector(X, c1)
    profile["positive_minus_negative_centroid_similarity"] = (
        profile["endpoint_positive_centroid_similarity"] - profile["endpoint_negative_centroid_similarity"]
    )
    profile["distance_to_endpoint_negative_centroid"] = euclidean_distance_to_vector(X, c0)
    profile["distance_to_endpoint_positive_centroid"] = euclidean_distance_to_vector(X, c1)
    profile["negative_minus_positive_distance"] = (
        profile["distance_to_endpoint_negative_centroid"] - profile["distance_to_endpoint_positive_centroid"]
    )

    # Metrics
    metrics = {}
    for metric in [
        "endpoint_positive_centroid_similarity",
        "positive_minus_negative_centroid_similarity",
        "distance_to_endpoint_positive_centroid",
        "negative_minus_positive_distance",
    ]:
        x0 = profile.loc[idx0, metric]
        x1 = profile.loc[idx1, metric]
        d = cohen_d(x0.values, x1.values)
        try:
            p = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit").pvalue
        except Exception:
            p = np.nan
        auc = safe_auc(y, profile[metric])
        metrics[f"{metric}_cohen_d"] = float(d) if pd.notna(d) else np.nan
        metrics[f"{metric}_p"] = float(p) if pd.notna(p) else np.nan
        metrics[f"{metric}_auc"] = float(auc) if pd.notna(auc) else np.nan
        metrics[f"{metric}_auc_distance"] = float(abs(auc - 0.5)) if pd.notna(auc) else np.nan

    return profile.reset_index().rename(columns={"index": "sample_id"}), metrics


def module_size_pattern(module_members: pd.DataFrame) -> List[int]:
    if module_members.empty:
        return []
    return module_members.groupby("BP_module_id").size().astype(int).tolist()


def create_random_module_members(all_bp_terms: List[str], size_pattern: List[int], rng: np.random.Generator) -> pd.DataFrame:
    needed = sum(size_pattern)
    if len(all_bp_terms) < needed:
        sampled = list(rng.choice(all_bp_terms, size=needed, replace=True))
    else:
        sampled = list(rng.choice(all_bp_terms, size=needed, replace=False))

    rows = []
    pos = 0
    for i, size in enumerate(size_pattern, start=1):
        mid = f"R{i:03d}"
        for bp in sampled[pos:pos + size]:
            rows.append({"BP_module_id": mid, "BP_term": bp, "BP_module_name": f"random_module_{i:03d}"})
        pos += size
    return pd.DataFrame(rows)


def summarize_validation_metrics(profile: pd.DataFrame, state_disc: pd.DataFrame, y: pd.Series) -> Dict:
    if profile.empty:
        return {}
    out = {}
    # primary metric
    metric = "positive_minus_negative_centroid_similarity"
    df = profile.set_index("sample_id") if "sample_id" in profile.columns else profile.copy()
    yy = y.loc[df.index]
    x0 = df.loc[yy == 0, metric]
    x1 = df.loc[yy == 1, metric]
    d = cohen_d(x0.values, x1.values)
    try:
        p = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit").pvalue
    except Exception:
        p = np.nan
    out["primary_centroid_similarity_d"] = float(d) if pd.notna(d) else np.nan
    out["primary_centroid_similarity_p"] = float(p) if pd.notna(p) else np.nan
    out["primary_centroid_similarity_auc"] = float(safe_auc(yy, df[metric]))
    out["mean_module_abs_d"] = float(state_disc["module_abs_cohen_d"].mean()) if not state_disc.empty else np.nan
    out["max_module_abs_d"] = float(state_disc["module_abs_cohen_d"].max()) if not state_disc.empty else np.nan
    out["mean_module_auc_distance"] = float(state_disc["module_auc_distance"].mean()) if not state_disc.empty else np.nan
    out["n_nominal_module_p_lt_0_05"] = int((state_disc["module_welch_p"] < 0.05).sum()) if not state_disc.empty else 0
    out["n_module_fdr_lt_0_05"] = int((state_disc["module_welch_fdr"] < 0.05).sum()) if not state_disc.empty else 0
    return out


def run_random_module_baseline(bp_mat: pd.DataFrame, y: pd.Series, module_members: pd.DataFrame,
                               all_candidate_bp: List[str]) -> pd.DataFrame:
    if N_RANDOM_MODULE_BASELINES <= 0 or module_members.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(RANDOM_STATE + 1000)
    size_pattern = module_size_pattern(module_members)
    rows = []
    for i in range(1, N_RANDOM_MODULE_BASELINES + 1):
        if i % 25 == 0:
            print(f"[INFO] Random BP-module baseline: {i}/{N_RANDOM_MODULE_BASELINES}")
        rand_members = create_random_module_members(all_candidate_bp, size_pattern, rng)
        rand_activity = compute_module_activity(bp_mat, rand_members)
        rand_disc, _ = compute_state_discriminability(rand_activity, y, pd.DataFrame())
        rand_profile, _ = compute_patient_state_profile(rand_activity, y)
        met = summarize_validation_metrics(rand_profile, rand_disc, y)
        met["iteration"] = i
        rows.append(met)
    return pd.DataFrame(rows)


def run_label_shuffle_control(module_activity: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    if N_LABEL_SHUFFLES <= 0 or module_activity.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(RANDOM_STATE + 2000)
    rows = []
    for i in range(1, N_LABEL_SHUFFLES + 1):
        if i % 25 == 0:
            print(f"[INFO] Endpoint-label shuffle control: {i}/{N_LABEL_SHUFFLES}")
        y_shuf = pd.Series(rng.permutation(y.values), index=y.index, name="label").astype(int)
        disc, _ = compute_state_discriminability(module_activity, y_shuf, pd.DataFrame())
        prof, _ = compute_patient_state_profile(module_activity, y_shuf)
        met = summarize_validation_metrics(prof, disc, y_shuf)
        met["iteration"] = i
        rows.append(met)
    return pd.DataFrame(rows)


def run_patient_scramble_control(module_activity: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    if N_PATIENT_SCRAMBLES <= 0 or module_activity.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(RANDOM_STATE + 3000)
    rows = []
    for i in range(1, N_PATIENT_SCRAMBLES + 1):
        if i % 25 == 0:
            print(f"[INFO] Patient-scramble control: {i}/{N_PATIENT_SCRAMBLES}")
        scrambled = module_activity.copy()
        for col in scrambled.columns:
            scrambled[col] = rng.permutation(scrambled[col].values)
        disc, _ = compute_state_discriminability(scrambled, y, pd.DataFrame())
        prof, _ = compute_patient_state_profile(scrambled, y)
        met = summarize_validation_metrics(prof, disc, y)
        met["iteration"] = i
        rows.append(met)
    return pd.DataFrame(rows)


def run_bootstrap_state_stability(module_activity: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    if N_BOOTSTRAPS_STATE <= 0 or module_activity.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(RANDOM_STATE + 4000)
    idx0 = np.array(y[y == 0].index)
    idx1 = np.array(y[y == 1].index)
    n0 = max(2, int(len(idx0) * BOOTSTRAP_FRACTION))
    n1 = max(2, int(len(idx1) * BOOTSTRAP_FRACTION))

    rows = []
    for i in range(1, N_BOOTSTRAPS_STATE + 1):
        if i % 25 == 0:
            print(f"[INFO] Bootstrap state stability: {i}/{N_BOOTSTRAPS_STATE}")
        s0 = rng.choice(idx0, size=n0, replace=True)
        s1 = rng.choice(idx1, size=n1, replace=True)
        selected = list(s0) + list(s1)
        Xb = module_activity.loc[selected].copy()
        # Duplicate sample IDs can happen in bootstrap; make them unique.
        Xb.index = [f"B{i}_{j}" for j in range(len(Xb))]
        yb = pd.Series([0] * len(s0) + [1] * len(s1), index=Xb.index, name="label")
        disc, _ = compute_state_discriminability(Xb, yb, pd.DataFrame())
        prof, _ = compute_patient_state_profile(Xb, yb)
        met = summarize_validation_metrics(prof, disc, yb)
        met["iteration"] = i
        rows.append(met)
    return pd.DataFrame(rows)


def run_leave_one_module_sensitivity(module_activity: pd.DataFrame, y: pd.Series,
                                     module_summary: pd.DataFrame) -> pd.DataFrame:
    if module_activity.empty or module_activity.shape[1] <= 1:
        return pd.DataFrame()
    rows = []
    for mid in module_activity.columns:
        reduced = module_activity.drop(columns=[mid])
        disc, info = compute_state_discriminability(reduced, y, module_summary[module_summary["BP_module_id"] != mid] if not module_summary.empty else pd.DataFrame())
        prof, _ = compute_patient_state_profile(reduced, y)
        met = summarize_validation_metrics(prof, disc, y)
        met["left_out_module_id"] = mid
        if not module_summary.empty and mid in set(module_summary["BP_module_id"]):
            sub = module_summary[module_summary["BP_module_id"] == mid].iloc[0]
            met["left_out_module_name"] = sub.get("BP_module_name", "")
            met["left_out_module_size"] = int(sub.get("BP_module_size", 0))
        rows.append(met)
    return pd.DataFrame(rows)


# =============================================================================
# Mechanism/readiness audit
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

        if "positive" in dominant_direction:
            mechanism_hint = "endpoint-positive-state associated BP program"
        elif "negative" in dominant_direction:
            mechanism_hint = "endpoint-negative-state associated BP program"
        else:
            mechanism_hint = "direction unclear; cautious interpretation"

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
    return audit.sort_values("mechanism_anchor_score", ascending=False).reset_index(drop=True)


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
        + q_rank_score(out.get("module_state_information_score", pd.Series(0, index=out.index)), True).fillna(0) * 0.30
        + q_rank_score(out.get("mechanism_anchor_score", pd.Series(0, index=out.index)), True).fillna(0) * 0.25
        + q_rank_score(out.get("n_Tier1_core_dual_D", pd.Series(0, index=out.index)), True).fillna(0) * 0.20
    )

    def readiness_class(row):
        if (
            row.get("readiness_score", 0) >= 0.75
            and row.get("module_state_information_score", 0) >= 0.60
            and row.get("mechanism_anchor_score", 0) >= 0.50
        ):
            return "interpretation_ready_high"
        if row.get("readiness_score", 0) >= 0.55 and row.get("module_state_information_score", 0) >= 0.40:
            return "interpretation_ready_moderate"
        if row.get("module_state_information_score", 0) >= 0.45:
            return "state_informative_but_mechanism_weak"
        return "exploratory_or_weak"

    out["interpretation_readiness_class"] = out.apply(readiness_class, axis=1)
    return out.sort_values("readiness_score", ascending=False).reset_index(drop=True)


# =============================================================================
# Secondary context scan
# =============================================================================

def run_secondary_context_scan(profile: pd.DataFrame, clinical: Optional[pd.DataFrame]) -> pd.DataFrame:
    if not RUN_SECONDARY_CONTEXT_SCAN or profile.empty or clinical is None:
        return pd.DataFrame()
    try:
        cdf = clinical_to_patient_index(clinical)
    except Exception:
        return pd.DataFrame()

    prof = profile.copy()
    prof["patient_barcode"] = prof["sample_id"].map(tcga_patient_barcode)
    metric = "positive_minus_negative_centroid_similarity"

    rows = []
    for group in SECONDARY_CONTEXT_KEYWORDS:
        col = find_column_by_keywords(cdf, [group])
        if col is None:
            continue

        tmp = prof[["patient_barcode", metric]].merge(
            cdf[[col]].reset_index().rename(columns={"index": "patient_barcode"}),
            on="patient_barcode",
            how="left"
        )
        tmp = tmp.dropna()
        if tmp.empty:
            continue

        # clean categories
        tmp[col] = tmp[col].astype(str)
        counts = tmp[col].value_counts()
        keep_cats = counts[counts >= 10].index.tolist()
        tmp = tmp[tmp[col].isin(keep_cats)]
        if tmp[col].nunique() < 2:
            continue

        groups = [sub[metric].values for _, sub in tmp.groupby(col)]
        try:
            kw = stats.kruskal(*groups)
            p = kw.pvalue
            stat = kw.statistic
        except Exception:
            p = np.nan
            stat = np.nan

        rows.append({
            "clinical_column": col,
            "keyword_group": "+".join(group),
            "n_samples": int(tmp.shape[0]),
            "n_groups": int(tmp[col].nunique()),
            "group_counts": json.dumps(tmp[col].value_counts().to_dict()),
            "kruskal_statistic": stat,
            "kruskal_p": p,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["kruskal_fdr"] = bh_fdr(out["kruskal_p"])
        out = out.sort_values("kruskal_p").reset_index(drop=True)
    return out


# =============================================================================
# Figures
# =============================================================================

def make_figures(out_dir: Path, dphy: pd.DataFrame, dclinical: pd.DataFrame, dual: pd.DataFrame,
                 state_disc: pd.DataFrame, mechanism_audit: pd.DataFrame,
                 profile: pd.DataFrame, y: pd.Series,
                 random_df: pd.DataFrame, shuffle_df: pd.DataFrame,
                 scramble_df: pd.DataFrame, bootstrap_df: pd.DataFrame,
                 dclinical_info: Dict, state_info: Dict) -> None:
    if not HAS_MPL:
        return

    fig_dir = ensure_dir(out_dir / "figures")

    def save_barh(df, label_col, value_col, title, xlabel, filename, n=20):
        if df is None or df.empty or value_col not in df.columns:
            return
        top = df.head(n).copy()
        labels = top[label_col].astype(str).str.replace("GOBP_", "", regex=False).str.replace("_", " ").str[:60]
        plt.figure(figsize=(10, max(5, 0.34 * len(top))))
        yy = np.arange(len(top))
        plt.barh(yy, top[value_col])
        plt.yticks(yy, labels)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=300)
        plt.close()

    save_barh(dphy, "BP_term", "abs_cohen_d", "Top D-PHY endpoint BP deviations", "|Cohen d|", "FIG01_top_DPHY_endpoint_BP_deviations.png")
    save_barh(dclinical, "BP_term", "DClinical_mean_abs_coef", "Top D-Clinical endpoint BP contributors", "Mean absolute coefficient", "FIG02_top_DClinical_endpoint_BP_contributors.png")
    save_barh(state_disc, "BP_module_name", "module_state_information_score", "Top reconstructed endpoint-state modules", "State information score", "FIG03_top_reconstructed_state_modules.png")
    save_barh(mechanism_audit, "BP_module_name", "mechanism_anchor_score", "Top oncogene/TSG anchored modules", "Mechanism anchor score", "FIG04_top_mechanism_anchored_modules.png")

    # Dual-D evidence
    if dual is not None and not dual.empty:
        plt.figure(figsize=(7, 6))
        for tier, sub in dual.groupby("DualD_tier"):
            plt.scatter(sub["DPHY_selection_score"], sub["DClinical_selection_score"], s=18, alpha=0.65, label=tier.replace("_", " "))
        plt.xlabel("D-PHY selection score")
        plt.ylabel("D-Clinical selection score")
        plt.title("Dual-D endpoint BP evidence space")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / "FIG05_DualD_endpoint_BP_evidence_space.png", dpi=300)
        plt.close()

    # Patient-level centroid similarity
    if profile is not None and not profile.empty:
        metric = "positive_minus_negative_centroid_similarity"
        pidx = profile.set_index("sample_id")
        yy = y.loc[pidx.index]
        vals0 = pidx.loc[yy == 0, metric]
        vals1 = pidx.loc[yy == 1, metric]
        plt.figure(figsize=(6, 5))
        plt.boxplot([vals0.dropna(), vals1.dropna()], labels=["Endpoint negative", "Endpoint positive"], showfliers=False)
        plt.scatter(np.random.normal(1, 0.04, len(vals0)), vals0, s=8, alpha=0.45)
        plt.scatter(np.random.normal(2, 0.04, len(vals1)), vals1, s=8, alpha=0.45)
        plt.ylabel("Positive-minus-negative centroid similarity")
        plt.title("Patient-level reconstructed endpoint-state alignment")
        plt.tight_layout()
        plt.savefig(fig_dir / "FIG06_patient_centroid_similarity.png", dpi=300)
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
        plt.title("Endpoint discriminability comparison")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "FIG07_endpoint_AUC_comparison.png", dpi=300)
        plt.close()

    # Null distributions
    observed_metric = None
    if profile is not None and not profile.empty:
        met = summarize_validation_metrics(profile, state_disc, y)
        observed_metric = met.get("primary_centroid_similarity_d", None)

    for df, name, title in [
        (random_df, "FIG08_random_module_baseline.png", "Random BP-module baseline"),
        (shuffle_df, "FIG09_label_shuffle_control.png", "Endpoint-label shuffle control"),
        (scramble_df, "FIG10_patient_scramble_control.png", "Patient-scramble control"),
        (bootstrap_df, "FIG11_bootstrap_state_stability.png", "Bootstrap state stability"),
    ]:
        if df is None or df.empty or "primary_centroid_similarity_d" not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(df["primary_centroid_similarity_d"].dropna(), bins=30)
        if observed_metric is not None:
            plt.axvline(observed_metric, linestyle="--")
        plt.xlabel("Primary centroid similarity Cohen d")
        plt.ylabel("Count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=300)
        plt.close()


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline() -> Path:
    out_dir = ensure_dir(Path(OUT_DIR))

    bp_gene_sets_raw, bp_gene_set_path = load_bp_gene_sets(BP_GENESET_FILE)

    config = {
        "CANCER_TYPE": CANCER_TYPE,
        "GE_FILE": GE_FILE,
        "CLINICAL_FILE": CLINICAL_FILE,
        "BP_GENESET_FILE": bp_gene_set_path,
        "OUT_DIR": OUT_DIR,
        "ENDPOINT_MODE": ENDPOINT_MODE,
        "KEEP_TCGA_PRIMARY_TUMOR_ONLY": KEEP_TCGA_PRIMARY_TUMOR_ONLY,
        "DUALD_POOL_MODE": DUALD_POOL_MODE,
        "TOP_K_DPHY": TOP_K_DPHY,
        "TOP_K_DCLINICAL": TOP_K_DCLINICAL,
        "N_RANDOM_MODULE_BASELINES": N_RANDOM_MODULE_BASELINES,
        "N_LABEL_SHUFFLES": N_LABEL_SHUFFLES,
        "N_PATIENT_SCRAMBLES": N_PATIENT_SCRAMBLES,
        "N_BOOTSTRAPS_STATE": N_BOOTSTRAPS_STATE,
        "note": "No TTU/drug. BRCA completion version: D-PHY/D-Clinical audited BP-Enrichment state reconstruction with old-version-style validation controls for direct comparison with the submitted IMU BPState line."
    }
    (out_dir / "00_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Load expression and endpoint
    expr = load_expression_matrix(GE_FILE)
    if ENDPOINT_MODE != "tumor_vs_normal" and KEEP_TCGA_PRIMARY_TUMOR_ONLY:
        expr = keep_tcga_primary_tumor_only(expr)

    clinical = None
    if ENDPOINT_MODE not in ["tumor_vs_normal", "custom_binary"]:
        clinical = read_clinical_table(CLINICAL_FILE)

    if ENDPOINT_MODE == "tumor_vs_normal":
        y_endpoint = make_tumor_normal_labels(expr.index)
    elif ENDPOINT_MODE == "custom_binary":
        y_endpoint = load_custom_binary_labels(CUSTOM_LABEL_FILE, CUSTOM_SAMPLE_COL, CUSTOM_LABEL_COL)
    elif ENDPOINT_MODE == "stage_early_late":
        y_endpoint = make_stage_labels(clinical)
    elif ENDPOINT_MODE == "node_negative_positive":
        y_endpoint = make_node_labels(clinical)
    elif ENDPOINT_MODE == "survival_short_long":
        y_endpoint = make_survival_labels(clinical)
    else:
        raise ValueError(f"Unknown ENDPOINT_MODE: {ENDPOINT_MODE}")

    expr, y = align_expression_endpoint(expr, y_endpoint)
    print(f"[INFO] After endpoint alignment: samples={expr.shape[0]}, genes={expr.shape[1]}")
    print(f"[INFO] Endpoint label counts: negative={int((y == 0).sum())}, positive={int((y == 1).sum())}")

    labels_out = pd.DataFrame({"sample_id": y.index, "label": y.values})
    labels_out["patient_barcode"] = labels_out["sample_id"].map(tcga_patient_barcode)
    labels_out["endpoint_mode"] = ENDPOINT_MODE
    labels_out["cancer_type"] = CANCER_TYPE
    labels_out["label_name"] = np.where(labels_out["label"] == 1, "endpoint_positive", "endpoint_negative")
    labels_out.to_csv(out_dir / "01_endpoint_labels.csv", index=False)

    if SAVE_EXPRESSION_MATRIX:
        expr.to_csv(out_dir / "01b_expression_endpoint_sample_gene_matrix.csv.gz", compression="gzip")

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
    dphy = bootstrap_dphy_stability(bp_mat, y, dphy, BOOTSTRAP_DPHY_N, BOOTSTRAP_DPHY_TOP_K, RANDOM_STATE)
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

    # Reconstructed state
    module_activity = compute_module_activity(bp_mat, module_members)
    module_activity.to_csv(out_dir / "09_reconstructed_state_module_activity.csv.gz", compression="gzip")
    state_disc, state_info = compute_state_discriminability(module_activity, y, module_summary)
    state_disc.to_csv(out_dir / "10_reconstructed_state_discriminability.csv", index=False)

    profile, centroid_metrics = compute_patient_state_profile(module_activity, y)
    profile.to_csv(out_dir / "11_patient_level_state_profile.csv", index=False)

    centroid_metrics_df = pd.DataFrame([centroid_metrics])
    centroid_metrics_df.to_csv(out_dir / "12_centroid_state_metrics.csv", index=False)

    # Validation controls
    all_candidate_bp = sorted(get_module_candidate_pool(dual))
    random_df = run_random_module_baseline(bp_mat, y, module_members, all_candidate_bp)
    random_df.to_csv(out_dir / "13_random_BP_module_baseline.csv", index=False)

    shuffle_df = run_label_shuffle_control(module_activity, y)
    shuffle_df.to_csv(out_dir / "14_endpoint_label_shuffle_control.csv", index=False)

    scramble_df = run_patient_scramble_control(module_activity, y)
    scramble_df.to_csv(out_dir / "15_patient_scramble_control.csv", index=False)

    bootstrap_df = run_bootstrap_state_stability(module_activity, y)
    bootstrap_df.to_csv(out_dir / "16_bootstrap_state_stability.csv", index=False)

    leave_one_df = run_leave_one_module_sensitivity(module_activity, y, module_summary)
    leave_one_df.to_csv(out_dir / "17_leave_one_module_sensitivity.csv", index=False)

    # Mechanism/readiness
    mechanism_audit = build_module_mechanism_audit(module_members, module_summary)
    mechanism_audit.to_csv(out_dir / "18_module_oncogene_TSG_mechanism_audit.csv", index=False)

    readiness = build_interpretation_readiness(module_summary, state_disc, mechanism_audit)
    readiness.to_csv(out_dir / "19_interpretation_readiness_audit.csv", index=False)

    # Secondary context scan
    secondary_df = run_secondary_context_scan(profile, clinical)
    secondary_df.to_csv(out_dir / "19b_secondary_clinical_context_scan.csv", index=False)

    # Empirical control summary
    observed_validation = summarize_validation_metrics(profile, state_disc, y)
    control_summary = {
        "observed": observed_validation,
        "random_module_empirical_p_primary_d": empirical_p_value(
            observed_validation.get("primary_centroid_similarity_d", np.nan),
            random_df["primary_centroid_similarity_d"] if not random_df.empty else [],
            direction="greater"
        ),
        "label_shuffle_empirical_p_primary_d": empirical_p_value(
            observed_validation.get("primary_centroid_similarity_d", np.nan),
            shuffle_df["primary_centroid_similarity_d"] if not shuffle_df.empty else [],
            direction="greater"
        ),
        "patient_scramble_empirical_p_primary_d": empirical_p_value(
            observed_validation.get("primary_centroid_similarity_d", np.nan),
            scramble_df["primary_centroid_similarity_d"] if not scramble_df.empty else [],
            direction="greater"
        ),
        "bootstrap_primary_d_median": float(bootstrap_df["primary_centroid_similarity_d"].median()) if not bootstrap_df.empty else np.nan,
        "bootstrap_primary_d_p05": float(bootstrap_df["primary_centroid_similarity_d"].quantile(0.05)) if not bootstrap_df.empty else np.nan,
        "bootstrap_primary_d_p95": float(bootstrap_df["primary_centroid_similarity_d"].quantile(0.95)) if not bootstrap_df.empty else np.nan,
        "leave_one_primary_d_min": float(leave_one_df["primary_centroid_similarity_d"].min()) if not leave_one_df.empty else np.nan,
        "leave_one_primary_d_median": float(leave_one_df["primary_centroid_similarity_d"].median()) if not leave_one_df.empty else np.nan,
    }

    # Old IMU-line comparison summary.
    # These are not copied outputs from the old manuscript; they are comparison descriptors
    # generated from the current run so that the new BRCA completion result can be compared
    # side-by-side with the already-submitted IMU BPState line.
    old_version_comparison = {
        "old_IMU_line": {
            "selection_logic": "nominal task-discriminative BP selection",
            "reported_BP_terms": 30,
            "reported_modules_components": 7,
            "main_state_metric": "late-centroid similarity",
            "major_validation_layers": [
                "random BP-module baseline",
                "stage-label shuffling",
                "patient scrambling",
                "bootstrap",
                "leave-one-module",
                "PAM50 and survival context"
            ],
            "does_include_DPHY": False,
            "does_include_DClinical": False,
            "does_include_oncogene_TSG_audit": False
        },
        "current_BRCA_V2_1_line": {
            "selection_logic": "D-PHY + D-Clinical + Dual-D evidence stratification",
            "D_PHY_selected_BP": int(len(dphy_selected)),
            "D_Clinical_selected_BP": int(len(dclinical_selected)),
            "DualD_core_BP": int(dual["selected_by_dualD_intersection"].sum()),
            "BP_modules_components": int(module_summary.shape[0]) if module_summary is not None else 0,
            "main_state_metric": "positive-minus-negative endpoint centroid similarity",
            "major_validation_layers": [
                "random BP-module baseline",
                "endpoint-label shuffling",
                "patient scrambling",
                "bootstrap",
                "leave-one-module",
                "oncogene/TSG mechanism audit",
                "interpretation-readiness audit"
            ],
            "does_include_DPHY": True,
            "does_include_DClinical": True,
            "does_include_oncogene_TSG_audit": True
        },
        "intended_interpretation": (
            "The old IMU line demonstrates feasibility of patient-level BP-state reconstruction. "
            "The current BRCA V2.1 line tests whether D-PHY/D-Clinical audited BP selection can "
            "support BP-Enrichment state reconstruction and mechanism-readiness audit."
        )
    }
    (out_dir / "19c_old_IMU_vs_current_BRCA_V2_1_comparison.json").write_text(
        json.dumps(old_version_comparison, indent=2), encoding="utf-8"
    )

    # Figures
    make_figures(
        out_dir, dphy, dclinical, dual, state_disc, mechanism_audit,
        profile, y, random_df, shuffle_df, scramble_df, bootstrap_df,
        dclinical_info, state_info
    )

    tier_counts = dual.loc[dual["selected_by_dualD_union"], "DualD_tier"].value_counts().to_dict()
    readiness_counts = readiness["interpretation_readiness_class"].value_counts().to_dict() if not readiness.empty else {}

    run_summary = {
        "run_timestamp": now_stamp(),
        "cancer_type": CANCER_TYPE,
        "endpoint_mode": ENDPOINT_MODE,
        "n_endpoint_samples": int(bp_mat.shape[0]),
        "n_endpoint_negative": int((y == 0).sum()),
        "n_endpoint_positive": int((y == 1).sum()),
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
        "centroid_state_metrics": centroid_metrics,
        "validation_control_summary": control_summary,
        "oncogene_TSG_audit": {
            "n_mechanism_audited_modules": int(mechanism_audit.shape[0]) if not mechanism_audit.empty else 0,
            "n_oncogene_anchored_modules": int((mechanism_audit["module_oncogene_count"] > 0).sum()) if not mechanism_audit.empty else 0,
            "n_TSG_anchored_modules": int((mechanism_audit["module_TSG_count"] > 0).sum()) if not mechanism_audit.empty else 0,
        },
        "interpretation_readiness_counts": {str(k): int(v) for k, v in readiness_counts.items()},
        "secondary_context_scan_n": int(secondary_df.shape[0]) if not secondary_df.empty else 0,
        "old_IMU_vs_current_BRCA_V2_1_comparison": old_version_comparison,
        "interpretation": (
            "This BRCA V2.1 completion run is designed for direct comparison with the old IMU Post-D/BPState submission, while adding D-PHY/D-Clinical/Dual-D audit layers. It uses D-PHY and D-Clinical "
            "to audit BP selection before BP-Enrichment state reconstruction, then applies patient-level centroid "
            "state profiling, random controls, bootstrap, leave-one-module sensitivity, and oncogene/TSG mechanism audit."
        )
    }
    (out_dir / "20_endpoint_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    # Excel workbook
    try:
        with pd.ExcelWriter(out_dir / "21_AIDO_DPHY_DClinical_StateValidation_summary.xlsx", engine="openpyxl") as writer:
            pd.DataFrame([run_summary]).to_excel(writer, sheet_name="run_summary", index=False)
            labels_out.to_excel(writer, sheet_name="endpoint_labels", index=False)
            dphy.head(500).to_excel(writer, sheet_name="DPHY_top500", index=False)
            dclinical.head(500).to_excel(writer, sheet_name="DClinical_top500", index=False)
            dual.to_excel(writer, sheet_name="DualD_candidates", index=False)
            h.head(1000).to_excel(writer, sheet_name="h_layer_top1000", index=False)
            module_summary.to_excel(writer, sheet_name="BP_modules", index=False)
            module_members.to_excel(writer, sheet_name="module_members", index=False)
            state_disc.to_excel(writer, sheet_name="state_discriminability", index=False)
            profile.to_excel(writer, sheet_name="patient_state_profile", index=False)
            pd.DataFrame([centroid_metrics]).to_excel(writer, sheet_name="centroid_metrics", index=False)
            random_df.head(5000).to_excel(writer, sheet_name="random_baseline", index=False)
            shuffle_df.head(5000).to_excel(writer, sheet_name="label_shuffle", index=False)
            scramble_df.head(5000).to_excel(writer, sheet_name="patient_scramble", index=False)
            bootstrap_df.head(5000).to_excel(writer, sheet_name="bootstrap", index=False)
            leave_one_df.to_excel(writer, sheet_name="leave_one_module", index=False)
            mechanism_audit.to_excel(writer, sheet_name="oncogene_TSG_audit", index=False)
            readiness.to_excel(writer, sheet_name="readiness_audit", index=False)
            secondary_df.to_excel(writer, sheet_name="secondary_context", index=False)
            pd.DataFrame([old_version_comparison["current_BRCA_V2_1_line"]]).to_excel(writer, sheet_name="old_vs_current", index=False)
    except Exception as e:
        print("[WARN] Excel export failed; CSV outputs are still available:", e)

    readme = f"""AIDO D-PHY/D-Clinical BRCA Endpoint-State Reconstruction Completion V2.1
======================================================================

No TTU. No drug task.

Cancer type: {CANCER_TYPE}
Endpoint mode: {ENDPOINT_MODE}

This pipeline is designed to be separated from the old submitted IMU Post-D/BPState
BRCA manuscript. It adds D-PHY, D-Clinical, Dual-D evidence stratification,
oncogene/TSG mechanism audit, and interpretation-readiness classes.

Main counts
-----------
Endpoint samples: {bp_mat.shape[0]}
Endpoint negative: {(y == 0).sum()}
Endpoint positive: {(y == 1).sum()}
Genes: {expr.shape[1]}
BP terms: {bp_mat.shape[1]}

D-PHY selected: {len(dphy_selected)}
D-Clinical selected: {len(dclinical_selected)}
Dual-D union: {int(dual['selected_by_dualD_union'].sum())}
Dual-D core/intersection: {int(dual['selected_by_dualD_intersection'].sum())}

D-Clinical:
{json.dumps(dclinical_info, indent=2)}

Reconstructed state:
{json.dumps(state_info, indent=2)}

Centroid profile:
{json.dumps(centroid_metrics, indent=2)}

Validation controls:
{json.dumps(control_summary, indent=2)}

Interpretation readiness:
{json.dumps({str(k): int(v) for k, v in readiness_counts.items()}, indent=2)}

Notes
-----
AUC close to 0 can still be informative if direction is reversed. Use AUC distance.

This BRCA V2.1 pipeline should be used first to understand the difference from the submitted IMU BRCA line; after that, switch CANCER_TYPE/GE_FILE/CLINICAL_FILE to KIRC/KIRP if needed.
"""
    (out_dir / "README_AIDO_DPHY_DClinical_StateValidation_V2.txt").write_text(readme, encoding="utf-8")

    zip_path = out_dir.parent / f"{out_dir.name}.zip"
    zip_dir(out_dir, zip_path)

    print("\n[DONE] AIDO D-PHY/D-Clinical Multi-Cancer Endpoint-State Reconstruction V2 completed.")
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
