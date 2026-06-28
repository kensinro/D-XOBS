#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DUAL-D BP Enrichment / BP-State Module Reconstruction Pipeline V1
================================================================

Purpose
-------
This script treats D as a modular selection layer:

    D-module = D-PHY + D-Clinical

It then passes the D-selected BP candidate pool into a downstream
BP-enrichment / BP-module reconstruction operation.

Core idea
---------
1. D-PHY:
   Local single-BP state deviation / process-level deviation.
   It is read from an existing D-PHY result table, usually:
       03_D_layer_ranked_BP_signals.csv

2. D-Clinical:
   Global clinical/cancer-state contribution in BP space.
   It is computed from a BP activity matrix and labels using repeated
   stratified logistic regression. Stable coefficients define
   clinical-state-contributive BPs.

3. Dual-D integration:
   Candidate BP pool = union(D-PHY-selected BP, D-Clinical-selected BP)
   Tier labels:
       - core_dual_D
       - DPHY_only
       - DClinical_only
       - weak_or_unselected

4. BP-Enrichment / BP-module reconstruction:
   D-selected BP candidates are grouped into BP-state modules using:
       - gene-overlap/Jaccard similarity when a GMT/BP-gene-set file is available
       - lexical fallback when gene sets are unavailable

5. Optional evidence merge:
   h-layer biological anchoring, PPI/network support, and drug/reversal
   annotations can be merged if available.

Outputs
-------
The script writes a complete output folder and a ZIP package containing:
    01_DPHY_input_normalized.csv
    02_DClinical_BP_coefficients.csv
    03_DualD_BP_candidates.csv
    04_BPenrichment_module_members.csv
    05_BPenrichment_module_summary.csv
    06_DualD_run_summary.json
    07_DualD_summary.xlsx
    figures/*.png
    DUAL_D_BP_Enrichment_*.zip

Recommended first use
---------------------
For TCGA tumor-vs-normal:
    python run_dual_d_bp_enrichment_v1.py ^
        --run_dir "D:/AIDO-Temp/D_PHY_TCGA_BRCA_tumor_vs_normal" ^
        --out_dir "D:/AIDO-Temp/DUAL_D_TCGA_TumorVsNormal_V1" ^
        --label_mode tumor_vs_normal ^
        --top_k_dphy 50 ^
        --top_k_dclinical 50

For a folder that already contains:
    03_D_layer_ranked_BP_signals.csv
    BP activity matrix
    phenotype/label file

The script will try to auto-detect common file names.
"""


import argparse
import json
import math
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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


def safe_read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read csv/tsv/txt/xlsx with simple delimiter auto-detection."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, **kwargs)

    if suffix in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)

    # try CSV first, then TSV
    try:
        df = pd.read_csv(path, low_memory=False, **kwargs)
        if df.shape[1] == 1:
            df2 = pd.read_csv(path, sep="\t", low_memory=False, **kwargs)
            if df2.shape[1] > df.shape[1]:
                return df2
        return df
    except Exception:
        return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)


def find_first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def normalize_bp_name(x: str) -> str:
    x = str(x)
    x = x.strip()
    x = x.replace(" ", "_")
    return x


def infer_bp_column(df: pd.DataFrame) -> str:
    candidates = ["BP_term", "bp_term", "pathway", "Pathway", "term", "Term", "gene_set", "GeneSet", "NAME", "name"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[0]


def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def q_rank_score(values: pd.Series, larger_better: bool = True) -> pd.Series:
    """Convert a numeric series into 0-1 percentile rank."""
    x = pd.to_numeric(values, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index)
    r = x.rank(pct=True, ascending=not larger_better)
    return r.fillna(0.0)


def cap_neglog10_p(p: float, cap: float = 50.0) -> float:
    try:
        p = float(p)
    except Exception:
        return np.nan
    if not np.isfinite(p) or p <= 0:
        return cap
    return min(-math.log10(max(p, 1e-300)), cap)


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
# Auto-detection
# =============================================================================

def autodetect_dphy_file(run_dir: Path) -> Optional[Path]:
    patterns = [
        "03_D_layer_ranked_BP_signals.csv",
        "*D_layer*ranked*BP*.csv",
        "*ranked_BP_signals*.csv",
        "*D_PHY*BP*.csv",
    ]
    for pat in patterns:
        hits = sorted(Path(run_dir).rglob(pat))
        if hits:
            return hits[0]
    return None


def autodetect_bp_matrix(run_dir: Path) -> Optional[Path]:
    patterns = [
        "01_BP_activity_matrix.csv",
        "BP_activity_matrix.csv",
        "*BP*activity*matrix*.csv",
        "*bp*activity*matrix*.csv",
        "*BP_matrix*.csv",
        "*bp_matrix*.csv",
        "*activity*.csv",
    ]
    for pat in patterns:
        hits = sorted(Path(run_dir).rglob(pat))
        # avoid summary/ranked tables
        hits = [h for h in hits if "ranked" not in h.name.lower() and "summary" not in h.name.lower()]
        if hits:
            return hits[0]
    return None


def autodetect_label_file(run_dir: Path) -> Optional[Path]:
    patterns = [
        "02_sample_labels.csv",
        "sample_labels.csv",
        "phenotype.csv",
        "phenotype.tsv",
        "Phenotype.tsv",
        "*label*.csv",
        "*label*.tsv",
        "*phenotype*.csv",
        "*phenotype*.tsv",
    ]
    for pat in patterns:
        hits = sorted(Path(run_dir).rglob(pat))
        if hits:
            return hits[0]
    return None


def autodetect_h_layer(run_dir: Path) -> Optional[Path]:
    patterns = [
        "05_h_layer_biological_anchoring.csv",
        "*h_layer*biological*anchoring*.csv",
        "*biological_anchoring*.csv",
    ]
    for pat in patterns:
        hits = sorted(Path(run_dir).rglob(pat))
        if hits:
            return hits[0]
    return None


def autodetect_ppi_layer(run_dir: Path) -> Optional[Path]:
    patterns = [
        "06_PPI_network_support.csv",
        "*PPI*support*.csv",
        "*network*support*.csv",
    ]
    for pat in patterns:
        hits = sorted(Path(run_dir).rglob(pat))
        if hits:
            return hits[0]
    return None


# =============================================================================
# D-PHY input
# =============================================================================

def load_dphy_table(path: Path) -> pd.DataFrame:
    df = safe_read_table(path)
    bp_col = infer_bp_column(df)
    if bp_col != "BP_term":
        df = df.rename(columns={bp_col: "BP_term"})
    df["BP_term"] = df["BP_term"].map(normalize_bp_name)

    # Standardize expected columns if present
    rename_map = {
        "auc": "auc_late_vs_early",
        "AUC": "auc_late_vs_early",
        "cohen_d": "cohen_d",
        "abs_d": "abs_cohen_d",
        "fdr": "welch_fdr",
        "FDR": "welch_fdr",
        "direction_label": "direction",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Numeric conversions
    for c in ["D_score", "abs_cohen_d", "cohen_d", "auc_late_vs_early", "welch_fdr",
              "bootstrap_topk_stability", "permutation_fdr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Add capped D score if possible
    if "D_score_capped50" not in df.columns:
        if {"abs_cohen_d", "auc_late_vs_early", "welch_fdr"}.issubset(df.columns):
            auc_dist = (df["auc_late_vs_early"] - 0.5).abs()
            neglog = df["welch_fdr"].map(lambda p: cap_neglog10_p(p, cap=50.0))
            df["D_score_capped50"] = df["abs_cohen_d"].fillna(0) * (1 + 2 * auc_dist.fillna(0)) * neglog.fillna(0)
        elif "D_score" in df.columns:
            df["D_score_capped50"] = df["D_score"].clip(upper=df["D_score"].quantile(0.99))
        else:
            df["D_score_capped50"] = np.nan

    # D-PHY selection score: robust composite
    # Prefer D_score_capped50, but also include effect and AUC if available.
    parts = []
    if "D_score_capped50" in df.columns:
        parts.append(q_rank_score(df["D_score_capped50"], larger_better=True))
    if "abs_cohen_d" in df.columns:
        parts.append(q_rank_score(df["abs_cohen_d"], larger_better=True))
    if "auc_late_vs_early" in df.columns:
        parts.append(q_rank_score((df["auc_late_vs_early"] - 0.5).abs(), larger_better=True))
    if "welch_fdr" in df.columns:
        parts.append(q_rank_score(-np.log10(df["welch_fdr"].clip(lower=1e-300)), larger_better=True))

    if parts:
        df["DPHY_selection_score"] = pd.concat(parts, axis=1).mean(axis=1)
    elif "D_score" in df.columns:
        df["DPHY_selection_score"] = q_rank_score(df["D_score"], larger_better=True)
    else:
        # fallback to table order
        df["DPHY_selection_score"] = np.linspace(1, 0, len(df))

    df = df.sort_values("DPHY_selection_score", ascending=False).reset_index(drop=True)
    df["DPHY_rank"] = np.arange(1, len(df) + 1)
    return df


def select_dphy_bp(df: pd.DataFrame,
                   top_k: int = 50,
                   fdr_cutoff: float = 0.05,
                   abs_d_cutoff: float = 0.30,
                   use_threshold_or_topk: bool = True) -> Set[str]:
    selected = set(df.head(top_k)["BP_term"].astype(str))

    if use_threshold_or_topk and {"welch_fdr", "abs_cohen_d"}.issubset(df.columns):
        mask = (df["welch_fdr"] <= fdr_cutoff) & (df["abs_cohen_d"].abs() >= abs_d_cutoff)
        selected |= set(df.loc[mask, "BP_term"].astype(str))

    return selected


# =============================================================================
# BP matrix and labels
# =============================================================================

def load_bp_matrix(path: Path) -> pd.DataFrame:
    df = safe_read_table(path)
    # Determine whether first column is sample ID or BP term
    first = df.columns[0]
    if first.lower() in ["sample", "sample_id", "samples", "id", "patient", "patient_id"]:
        df = df.rename(columns={first: "sample_id"})
        df["sample_id"] = df["sample_id"].astype(str)
        df = df.set_index("sample_id")
    else:
        # If first column is nonnumeric and many other numeric, use it as index
        if not pd.api.types.is_numeric_dtype(df[first]):
            numeric_frac = df.drop(columns=[first]).apply(lambda s: pd.to_numeric(s, errors="coerce").notna().mean()).mean()
            if numeric_frac > 0.70:
                df = df.rename(columns={first: "sample_id"})
                df["sample_id"] = df["sample_id"].astype(str)
                df = df.set_index("sample_id")

    # Convert all columns to numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remove all-empty columns and rows
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = [normalize_bp_name(c) for c in df.columns]
    df.index = df.index.astype(str)
    return df


def parse_tcga_sample_type(sample_id: str) -> Optional[int]:
    """
    TCGA barcode sample type:
      01 = primary tumor
      11 = solid tissue normal
    """
    s = str(sample_id)
    parts = s.split("-")
    if len(parts) >= 4:
        code = parts[3][:2]
        if code.isdigit():
            return int(code)
    return None


def labels_from_tcga_barcodes(samples: Sequence[str], label_mode: str) -> Optional[pd.Series]:
    if label_mode != "tumor_vs_normal":
        return None
    labels = {}
    for sid in samples:
        st = parse_tcga_sample_type(str(sid))
        if st == 11:
            labels[sid] = 0
        elif st == 1:
            labels[sid] = 1
    if len(labels) < 10:
        return None
    return pd.Series(labels, name="label")


def load_labels(path: Optional[Path],
                bp_matrix: pd.DataFrame,
                label_col: Optional[str] = None,
                sample_col: Optional[str] = None,
                label_mode: str = "auto") -> pd.Series:
    # If label file provided
    if path is not None and Path(path).exists():
        df = safe_read_table(path)

        if sample_col is None:
            for c in ["sample_id", "sample", "Sample", "SAMPLE", "patient_id", "Patient", "id", "ID"]:
                if c in df.columns:
                    sample_col = c
                    break
        if sample_col is None:
            sample_col = df.columns[0]

        if label_col is None:
            for c in ["label", "class", "target", "y", "phenotype", "status", "group",
                      "endpoint", "tumor_vs_normal", "early_vs_late", "node_status"]:
                if c in df.columns:
                    label_col = c
                    break
        if label_col is None:
            # choose first non-sample column with 2 unique values if possible
            for c in df.columns:
                if c == sample_col:
                    continue
                vals = df[c].dropna().astype(str).unique()
                if 2 <= len(vals) <= 5:
                    label_col = c
                    break
        if label_col is None:
            raise ValueError("Cannot infer label column. Please provide --label_col.")

        tmp = df[[sample_col, label_col]].dropna().copy()
        tmp[sample_col] = tmp[sample_col].astype(str)
        y_raw = tmp[label_col]

        # convert labels to 0/1
        if pd.api.types.is_numeric_dtype(y_raw):
            y = pd.to_numeric(y_raw, errors="coerce")
            uniq = sorted(y.dropna().unique())
            if len(uniq) > 2:
                raise ValueError(f"Label column has more than two numeric classes: {uniq[:10]}")
            mapping = {uniq[0]: 0, uniq[-1]: 1}
            y01 = y.map(mapping)
        else:
            s = y_raw.astype(str).str.strip().str.lower()
            positive_tokens = {"1", "tumor", "tumour", "late", "late_like", "high", "positive", "node_positive",
                               "node-positive", "poor", "case", "cancer", "yes", "true", "response", "pcr"}
            negative_tokens = {"0", "normal", "early", "early_like", "low", "negative", "node_negative",
                               "node-negative", "good", "control", "no", "false", "rd"}
            y01 = []
            for v in s:
                if v in positive_tokens:
                    y01.append(1)
                elif v in negative_tokens:
                    y01.append(0)
                else:
                    y01.append(np.nan)
            y01 = pd.Series(y01, index=tmp.index)

            if y01.notna().sum() < 10:
                # Fallback: two categories alphabetical
                uniq = sorted(s.dropna().unique())
                if len(uniq) != 2:
                    raise ValueError(f"Cannot convert labels to binary. Unique labels: {uniq[:20]}")
                mapping = {uniq[0]: 0, uniq[1]: 1}
                y01 = s.map(mapping)

        labels = pd.Series(y01.values, index=tmp[sample_col].astype(str).values, name="label").dropna().astype(int)
        return labels

    # Auto from TCGA barcode
    y = labels_from_tcga_barcodes(bp_matrix.index, label_mode=label_mode)
    if y is not None:
        return y.astype(int)

    raise FileNotFoundError(
        "No label file provided/found and labels could not be inferred. "
        "Please provide --label_file and optionally --sample_col/--label_col."
    )


def align_matrix_and_labels(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    common = [s for s in X.index.astype(str) if s in set(y.index.astype(str))]
    if len(common) < 20:
        # try truncating TCGA barcodes to first 15 chars
        X2 = X.copy()
        X2.index = X2.index.astype(str).str[:15]
        y2 = y.copy()
        y2.index = y2.index.astype(str).str[:15]
        common = sorted(set(X2.index) & set(y2.index))
        if len(common) >= 20:
            X_aligned = X2.loc[common]
            y_aligned = y2.loc[common]
            return X_aligned, y_aligned.astype(int)

    if len(common) < 20:
        raise ValueError(f"Too few matched samples between BP matrix and labels: {len(common)}")

    X_aligned = X.loc[common]
    y_aligned = y.loc[common]
    return X_aligned, y_aligned.astype(int)


# =============================================================================
# D-Clinical
# =============================================================================

def compute_dclinical(X: pd.DataFrame,
                      y: pd.Series,
                      n_splits: int = 5,
                      n_repeats: int = 10,
                      random_state: int = 42,
                      max_iter: int = 3000,
                      C: float = 0.25,
                      top_k: int = 50) -> Tuple[pd.DataFrame, Dict]:
    """
    Repeated stratified CV logistic regression.
    D-Clinical score is based on absolute standardized coefficient stability.
    """
    X = X.copy()
    y = y.loc[X.index].astype(int)

    # Remove near-constant columns
    nunique = X.nunique(dropna=True)
    keep_cols = nunique[nunique > 1].index.tolist()
    X = X[keep_cols]

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    coef_records = []
    aucs = []
    fold_id = 0

    for train_idx, test_idx in cv.split(X, y):
        fold_id += 1
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

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

    # Composite D-Clinical selection score
    out["DClinical_selection_score"] = (
        q_rank_score(out["DClinical_mean_abs_coef"], larger_better=True) * 0.55
        + q_rank_score(out["DClinical_selection_frequency"], larger_better=True) * 0.25
        + q_rank_score(out["DClinical_sign_consistency"], larger_better=True) * 0.20
    )

    out = out.sort_values("DClinical_selection_score", ascending=False).reset_index(drop=True)
    out["DClinical_rank"] = np.arange(1, len(out) + 1)
    out["DClinical_topK"] = out["DClinical_rank"] <= top_k

    info = {
        "n_samples": int(X.shape[0]),
        "n_bp_features": int(X.shape[1]),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "cv_auc_mean": float(np.nanmean(aucs)),
        "cv_auc_sd": float(np.nanstd(aucs)),
        "cv_auc_median": float(np.nanmedian(aucs)),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "C": C,
    }
    return out, info


def select_dclinical_bp(df: pd.DataFrame, top_k: int = 50, score_quantile: Optional[float] = None) -> Set[str]:
    selected = set(df.head(top_k)["BP_term"].astype(str))
    if score_quantile is not None:
        cutoff = df["DClinical_selection_score"].quantile(score_quantile)
        selected |= set(df.loc[df["DClinical_selection_score"] >= cutoff, "BP_term"].astype(str))
    return selected


# =============================================================================
# Dual-D integration
# =============================================================================

def build_dual_d_table(dphy: pd.DataFrame,
                       dclinical: pd.DataFrame,
                       dphy_selected: Set[str],
                       dclinical_selected: Set[str],
                       h_layer: Optional[pd.DataFrame] = None,
                       ppi_layer: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    all_bp = sorted(set(dphy["BP_term"].astype(str)) | set(dclinical["BP_term"].astype(str)))

    base = pd.DataFrame({"BP_term": all_bp})
    keep_dphy_cols = [
        "BP_term", "DPHY_rank", "DPHY_selection_score", "D_score", "D_score_capped50",
        "abs_cohen_d", "cohen_d", "auc_late_vs_early", "welch_fdr",
        "direction", "bootstrap_topk_stability", "permutation_fdr"
    ]
    keep_dphy_cols = [c for c in keep_dphy_cols if c in dphy.columns]
    base = base.merge(dphy[keep_dphy_cols].drop_duplicates("BP_term"), on="BP_term", how="left")

    keep_dc_cols = [
        "BP_term", "DClinical_rank", "DClinical_selection_score",
        "DClinical_mean_coef", "DClinical_mean_abs_coef",
        "DClinical_sd_coef", "DClinical_selection_frequency",
        "DClinical_sign_consistency"
    ]
    keep_dc_cols = [c for c in keep_dc_cols if c in dclinical.columns]
    base = base.merge(dclinical[keep_dc_cols].drop_duplicates("BP_term"), on="BP_term", how="left")

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

    # Direction interpretation
    def direction_class(row):
        direction = str(row.get("direction", "")).lower()
        coef = row.get("DClinical_mean_coef", np.nan)
        if "tumor_up" in direction or "late_up" in direction or "node_positive_up" in direction:
            return "cancerized_up_or_bad_state_up"
        if "normal_up" in direction or "early_up" in direction or "node_negative_up" in direction:
            return "good_state_up_or_cancerized_suppressed"
        if pd.notna(coef):
            return "clinical_positive_contributor" if coef > 0 else "clinical_negative_contributor"
        return "unknown_direction"

    base["BP_state_direction_class"] = base.apply(direction_class, axis=1)

    # Optional h-layer merge
    if h_layer is not None and not h_layer.empty:
        h = h_layer.copy()
        bp_col = infer_bp_column(h)
        if bp_col != "BP_term":
            h = h.rename(columns={bp_col: "BP_term"})
        h["BP_term"] = h["BP_term"].map(normalize_bp_name)
        h_cols = ["BP_term"]
        for c in [
            "h_overlap_total", "h_best_fdr", "h_score",
            "cancer_gene_overlap_n", "oncogene_overlap_n", "tumor_suppressor_overlap_n",
            "cancer_gene_overlap_genes", "oncogene_overlap_genes", "tumor_suppressor_overlap_genes",
            "flag_biologically_anchored"
        ]:
            if c in h.columns:
                h_cols.append(c)
        base = base.merge(h[h_cols].drop_duplicates("BP_term"), on="BP_term", how="left")

    # Optional PPI layer merge
    if ppi_layer is not None and not ppi_layer.empty:
        p = ppi_layer.copy()
        bp_col = infer_bp_column(p)
        if bp_col != "BP_term":
            p = p.rename(columns={bp_col: "BP_term"})
        p["BP_term"] = p["BP_term"].map(normalize_bp_name)
        p_cols = ["BP_term"]
        for c in [
            "ppi_edges", "ppi_lcc_size", "ppi_density", "ppi_score",
            "network_supported", "flag_network_supported"
        ]:
            if c in p.columns:
                p_cols.append(c)
        base = base.merge(p[p_cols].drop_duplicates("BP_term"), on="BP_term", how="left")

    # Final utility-like preliminary score
    score_parts = []
    if "DPHY_selection_score" in base.columns:
        score_parts.append(base["DPHY_selection_score"].fillna(0) * 0.45)
    if "DClinical_selection_score" in base.columns:
        score_parts.append(base["DClinical_selection_score"].fillna(0) * 0.45)
    if "h_score" in base.columns:
        score_parts.append(q_rank_score(base["h_score"], larger_better=True).fillna(0) * 0.10)
    if score_parts:
        base["DualD_preliminary_priority_score"] = sum(score_parts)
    else:
        base["DualD_preliminary_priority_score"] = 0.0

    # Put selected candidates first
    tier_order = {
        "Tier1_core_dual_D": 1,
        "Tier2_DPHY_only_direct_BP_deviation": 2,
        "Tier3_DClinical_only_distributed_state_contributor": 3,
        "Tier4_weak_or_unselected": 4,
    }
    base["tier_order"] = base["DualD_tier"].map(tier_order)
    base = base.sort_values(
        ["tier_order", "DualD_preliminary_priority_score"],
        ascending=[True, False]
    ).drop(columns=["tier_order"]).reset_index(drop=True)

    return base


# =============================================================================
# Gene sets and BP module reconstruction
# =============================================================================

def read_gmt(path: Path) -> Dict[str, Set[str]]:
    gene_sets = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                name = normalize_bp_name(parts[0])
                genes = set(g.strip().upper() for g in parts[2:] if g.strip())
                if genes:
                    gene_sets[name] = genes
    return gene_sets


def read_gene_set_table(path: Path) -> Dict[str, Set[str]]:
    df = safe_read_table(path)
    bp_col = infer_bp_column(df)

    # Common formats:
    # 1. BP_term, gene
    # 2. BP_term, genes/comma-separated
    gene_col = None
    for c in ["gene", "Gene", "genes", "Genes", "gene_symbol", "GENE_SYMBOL", "member_gene", "members"]:
        if c in df.columns:
            gene_col = c
            break

    if gene_col is None:
        raise ValueError("Cannot infer gene column from gene-set table.")

    gene_sets = {}
    for bp, sub in df.groupby(bp_col):
        genes = set()
        for val in sub[gene_col].dropna().astype(str):
            # split comma/semicolon/pipe/space if needed
            for g in re.split(r"[,;|\s]+", val):
                g = g.strip().upper()
                if g:
                    genes.add(g)
        if genes:
            gene_sets[normalize_bp_name(bp)] = genes
    return gene_sets


def load_gene_sets(path: Optional[Path]) -> Dict[str, Set[str]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    if path.suffix.lower() in [".gmt", ".gmx"]:
        return read_gmt(path)
    return read_gene_set_table(path)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def lexical_tokens(bp: str) -> Set[str]:
    bp = str(bp).upper()
    bp = bp.replace("GOBP_", "")
    toks = re.split(r"[_\W]+", bp)
    stop = {
        "PROCESS", "REGULATION", "POSITIVE", "NEGATIVE", "BIOLOGICAL",
        "CELLULAR", "CELL", "OF", "TO", "IN", "BY", "AND", "OR",
        "RESPONSE", "INVOLVED"
    }
    toks = {t for t in toks if len(t) >= 4 and t not in stop}
    return toks


def lexical_similarity(a: str, b: str) -> float:
    ta = lexical_tokens(a)
    tb = lexical_tokens(b)
    return jaccard(ta, tb)


def build_similarity_edges(candidates: List[str],
                           gene_sets: Dict[str, Set[str]],
                           min_jaccard: float = 0.12,
                           min_lexical: float = 0.20) -> pd.DataFrame:
    rows = []
    n = len(candidates)
    for i in range(n):
        a = candidates[i]
        ga = gene_sets.get(a, set())
        for j in range(i + 1, n):
            b = candidates[j]
            gb = gene_sets.get(b, set())

            if ga and gb:
                sim = jaccard(ga, gb)
                sim_type = "gene_jaccard"
                keep = sim >= min_jaccard
            else:
                sim = lexical_similarity(a, b)
                sim_type = "lexical_fallback"
                keep = sim >= min_lexical

            if keep:
                rows.append({
                    "BP1": a,
                    "BP2": b,
                    "similarity": sim,
                    "similarity_type": sim_type,
                })
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

    roots = {}
    comp_id = {}
    next_id = 1
    for n in nodes:
        r = find(n)
        if r not in roots:
            roots[r] = next_id
            next_id += 1
        comp_id[n] = roots[r]
    return comp_id


def module_name_from_bps(bps: List[str]) -> str:
    tokens = []
    for bp in bps:
        tokens.extend(list(lexical_tokens(bp)))
    if not tokens:
        return "miscellaneous_BP_state_module"

    counts = pd.Series(tokens).value_counts()
    top = [t.lower() for t in counts.head(4).index.tolist()]
    return "_".join(top) + "_module"


def reconstruct_bp_modules(dual: pd.DataFrame,
                           gene_sets: Dict[str, Set[str]],
                           min_jaccard: float = 0.12,
                           min_lexical: float = 0.20,
                           min_module_size: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cand = dual.loc[dual["selected_by_dualD_union"], "BP_term"].astype(str).tolist()
    cand = sorted(set(cand))
    if not cand:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    edges = build_similarity_edges(cand, gene_sets, min_jaccard=min_jaccard, min_lexical=min_lexical)
    comp_map = connected_components(cand, edges)

    members = dual[dual["BP_term"].isin(cand)].copy()
    members["BP_module_id"] = members["BP_term"].map(lambda x: f"M{comp_map[x]:03d}")

    # Module name
    name_map = {}
    for mid, sub in members.groupby("BP_module_id"):
        bps = sub.sort_values("DualD_preliminary_priority_score", ascending=False)["BP_term"].head(8).tolist()
        name_map[mid] = module_name_from_bps(bps)
    members["BP_module_name"] = members["BP_module_id"].map(name_map)

    # Optional: mark singleton modules
    sizes = members["BP_module_id"].value_counts()
    members["BP_module_size"] = members["BP_module_id"].map(sizes)
    members["BP_module_status"] = np.where(
        members["BP_module_size"] >= min_module_size,
        "module",
        "singleton_candidate"
    )

    # Module summary
    summary_rows = []
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
            "top_BP_terms": ";".join(sub.sort_values("DualD_preliminary_priority_score", ascending=False)["BP_term"].head(10).tolist()),
        }
        if "h_score" in sub.columns:
            row["mean_h_score"] = float(pd.to_numeric(sub["h_score"], errors="coerce").mean())
            row["n_h_anchored"] = int(pd.to_numeric(sub.get("h_overlap_total", pd.Series(index=sub.index, dtype=float)), errors="coerce").fillna(0).gt(0).sum())
        summary_rows.append(row)

    module_summary = pd.DataFrame(summary_rows).sort_values(
        ["n_Tier1_core_dual_D", "mean_DualD_priority_score", "BP_module_size"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return members, module_summary, edges


# =============================================================================
# Figures
# =============================================================================

def make_figures(out_dir: Path,
                 dual: pd.DataFrame,
                 module_summary: pd.DataFrame,
                 dclinical_info: Dict) -> List[Path]:
    figs = []
    if not HAS_MPL:
        return figs

    fig_dir = ensure_dir(out_dir / "figures")

    # Figure 1: Dual-D tier counts
    tier_counts = dual.loc[dual["selected_by_dualD_union"], "DualD_tier"].value_counts()
    if not tier_counts.empty:
        plt.figure(figsize=(9, 5))
        plt.bar(tier_counts.index.astype(str), tier_counts.values)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("BP count")
        plt.title("Dual-D selected BP tiers")
        plt.tight_layout()
        f = fig_dir / "FIG01_DualD_tier_counts.png"
        plt.savefig(f, dpi=300)
        plt.close()
        figs.append(f)

    # Figure 2: D-PHY vs D-Clinical score scatter
    plot_df = dual.copy()
    if {"DPHY_selection_score", "DClinical_selection_score"}.issubset(plot_df.columns):
        plt.figure(figsize=(7, 6))
        for tier, sub in plot_df.groupby("DualD_tier"):
            plt.scatter(
                sub["DPHY_selection_score"],
                sub["DClinical_selection_score"],
                s=18,
                alpha=0.65,
                label=tier.replace("_", " ")
            )
        plt.xlabel("D-PHY selection score")
        plt.ylabel("D-Clinical selection score")
        plt.title("Dual-D BP evidence space")
        plt.legend(fontsize=7, loc="best")
        plt.tight_layout()
        f = fig_dir / "FIG02_DPHY_vs_DClinical_score_space.png"
        plt.savefig(f, dpi=300)
        plt.close()
        figs.append(f)

    # Figure 3: Module sizes
    if module_summary is not None and not module_summary.empty:
        top = module_summary.head(20).copy()
        plt.figure(figsize=(10, max(5, 0.28 * len(top) + 2)))
        y = np.arange(len(top))
        plt.barh(y, top["BP_module_size"])
        plt.yticks(y, top["BP_module_name"].astype(str))
        plt.xlabel("BP count")
        plt.title("Top Dual-D BP-enrichment modules")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        f = fig_dir / "FIG03_top_BP_modules_by_size.png"
        plt.savefig(f, dpi=300)
        plt.close()
        figs.append(f)

    # Figure 4: Module evidence composition
    if module_summary is not None and not module_summary.empty:
        top = module_summary.head(15).copy()
        y = np.arange(len(top))
        plt.figure(figsize=(10, max(5, 0.35 * len(top) + 2)))
        left = np.zeros(len(top))
        for c, lab in [
            ("n_Tier1_core_dual_D", "Core dual-D"),
            ("n_Tier2_DPHY_only", "D-PHY only"),
            ("n_Tier3_DClinical_only", "D-Clinical only"),
        ]:
            vals = top[c].fillna(0).values
            plt.barh(y, vals, left=left, label=lab)
            left += vals
        plt.yticks(y, top["BP_module_name"].astype(str))
        plt.xlabel("BP count")
        plt.title("Evidence composition of top BP modules")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.tight_layout()
        f = fig_dir / "FIG04_BP_module_evidence_composition.png"
        plt.savefig(f, dpi=300)
        plt.close()
        figs.append(f)

    # Figure 5: D-Clinical CV AUC annotation
    plt.figure(figsize=(6, 4))
    vals = [
        dclinical_info.get("cv_auc_mean", np.nan),
        dclinical_info.get("cv_auc_median", np.nan),
    ]
    plt.bar(["Mean CV AUC", "Median CV AUC"], vals)
    plt.ylim(0.45, 1.05)
    plt.ylabel("AUC")
    plt.title("D-Clinical full BP-space performance")
    plt.tight_layout()
    f = fig_dir / "FIG05_DClinical_CV_AUC.png"
    plt.savefig(f, dpi=300)
    plt.close()
    figs.append(f)

    return figs


# =============================================================================
# Main
# =============================================================================

def run(args) -> Path:
    run_dir = Path(args.run_dir) if args.run_dir else Path(".")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"DUAL_D_BP_Enrichment_Run_{now_stamp()}")
    ensure_dir(out_dir)

    # Detect inputs
    dphy_file = Path(args.dphy_file) if args.dphy_file else autodetect_dphy_file(run_dir)
    bp_matrix_file = Path(args.bp_matrix_file) if args.bp_matrix_file else autodetect_bp_matrix(run_dir)
    label_file = Path(args.label_file) if args.label_file else autodetect_label_file(run_dir)
    h_file = Path(args.h_layer_file) if args.h_layer_file else autodetect_h_layer(run_dir)
    ppi_file = Path(args.ppi_layer_file) if args.ppi_layer_file else autodetect_ppi_layer(run_dir)
    gene_set_file = Path(args.gene_set_file) if args.gene_set_file else None

    if dphy_file is None or not dphy_file.exists():
        raise FileNotFoundError("Cannot find D-PHY file. Please provide --dphy_file.")
    if bp_matrix_file is None or not bp_matrix_file.exists():
        raise FileNotFoundError("Cannot find BP matrix file. Please provide --bp_matrix_file.")

    print(f"[INFO] D-PHY file: {dphy_file}")
    print(f"[INFO] BP matrix file: {bp_matrix_file}")
    print(f"[INFO] Label file: {label_file if label_file else 'AUTO/None'}")
    print(f"[INFO] h-layer file: {h_file if h_file else 'None'}")
    print(f"[INFO] PPI-layer file: {ppi_file if ppi_file else 'None'}")
    print(f"[INFO] Gene-set file: {gene_set_file if gene_set_file else 'None'}")

    # Load D-PHY
    dphy = load_dphy_table(dphy_file)
    dphy_selected = select_dphy_bp(
        dphy,
        top_k=args.top_k_dphy,
        fdr_cutoff=args.dphy_fdr_cutoff,
        abs_d_cutoff=args.dphy_abs_d_cutoff,
        use_threshold_or_topk=True
    )

    # Load BP matrix and labels
    X = load_bp_matrix(bp_matrix_file)
    y = load_labels(
        label_file,
        X,
        label_col=args.label_col,
        sample_col=args.sample_col,
        label_mode=args.label_mode
    )
    X, y = align_matrix_and_labels(X, y)

    # Keep BPs common to D-PHY if possible, but do not over-restrict
    common_bp = sorted(set(X.columns) & set(dphy["BP_term"]))
    if len(common_bp) >= 20:
        X = X[common_bp]
    else:
        print(f"[WARN] Few BP overlap between BP matrix and D-PHY table: {len(common_bp)}. Using all BP matrix columns.")

    # Compute D-Clinical
    dclinical, dclinical_info = compute_dclinical(
        X, y,
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        random_state=args.random_state,
        max_iter=args.max_iter,
        C=args.logistic_C,
        top_k=args.top_k_dclinical
    )
    dclinical_selected = select_dclinical_bp(
        dclinical,
        top_k=args.top_k_dclinical,
        score_quantile=args.dclinical_score_quantile
    )

    # Optional evidence layers
    h_layer = safe_read_table(h_file) if h_file and Path(h_file).exists() else None
    ppi_layer = safe_read_table(ppi_file) if ppi_file and Path(ppi_file).exists() else None

    # Dual-D integration
    dual = build_dual_d_table(
        dphy=dphy,
        dclinical=dclinical,
        dphy_selected=dphy_selected,
        dclinical_selected=dclinical_selected,
        h_layer=h_layer,
        ppi_layer=ppi_layer
    )

    # BP-enrichment / module reconstruction
    gene_sets = load_gene_sets(gene_set_file)
    module_members, module_summary, module_edges = reconstruct_bp_modules(
        dual,
        gene_sets=gene_sets,
        min_jaccard=args.min_gene_jaccard,
        min_lexical=args.min_lexical_similarity,
        min_module_size=args.min_module_size
    )

    # Write outputs
    dphy_out = out_dir / "01_DPHY_input_normalized.csv"
    dclinical_out = out_dir / "02_DClinical_BP_coefficients.csv"
    dual_out = out_dir / "03_DualD_BP_candidates.csv"
    members_out = out_dir / "04_BPenrichment_module_members.csv"
    module_out = out_dir / "05_BPenrichment_module_summary.csv"
    edges_out = out_dir / "05b_BPenrichment_BP_similarity_edges.csv"
    summary_out = out_dir / "06_DualD_run_summary.json"
    xlsx_out = out_dir / "07_DualD_summary.xlsx"

    dphy.to_csv(dphy_out, index=False)
    dclinical.to_csv(dclinical_out, index=False)
    dual.to_csv(dual_out, index=False)
    module_members.to_csv(members_out, index=False)
    module_summary.to_csv(module_out, index=False)
    module_edges.to_csv(edges_out, index=False)

    # Figures
    figs = make_figures(out_dir, dual, module_summary, dclinical_info)

    tier_counts = dual.loc[dual["selected_by_dualD_union"], "DualD_tier"].value_counts().to_dict()

    run_summary = {
        "run_timestamp": now_stamp(),
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "input_files": {
            "dphy_file": str(dphy_file),
            "bp_matrix_file": str(bp_matrix_file),
            "label_file": str(label_file) if label_file else None,
            "h_layer_file": str(h_file) if h_file else None,
            "ppi_layer_file": str(ppi_file) if ppi_file else None,
            "gene_set_file": str(gene_set_file) if gene_set_file else None,
        },
        "label_summary": {
            "n_samples": int(len(y)),
            "n_negative_or_good_state": int((y == 0).sum()),
            "n_positive_or_cancerized_state": int((y == 1).sum()),
            "label_mode": args.label_mode,
        },
        "DClinical_performance": dclinical_info,
        "selection_settings": {
            "top_k_dphy": args.top_k_dphy,
            "top_k_dclinical": args.top_k_dclinical,
            "dphy_fdr_cutoff": args.dphy_fdr_cutoff,
            "dphy_abs_d_cutoff": args.dphy_abs_d_cutoff,
            "dclinical_score_quantile": args.dclinical_score_quantile,
        },
        "DualD_counts": {
            "n_DPHY_selected": int(len(dphy_selected)),
            "n_DClinical_selected": int(len(dclinical_selected)),
            "n_union_selected": int(dual["selected_by_dualD_union"].sum()),
            "n_intersection_selected": int(dual["selected_by_dualD_intersection"].sum()),
            "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        },
        "BP_enrichment": {
            "n_modules_total": int(module_summary.shape[0]) if module_summary is not None else 0,
            "n_module_members": int(module_members.shape[0]) if module_members is not None else 0,
            "n_similarity_edges": int(module_edges.shape[0]) if module_edges is not None else 0,
            "gene_sets_loaded": int(len(gene_sets)),
        },
        "interpretation_note": (
            "Dual-D is a modular selection layer. D-PHY captures local BP-state deviation, "
            "D-Clinical captures global clinical/cancer-state contribution, and the union "
            "candidate pool is passed to BP-enrichment / BP-state module reconstruction."
        ),
    }
    summary_out.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    # Excel workbook
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
        pd.DataFrame([run_summary["DualD_counts"]]).to_excel(writer, sheet_name="run_counts", index=False)
        dclinical.to_excel(writer, sheet_name="DClinical", index=False)
        dual.to_excel(writer, sheet_name="DualD_candidates", index=False)
        module_summary.to_excel(writer, sheet_name="BP_modules", index=False)
        module_members.to_excel(writer, sheet_name="module_members", index=False)

    # Notes
    notes = []
    notes.append("DUAL-D BP Enrichment V1")
    notes.append("=" * 80)
    notes.append("")
    notes.append("Concept:")
    notes.append("D-module = D-PHY + D-Clinical.")
    notes.append("D-PHY captures local BP-state deviation.")
    notes.append("D-Clinical captures global clinical/cancer-state contribution.")
    notes.append("The union of selected BPs enters BP-enrichment / BP-module reconstruction.")
    notes.append("")
    notes.append("Main counts:")
    notes.append(f"- D-PHY selected: {len(dphy_selected)}")
    notes.append(f"- D-Clinical selected: {len(dclinical_selected)}")
    notes.append(f"- Dual-D union: {int(dual['selected_by_dualD_union'].sum())}")
    notes.append(f"- Dual-D intersection/core: {int(dual['selected_by_dualD_intersection'].sum())}")
    notes.append(f"- BP modules: {module_summary.shape[0] if module_summary is not None else 0}")
    notes.append("")
    notes.append("Tier interpretation:")
    notes.append("- Tier1_core_dual_D: both local BP deviation and clinical-state contribution.")
    notes.append("- Tier2_DPHY_only: direct BP-state deviation; clinical claim should be cautious.")
    notes.append("- Tier3_DClinical_only: distributed clinical-state contributor; avoid single-BP overinterpretation.")
    notes.append("- Tier4: weak or unselected.")
    notes_path = out_dir / "README_DUAL_D_INTERPRETATION.txt"
    notes_path.write_text("\n".join(notes), encoding="utf-8")

    # Zip package
    zip_path = out_dir.parent / f"{out_dir.name}.zip"
    zip_dir(out_dir, zip_path)

    print("")
    print("[DONE] DUAL-D BP Enrichment completed.")
    print(f"[DONE] Output folder: {out_dir}")
    print(f"[DONE] ZIP package: {zip_path}")
    print("")
    print(json.dumps(run_summary["DualD_counts"], indent=2))
    print("")
    print(f"D-Clinical CV AUC mean: {dclinical_info.get('cv_auc_mean', np.nan):.4f}")
    print(f"D-Clinical CV AUC SD:   {dclinical_info.get('cv_auc_sd', np.nan):.4f}")
    return zip_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DUAL-D BP Enrichment / BP-State Module Reconstruction Pipeline V1"
    )

    p.add_argument("--run_dir", type=str, default=".", help="Folder containing previous D-PHY outputs.")
    p.add_argument("--out_dir", type=str, default=None, help="Output folder.")

    p.add_argument("--dphy_file", type=str, default=None, help="D-PHY ranked BP table.")
    p.add_argument("--bp_matrix_file", type=str, default=None, help="Samples x BP activity matrix.")
    p.add_argument("--label_file", type=str, default=None, help="Sample labels file.")
    p.add_argument("--sample_col", type=str, default=None, help="Sample ID column in label file.")
    p.add_argument("--label_col", type=str, default=None, help="Binary label column in label file.")
    p.add_argument("--label_mode", type=str, default="auto",
                   choices=["auto", "tumor_vs_normal"],
                   help="Auto label mode. tumor_vs_normal can infer labels from TCGA barcodes.")

    p.add_argument("--h_layer_file", type=str, default=None, help="Optional h-layer biological anchoring CSV.")
    p.add_argument("--ppi_layer_file", type=str, default=None, help="Optional PPI/network support CSV.")
    p.add_argument("--gene_set_file", type=str, default=None,
                   help="Optional BP gene-set file, GMT or table with BP/gene columns.")

    p.add_argument("--top_k_dphy", type=int, default=50, help="Top K D-PHY BPs.")
    p.add_argument("--top_k_dclinical", type=int, default=50, help="Top K D-Clinical BPs.")
    p.add_argument("--dphy_fdr_cutoff", type=float, default=0.05, help="D-PHY FDR cutoff.")
    p.add_argument("--dphy_abs_d_cutoff", type=float, default=0.30, help="D-PHY effect-size cutoff.")
    p.add_argument("--dclinical_score_quantile", type=float, default=None,
                   help="Optional D-Clinical score quantile cutoff, e.g. 0.99.")

    p.add_argument("--cv_splits", type=int, default=5, help="CV splits for D-Clinical.")
    p.add_argument("--cv_repeats", type=int, default=10, help="CV repeats for D-Clinical.")
    p.add_argument("--random_state", type=int, default=42, help="Random seed.")
    p.add_argument("--max_iter", type=int, default=3000, help="Logistic regression max iterations.")
    p.add_argument("--logistic_C", type=float, default=0.25, help="Logistic regression regularization C.")

    p.add_argument("--min_gene_jaccard", type=float, default=0.12,
                   help="Minimum gene-overlap Jaccard for BP module edges.")
    p.add_argument("--min_lexical_similarity", type=float, default=0.20,
                   help="Minimum lexical similarity for fallback BP module edges.")
    p.add_argument("--min_module_size", type=int, default=2,
                   help="Minimum module size to call a reconstructed module.")

    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)
