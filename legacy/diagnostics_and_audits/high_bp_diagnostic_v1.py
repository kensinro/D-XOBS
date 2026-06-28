# -*- coding: utf-8 -*-
r"""
D-Clinical high BP diagnostic V1

Question addressed:
For high D-Clinical BP terms:
1. What are their BP-level means/mu in each class?
2. What are their D-PHY characteristics?
3. Are they no better than random BP sets?

This is a diagnostic / SERIOUS FILTER test module.

Inputs:
- Existing D-PHY run folders under D:/AIDO-Temp
- Existing SERIOUS FILTER test folder if available:
  D:/AIDO-Temp/D_PHY_SERIOUS_FILTER_Test_*/

Main outputs:
For each target:
- DClinical_high_BP_mu_DPHY_features.csv
- DClinical_high_BP_vs_random_summary.csv
- DClinical_high_BP_random_null_distribution.csv
- DClinical_high_BP_diagnostic.xlsx

Across all targets:
- ALL_DClinical_high_BP_vs_random_summary.csv
- DClinical_high_BP_Diagnostics_SUMMARY.xlsx

Notes:
- μ_early_like and μ_late_like are means of BP observable scores in the two classes.
- For tumor_vs_normal:
    early-like = normal
    late-like = tumor
- For stage/node:
    early-like = early-stage/node-negative
    late-like = late-stage/node-positive
- D-Clinical-high BP terms are taken from DClinical_first_all_BP_ranked.csv if available,
  otherwise computed from existing SERIOUS FILTER outputs.
"""

import os
import re
import gc
import json
import time
import warnings
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

from scipy.stats import mannwhitneyu


# ============================================================
# 0. CONFIG
# ============================================================

AIDO_TEMP_ROOT = Path("D:/AIDO-Temp")
OUTPUT_ROOT = AIDO_TEMP_ROOT / f"DClinical_high_BP_Diagnostic_{time.strftime('%Y%m%d_%H%M%S')}"

RUN_FOLDER_PATTERNS = [
    "D_PHY_BioSystems_V5_Run_*",
    "D_PHY_METABRIC_External_V2_Run_*",
    "D_PHY_GSE96058_External_V2_Run_*",
    "D_PHY_TCGA_NewTarget_Run_*",
]

SERIOUS_FILTER_PATTERN = "D_PHY_SERIOUS_FILTER_Test_*"

TOPK_DCLINICAL = 50

# Random test:
# Random BP sets are sampled from the same BP universe with the same K.
RANDOM_REPS = 500

# To avoid overlong runtime, CV-AUC random null is optional and limited.
COMPUTE_RANDOM_CV_AUC = True
RANDOM_AUC_REPS = 100

RANDOM_SEED = 42
FDR_THRESHOLD = 0.05


# ============================================================
# 1. HELPERS
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^\w\-.()]+", "_", x)
    return x[:150]


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return None


def safe_read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def infer_dataset_endpoint(folder, summary):
    folder = Path(folder)
    name = folder.name.lower()

    dataset = summary.get("cohort_name") or summary.get("cancer_name") or folder.name
    endpoint = (
        summary.get("target_mode")
        or summary.get("endpoint_type")
        or summary.get("endpoint")
        or summary.get("endpoint_name")
        or "unknown_endpoint"
    )

    if "tcga_brca_tumor_vs_normal" in name:
        dataset = "TCGA-BRCA"
        endpoint = "tumor_vs_normal"
    elif "metabric" in name:
        dataset = "METABRIC"
        if endpoint in ["stage", "unknown_endpoint", ""]:
            endpoint = "early_vs_late_stage"
    elif "gse96058" in name:
        dataset = "GSE96058"
        if endpoint in ["node", "unknown_endpoint", ""]:
            endpoint = "lymph_node_status"
    elif "brca" in name:
        dataset = "TCGA-BRCA"
        if endpoint in ["unknown_endpoint", ""]:
            endpoint = "early_vs_late_stage"

    return str(dataset), str(endpoint)


def find_result_folders():
    run_roots = []
    for pattern in RUN_FOLDER_PATTERNS:
        run_roots.extend(AIDO_TEMP_ROOT.glob(pattern))

    run_roots = sorted(set([p for p in run_roots if p.is_dir()]))

    result_folders = []

    for root in run_roots:
        for p in root.rglob("*"):
            if not p.is_dir():
                continue

            needed = [
                p / "SUMMARY.json",
                p / "02_BP_observable_matrix.csv",
                p / "01_input_endpoint_definitions.csv",
                p / "03_D_layer_ranked_BP_signals.csv",
            ]

            if all(x.exists() for x in needed):
                result_folders.append(p)

    return sorted(set(result_folders))


def latest_serious_filter_root():
    roots = sorted(AIDO_TEMP_ROOT.glob(SERIOUS_FILTER_PATTERN))
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        return None
    return roots[-1]


def load_bp_matrix(folder):
    bp_path = Path(folder) / "02_BP_observable_matrix.csv"
    bp = pd.read_csv(bp_path, index_col=0)
    bp.index = bp.index.astype(str)
    bp = bp.apply(pd.to_numeric, errors="coerce")
    bp = bp.loc[:, bp.notna().mean(axis=0) > 0.80]
    bp = bp.fillna(bp.median(axis=0))
    return bp


def load_labels(folder):
    labels_path = Path(folder) / "01_input_endpoint_definitions.csv"
    df = pd.read_csv(labels_path)

    sample_col = None
    label_col = None

    for c in df.columns:
        cl = str(c).lower()
        if cl in ["sample_id", "sample", "id"] or "sample" in cl:
            sample_col = c
            break

    for c in df.columns:
        cl = str(c).lower()
        if cl in ["stage_group", "target_group", "label", "group", "endpoint_group"]:
            label_col = c
            break

    if sample_col is None:
        sample_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[1]

    labels = df.dropna(subset=[sample_col, label_col]).copy()
    labels[sample_col] = labels[sample_col].astype(str)
    labels = labels.drop_duplicates(sample_col)
    labels = labels.set_index(sample_col)[label_col]
    labels.index = labels.index.astype(str)

    return labels


def align_bp_labels(bp, labels):
    common = bp.index.intersection(labels.index)
    bp2 = bp.loc[common].copy()
    y = labels.loc[common].copy()

    classes = sorted(y.dropna().unique().tolist())
    if len(classes) != 2:
        raise ValueError(f"Expected two classes, found {classes}")

    # Try to make late/tumor/node-positive/basal-like the positive class.
    pos_candidates = ["late", "tumor", "positive", "basal", "high", "1", "yes", "true"]

    pos_class = None
    for c in classes:
        s = str(c).lower()
        if any(k in s for k in pos_candidates):
            pos_class = c
            break

    if pos_class is None:
        pos_class = classes[-1]

    y_bin = (y == pos_class).astype(int)

    neg_class = [c for c in classes if c != pos_class][0]

    return bp2, y, y_bin, pos_class, neg_class, classes


def cohen_d(x0, x1):
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x0 = x0[np.isfinite(x0)]
    x1 = x1[np.isfinite(x1)]

    if len(x0) < 2 or len(x1) < 2:
        return np.nan

    n0, n1 = len(x0), len(x1)
    v0, v1 = np.var(x0, ddof=1), np.var(x1, ddof=1)
    sp = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / max(n0 + n1 - 2, 1))

    if sp == 0:
        return 0.0

    return (np.mean(x1) - np.mean(x0)) / sp


def auc_single_bp(y_bin, values):
    try:
        return roc_auc_score(y_bin, values)
    except Exception:
        return np.nan


def cv_auc_for_terms(bp, y_bin, terms):
    terms = [t for t in terms if t in bp.columns]

    if len(terms) == 0:
        return {
            "cv_auc": np.nan,
            "cv_balanced_accuracy": np.nan,
            "cv_accuracy": np.nan,
            "n_features": 0,
        }

    y_bin = np.asarray(y_bin).astype(int)
    min_class = int(np.bincount(y_bin).min())
    n_splits = min(5, min_class)

    if n_splits < 2:
        return {
            "cv_auc": np.nan,
            "cv_balanced_accuracy": np.nan,
            "cv_accuracy": np.nan,
            "n_features": len(terms),
        }

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED
        ))
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    X = bp[terms].values
    prob = cross_val_predict(clf, X, y_bin, cv=cv, method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(int)

    return {
        "cv_auc": roc_auc_score(y_bin, prob),
        "cv_balanced_accuracy": balanced_accuracy_score(y_bin, pred),
        "cv_accuracy": accuracy_score(y_bin, pred),
        "n_features": len(terms),
    }


# ============================================================
# 2. FIND D-CLINICAL FIRST OUTPUT
# ============================================================

def find_serious_filter_output_for_dataset_endpoint(dataset, endpoint):
    """
    Locate per-target serious filter folder by name.
    """
    root = latest_serious_filter_root()
    if root is None:
        return None

    # Serious filter folder naming: safe(dataset)__safe(endpoint)
    target_name = f"{safe_name(dataset)}__{safe_name(endpoint)}"
    direct = root / target_name
    if direct.exists():
        return direct

    # Fallback fuzzy search
    candidates = [p for p in root.iterdir() if p.is_dir()]
    dl = safe_name(dataset).lower()
    el = safe_name(endpoint).lower()

    scored = []
    for p in candidates:
        name = p.name.lower()
        score = 0
        if dl in name:
            score += 10
        if el in name:
            score += 10
        scored.append((score, p))

    scored = sorted(scored, reverse=True)
    if scored and scored[0][0] >= 10:
        return scored[0][1]

    return None


def load_dclinical_first_ranking(dataset, endpoint):
    sf_dir = find_serious_filter_output_for_dataset_endpoint(dataset, endpoint)
    if sf_dir is None:
        return None, None

    rank_path = sf_dir / "DClinical_first_all_BP_ranked.csv"
    dclin = safe_read_csv(rank_path)

    return dclin, sf_dir


# ============================================================
# 3. DIAGNOSTIC TABLES
# ============================================================

def build_dclinical_high_bp_table(folder, dataset, endpoint, bp, y, y_bin, pos_class, neg_class, dclinical_rank, dphy_table):
    """
    Build per-BP table for high D-Clinical BP terms.
    """
    if dclinical_rank is None or len(dclinical_rank) == 0:
        raise ValueError("Missing DClinical_first_all_BP_ranked.csv. Please run serious_filter_dphy_dclinical_v1.py first.")

    dclinical_rank = dclinical_rank.copy()

    if "BP_term" not in dclinical_rank.columns:
        raise ValueError("DClinical ranking lacks BP_term column.")

    dclinical_rank["BP_term"] = dclinical_rank["BP_term"].astype(str)

    # Top K high DClinical BP
    top_terms = dclinical_rank.head(min(TOPK_DCLINICAL, len(dclinical_rank)))["BP_term"].tolist()
    top_terms = [t for t in top_terms if t in bp.columns]

    # Prepare D-PHY table
    dphy = dphy_table.copy()
    dphy["BP_term"] = dphy["BP_term"].astype(str)

    dphy_keep = [
        "BP_term", "DPHY_rank", "D_score", "abs_cohen_d", "cohen_d_late_minus_early",
        "auc_late_vs_early", "auc_distance", "welch_p", "welch_fdr",
        "mannwhitney_p", "mannwhitney_fdr", "direction",
        "mean_early_like", "mean_late_like", "n_early_like", "n_late_like"
    ]
    dphy_keep = [c for c in dphy_keep if c in dphy.columns]
    dphy = dphy[dphy_keep]

    rows = []

    neg_idx = y[y == neg_class].index
    pos_idx = y[y == pos_class].index

    for term in top_terms:
        values_neg = bp.loc[neg_idx, term].values
        values_pos = bp.loc[pos_idx, term].values
        values_all = bp[term].values

        mu_neg = float(np.nanmean(values_neg))
        mu_pos = float(np.nanmean(values_pos))

        sd_neg = float(np.nanstd(values_neg, ddof=1))
        sd_pos = float(np.nanstd(values_pos, ddof=1))
        sd_all = float(np.nanstd(values_all, ddof=1))

        d = cohen_d(values_neg, values_pos)
        auc = auc_single_bp(y_bin.values if hasattr(y_bin, "values") else y_bin, values_all)

        try:
            mw_p = mannwhitneyu(values_pos, values_neg, alternative="two-sided").pvalue
        except Exception:
            mw_p = np.nan

        rows.append({
            "dataset": dataset,
            "endpoint": endpoint,
            "BP_term": term,
            "negative_class": neg_class,
            "positive_class": pos_class,
            "mu_negative_class": mu_neg,
            "mu_positive_class": mu_pos,
            "mu_delta_positive_minus_negative": mu_pos - mu_neg,
            "abs_mu_delta": abs(mu_pos - mu_neg),
            "sd_negative_class": sd_neg,
            "sd_positive_class": sd_pos,
            "sd_all_samples": sd_all,
            "cohen_d_positive_minus_negative_recomputed": d,
            "abs_cohen_d_recomputed": abs(d) if np.isfinite(d) else np.nan,
            "single_BP_auc_recomputed": auc,
            "single_BP_auc_distance_recomputed": abs(auc - 0.5) if np.isfinite(auc) else np.nan,
            "mannwhitney_p_recomputed": mw_p,
        })

    out = pd.DataFrame(rows)

    # Merge D-Clinical ranking metrics
    dclin_keep = [
        "BP_term", "DClinical_rank", "DClinical_mean_abs_coef",
        "DClinical_mean_signed_coef", "DClinical_coef_sd",
        "DClinical_selection_stability_top50",
        "DClinical_sign_stability",
        "DClinical_direction"
    ]
    dclin_keep = [c for c in dclin_keep if c in dclinical_rank.columns]

    out = out.merge(dclinical_rank[dclin_keep], on="BP_term", how="left")
    out = out.merge(dphy, on="BP_term", how="left", suffixes=("", "_from_DPHY_table"))

    # Additional D-PHY flags
    out["DPHY_significant_FDR05"] = pd.to_numeric(out.get("welch_fdr", np.nan), errors="coerce") <= FDR_THRESHOLD
    out["DPHY_abs_d_ge_0p30"] = pd.to_numeric(out.get("abs_cohen_d", np.nan), errors="coerce") >= 0.30
    out["DPHY_abs_d_ge_0p50"] = pd.to_numeric(out.get("abs_cohen_d", np.nan), errors="coerce") >= 0.50

    # Move useful columns to front
    front = [
        "dataset", "endpoint", "BP_term",
        "DClinical_rank", "DClinical_mean_abs_coef", "DClinical_mean_signed_coef",
        "DClinical_selection_stability_top50", "DClinical_sign_stability",
        "negative_class", "positive_class",
        "mu_negative_class", "mu_positive_class", "mu_delta_positive_minus_negative", "abs_mu_delta",
        "cohen_d_positive_minus_negative_recomputed", "single_BP_auc_recomputed",
        "DPHY_rank", "D_score", "abs_cohen_d", "auc_late_vs_early",
        "welch_fdr", "direction",
        "DPHY_significant_FDR05", "DPHY_abs_d_ge_0p30", "DPHY_abs_d_ge_0p50"
    ]
    front = [c for c in front if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    return out, top_terms


def random_bp_null(bp, y, y_bin, dphy_table, observed_terms):
    """
    Compare high D-Clinical BP features with random BP sets of same size.

    Random metrics:
    - mean/median abs D-PHY effect size
    - number significant FDR<0.05
    - mean AUC distance
    - feature-set CV AUC for random sets, optional limited reps
    """
    rng = np.random.default_rng(RANDOM_SEED)

    all_terms = np.array([str(c) for c in bp.columns])
    k = min(len(observed_terms), len(all_terms))

    dphy = dphy_table.copy()
    dphy["BP_term"] = dphy["BP_term"].astype(str)

    dphy_metrics = dphy.set_index("BP_term")

    def set_metrics(terms, compute_auc=False):
        terms = [t for t in terms if t in dphy_metrics.index and t in bp.columns]
        sub = dphy_metrics.loc[terms]

        abs_d = pd.to_numeric(sub.get("abs_cohen_d", np.nan), errors="coerce")
        fdr = pd.to_numeric(sub.get("welch_fdr", np.nan), errors="coerce")
        auc_dist = pd.to_numeric(sub.get("auc_distance", np.nan), errors="coerce")

        out = {
            "n_terms": len(terms),
            "mean_abs_cohen_d": float(np.nanmean(abs_d)) if len(abs_d) else np.nan,
            "median_abs_cohen_d": float(np.nanmedian(abs_d)) if len(abs_d) else np.nan,
            "max_abs_cohen_d": float(np.nanmax(abs_d)) if len(abs_d) else np.nan,
            "n_DPHY_sig_FDR05": int(np.nansum(fdr <= FDR_THRESHOLD)) if len(fdr) else 0,
            "fraction_DPHY_sig_FDR05": float(np.nanmean(fdr <= FDR_THRESHOLD)) if len(fdr) else np.nan,
            "mean_auc_distance": float(np.nanmean(auc_dist)) if len(auc_dist) else np.nan,
            "median_auc_distance": float(np.nanmedian(auc_dist)) if len(auc_dist) else np.nan,
        }

        if compute_auc:
            ev = cv_auc_for_terms(bp, y_bin, terms)
            out["feature_set_cv_auc"] = ev["cv_auc"]
            out["feature_set_balanced_accuracy"] = ev["cv_balanced_accuracy"]
        else:
            out["feature_set_cv_auc"] = np.nan
            out["feature_set_balanced_accuracy"] = np.nan

        return out

    observed_metrics = set_metrics(observed_terms, compute_auc=True)

    random_rows = []
    for i in range(RANDOM_REPS):
        terms = rng.choice(all_terms, size=k, replace=False).tolist()
        compute_auc = COMPUTE_RANDOM_CV_AUC and (i < RANDOM_AUC_REPS)
        m = set_metrics(terms, compute_auc=compute_auc)
        m["random_rep"] = i + 1
        random_rows.append(m)

    random_df = pd.DataFrame(random_rows)

    summary_rows = []
    for metric, obs_val in observed_metrics.items():
        if metric == "n_terms":
            continue

        rand_vals = pd.to_numeric(random_df[metric], errors="coerce").dropna()

        if len(rand_vals) == 0:
            summary_rows.append({
                "metric": metric,
                "observed_DClinical_high_BP": obs_val,
                "random_mean": np.nan,
                "random_median": np.nan,
                "random_p05": np.nan,
                "random_p95": np.nan,
                "empirical_p_random_ge_observed": np.nan,
                "empirical_percentile_observed": np.nan,
                "n_random_reps_used": 0,
            })
            continue

        emp_p = (np.sum(rand_vals >= obs_val) + 1) / (len(rand_vals) + 1) if np.isfinite(obs_val) else np.nan
        percentile = float(np.mean(rand_vals <= obs_val)) if np.isfinite(obs_val) else np.nan

        summary_rows.append({
            "metric": metric,
            "observed_DClinical_high_BP": obs_val,
            "random_mean": float(np.mean(rand_vals)),
            "random_median": float(np.median(rand_vals)),
            "random_p05": float(np.quantile(rand_vals, 0.05)),
            "random_p95": float(np.quantile(rand_vals, 0.95)),
            "empirical_p_random_ge_observed": emp_p,
            "empirical_percentile_observed": percentile,
            "n_random_reps_used": int(len(rand_vals)),
        })

    summary_df = pd.DataFrame(summary_rows)

    return observed_metrics, random_df, summary_df


# ============================================================
# 4. RUN ONE TARGET
# ============================================================

def run_one_target(folder):
    folder = Path(folder)
    summary = safe_read_json(folder / "SUMMARY.json")
    dataset, endpoint = infer_dataset_endpoint(folder, summary)

    out_dir = OUTPUT_ROOT / f"{safe_name(dataset)}__{safe_name(endpoint)}"
    ensure_dir(out_dir)

    log("=" * 80)
    log(f"D-Clinical high BP diagnostic: {dataset} | {endpoint}")
    log(f"Folder: {folder}")

    bp = load_bp_matrix(folder)
    labels = load_labels(folder)
    bp, y, y_bin, pos_class, neg_class, classes = align_bp_labels(bp, labels)

    dphy_table = safe_read_csv(folder / "03_D_layer_ranked_BP_signals.csv")
    if dphy_table is None:
        raise ValueError("Missing D-PHY table.")

    dclinical_rank, serious_dir = load_dclinical_first_ranking(dataset, endpoint)
    if dclinical_rank is None:
        raise ValueError(
            f"Missing D-Clinical-first ranking for {dataset} | {endpoint}. "
            "Please run serious_filter_dphy_dclinical_v1.py first."
        )

    high_bp_table, top_terms = build_dclinical_high_bp_table(
        folder=folder,
        dataset=dataset,
        endpoint=endpoint,
        bp=bp,
        y=y,
        y_bin=y_bin,
        pos_class=pos_class,
        neg_class=neg_class,
        dclinical_rank=dclinical_rank,
        dphy_table=dphy_table
    )

    high_bp_table.to_csv(out_dir / "DClinical_high_BP_mu_DPHY_features.csv", index=False)

    observed_metrics, random_df, random_summary = random_bp_null(
        bp=bp,
        y=y,
        y_bin=y_bin,
        dphy_table=dphy_table,
        observed_terms=top_terms
    )

    random_df.to_csv(out_dir / "DClinical_high_BP_random_null_distribution.csv", index=False)
    random_summary.to_csv(out_dir / "DClinical_high_BP_vs_random_summary.csv", index=False)

    # Compact target-level summary
    n_sig = int(high_bp_table["DPHY_significant_FDR05"].fillna(False).sum()) if "DPHY_significant_FDR05" in high_bp_table.columns else np.nan
    n_abs030 = int(high_bp_table["DPHY_abs_d_ge_0p30"].fillna(False).sum()) if "DPHY_abs_d_ge_0p30" in high_bp_table.columns else np.nan
    n_abs050 = int(high_bp_table["DPHY_abs_d_ge_0p50"].fillna(False).sum()) if "DPHY_abs_d_ge_0p50" in high_bp_table.columns else np.nan

    # Random p for feature-set CV AUC
    auc_row = random_summary[random_summary["metric"] == "feature_set_cv_auc"]
    auc_p = auc_row["empirical_p_random_ge_observed"].iloc[0] if len(auc_row) else np.nan
    auc_obs = auc_row["observed_DClinical_high_BP"].iloc[0] if len(auc_row) else np.nan
    auc_rand_mean = auc_row["random_mean"].iloc[0] if len(auc_row) else np.nan
    auc_rand_p95 = auc_row["random_p95"].iloc[0] if len(auc_row) else np.nan

    absd_row = random_summary[random_summary["metric"] == "mean_abs_cohen_d"]
    absd_p = absd_row["empirical_p_random_ge_observed"].iloc[0] if len(absd_row) else np.nan
    absd_obs = absd_row["observed_DClinical_high_BP"].iloc[0] if len(absd_row) else np.nan
    absd_rand_mean = absd_row["random_mean"].iloc[0] if len(absd_row) else np.nan

    target_summary = {
        "dataset": dataset,
        "endpoint": endpoint,
        "source_folder": str(folder),
        "serious_filter_folder": str(serious_dir),
        "output_dir": str(out_dir),
        "n_samples": int(bp.shape[0]),
        "n_bp_terms": int(bp.shape[1]),
        "negative_class": str(neg_class),
        "positive_class": str(pos_class),
        "topK_DClinical": int(len(top_terms)),
        "DClinical_high_n_DPHY_FDR05": n_sig,
        "DClinical_high_n_abs_d_ge_0p30": n_abs030,
        "DClinical_high_n_abs_d_ge_0p50": n_abs050,
        "DClinical_high_mean_abs_d": high_bp_table["abs_cohen_d"].mean() if "abs_cohen_d" in high_bp_table.columns else np.nan,
        "DClinical_high_median_abs_d": high_bp_table["abs_cohen_d"].median() if "abs_cohen_d" in high_bp_table.columns else np.nan,
        "DClinical_high_mean_auc_distance": high_bp_table["single_BP_auc_distance_recomputed"].mean(),
        "DClinical_high_feature_set_cv_auc": auc_obs,
        "Random_feature_set_cv_auc_mean": auc_rand_mean,
        "Random_feature_set_cv_auc_p95": auc_rand_p95,
        "Empirical_p_random_auc_ge_observed": auc_p,
        "DClinical_high_mean_abs_d_observed": absd_obs,
        "Random_mean_abs_d": absd_rand_mean,
        "Empirical_p_random_mean_abs_d_ge_observed": absd_p,
        "top1_DClinical_BP": high_bp_table.iloc[0]["BP_term"] if len(high_bp_table) else "",
        "top1_DClinical_mu_negative": high_bp_table.iloc[0]["mu_negative_class"] if len(high_bp_table) else np.nan,
        "top1_DClinical_mu_positive": high_bp_table.iloc[0]["mu_positive_class"] if len(high_bp_table) else np.nan,
        "top1_DClinical_abs_d": high_bp_table.iloc[0].get("abs_cohen_d", np.nan) if len(high_bp_table) else np.nan,
        "top1_DClinical_DPHY_rank": high_bp_table.iloc[0].get("DPHY_rank", np.nan) if len(high_bp_table) else np.nan,
    }

    with open(out_dir / "DClinical_high_BP_diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(target_summary, f, indent=2)

    with pd.ExcelWriter(out_dir / "DClinical_high_BP_diagnostic.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([target_summary]).to_excel(writer, sheet_name="summary", index=False)
        high_bp_table.to_excel(writer, sheet_name="DClinical_high_BP_mu_DPHY", index=False)
        random_summary.to_excel(writer, sheet_name="vs_random_summary", index=False)
        random_df.to_excel(writer, sheet_name="random_null_distribution", index=False)

    # Interpretation note
    with open(out_dir / "INTERPRETATION_NOTE.txt", "w", encoding="utf-8") as f:
        f.write(f"D-Clinical high BP diagnostic\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Endpoint: {endpoint}\n")
        f.write(f"Classes: negative={neg_class}; positive={pos_class}\n")
        f.write(f"Top K D-Clinical BP: {len(top_terms)}\n\n")
        f.write("Key interpretation:\n")
        f.write("- μ_negative_class and μ_positive_class are BP observable means in each class.\n")
        f.write("- D-PHY features show whether these D-Clinical-high BP terms are also individually discriminative.\n")
        f.write("- Random comparison tests whether D-Clinical-high BP terms are stronger than random BP sets of the same size.\n\n")
        for k, v in target_summary.items():
            f.write(f"{k}: {v}\n")

    del bp, high_bp_table, random_df
    gc.collect()

    return target_summary


# ============================================================
# 5. MAIN
# ============================================================

def main():
    ensure_dir(OUTPUT_ROOT)

    log("=" * 80)
    log("D-Clinical high BP diagnostic V1")
    log(f"Scan root: {AIDO_TEMP_ROOT}")
    log(f"Output root: {OUTPUT_ROOT}")
    log("=" * 80)

    folders = find_result_folders()
    pd.DataFrame({"result_folder": [str(x) for x in folders]}).to_csv(
        OUTPUT_ROOT / "00_detected_result_folders.csv",
        index=False
    )

    log(f"Detected result folders: {len(folders)}")
    log(f"Latest serious filter root: {latest_serious_filter_root()}")

    summaries = []
    failures = []

    for folder in folders:
        try:
            summaries.append(run_one_target(folder))
        except Exception as e:
            log(f"FAILED: {folder} | {e}")
            failures.append({
                "result_folder": str(folder),
                "error": str(e)
            })

    summary_df = pd.DataFrame(summaries)
    failure_df = pd.DataFrame(failures)

    summary_df.to_csv(OUTPUT_ROOT / "ALL_DClinical_high_BP_vs_random_summary.csv", index=False)
    failure_df.to_csv(OUTPUT_ROOT / "ALL_DClinical_high_BP_failures.csv", index=False)

    xlsx = OUTPUT_ROOT / "DClinical_high_BP_Diagnostics_SUMMARY.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        failure_df.to_excel(writer, sheet_name="failures", index=False)

    readme = f"""
D-Clinical high BP diagnostic V1

This module answers:
1. For high D-Clinical BP terms, what are the BP-level μ/mean values in each class?
2. What D-PHY characteristics do these BP terms have?
3. Are high D-Clinical BP terms better than random BP sets?

Output root:
{OUTPUT_ROOT}

Main files:
- ALL_DClinical_high_BP_vs_random_summary.csv
- DClinical_high_BP_Diagnostics_SUMMARY.xlsx

Per target:
- DClinical_high_BP_mu_DPHY_features.csv
- DClinical_high_BP_vs_random_summary.csv
- DClinical_high_BP_random_null_distribution.csv
- DClinical_high_BP_diagnostic.xlsx

Important interpretation:
If high D-Clinical BP terms have weak D-PHY features but outperform random BP sets in multivariate AUC,
the endpoint signal is likely distributed/combinatorial.
If they also show strong D-PHY features, the endpoint is more local-process-driven.
"""

    with open(OUTPUT_ROOT / "README_DClinical_high_BP_diagnostic.txt", "w", encoding="utf-8") as f:
        f.write(readme)

    log("=" * 80)
    log("D-CLINICAL HIGH BP DIAGNOSTIC COMPLETE")
    log(f"Successful targets: {len(summary_df)}")
    log(f"Failed targets: {len(failure_df)}")
    log(f"Output: {OUTPUT_ROOT}")
    log("=" * 80)


if __name__ == "__main__":
    main()
