# -*- coding: utf-8 -*-
r"""
SERIOUS FILTER TEST V1
D-PHY-first vs D-Clinical-first BP filtering

Purpose
-------
This is a test script, not a final manuscript analysis.

It compares two BP filtering routes using already completed D-PHY run folders:

Route A. D-PHY-first
    All BP terms
    -> rank/filter by univariate process-level discriminability
    -> ask whether selected BP terms contribute to target observability

Route B. D-Clinical-first
    All BP terms
    -> repeated CV logistic regression on full BP space
    -> rank BP terms by standardized coefficient importance and stability
    -> ask whether clinically important BP terms are also D-PHY-discriminative

Main outputs
------------
For each completed run folder:
- DPHY_first_topK_BP.csv
- DClinical_first_topK_BP.csv
- SERIOUS_FILTER_2x2_BP_categories.csv
- SERIOUS_FILTER_overlap_metrics.csv
- SERIOUS_FILTER_summary.json
- SERIOUS_FILTER_results.xlsx

Across all runs:
- ALL_SERIOUS_FILTER_SUMMARY.csv
- DPHY_DClinical_SERIOUS_FILTER_SUMMARY.xlsx

Default scan root:
D:/AIDO-Temp/

Author:
AIDO / Kong Sin Guan
"""

import os
import re
import gc
import json
import time
import glob
import warnings
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score


# ============================================================
# 0. CONFIG
# ============================================================

AIDO_TEMP_ROOT = Path("D:/AIDO-Temp")
OUTPUT_ROOT = AIDO_TEMP_ROOT / f"D_PHY_SERIOUS_FILTER_Test_{time.strftime('%Y%m%d_%H%M%S')}"

RUN_FOLDER_PATTERNS = [
    "D_PHY_BioSystems_V5_Run_*",          # TCGA-BRCA early vs late stage
    "D_PHY_METABRIC_External_V2_Run_*",   # METABRIC early vs late stage
    "D_PHY_GSE96058_External_V2_Run_*",   # GSE96058 lymph node
    "D_PHY_TCGA_NewTarget_Run_*",         # TCGA-BRCA tumor vs normal
]

TOP_K_LIST = [10, 25, 50, 100, 200]

# Used for 2x2 classification
DPHY_HIGH_FDR = 0.05
DPHY_HIGH_ABS_D = 0.30

# D-Clinical high definition:
# top K by repeated-CV coefficient importance.
DCLINICAL_HIGH_TOPK = 50

# Repeated CV for D-Clinical coefficient stability.
N_SPLITS = 5
N_REPEATS = 20

RANDOM_SEED = 42


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

    return dataset, endpoint


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
        # choose second column if possible
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

    # Try to keep original AIDO convention:
    # late/tumor/node-positive/basal-like should be positive class when possible.
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

    return bp2, y, y_bin, pos_class, classes


# ============================================================
# 2. D-CLINICAL FIRST RANKING
# ============================================================

def compute_dclinical_importance(bp, y_bin):
    """
    Repeated stratified CV logistic regression coefficient ranking.

    Uses standardized BP matrix and balanced logistic regression.

    Outputs:
    - mean_abs_coef
    - mean_signed_coef
    - coef_sd
    - selection_stability_top50
    - sign_stability
    - rank
    """
    X = bp.values
    feature_names = np.array(bp.columns)

    y_bin = np.asarray(y_bin).astype(int)

    min_class = int(np.bincount(y_bin).min())
    n_splits = min(N_SPLITS, min_class)

    if n_splits < 2:
        raise ValueError("Not enough samples in one class for repeated CV coefficient ranking.")

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    rkf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED
    )

    coef_records = []
    top_counter = Counter()
    pos_counter = Counter()
    neg_counter = Counter()

    for fold_id, (train_idx, test_idx) in enumerate(rkf.split(Xz, y_bin), start=1):
        clf = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED + fold_id
        )

        clf.fit(Xz[train_idx], y_bin[train_idx])
        coef = clf.coef_.ravel()

        abs_coef = np.abs(coef)
        order = np.argsort(-abs_coef)
        top50 = feature_names[order[:min(DCLINICAL_HIGH_TOPK, len(feature_names))]]

        for t in top50:
            top_counter[str(t)] += 1

        for fname, c in zip(feature_names, coef):
            if c > 0:
                pos_counter[str(fname)] += 1
            elif c < 0:
                neg_counter[str(fname)] += 1

        coef_records.append(coef)

    coef_mat = np.vstack(coef_records)
    n_models = coef_mat.shape[0]

    mean_signed = coef_mat.mean(axis=0)
    mean_abs = np.abs(coef_mat).mean(axis=0)
    sd = coef_mat.std(axis=0)

    rows = []
    for j, fname in enumerate(feature_names):
        fname = str(fname)
        pos_n = pos_counter[fname]
        neg_n = neg_counter[fname]
        sign_stability = max(pos_n, neg_n) / n_models

        rows.append({
            "BP_term": fname,
            "DClinical_mean_abs_coef": mean_abs[j],
            "DClinical_mean_signed_coef": mean_signed[j],
            "DClinical_coef_sd": sd[j],
            "DClinical_selection_stability_top50": top_counter[fname] / n_models,
            "DClinical_sign_stability": sign_stability,
            "DClinical_direction": "positive_class_up" if mean_signed[j] > 0 else "negative_class_up",
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("DClinical_mean_abs_coef", ascending=False).reset_index(drop=True)
    out["DClinical_rank"] = np.arange(1, len(out) + 1)

    return out


def evaluate_feature_set_auc(bp, y_bin, feature_terms, label):
    terms = [t for t in feature_terms if t in bp.columns]

    if len(terms) == 0:
        return {
            "feature_set": label,
            "n_features": 0,
            "cv_auc": np.nan,
            "cv_balanced_accuracy": np.nan,
            "cv_accuracy": np.nan,
        }

    X = bp[terms].values
    y_bin = np.asarray(y_bin).astype(int)

    min_class = int(np.bincount(y_bin).min())
    n_splits = min(5, min_class)

    if n_splits < 2:
        return {
            "feature_set": label,
            "n_features": len(terms),
            "cv_auc": np.nan,
            "cv_balanced_accuracy": np.nan,
            "cv_accuracy": np.nan,
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

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    prob = cross_val_predict(clf, X, y_bin, cv=cv, method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(int)

    return {
        "feature_set": label,
        "n_features": len(terms),
        "cv_auc": roc_auc_score(y_bin, prob),
        "cv_balanced_accuracy": balanced_accuracy_score(y_bin, pred),
        "cv_accuracy": accuracy_score(y_bin, pred),
    }


# ============================================================
# 3. D-PHY FIRST RANKING AND COMPARISON
# ============================================================

def prepare_dphy_ranking(folder):
    d = safe_read_csv(Path(folder) / "03_D_layer_ranked_BP_signals.csv")
    if d is None:
        raise ValueError("Missing D layer table.")

    needed = ["BP_term", "D_score", "abs_cohen_d", "welch_fdr", "direction"]
    for c in needed:
        if c not in d.columns:
            d[c] = np.nan

    d["D_score"] = pd.to_numeric(d["D_score"], errors="coerce")
    d["abs_cohen_d"] = pd.to_numeric(d["abs_cohen_d"], errors="coerce")
    d["welch_fdr"] = pd.to_numeric(d["welch_fdr"], errors="coerce")

    d = d.sort_values(["D_score", "abs_cohen_d"], ascending=False).reset_index(drop=True)
    d["DPHY_rank"] = np.arange(1, len(d) + 1)

    d["DPHY_high_flag"] = (
        (d["welch_fdr"] <= DPHY_HIGH_FDR)
        & (d["abs_cohen_d"] >= DPHY_HIGH_ABS_D)
    )

    return d


def topk_overlap_metrics(dphy, dclinical, bp, y_bin):
    rows = []

    for k in TOP_K_LIST:
        dphy_top = dphy.head(min(k, len(dphy)))["BP_term"].astype(str).tolist()
        dclin_top = dclinical.head(min(k, len(dclinical)))["BP_term"].astype(str).tolist()

        set_a = set(dphy_top)
        set_b = set(dclin_top)

        inter = set_a.intersection(set_b)
        union = set_a.union(set_b)

        jaccard = len(inter) / len(union) if union else np.nan
        overlap_fraction_of_k = len(inter) / k if k else np.nan

        auc_dphy = evaluate_feature_set_auc(bp, y_bin, dphy_top, f"DPHY_top_{k}")
        auc_dclin = evaluate_feature_set_auc(bp, y_bin, dclin_top, f"DClinical_top_{k}")
        auc_inter = evaluate_feature_set_auc(bp, y_bin, list(inter), f"Intersection_top_{k}")

        rows.append({
            "top_k": k,
            "n_DPHY_top": len(set_a),
            "n_DClinical_top": len(set_b),
            "n_overlap": len(inter),
            "jaccard": jaccard,
            "overlap_fraction_of_k": overlap_fraction_of_k,
            "DPHY_topK_auc": auc_dphy["cv_auc"],
            "DClinical_topK_auc": auc_dclin["cv_auc"],
            "Intersection_topK_auc": auc_inter["cv_auc"],
            "DPHY_topK_balanced_accuracy": auc_dphy["cv_balanced_accuracy"],
            "DClinical_topK_balanced_accuracy": auc_dclin["cv_balanced_accuracy"],
            "Intersection_topK_balanced_accuracy": auc_inter["cv_balanced_accuracy"],
        })

    return pd.DataFrame(rows)


def build_2x2_categories(dphy, dclinical):
    """
    Merge D-PHY and D-Clinical rankings and classify each BP.

    Axes:
    - D-PHY high:
        FDR <= DPHY_HIGH_FDR and abs_d >= DPHY_HIGH_ABS_D
    - D-Clinical high:
        top DCLINICAL_HIGH_TOPK by coefficient ranking
    """
    left_cols = [
        "BP_term", "DPHY_rank", "D_score", "abs_cohen_d", "welch_fdr",
        "direction", "DPHY_high_flag"
    ]

    for c in left_cols:
        if c not in dphy.columns:
            dphy[c] = np.nan

    right_cols = [
        "BP_term", "DClinical_rank", "DClinical_mean_abs_coef",
        "DClinical_mean_signed_coef", "DClinical_selection_stability_top50",
        "DClinical_sign_stability", "DClinical_direction"
    ]

    for c in right_cols:
        if c not in dclinical.columns:
            dclinical[c] = np.nan

    m = dphy[left_cols].merge(dclinical[right_cols], on="BP_term", how="outer")

    m["DPHY_rank"] = pd.to_numeric(m["DPHY_rank"], errors="coerce")
    m["DClinical_rank"] = pd.to_numeric(m["DClinical_rank"], errors="coerce")

    m["DClinical_high_flag"] = m["DClinical_rank"] <= DCLINICAL_HIGH_TOPK

    def classify(row):
        a = bool(row.get("DPHY_high_flag", False))
        b = bool(row.get("DClinical_high_flag", False))

        if a and b:
            return "core_overlap_DPHY_high_and_DClinical_high"
        if a and not b:
            return "local_process_discriminative_but_clinically_non_dominant"
        if (not a) and b:
            return "clinically_contributive_but_individually_weak"
        return "weak_or_non_priority"

    m["serious_filter_category"] = m.apply(classify, axis=1)

    # Rank agreement
    valid = m.dropna(subset=["DPHY_rank", "DClinical_rank"]).copy()
    if len(valid) >= 3:
        rho, p = spearmanr(valid["DPHY_rank"], valid["DClinical_rank"])
    else:
        rho, p = np.nan, np.nan

    m.attrs["rank_spearman_rho"] = rho
    m.attrs["rank_spearman_p"] = p

    # Sort categories first, then rankings
    category_order = {
        "core_overlap_DPHY_high_and_DClinical_high": 0,
        "clinically_contributive_but_individually_weak": 1,
        "local_process_discriminative_but_clinically_non_dominant": 2,
        "weak_or_non_priority": 3,
    }

    m["_cat_order"] = m["serious_filter_category"].map(category_order)
    m = m.sort_values(
        ["_cat_order", "DClinical_rank", "DPHY_rank"],
        ascending=[True, True, True],
        na_position="last"
    ).drop(columns=["_cat_order"]).reset_index(drop=True)

    return m


# ============================================================
# 4. RUN ONE FOLDER
# ============================================================

def run_one_folder(folder, output_root):
    folder = Path(folder)

    summary = safe_read_json(folder / "SUMMARY.json")
    dataset, endpoint = infer_dataset_endpoint(folder, summary)

    label = f"{safe_name(dataset)}__{safe_name(endpoint)}"
    out_dir = Path(output_root) / label
    ensure_dir(out_dir)

    log("=" * 80)
    log(f"SERIOUS FILTER TEST: {dataset} | {endpoint}")
    log(f"Folder: {folder}")

    bp = load_bp_matrix(folder)
    labels = load_labels(folder)
    bp, y, y_bin, pos_class, classes = align_bp_labels(bp, labels)

    log(f"Aligned BP matrix: {bp.shape}; classes={classes}; positive={pos_class}")

    dphy = prepare_dphy_ranking(folder)
    dclinical = compute_dclinical_importance(bp, y_bin)

    # Restrict both to BP matrix columns
    dphy = dphy[dphy["BP_term"].astype(str).isin(bp.columns.astype(str))].reset_index(drop=True)
    dclinical = dclinical[dclinical["BP_term"].astype(str).isin(bp.columns.astype(str))].reset_index(drop=True)

    dphy.to_csv(out_dir / "DPHY_first_all_BP_ranked.csv", index=False)
    dclinical.to_csv(out_dir / "DClinical_first_all_BP_ranked.csv", index=False)

    for k in TOP_K_LIST:
        dphy.head(min(k, len(dphy))).to_csv(out_dir / f"DPHY_first_top{k}_BP.csv", index=False)
        dclinical.head(min(k, len(dclinical))).to_csv(out_dir / f"DClinical_first_top{k}_BP.csv", index=False)

    overlap_df = topk_overlap_metrics(dphy, dclinical, bp, y_bin)
    overlap_df.to_csv(out_dir / "SERIOUS_FILTER_overlap_metrics.csv", index=False)

    cat_df = build_2x2_categories(dphy, dclinical)
    cat_df.to_csv(out_dir / "SERIOUS_FILTER_2x2_BP_categories.csv", index=False)

    cat_counts = cat_df["serious_filter_category"].value_counts().to_dict()

    # Also evaluate four sets from 2x2 category, where available
    set_eval_rows = []
    for cat in [
        "core_overlap_DPHY_high_and_DClinical_high",
        "local_process_discriminative_but_clinically_non_dominant",
        "clinically_contributive_but_individually_weak",
    ]:
        terms = cat_df.loc[cat_df["serious_filter_category"] == cat, "BP_term"].dropna().astype(str).tolist()
        terms = terms[:200]  # cap for simple evaluation
        ev = evaluate_feature_set_auc(bp, y_bin, terms, cat)
        set_eval_rows.append(ev)

    set_eval_df = pd.DataFrame(set_eval_rows)
    set_eval_df.to_csv(out_dir / "SERIOUS_FILTER_category_feature_set_AUC.csv", index=False)

    # Rank correlation from attrs
    rho = cat_df.attrs.get("rank_spearman_rho", np.nan)
    pval = cat_df.attrs.get("rank_spearman_p", np.nan)

    # Main top50 overlap row
    top50 = overlap_df[overlap_df["top_k"] == 50].iloc[0].to_dict() if (overlap_df["top_k"] == 50).any() else {}

    n_dphy_high = int(cat_df["DPHY_high_flag"].fillna(False).sum())
    n_dclinical_high = int(cat_df["DClinical_high_flag"].fillna(False).sum())

    run_summary = {
        "dataset": dataset,
        "endpoint": endpoint,
        "source_folder": str(folder),
        "output_dir": str(out_dir),
        "n_samples": int(bp.shape[0]),
        "n_bp_terms": int(bp.shape[1]),
        "classes": ";".join(map(str, classes)),
        "positive_class": str(pos_class),
        "n_DPHY_high": n_dphy_high,
        "n_DClinical_high_top50": n_dclinical_high,
        "n_core_overlap_DPHY_high_and_DClinical_high": int(cat_counts.get("core_overlap_DPHY_high_and_DClinical_high", 0)),
        "n_local_process_discriminative_but_clinically_non_dominant": int(cat_counts.get("local_process_discriminative_but_clinically_non_dominant", 0)),
        "n_clinically_contributive_but_individually_weak": int(cat_counts.get("clinically_contributive_but_individually_weak", 0)),
        "n_weak_or_non_priority": int(cat_counts.get("weak_or_non_priority", 0)),
        "rank_spearman_rho_DPHY_vs_DClinical": rho,
        "rank_spearman_p_DPHY_vs_DClinical": pval,
        "top50_n_overlap": top50.get("n_overlap", np.nan),
        "top50_jaccard": top50.get("jaccard", np.nan),
        "top50_overlap_fraction_of_k": top50.get("overlap_fraction_of_k", np.nan),
        "top50_DPHY_auc": top50.get("DPHY_topK_auc", np.nan),
        "top50_DClinical_auc": top50.get("DClinical_topK_auc", np.nan),
        "top50_Intersection_auc": top50.get("Intersection_topK_auc", np.nan),
        "top1_DPHY_BP": dphy.iloc[0]["BP_term"] if len(dphy) else "",
        "top1_DClinical_BP": dclinical.iloc[0]["BP_term"] if len(dclinical) else "",
    }

    with open(out_dir / "SERIOUS_FILTER_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    with pd.ExcelWriter(out_dir / "SERIOUS_FILTER_results.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([run_summary]).to_excel(writer, sheet_name="summary", index=False)
        overlap_df.to_excel(writer, sheet_name="topk_overlap", index=False)
        cat_df.head(2000).to_excel(writer, sheet_name="2x2_categories_top2000", index=False)
        dphy.head(500).to_excel(writer, sheet_name="DPHY_first_top500", index=False)
        dclinical.head(500).to_excel(writer, sheet_name="DClinical_first_top500", index=False)
        set_eval_df.to_excel(writer, sheet_name="category_auc", index=False)

    # Create a short note
    with open(out_dir / "SERIOUS_FILTER_INTERPRETATION_NOTE.txt", "w", encoding="utf-8") as f:
        f.write(f"SERIOUS FILTER TEST\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Endpoint: {endpoint}\n")
        f.write(f"Samples: {bp.shape[0]}\n")
        f.write(f"BP terms: {bp.shape[1]}\n\n")
        f.write("Interpretation of categories:\n")
        f.write("- core_overlap_DPHY_high_and_DClinical_high: BP terms passing both process-first and clinical-first filters.\n")
        f.write("- local_process_discriminative_but_clinically_non_dominant: individually discriminative BP terms not among top D-Clinical contributors.\n")
        f.write("- clinically_contributive_but_individually_weak: clinically important BP terms not individually strong by D-PHY.\n")
        f.write("- weak_or_non_priority: neither filter prioritizes the BP strongly.\n\n")
        f.write("Key numbers:\n")
        for k, v in run_summary.items():
            f.write(f"{k}: {v}\n")

    del bp, dphy, dclinical, cat_df
    gc.collect()

    return run_summary


# ============================================================
# 5. MAIN
# ============================================================

def main():
    ensure_dir(OUTPUT_ROOT)

    log("=" * 80)
    log("D-PHY / D-Clinical SERIOUS FILTER TEST V1")
    log(f"Scan root: {AIDO_TEMP_ROOT}")
    log(f"Output root: {OUTPUT_ROOT}")
    log("=" * 80)

    folders = find_result_folders()

    pd.DataFrame({"result_folder": [str(x) for x in folders]}).to_csv(
        OUTPUT_ROOT / "00_detected_result_folders.csv",
        index=False
    )

    log(f"Detected result folders: {len(folders)}")

    summaries = []
    failures = []

    # Deduplicate folders by dataset+endpoint, keeping latest/most samples later.
    # First run all; summary table will expose duplicates if any.
    for folder in folders:
        try:
            summaries.append(run_one_folder(folder, OUTPUT_ROOT))
        except Exception as e:
            log(f"FAILED: {folder} | {e}")
            failures.append({
                "result_folder": str(folder),
                "error": str(e)
            })

    summary_df = pd.DataFrame(summaries)
    failure_df = pd.DataFrame(failures)

    if len(summary_df) > 0:
        summary_df.to_csv(OUTPUT_ROOT / "ALL_SERIOUS_FILTER_SUMMARY.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_ROOT / "ALL_SERIOUS_FILTER_SUMMARY.csv", index=False)

    failure_df.to_csv(OUTPUT_ROOT / "ALL_SERIOUS_FILTER_FAILURES.csv", index=False)

    xlsx = OUTPUT_ROOT / "DPHY_DClinical_SERIOUS_FILTER_SUMMARY.xlsx"

    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        failure_df.to_excel(writer, sheet_name="failures", index=False)

    readme = f"""
SERIOUS FILTER TEST V1

This test compares:
1. D-PHY-first BP filtering:
   univariate process-level discriminability first.

2. D-Clinical-first BP filtering:
   repeated-CV logistic-regression coefficient importance in full BP space first.

The two routes do not have to produce identical BP lists.
Their overlap and discordance are the main diagnostic outputs.

Key output:
{OUTPUT_ROOT}

Most important files:
- ALL_SERIOUS_FILTER_SUMMARY.csv
- DPHY_DClinical_SERIOUS_FILTER_SUMMARY.xlsx
- Per target:
  - DPHY_first_top50_BP.csv
  - DClinical_first_top50_BP.csv
  - SERIOUS_FILTER_2x2_BP_categories.csv
  - SERIOUS_FILTER_overlap_metrics.csv

Category meaning:
- core_overlap_DPHY_high_and_DClinical_high:
  BP terms prioritized by both local process discriminability and global clinical contribution.

- local_process_discriminative_but_clinically_non_dominant:
  BP terms individually discriminative but not dominant in the full BP clinical model.

- clinically_contributive_but_individually_weak:
  BP terms not individually strong but useful in multivariate clinical observability.

- weak_or_non_priority:
  BP terms not strongly prioritized by either route.
"""

    with open(OUTPUT_ROOT / "README_SERIOUS_FILTER_TEST.txt", "w", encoding="utf-8") as f:
        f.write(readme)

    log("=" * 80)
    log("SERIOUS FILTER TEST COMPLETE")
    log(f"Successful folders: {len(summary_df)}")
    log(f"Failed folders: {len(failure_df)}")
    log(f"Output: {OUTPUT_ROOT}")
    log("=" * 80)


if __name__ == "__main__":
    main()
