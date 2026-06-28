# -*- coding: utf-8 -*-
r"""
D-PHY vs D-Clinical comparison V1

Purpose:
Aggregate and compare existing AIDO/D-PHY run outputs under D:/AIDO-Temp.

This script compares:
1. D-Clinical:
   - full BP-space clinical/endpoint observability
   - from 07A_DPHY_vs_DClinical_concordance.csv
   - row: D-Clinical_all_BP_space

2. D-PHY:
   - compact top-BP representation selected by D layer
   - from 07A_DPHY_vs_DClinical_concordance.csv
   - row: D-PHY_top_N_BP_space

3. D layer:
   - number of significant BP terms
   - top BP terms and effect sizes

4. Interpretation readiness:
   - strong/moderate/weak category counts
   - whether significant BP signals become interpretation-ready

Default input:
D:/AIDO-Temp/

Default output:
D:/AIDO-Temp/D_PHY_vs_DClinical_Comparison_<timestamp>/

Author:
AIDO / Kong Sin Guan
"""

import os
import re
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIG
# ============================================================

AIDO_TEMP_ROOT = Path("D:/AIDO-Temp")

OUTPUT_ROOT = AIDO_TEMP_ROOT / f"D_PHY_vs_DClinical_Comparison_{time.strftime('%Y%m%d_%H%M%S')}"

# Scan these run-folder patterns.
# Add more patterns if future runs have new names.
RUN_FOLDER_PATTERNS = [
    "D_PHY_BioSystems_V5_Run_*",          # TCGA-BRCA early vs late stage
    "D_PHY_METABRIC_External_V2_Run_*",   # METABRIC early vs late stage
    "D_PHY_GSE96058_External_V2_Run_*",   # GSE96058 node status
    "D_PHY_TCGA_NewTarget_Run_*",         # TCGA-BRCA tumor vs normal / subtype
    "D_PHY_BioSystems_Run_*",
    "D_PHY_BioSystems_V4_Run_*",
    "D_PHY_METABRIC_External_Run_*",
    "D_PHY_GSE96058_External_Run_*",
]

FDR_THRESHOLD = 0.05


# ============================================================
# 1. BASIC HELPERS
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


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


def short_run_label(path):
    p = Path(path)
    s = p.name

    # Clean common long names
    s = s.replace("D_PHY_internal_", "")
    s = s.replace("D_PHY_external_", "")
    s = s.replace("D_PHY_TCGA_BRCA_", "TCGA_BRCA_")
    s = s.replace("Breast_Cancer_(BRCA)", "BRCA")

    return s


def infer_dataset_and_endpoint(folder, summary):
    """
    Infer readable dataset/endpoint from folder name and summary fields.
    """
    folder_name = Path(folder).name
    folder_lower = folder_name.lower()

    dataset = summary.get("cohort_name") or summary.get("cancer_name") or folder_name
    endpoint = (
        summary.get("target_mode")
        or summary.get("endpoint_type")
        or summary.get("endpoint")
        or summary.get("endpoint_name")
        or "unknown_endpoint"
    )

    # More explicit labels from known folder patterns
    if "tcga_brca_tumor_vs_normal" in folder_lower:
        dataset = "TCGA-BRCA"
        endpoint = "tumor_vs_normal"
    elif "tcga" in folder_lower and "brca" in folder_lower and "tumor_vs_normal" not in folder_lower:
        dataset = "TCGA-BRCA"
        if endpoint == "unknown_endpoint":
            endpoint = "early_vs_late_stage"
    elif "metabric" in folder_lower:
        dataset = "METABRIC"
        if endpoint in ["unknown_endpoint", "stage"]:
            endpoint = "early_vs_late_stage"
    elif "gse96058" in folder_lower:
        dataset = "GSE96058"
        if endpoint == "node":
            endpoint = "lymph_node_status"
    elif "brca" in folder_lower:
        dataset = "TCGA-BRCA"

    return dataset, endpoint


def find_all_result_folders():
    """
    Find per-cohort result folders that contain:
    - SUMMARY.json
    - 07A_DPHY_vs_DClinical_concordance.csv
    """
    run_roots = []

    for pattern in RUN_FOLDER_PATTERNS:
        run_roots.extend(AIDO_TEMP_ROOT.glob(pattern))

    run_roots = sorted(set([p for p in run_roots if p.is_dir()]))

    result_folders = []

    for root in run_roots:
        for p in root.rglob("*"):
            if not p.is_dir():
                continue

            summary_path = p / "SUMMARY.json"
            conc_path = p / "07A_DPHY_vs_DClinical_concordance.csv"

            if summary_path.exists() and conc_path.exists():
                result_folders.append(p)

    result_folders = sorted(set(result_folders))

    return run_roots, result_folders


# ============================================================
# 2. EXTRACT ONE RESULT FOLDER
# ============================================================

def extract_concordance_metrics(conc_df):
    """
    Return D-Clinical and D-PHY top-BP metrics from 07A file.
    """
    out = {
        "DClinical_n_features": np.nan,
        "DClinical_auc": np.nan,
        "DClinical_balanced_accuracy": np.nan,
        "DClinical_accuracy": np.nan,
        "DPHY_feature_set": "",
        "DPHY_n_features": np.nan,
        "DPHY_auc": np.nan,
        "DPHY_balanced_accuracy": np.nan,
        "DPHY_accuracy": np.nan,
        "DPHY_minus_DClinical_auc": np.nan,
        "DPHY_minus_DClinical_balanced_accuracy": np.nan,
    }

    if conc_df is None or len(conc_df) == 0 or "feature_set" not in conc_df.columns:
        return out

    # D-Clinical row
    dclin = conc_df[conc_df["feature_set"].astype(str).str.contains("D-Clinical", case=False, na=False)]
    if len(dclin) > 0:
        r = dclin.iloc[0]
        out["DClinical_n_features"] = r.get("n_features", np.nan)
        out["DClinical_auc"] = r.get("cv_auc", np.nan)
        out["DClinical_balanced_accuracy"] = r.get("cv_balanced_accuracy", np.nan)
        out["DClinical_accuracy"] = r.get("cv_accuracy", np.nan)

    # D-PHY row
    dphy = conc_df[conc_df["feature_set"].astype(str).str.contains("D-PHY", case=False, na=False)]
    if len(dphy) > 0:
        r = dphy.iloc[0]
        out["DPHY_feature_set"] = r.get("feature_set", "")
        out["DPHY_n_features"] = r.get("n_features", np.nan)
        out["DPHY_auc"] = r.get("cv_auc", np.nan)
        out["DPHY_balanced_accuracy"] = r.get("cv_balanced_accuracy", np.nan)
        out["DPHY_accuracy"] = r.get("cv_accuracy", np.nan)

    try:
        out["DPHY_minus_DClinical_auc"] = float(out["DPHY_auc"]) - float(out["DClinical_auc"])
    except Exception:
        pass

    try:
        out["DPHY_minus_DClinical_balanced_accuracy"] = (
            float(out["DPHY_balanced_accuracy"]) - float(out["DClinical_balanced_accuracy"])
        )
    except Exception:
        pass

    return out


def extract_d_layer_metrics(d_df):
    """
    Extract D-layer signal count and top BP summary.
    """
    out = {
        "n_D_significant_FDR05_from_D_table": np.nan,
        "top1_BP": "",
        "top1_direction": "",
        "top1_abs_cohen_d": np.nan,
        "top1_auc": np.nan,
        "top1_fdr": np.nan,
        "top10_median_abs_cohen_d": np.nan,
        "top50_median_abs_cohen_d": np.nan,
        "top50_median_fdr": np.nan,
        "top50_late_up_n": np.nan,
        "top50_early_up_n": np.nan,
    }

    if d_df is None or len(d_df) == 0:
        return out

    if "welch_fdr" in d_df.columns:
        out["n_D_significant_FDR05_from_D_table"] = int((pd.to_numeric(d_df["welch_fdr"], errors="coerce") <= FDR_THRESHOLD).sum())

    first = d_df.iloc[0]
    out["top1_BP"] = first.get("BP_term", "")
    out["top1_direction"] = first.get("direction", "")
    out["top1_abs_cohen_d"] = first.get("abs_cohen_d", np.nan)
    out["top1_auc"] = first.get("auc_late_vs_early", np.nan)
    out["top1_fdr"] = first.get("welch_fdr", np.nan)

    if "abs_cohen_d" in d_df.columns:
        out["top10_median_abs_cohen_d"] = pd.to_numeric(d_df.head(10)["abs_cohen_d"], errors="coerce").median()
        out["top50_median_abs_cohen_d"] = pd.to_numeric(d_df.head(50)["abs_cohen_d"], errors="coerce").median()

    if "welch_fdr" in d_df.columns:
        out["top50_median_fdr"] = pd.to_numeric(d_df.head(50)["welch_fdr"], errors="coerce").median()

    if "direction" in d_df.columns:
        top50_dir = d_df.head(50)["direction"].astype(str)
        out["top50_late_up_n"] = int((top50_dir == "late_up").sum())
        out["top50_early_up_n"] = int((top50_dir == "early_up").sum())

    return out


def extract_readiness_metrics(readiness_df):
    """
    Extract interpretation-readiness category counts.
    """
    out = {
        "n_readiness_total": np.nan,
        "n_interpretation_ready_strong_from_profile": 0,
        "n_interpretation_ready_moderate_from_profile": 0,
        "n_discriminative_but_statistically_unstable": 0,
        "n_statistically_supported_but_weakly_anchored": 0,
        "n_biologically_anchored_but_endpoint_weak": 0,
        "n_exploratory_or_weak": 0,
        "readiness_strong_moderate_total": 0,
        "readiness_strong_moderate_fraction": np.nan,
    }

    if readiness_df is None or len(readiness_df) == 0 or "interpretation_readiness_class" not in readiness_df.columns:
        return out

    cls = readiness_df["interpretation_readiness_class"].astype(str)
    counts = cls.value_counts().to_dict()

    out["n_readiness_total"] = int(len(readiness_df))

    mapping = {
        "interpretation_ready_strong": "n_interpretation_ready_strong_from_profile",
        "interpretation_ready_moderate": "n_interpretation_ready_moderate_from_profile",
        "discriminative_but_statistically_unstable": "n_discriminative_but_statistically_unstable",
        "statistically_supported_but_weakly_anchored": "n_statistically_supported_but_weakly_anchored",
        "biologically_anchored_but_endpoint_weak": "n_biologically_anchored_but_endpoint_weak",
        "exploratory_or_weak": "n_exploratory_or_weak",
    }

    for k, out_col in mapping.items():
        out[out_col] = int(counts.get(k, 0))

    out["readiness_strong_moderate_total"] = (
        out["n_interpretation_ready_strong_from_profile"]
        + out["n_interpretation_ready_moderate_from_profile"]
    )

    if out["n_readiness_total"] and out["n_readiness_total"] > 0:
        out["readiness_strong_moderate_fraction"] = out["readiness_strong_moderate_total"] / out["n_readiness_total"]

    return out


def extract_one_result_folder(folder):
    folder = Path(folder)

    summary = safe_read_json(folder / "SUMMARY.json")

    conc_df = safe_read_csv(folder / "07A_DPHY_vs_DClinical_concordance.csv")
    d_df = safe_read_csv(folder / "03_D_layer_ranked_BP_signals.csv")
    readiness_df = safe_read_csv(folder / "08_interpretation_readiness_profile.csv")
    metadata_df = safe_read_csv(folder / "00_run_metadata.csv")
    endpoint_summary_df = safe_read_csv(folder / "07B_endpoint_summary.csv")

    dataset, endpoint = infer_dataset_and_endpoint(folder, summary)

    row = {
        "result_folder": str(folder),
        "run_label": short_run_label(folder),
        "dataset": dataset,
        "endpoint": endpoint,
        "cohort_name": summary.get("cohort_name", summary.get("cancer_name", "")),
        "cancer_code": summary.get("cancer_code", summary.get("cohort_code", "")),
        "target_mode": summary.get("target_mode", ""),
        "endpoint_type": summary.get("endpoint_type", ""),
        "endpoint_column": summary.get("endpoint_column", ""),
        "alignment_column": summary.get("alignment_column", ""),
        "label_source": summary.get("label_source", ""),
        "n_samples": summary.get("n_samples", np.nan),
        "n_early_like": summary.get("n_early_like", summary.get("n_early", np.nan)),
        "n_late_like": summary.get("n_late_like", summary.get("n_late", np.nan)),
        "n_genes": summary.get("n_genes", np.nan),
        "n_bp_terms": summary.get("n_bp_terms", np.nan),
        "n_D_significant_FDR05": summary.get("n_D_significant_FDR05", np.nan),
        "n_interpretation_ready_strong": summary.get("n_interpretation_ready_strong", np.nan),
        "n_interpretation_ready_moderate": summary.get("n_interpretation_ready_moderate", np.nan),
    }

    row.update(extract_concordance_metrics(conc_df))
    row.update(extract_d_layer_metrics(d_df))
    row.update(extract_readiness_metrics(readiness_df))

    # Add metadata columns if available
    if metadata_df is not None and len(metadata_df) > 0:
        meta = metadata_df.iloc[0].to_dict()
        for k, v in meta.items():
            key = f"metadata_{k}"
            if key not in row:
                row[key] = v

    return row


# ============================================================
# 3. OUTPUT TABLES AND FIGURES
# ============================================================

def classify_observability(row):
    """
    Simple descriptive class for the comparison table.
    """
    auc = row.get("DClinical_auc", np.nan)
    n_sig = row.get("n_D_significant_FDR05", np.nan)
    n_ready = row.get("readiness_strong_moderate_total", np.nan)

    try:
        auc = float(auc)
    except Exception:
        auc = np.nan

    try:
        n_sig = float(n_sig)
    except Exception:
        n_sig = np.nan

    try:
        n_ready = float(n_ready)
    except Exception:
        n_ready = np.nan

    if np.isfinite(auc) and auc >= 0.90 and n_ready > 0:
        return "strong_process_observability_and_interpretation_ready"

    if np.isfinite(auc) and auc >= 0.80:
        return "strong_process_observability"

    if np.isfinite(auc) and auc >= 0.65 and n_sig >= 50:
        return "moderate_process_observability"

    if np.isfinite(auc) and auc >= 0.58 and n_sig >= 1:
        return "partial_process_observability"

    if np.isfinite(auc):
        return "weak_or_unclear_process_observability"

    return "not_evaluable"


def make_comparison_figures(summary_df, out_dir):
    """
    Save simple figures:
    1. D-Clinical vs D-PHY AUC
    2. D-PHY minus D-Clinical AUC
    3. Significant BP count
    4. Interpretation-ready count
    """
    ensure_dir(out_dir)

    df = summary_df.copy()
    df["comparison_label"] = df["dataset"].astype(str) + "\n" + df["endpoint"].astype(str)

    # Sort: put tumor_vs_normal first if available, then by dataset
    def sort_key(row):
        ep = str(row["endpoint"]).lower()
        if "tumor" in ep and "normal" in ep:
            return 0
        if "stage" in ep:
            return 1
        if "node" in ep:
            return 2
        return 3

    df["_sort"] = df.apply(sort_key, axis=1)
    df = df.sort_values(["_sort", "dataset", "endpoint"]).reset_index(drop=True)

    # 1. AUC comparison
    plt.figure(figsize=(max(8, len(df) * 1.8), 5))
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, pd.to_numeric(df["DClinical_auc"], errors="coerce"), width, label="D-Clinical full BP")
    plt.bar(x + width/2, pd.to_numeric(df["DPHY_auc"], errors="coerce"), width, label="D-PHY top BP")
    plt.xticks(x, df["comparison_label"], rotation=45, ha="right")
    plt.ylabel("Cross-validated AUC")
    plt.title("D-Clinical vs D-PHY clinical/endpoint observability")
    plt.ylim(0.45, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "FIG01_DClinical_vs_DPHY_AUC.png", dpi=300)
    plt.close()

    # 2. Delta AUC
    plt.figure(figsize=(max(8, len(df) * 1.8), 5))
    plt.bar(x, pd.to_numeric(df["DPHY_minus_DClinical_auc"], errors="coerce"))
    plt.axhline(0, linestyle="--")
    plt.xticks(x, df["comparison_label"], rotation=45, ha="right")
    plt.ylabel("AUC difference: D-PHY top BP - D-Clinical full BP")
    plt.title("Compact D-PHY representation gain/loss over full BP space")
    plt.tight_layout()
    plt.savefig(out_dir / "FIG02_DPHY_minus_DClinical_AUC.png", dpi=300)
    plt.close()

    # 3. Significant BP count
    plt.figure(figsize=(max(8, len(df) * 1.8), 5))
    plt.bar(x, pd.to_numeric(df["n_D_significant_FDR05"], errors="coerce"))
    plt.xticks(x, df["comparison_label"], rotation=45, ha="right")
    plt.ylabel("Number of significant BP terms, FDR < 0.05")
    plt.title("D-layer significant biological-process signals")
    plt.tight_layout()
    plt.savefig(out_dir / "FIG03_significant_BP_count.png", dpi=300)
    plt.close()

    # 4. Interpretation-ready count
    plt.figure(figsize=(max(8, len(df) * 1.8), 5))
    plt.bar(x, pd.to_numeric(df["readiness_strong_moderate_total"], errors="coerce"))
    plt.xticks(x, df["comparison_label"], rotation=45, ha="right")
    plt.ylabel("Strong + moderate interpretation-ready BP count")
    plt.title("Integrated interpretation-readiness output")
    plt.tight_layout()
    plt.savefig(out_dir / "FIG04_interpretation_ready_count.png", dpi=300)
    plt.close()


def write_interpretation_notes(summary_df, out_dir):
    """
    Create a manuscript-oriented interpretation note.
    """
    df = summary_df.copy()

    lines = []
    lines.append("D-PHY vs D-Clinical comparison notes")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Definitions")
    lines.append("- D-Clinical: endpoint observability using the full BP-space representation.")
    lines.append("- D-PHY: endpoint observability using the compact top-BP representation selected by the D layer.")
    lines.append("- Positive D-PHY minus D-Clinical AUC means the compact D-selected BP space outperformed the full BP space.")
    lines.append("- Negative or near-zero difference means the endpoint signal is distributed across BP space or the D-selected top BP set does not improve compact discrimination.")
    lines.append("")

    for _, r in df.iterrows():
        label = f"{r.get('dataset', '')} | {r.get('endpoint', '')}"
        lines.append(label)
        lines.append("-" * len(label))
        lines.append(f"Samples: {r.get('n_samples', '')}; early-like={r.get('n_early_like', '')}; late-like={r.get('n_late_like', '')}")
        lines.append(f"D-Clinical AUC: {r.get('DClinical_auc', np.nan)}")
        lines.append(f"D-PHY top-BP AUC: {r.get('DPHY_auc', np.nan)}")
        lines.append(f"D-PHY minus D-Clinical AUC: {r.get('DPHY_minus_DClinical_auc', np.nan)}")
        lines.append(f"Significant BP terms, FDR<0.05: {r.get('n_D_significant_FDR05', np.nan)}")
        lines.append(f"Strong+moderate interpretation-ready BP count: {r.get('readiness_strong_moderate_total', np.nan)}")
        lines.append(f"Observability class: {r.get('observability_class', '')}")
        lines.append(f"Top BP: {r.get('top1_BP', '')}")
        lines.append("")

    lines.append("Suggested manuscript-level punchline")
    lines.append("-" * 80)
    lines.append(
        "Biological-process observability is target-dependent. "
        "Strong biological contrasts such as tumor-versus-normal can yield highly discriminative, "
        "statistically reliable, biologically anchored, and interpretation-ready BP observables. "
        "In contrast, clinical progression-related endpoints such as stage or lymph-node status "
        "may show only partial or distributed BP-level observability, even when many BP terms are statistically significant. "
        "Therefore, BP-level interpretation should not be based on D scores alone, nor on biological anchoring alone, "
        "but should be assessed through an integrated interpretation-readiness profile."
    )

    with open(out_dir / "INTERPRETATION_NOTES.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 4. MAIN
# ============================================================

def main():
    ensure_dir(OUTPUT_ROOT)

    log("=" * 80)
    log("D-PHY vs D-Clinical comparison V1")
    log(f"Scanning: {AIDO_TEMP_ROOT}")
    log(f"Output: {OUTPUT_ROOT}")
    log("=" * 80)

    run_roots, result_folders = find_all_result_folders()

    pd.DataFrame({"run_root": [str(x) for x in run_roots]}).to_csv(
        OUTPUT_ROOT / "00_detected_run_roots.csv",
        index=False
    )

    pd.DataFrame({"result_folder": [str(x) for x in result_folders]}).to_csv(
        OUTPUT_ROOT / "00_detected_result_folders.csv",
        index=False
    )

    if not result_folders:
        raise ValueError("No result folders found. Check D:/AIDO-Temp and run folder patterns.")

    rows = []
    failures = []

    for folder in result_folders:
        try:
            rows.append(extract_one_result_folder(folder))
        except Exception as e:
            failures.append({
                "result_folder": str(folder),
                "error": str(e)
            })

    summary_df = pd.DataFrame(rows)
    failure_df = pd.DataFrame(failures)

    if len(summary_df) == 0:
        failure_df.to_csv(OUTPUT_ROOT / "EXTRACTION_FAILURES.csv", index=False)
        raise ValueError("No result folders could be extracted.")

    summary_df["observability_class"] = summary_df.apply(classify_observability, axis=1)

    # Deduplicate: keep newest / most informative for same dataset+endpoint.
    # If multiple old runs exist, keep the row with largest n_samples then newest folder string.
    summary_df["n_samples_numeric"] = pd.to_numeric(summary_df["n_samples"], errors="coerce")
    summary_df = summary_df.sort_values(
        ["dataset", "endpoint", "n_samples_numeric", "result_folder"],
        ascending=[True, True, False, False]
    )

    dedup_df = summary_df.drop_duplicates(["dataset", "endpoint"], keep="first").copy()

    # Sort for readability.
    def endpoint_order(ep):
        ep = str(ep).lower()
        if "tumor" in ep and "normal" in ep:
            return 0
        if "stage" in ep:
            return 1
        if "node" in ep:
            return 2
        return 3

    dedup_df["_endpoint_order"] = dedup_df["endpoint"].map(endpoint_order)
    dedup_df = dedup_df.sort_values(["_endpoint_order", "dataset", "endpoint"]).drop(columns=["_endpoint_order"])

    # Main compact columns
    compact_cols = [
        "dataset",
        "endpoint",
        "target_mode",
        "endpoint_type",
        "endpoint_column",
        "alignment_column",
        "n_samples",
        "n_early_like",
        "n_late_like",
        "n_genes",
        "n_bp_terms",
        "DClinical_auc",
        "DPHY_auc",
        "DPHY_minus_DClinical_auc",
        "DClinical_balanced_accuracy",
        "DPHY_balanced_accuracy",
        "DPHY_minus_DClinical_balanced_accuracy",
        "n_D_significant_FDR05",
        "readiness_strong_moderate_total",
        "n_interpretation_ready_strong_from_profile",
        "n_interpretation_ready_moderate_from_profile",
        "n_statistically_supported_but_weakly_anchored",
        "n_biologically_anchored_but_endpoint_weak",
        "n_exploratory_or_weak",
        "top1_BP",
        "top1_direction",
        "top1_abs_cohen_d",
        "top1_auc",
        "top1_fdr",
        "top50_median_abs_cohen_d",
        "top50_median_fdr",
        "top50_late_up_n",
        "top50_early_up_n",
        "observability_class",
        "result_folder",
    ]

    compact_cols = [c for c in compact_cols if c in dedup_df.columns]

    compact_df = dedup_df[compact_cols].copy()

    # Save tables
    summary_df.to_csv(OUTPUT_ROOT / "01_all_extracted_DPHY_DClinical_rows.csv", index=False)
    dedup_df.to_csv(OUTPUT_ROOT / "02_deduplicated_full_summary.csv", index=False)
    compact_df.to_csv(OUTPUT_ROOT / "03_DPHY_vs_DClinical_compact_comparison.csv", index=False)
    failure_df.to_csv(OUTPUT_ROOT / "EXTRACTION_FAILURES.csv", index=False)

    # Excel workbook
    xlsx_path = OUTPUT_ROOT / "D_PHY_vs_DClinical_Comparison.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        compact_df.to_excel(writer, sheet_name="compact_comparison", index=False)
        dedup_df.to_excel(writer, sheet_name="deduplicated_full", index=False)
        summary_df.to_excel(writer, sheet_name="all_extracted_rows", index=False)
        failure_df.to_excel(writer, sheet_name="extraction_failures", index=False)

    # Figures and notes
    make_comparison_figures(compact_df, OUTPUT_ROOT)
    write_interpretation_notes(compact_df, OUTPUT_ROOT)

    log("=" * 80)
    log("COMPARISON COMPLETE")
    log(f"Detected result folders: {len(result_folders)}")
    log(f"Extracted rows: {len(summary_df)}")
    log(f"Deduplicated comparisons: {len(compact_df)}")
    log(f"Output: {OUTPUT_ROOT}")
    log("=" * 80)


if __name__ == "__main__":
    main()
