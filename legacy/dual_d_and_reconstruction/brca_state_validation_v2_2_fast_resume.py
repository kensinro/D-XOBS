# -*- coding: utf-8 -*-
"""
AIDO BRCA D-PHY/D-Clinical StateValidation V2.2 FAST FINISH / RESUME
====================================================================

Purpose
-------
This script is for finishing a V2.1 run that already produced files up to
16_bootstrap_state_stability.csv but got stuck at the slow leave-one-module step.

It DOES NOT rerun:
    GE loading
    BP activity construction
    D-PHY
    D-Clinical
    Dual-D
    BP-Enrichment
    random baseline
    label shuffle
    patient scramble
    bootstrap

It only reads existing V2.1 outputs and finishes:

    17_leave_one_module_sensitivity_FAST.csv
    18_module_oncogene_TSG_mechanism_audit.csv
    19_interpretation_readiness_audit.csv
    19c_old_IMU_vs_current_BRCA_V2_2_comparison.json
    20_endpoint_run_summary.json
    21_AIDO_DPHY_DClinical_StateValidation_summary.xlsx
    figures/summary figures
    ZIP package

Important change from V2.1
--------------------------
1. Skip or cap leave-one-module sensitivity.
   The old full leave-one-module step was slow because it reran CV/state calculations
   for every module. With 262 modules it can hang for a long time.

2. Patient scramble is NOT used as a main negative-control claim here.
   The V2.1 patient-scramble implementation can create artificial centroid separation.
   We retain the file if present but mark it as diagnostic/unreliable.

3. Random module and label shuffle remain the main controls.

How to use
----------
Set EXISTING_OUT_DIR to your completed/partial V2.1 output folder, then run.

Default:
    EXISTING_OUT_DIR =
    r"D:/AIDO-Temp/AIDO_DPHY_DClinical_StateValidation_BRCA_Stage_V2_1_COMPLETE"

If you extracted the zip somewhere else, point EXISTING_OUT_DIR to that folder.
"""

# =============================================================================
# CONFIG
# =============================================================================

EXISTING_OUT_DIR = r"D:/AIDO-Temp/AIDO_DPHY_DClinical_StateValidation_BRCA_Stage_V2_1_COMPLETE"

# Fast leave-one-module settings
RUN_FAST_LEAVE_ONE = True
MAX_LEAVE_ONE_MODULES = 40       # rank-top modules only; avoids 262 × expensive recalculation
LEAVE_ONE_SELECTION = "top_readiness_proxy"  # currently uses module/state ranking proxy

# Whether to create figures and Excel
MAKE_FIGURES = True
MAKE_EXCEL = True
MAKE_ZIP = True

# Old IMU reference, for comparison only
OLD_IMU_REFERENCE = {
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
}


# =============================================================================
# Imports
# =============================================================================

import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# =============================================================================
# Utilities
# =============================================================================

def safe_read_csv(path, **kwargs):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] Missing file: {path.name}")
        return pd.DataFrame()
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip", low_memory=False, **kwargs)
    return pd.read_csv(path, low_memory=False, **kwargs)


def cohen_d(x0, x1):
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x0 = x0[np.isfinite(x0)]
    x1 = x1[np.isfinite(x1)]
    if len(x0) < 2 or len(x1) < 2:
        return np.nan
    s0 = np.var(x0, ddof=1)
    s1 = np.var(x1, ddof=1)
    sp = math.sqrt(((len(x0)-1)*s0 + (len(x1)-1)*s1) / max(len(x0)+len(x1)-2, 1))
    if sp == 0:
        return np.nan
    return (np.mean(x1) - np.mean(x0)) / sp


def empirical_p_value(observed, null_values, direction="greater"):
    vals = pd.to_numeric(pd.Series(null_values), errors="coerce").dropna().values
    if len(vals) == 0 or not np.isfinite(observed):
        return np.nan
    if direction == "greater":
        return float((1 + np.sum(vals >= observed)) / (len(vals) + 1))
    if direction == "less":
        return float((1 + np.sum(vals <= observed)) / (len(vals) + 1))
    return np.nan


def q_rank_score(values, larger_better=True):
    x = pd.to_numeric(values, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(0.0, index=values.index)
    return x.rank(pct=True, ascending=larger_better).fillna(0.0)


def zip_dir(src_dir, zip_path):
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in src_dir.rglob("*"):
            if f.is_file() and f.resolve() != zip_path.resolve():
                zf.write(f, f.relative_to(src_dir))
    return zip_path


def summarize_profile(profile):
    if profile.empty:
        return {}

    metric_candidates = [
        "positive_minus_negative_centroid_similarity",
        "endpoint_positive_centroid_similarity",
    ]
    metric = next((m for m in metric_candidates if m in profile.columns), None)
    if metric is None or "label" not in profile.columns:
        return {}

    y = profile["label"].astype(int)
    x0 = pd.to_numeric(profile.loc[y == 0, metric], errors="coerce")
    x1 = pd.to_numeric(profile.loc[y == 1, metric], errors="coerce")

    try:
        p = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit").pvalue
    except Exception:
        p = np.nan

    return {
        "primary_metric": metric,
        "primary_centroid_similarity_d": float(cohen_d(x0, x1)),
        "primary_centroid_similarity_p": float(p) if pd.notna(p) else np.nan,
        "endpoint_negative_mean": float(np.nanmean(x0)),
        "endpoint_positive_mean": float(np.nanmean(x1)),
        "endpoint_positive_minus_negative_mean": float(np.nanmean(x1) - np.nanmean(x0)),
    }


def collect_genes(series):
    genes = set()
    if series is None:
        return genes
    for val in series.dropna().astype(str):
        for g in val.split(";"):
            g = g.strip().upper()
            if g:
                genes.add(g)
    return genes


# =============================================================================
# Mechanism audit and readiness
# =============================================================================

def build_module_mechanism_audit(module_members, module_summary):
    if module_members.empty:
        return pd.DataFrame()

    rows = []
    for mid, sub in module_members.groupby("BP_module_id"):
        oncogenes = collect_genes(sub["oncogene_overlap_genes"]) if "oncogene_overlap_genes" in sub.columns else set()
        tsg = collect_genes(sub["tumor_suppressor_overlap_genes"]) if "tumor_suppressor_overlap_genes" in sub.columns else set()
        cancer_genes = collect_genes(sub["cancer_gene_overlap_genes"]) if "cancer_gene_overlap_genes" in sub.columns else set()

        if "BP_state_direction_class" in sub.columns:
            direction_counts = sub["BP_state_direction_class"].value_counts().to_dict()
            dominant_direction = max(direction_counts.items(), key=lambda kv: kv[1])[0] if direction_counts else "unknown"
        else:
            dominant_direction = "unknown"

        if "positive" in str(dominant_direction):
            mechanism_hint = "endpoint-positive-state associated BP program"
        elif "negative" in str(dominant_direction):
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

    if not module_summary.empty:
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


def build_readiness(module_summary, state_disc, mechanism_audit):
    if module_summary.empty:
        return pd.DataFrame()

    out = module_summary.copy()

    if not state_disc.empty:
        cols = [
            "BP_module_id", "module_abs_cohen_d", "module_auc_positive_vs_negative",
            "module_auc_distance", "module_welch_fdr", "module_state_information_score",
            "module_direction"
        ]
        cols = [c for c in cols if c in state_disc.columns]
        out = out.merge(state_disc[cols], on="BP_module_id", how="left")

    if not mechanism_audit.empty:
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

    def cls(row):
        if row.get("readiness_score", 0) >= 0.75 and row.get("module_state_information_score", 0) >= 0.60 and row.get("mechanism_anchor_score", 0) >= 0.50:
            return "interpretation_ready_high"
        if row.get("readiness_score", 0) >= 0.55 and row.get("module_state_information_score", 0) >= 0.40:
            return "interpretation_ready_moderate"
        if row.get("module_state_information_score", 0) >= 0.45:
            return "state_informative_but_mechanism_weak"
        return "exploratory_or_weak"

    out["interpretation_readiness_class"] = out.apply(cls, axis=1)
    return out.sort_values("readiness_score", ascending=False).reset_index(drop=True)


def fast_leave_one(module_activity, labels, module_summary, max_modules=40):
    """
    Fast leave-one: no CV rerun.
    For top modules only, recompute centroid-similarity D after removing one module.
    This is enough to check whether patient-level state alignment is dominated by one module.
    """
    if module_activity.empty or labels.empty or module_activity.shape[1] <= 1:
        return pd.DataFrame()

    if module_summary.empty:
        modules = list(module_activity.columns[:max_modules])
    else:
        rank_cols = [c for c in ["mean_DualD_priority_score", "mean_h_score", "BP_module_size"] if c in module_summary.columns]
        ms = module_summary.copy()
        if rank_cols:
            ms["_proxy_rank"] = 0.0
            for c in rank_cols:
                ms["_proxy_rank"] += q_rank_score(ms[c], True)
            ms = ms.sort_values("_proxy_rank", ascending=False)
        modules = [m for m in ms["BP_module_id"].astype(str).tolist() if m in module_activity.columns][:max_modules]

    y = labels.set_index("sample_id")["label"].astype(int)
    y = y.loc[module_activity.index]

    rows = []
    for mid in modules:
        X = module_activity.drop(columns=[mid]).copy()
        X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median(axis=0))

        idx0 = y[y == 0].index
        idx1 = y[y == 1].index
        c0 = X.loc[idx0].mean(axis=0).values
        c1 = X.loc[idx1].mean(axis=0).values

        arr = X.values.astype(float)
        def cos_to(c):
            num = np.nansum(arr * c, axis=1)
            den = np.sqrt(np.nansum(arr ** 2, axis=1)) * np.sqrt(np.nansum(c ** 2))
            return pd.Series(np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den != 0), index=X.index)

        sim = cos_to(c1) - cos_to(c0)
        d = cohen_d(sim.loc[idx0].values, sim.loc[idx1].values)
        try:
            p = stats.ttest_ind(sim.loc[idx1], sim.loc[idx0], equal_var=False, nan_policy="omit").pvalue
        except Exception:
            p = np.nan

        row = {
            "left_out_module_id": mid,
            "remaining_module_count": X.shape[1],
            "primary_centroid_similarity_d": d,
            "primary_centroid_similarity_p": p,
            "endpoint_negative_mean": float(sim.loc[idx0].mean()),
            "endpoint_positive_mean": float(sim.loc[idx1].mean()),
        }
        if not module_summary.empty and mid in set(module_summary["BP_module_id"]):
            sub = module_summary[module_summary["BP_module_id"] == mid].iloc[0]
            row["left_out_module_name"] = sub.get("BP_module_name", "")
            row["left_out_module_size"] = sub.get("BP_module_size", np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def make_figures(out_dir, profile, random_df, shuffle_df, bootstrap_df, readiness):
    if not HAS_MPL or not MAKE_FIGURES:
        return

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)

    if not profile.empty and "positive_minus_negative_centroid_similarity" in profile.columns:
        y = profile["label"].astype(int)
        vals0 = profile.loc[y == 0, "positive_minus_negative_centroid_similarity"]
        vals1 = profile.loc[y == 1, "positive_minus_negative_centroid_similarity"]
        plt.figure(figsize=(6, 5))
        plt.boxplot([vals0.dropna(), vals1.dropna()], labels=["Endpoint negative", "Endpoint positive"], showfliers=False)
        plt.ylabel("Positive-minus-negative centroid similarity")
        plt.title("Patient-level reconstructed endpoint-state alignment")
        plt.tight_layout()
        plt.savefig(fig_dir / "FIG_finish_patient_centroid_similarity.png", dpi=300)
        plt.close()

    observed = summarize_profile(profile).get("primary_centroid_similarity_d", np.nan)

    for df, fname, title in [
        (random_df, "FIG_finish_random_baseline.png", "Random BP-module baseline"),
        (shuffle_df, "FIG_finish_label_shuffle.png", "Endpoint-label shuffle control"),
        (bootstrap_df, "FIG_finish_bootstrap.png", "Bootstrap stability"),
    ]:
        if df.empty or "primary_centroid_similarity_d" not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(pd.to_numeric(df["primary_centroid_similarity_d"], errors="coerce").dropna(), bins=30)
        if np.isfinite(observed):
            plt.axvline(observed, linestyle="--")
        plt.xlabel("Centroid similarity Cohen d")
        plt.ylabel("Count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=300)
        plt.close()

    if not readiness.empty and "readiness_score" in readiness.columns:
        top = readiness.head(20)
        labels = top["BP_module_name"].fillna(top["BP_module_id"]).astype(str)
        plt.figure(figsize=(10, max(5, 0.35 * len(top))))
        yy = np.arange(len(top))
        plt.barh(yy, top["readiness_score"])
        plt.yticks(yy, labels)
        plt.xlabel("Readiness score")
        plt.title("Top interpretation-ready modules")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(fig_dir / "FIG_finish_top_readiness_modules.png", dpi=300)
        plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    out_dir = Path(EXISTING_OUT_DIR)
    if not out_dir.exists():
        raise FileNotFoundError(f"EXISTING_OUT_DIR not found: {out_dir}")

    print("[INFO] Fast finishing existing folder:", out_dir)

    labels = safe_read_csv(out_dir / "01_endpoint_labels.csv")
    dphy = safe_read_csv(out_dir / "03_DPHY_ranked_BP_signals.csv")
    dclinical = safe_read_csv(out_dir / "04_DClinical_BP_coefficients.csv")
    dual = safe_read_csv(out_dir / "06_DualD_BP_candidates.csv")
    module_members = safe_read_csv(out_dir / "07_BPenrichment_module_members.csv")
    module_summary = safe_read_csv(out_dir / "08_BPenrichment_module_summary.csv")
    module_activity = safe_read_csv(out_dir / "09_reconstructed_state_module_activity.csv.gz")
    state_disc = safe_read_csv(out_dir / "10_reconstructed_state_discriminability.csv")
    profile = safe_read_csv(out_dir / "11_patient_level_state_profile.csv")
    centroid_metrics = safe_read_csv(out_dir / "12_centroid_state_metrics.csv")
    random_df = safe_read_csv(out_dir / "13_random_BP_module_baseline.csv")
    shuffle_df = safe_read_csv(out_dir / "14_endpoint_label_shuffle_control.csv")
    scramble_df = safe_read_csv(out_dir / "15_patient_scramble_control.csv")
    bootstrap_df = safe_read_csv(out_dir / "16_bootstrap_state_stability.csv")

    if "sample_id" in module_activity.columns:
        module_activity = module_activity.set_index("sample_id")
    elif module_activity.columns[0].startswith("Unnamed"):
        module_activity = module_activity.set_index(module_activity.columns[0])
        module_activity.index.name = "sample_id"

    if "sample_id" in labels.columns:
        labels["sample_id"] = labels["sample_id"].astype(str)
    if not module_activity.empty:
        module_activity.index = module_activity.index.astype(str)

    # Finish leave-one fast.
    if RUN_FAST_LEAVE_ONE:
        print("[INFO] Running FAST leave-one-module sensitivity...")
        leave_one = fast_leave_one(module_activity, labels, module_summary, MAX_LEAVE_ONE_MODULES)
    else:
        leave_one = pd.DataFrame()
    leave_one.to_csv(out_dir / "17_leave_one_module_sensitivity_FAST.csv", index=False)

    # Mechanism + readiness.
    print("[INFO] Building oncogene/TSG mechanism audit...")
    mechanism = build_module_mechanism_audit(module_members, module_summary)
    mechanism.to_csv(out_dir / "18_module_oncogene_TSG_mechanism_audit.csv", index=False)

    print("[INFO] Building interpretation readiness audit...")
    readiness = build_readiness(module_summary, state_disc, mechanism)
    readiness.to_csv(out_dir / "19_interpretation_readiness_audit.csv", index=False)

    observed = summarize_profile(profile)

    control_summary = {
        "observed": observed,
        "random_module_empirical_p_primary_d": empirical_p_value(
            observed.get("primary_centroid_similarity_d", np.nan),
            random_df["primary_centroid_similarity_d"] if "primary_centroid_similarity_d" in random_df.columns else [],
            direction="greater",
        ),
        "label_shuffle_empirical_p_primary_d": empirical_p_value(
            observed.get("primary_centroid_similarity_d", np.nan),
            shuffle_df["primary_centroid_similarity_d"] if "primary_centroid_similarity_d" in shuffle_df.columns else [],
            direction="greater",
        ),
        "patient_scramble_note": (
            "V2.1 patient-scramble control is retained only as diagnostic/unreliable. "
            "It should not be used as a main negative-control claim because independent column-wise scrambling "
            "can inflate artificial centroid separation in high-dimensional module space."
        ),
        "bootstrap_primary_d_median": float(pd.to_numeric(bootstrap_df.get("primary_centroid_similarity_d", pd.Series(dtype=float)), errors="coerce").median()) if not bootstrap_df.empty else np.nan,
        "bootstrap_primary_d_p05": float(pd.to_numeric(bootstrap_df.get("primary_centroid_similarity_d", pd.Series(dtype=float)), errors="coerce").quantile(0.05)) if not bootstrap_df.empty else np.nan,
        "bootstrap_primary_d_p95": float(pd.to_numeric(bootstrap_df.get("primary_centroid_similarity_d", pd.Series(dtype=float)), errors="coerce").quantile(0.95)) if not bootstrap_df.empty else np.nan,
        "fast_leave_one_module_count": int(leave_one.shape[0]),
        "fast_leave_one_primary_d_min": float(pd.to_numeric(leave_one.get("primary_centroid_similarity_d", pd.Series(dtype=float)), errors="coerce").min()) if not leave_one.empty else np.nan,
        "fast_leave_one_primary_d_median": float(pd.to_numeric(leave_one.get("primary_centroid_similarity_d", pd.Series(dtype=float)), errors="coerce").median()) if not leave_one.empty else np.nan,
    }

    comparison = {
        "old_IMU_line": OLD_IMU_REFERENCE,
        "current_BRCA_V2_2_finished_line": {
            "selection_logic": "D-PHY + D-Clinical + Dual-D evidence stratification",
            "D_PHY_selected_BP": int(dual["selected_by_DPHY"].sum()) if "selected_by_DPHY" in dual.columns else None,
            "D_Clinical_selected_BP": int(dual["selected_by_DClinical"].sum()) if "selected_by_DClinical" in dual.columns else None,
            "DualD_core_BP": int(dual["selected_by_dualD_intersection"].sum()) if "selected_by_dualD_intersection" in dual.columns else None,
            "BP_modules_components": int(module_summary.shape[0]),
            "does_include_DPHY": True,
            "does_include_DClinical": True,
            "does_include_oncogene_TSG_audit": True,
            "validation_controls_finished": [
                "random BP-module baseline",
                "endpoint-label shuffling",
                "bootstrap stability",
                "fast leave-one-module sensitivity",
                "oncogene/TSG mechanism audit",
                "interpretation-readiness audit"
            ],
            "patient_scramble_status": "diagnostic/unreliable, not used for main claim",
        },
        "intended_comparison": (
            "The old IMU line demonstrates BRCA patient-level BP-state reconstruction feasibility. "
            "The current V2.2 finished BRCA line tests whether D-PHY/D-Clinical audited BP selection "
            "supports BP-Enrichment state reconstruction plus mechanism-readiness audit."
        )
    }
    (out_dir / "19c_old_IMU_vs_current_BRCA_V2_2_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    readiness_counts = readiness["interpretation_readiness_class"].value_counts().to_dict() if "interpretation_readiness_class" in readiness.columns else {}
    label_counts = labels["label"].value_counts().to_dict() if "label" in labels.columns else {}

    summary = {
        "run_status": "finished_by_v2_2_fast_resume",
        "input_folder": str(out_dir),
        "n_endpoint_samples": int(labels.shape[0]),
        "endpoint_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "n_DPHY_rows": int(dphy.shape[0]),
        "n_DClinical_rows": int(dclinical.shape[0]),
        "n_DualD_rows": int(dual.shape[0]),
        "n_BP_modules": int(module_summary.shape[0]),
        "n_module_members": int(module_members.shape[0]),
        "n_state_disc_modules": int(state_disc.shape[0]),
        "n_mechanism_audited_modules": int(mechanism.shape[0]),
        "interpretation_readiness_counts": {str(k): int(v) for k, v in readiness_counts.items()},
        "control_summary": control_summary,
        "old_IMU_vs_current_BRCA_V2_2_comparison": comparison,
        "main_interpretation_note": (
            "This fast-finished run avoids the slow full leave-one-module CV loop. "
            "Use random module baseline, label shuffle, bootstrap, and fast leave-one as the main validation outputs. "
            "Do not use the V2.1 patient-scramble file as a main negative-control claim."
        ),
    }
    (out_dir / "20_endpoint_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Figures
    make_figures(out_dir, profile, random_df, shuffle_df, bootstrap_df, readiness)

    # Excel
    if MAKE_EXCEL:
        print("[INFO] Writing Excel summary...")
        try:
            with pd.ExcelWriter(out_dir / "21_AIDO_DPHY_DClinical_StateValidation_summary.xlsx", engine="openpyxl") as writer:
                pd.DataFrame([summary]).to_excel(writer, sheet_name="run_summary", index=False)
                labels.to_excel(writer, sheet_name="endpoint_labels", index=False)
                dphy.head(500).to_excel(writer, sheet_name="DPHY_top500", index=False)
                dclinical.head(500).to_excel(writer, sheet_name="DClinical_top500", index=False)
                dual.to_excel(writer, sheet_name="DualD_candidates", index=False)
                module_summary.to_excel(writer, sheet_name="BP_modules", index=False)
                module_members.to_excel(writer, sheet_name="module_members", index=False)
                state_disc.to_excel(writer, sheet_name="state_discriminability", index=False)
                profile.to_excel(writer, sheet_name="patient_state_profile", index=False)
                centroid_metrics.to_excel(writer, sheet_name="centroid_metrics", index=False)
                random_df.to_excel(writer, sheet_name="random_baseline", index=False)
                shuffle_df.to_excel(writer, sheet_name="label_shuffle", index=False)
                bootstrap_df.to_excel(writer, sheet_name="bootstrap", index=False)
                leave_one.to_excel(writer, sheet_name="fast_leave_one", index=False)
                mechanism.to_excel(writer, sheet_name="oncogene_TSG_audit", index=False)
                readiness.to_excel(writer, sheet_name="readiness_audit", index=False)
                pd.DataFrame([comparison["current_BRCA_V2_2_finished_line"]]).to_excel(writer, sheet_name="old_vs_current", index=False)
        except Exception as e:
            print("[WARN] Excel export failed:", e)

    readme = f"""AIDO BRCA D-PHY/D-Clinical StateValidation V2.2 FAST FINISH
================================================================

This folder was finished by V2.2 fast-resume because V2.1 stalled after bootstrap.

Main fixes
----------
1. Full leave-one-module CV loop was replaced by fast leave-one centroid-sensitivity.
2. V2.1 patient-scramble control is marked diagnostic/unreliable and should not be used as a main claim.
3. Mechanism audit, readiness audit, old-vs-current comparison, summary, Excel, and figures are generated.

Main validation controls to use
-------------------------------
- random BP-module baseline
- endpoint-label shuffle
- bootstrap stability
- fast leave-one-module sensitivity
- oncogene/TSG mechanism audit
- interpretation-readiness audit

Control summary
---------------
{json.dumps(control_summary, indent=2)}

Readiness counts
----------------
{json.dumps({str(k): int(v) for k, v in readiness_counts.items()}, indent=2)}
"""
    (out_dir / "README_V2_2_FAST_FINISH.txt").write_text(readme, encoding="utf-8")

    if MAKE_ZIP:
        zip_path = out_dir.parent / f"{out_dir.name}_V2_2_FAST_FINISHED.zip"
        zip_dir(out_dir, zip_path)
        print("[DONE] ZIP:", zip_path)

    print("[DONE] V2.2 fast finish completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
