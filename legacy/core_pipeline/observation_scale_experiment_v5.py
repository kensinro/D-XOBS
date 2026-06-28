
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
D-PHY observation-scale sensitivity experiment V5
==================================================

This version uses the exact biological-process observable universe from the
original AIDO-D-PHY-I run:

    5,065 retained GO Biological Process observables
    1,073 TCGA-BRCA samples
    803 early-stage samples
    270 late-stage samples

It does NOT rebuild BP scores from the current GMT file. Instead, it loads the
exact BP activity matrix and matched endpoint file from either:

1. an extracted AIDO-D-PHY-I folder, or
2. AIDO-D-PHY-I.zip.

This guarantees that Supplementary Table S9 and Figure S5 use the same
observable universe as the manuscript.

Primary analysis
----------------
Leakage-controlled repeated stratified cross-validation:

1. Within each training fold, rank the 5,065 BP observables by Mann-Whitney
   discriminability.
2. Select the top K observables.
3. Fit a class-balanced logistic-regression model.
4. Evaluate the held-out fold.
5. Repeat over K = 1, 3, 5, 10, 15, 20, 30, 40, 50.

Secondary descriptive analysis
------------------------------
For each K, calculate a signed aggregate score from the full-data BP ranking:

    S_p^(K) = mean_j [ direction_j * z(BP_j,p) ]

and report AUC, nominal P, D = -log10(P), and BH-FDR q.

Outputs
-------
D:/AIDO-Temp/D_PHY_ObservationScale_V5_<timestamp>/

Main files:
- Table_S9_observation_scale_results.csv
- Table_S9_observation_scale_results.xlsx
- Figure_S5_observation_scale.png
- Figure_S5_observation_scale.pdf
- nested_cv_fold_results.csv
- nested_cv_repeat_results.csv
- full_data_bp_ranking.csv
- full_data_aggregate_results.csv
- selection_stability_by_K.csv
- source_alignment_report.txt
- run_summary.txt
"""


import io
import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

OUTPUT_ROOT = Path(r"D:\AIDO-Temp")

# Set these manually only when auto-detection cannot find the files.
# Example extracted folder:
# AIDO_DPHY_FOLDER = Path(r"D:\AIDO-Temp\AIDO-D-PHY-I")
AIDO_DPHY_FOLDER = None

# Example ZIP:
# AIDO_DPHY_ZIP = Path(r"D:\AIDO-Temp\AIDO-D-PHY-I.zip")
AIDO_DPHY_ZIP = None

# Auto-search roots. The script searches these recursively for AIDO-D-PHY-I.zip
# or the extracted AIDO-D-PHY-I folder.
AUTO_SEARCH_ROOTS = [
    Path.cwd(),
    Path(r"D:\AIDO-Temp"),
    Path(r"D:\AIDO-Data"),
    Path.home() / "Downloads",
    Path.home() / "Desktop",
]

BP_MATRIX_RELATIVE = Path("02_bp_scores") / "BP_activity_samples_by_bp.tsv"
ENDPOINT_RELATIVE = Path("01_preprocessed") / "endpoint_matched.tsv"
DISCRIMINABILITY_RELATIVE = Path("04_discriminability") / "BP_discriminability_table.tsv"


# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

K_VALUES = [1, 3, 5, 10, 15, 20, 30, 40, 50]

N_SPLITS = 5
N_REPEATS = 20
RANDOM_STATE = 20260618
MAX_ITER = 5000

# Permutation is disabled for the main alignment run.
# It can be enabled later after Table S9 is confirmed.
N_PERMUTATIONS = 0
PERMUTATION_CV_SPLITS = 5

EXPECTED_TOTAL = 1073
EXPECTED_EARLY = 803
EXPECTED_LATE = 270
EXPECTED_BP_COUNT = 5065

EPS = 1e-300
TCGA_PATIENT_ID_LENGTH = 12


@dataclass
class RunConfig:
    output_root: str
    k_values: List[int]
    n_splits: int
    n_repeats: int
    random_state: int
    n_permutations: int
    expected_total: int
    expected_early: int
    expected_late: int
    expected_bp_count: int
    source_folder: str | None
    source_zip: str | None


# =============================================================================
# UTILITIES
# =============================================================================

def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def normalize_sample_id(value: object) -> str:
    s = str(value).strip().replace(".", "-").upper()
    if s.startswith("TCGA-") and len(s) >= TCGA_PATIENT_ID_LENGTH:
        return s[:TCGA_PATIENT_ID_LENGTH]
    return s


def orient_auc(y: np.ndarray, score: np.ndarray) -> float:
    auc = roc_auc_score(y, score)
    return float(max(auc, 1.0 - auc))


def find_source_folder() -> Path | None:
    if AIDO_DPHY_FOLDER is not None:
        folder = Path(AIDO_DPHY_FOLDER)
        if folder.exists():
            return folder
        raise FileNotFoundError(f"Configured AIDO_DPHY_FOLDER does not exist: {folder}")

    for root in AUTO_SEARCH_ROOTS:
        try:
            direct = root / "AIDO-D-PHY-I"
            if (direct / BP_MATRIX_RELATIVE).exists():
                return direct

            if root.exists():
                matches = list(root.glob("**/AIDO-D-PHY-I/02_bp_scores/BP_activity_samples_by_bp.tsv"))
                if matches:
                    return matches[0].parents[2]
        except (PermissionError, OSError):
            continue

    return None


def find_source_zip() -> Path | None:
    if AIDO_DPHY_ZIP is not None:
        path = Path(AIDO_DPHY_ZIP)
        if path.exists():
            return path
        raise FileNotFoundError(f"Configured AIDO_DPHY_ZIP does not exist: {path}")

    for root in AUTO_SEARCH_ROOTS:
        try:
            direct = root / "AIDO-D-PHY-I.zip"
            if direct.exists():
                return direct

            if root.exists():
                matches = list(root.glob("**/AIDO-D-PHY-I.zip"))
                if matches:
                    return matches[0]
        except (PermissionError, OSError):
            continue

    return None


def locate_member(zf: zipfile.ZipFile, relative_path: Path) -> str:
    target = str(relative_path).replace("\\", "/")
    candidates = [
        name for name in zf.namelist()
        if name.replace("\\", "/").endswith(target)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {target} in ZIP. "
            f"First ZIP members: {zf.namelist()[:20]}"
        )
    if len(candidates) > 1:
        candidates.sort(key=len)
    return candidates[0]


# =============================================================================
# EXACT MANUSCRIPT SOURCE LOADING
# =============================================================================

def load_from_extracted_folder(folder: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame | None, str]:
    bp_path = folder / BP_MATRIX_RELATIVE
    endpoint_path = folder / ENDPOINT_RELATIVE
    discrim_path = folder / DISCRIMINABILITY_RELATIVE

    if not bp_path.exists():
        raise FileNotFoundError(f"BP matrix not found: {bp_path}")
    if not endpoint_path.exists():
        raise FileNotFoundError(f"Endpoint file not found: {endpoint_path}")

    log(f"Loading exact manuscript BP matrix: {bp_path}")
    bp = pd.read_csv(bp_path, sep="\t", index_col=0, low_memory=False)

    log(f"Loading exact matched endpoint: {endpoint_path}")
    endpoint_df = pd.read_csv(endpoint_path, sep="\t", low_memory=False)

    discrim = None
    if discrim_path.exists():
        log(f"Loading original discriminability table: {discrim_path}")
        discrim = pd.read_csv(discrim_path, sep="\t", low_memory=False)

    source = f"Extracted folder: {folder}"
    return bp, parse_endpoint_dataframe(endpoint_df), discrim, source


def load_from_zip(zip_path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame | None, str]:
    log(f"Loading exact manuscript files from ZIP: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        bp_member = locate_member(zf, BP_MATRIX_RELATIVE)
        endpoint_member = locate_member(zf, ENDPOINT_RELATIVE)

        log(f"ZIP BP matrix member: {bp_member}")
        with zf.open(bp_member) as handle:
            bp = pd.read_csv(handle, sep="\t", index_col=0, low_memory=False)

        log(f"ZIP endpoint member: {endpoint_member}")
        with zf.open(endpoint_member) as handle:
            endpoint_df = pd.read_csv(handle, sep="\t", low_memory=False)

        discrim = None
        try:
            discrim_member = locate_member(zf, DISCRIMINABILITY_RELATIVE)
            log(f"ZIP discriminability member: {discrim_member}")
            with zf.open(discrim_member) as handle:
                discrim = pd.read_csv(handle, sep="\t", low_memory=False)
        except FileNotFoundError:
            log("Original discriminability table not found in ZIP; continuing.")

    source = f"ZIP archive: {zip_path}"
    return bp, parse_endpoint_dataframe(endpoint_df), discrim, source


def parse_endpoint_dataframe(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("Endpoint table is empty.")

    sample_candidates = [
        c for c in df.columns
        if any(token in str(c).lower() for token in ["sample", "patient", "barcode", "id"])
    ]
    sample_col = sample_candidates[0] if sample_candidates else df.columns[0]

    endpoint_candidates = [
        c for c in df.columns
        if c != sample_col and any(
            token in str(c).lower()
            for token in ["endpoint", "binary", "group", "label", "stage"]
        )
    ]
    endpoint_col = endpoint_candidates[0] if endpoint_candidates else df.columns[1]

    sample_ids = df[sample_col].map(normalize_sample_id)
    endpoint = pd.to_numeric(df[endpoint_col], errors="coerce")

    out = pd.Series(endpoint.to_numpy(), index=sample_ids, name="endpoint").dropna().astype(int)
    out = out[~out.index.duplicated(keep="first")]

    invalid = sorted(set(out.unique()) - {0, 1})
    if invalid:
        raise ValueError(
            f"Endpoint column {endpoint_col!r} contains non-binary values: {invalid}"
        )

    return out


def load_exact_manuscript_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame | None, str]:
    folder = find_source_folder()
    if folder is not None:
        return load_from_extracted_folder(folder)

    zip_path = find_source_zip()
    if zip_path is not None:
        return load_from_zip(zip_path)

    raise FileNotFoundError(
        "Could not find the exact AIDO-D-PHY-I source.\n"
        "Set either AIDO_DPHY_FOLDER or AIDO_DPHY_ZIP near the top of the script.\n"
        "Required source files:\n"
        f"  {BP_MATRIX_RELATIVE}\n"
        f"  {ENDPOINT_RELATIVE}"
    )


def align_and_validate(
    bp: pd.DataFrame,
    labels: pd.Series,
    original_discrim: pd.DataFrame | None,
) -> Tuple[pd.DataFrame, pd.Series, str]:

    bp.index = [normalize_sample_id(x) for x in bp.index]
    bp = bp.groupby(level=0).mean()

    # Remove accidental unnamed columns, if any survived index parsing.
    bp = bp.loc[:, ~bp.columns.astype(str).str.startswith("Unnamed:")]

    shared = bp.index.intersection(labels.index)
    bp = bp.loc[shared]
    labels = labels.loc[shared]

    # Coerce matrix to numeric and remove invalid constant columns.
    bp = bp.apply(pd.to_numeric, errors="coerce")
    bp = bp.loc[:, bp.notna().sum(axis=0) >= max(20, int(0.8 * len(bp)))]
    bp = bp.loc[:, bp.std(axis=0, ddof=1, skipna=True) > 0]

    counts = labels.value_counts().sort_index().to_dict()
    early = int(counts.get(0, 0))
    late = int(counts.get(1, 0))
    total = int(len(labels))
    bp_count = int(bp.shape[1])

    report_lines = [
        "Exact manuscript-source alignment report",
        f"Matched samples: {total}",
        f"Early: {early}",
        f"Late: {late}",
        f"Retained BP observables: {bp_count}",
        f"Expected samples: {EXPECTED_TOTAL}",
        f"Expected early: {EXPECTED_EARLY}",
        f"Expected late: {EXPECTED_LATE}",
        f"Expected BP observables: {EXPECTED_BP_COUNT}",
    ]

    if original_discrim is not None:
        report_lines.append(f"Original discriminability table rows: {len(original_discrim)}")

        if "BP_Name" in original_discrim.columns:
            original_terms = set(original_discrim["BP_Name"].astype(str))
            matrix_terms = set(bp.columns.astype(str))
            overlap = len(original_terms & matrix_terms)
            report_lines.append(f"BP-name overlap with original discriminability table: {overlap}")

    report = "\n".join(report_lines)

    log(report)

    errors = []
    if total != EXPECTED_TOTAL:
        errors.append(f"total samples {total} != {EXPECTED_TOTAL}")
    if early != EXPECTED_EARLY:
        errors.append(f"early samples {early} != {EXPECTED_EARLY}")
    if late != EXPECTED_LATE:
        errors.append(f"late samples {late} != {EXPECTED_LATE}")
    if bp_count != EXPECTED_BP_COUNT:
        errors.append(f"BP count {bp_count} != {EXPECTED_BP_COUNT}")

    if errors:
        raise ValueError(
            "Exact manuscript alignment failed: " + "; ".join(errors)
        )

    return bp, labels, report


# =============================================================================
# DISCRIMINABILITY AND MODELLING
# =============================================================================

def mannwhitney_rank(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    y_arr = y.to_numpy(dtype=int)
    mask0 = y_arr == 0
    mask1 = y_arr == 1

    for col in X.columns:
        vals = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(vals)
        if valid.sum() < 10:
            continue

        a = vals[mask0 & valid]
        b = vals[mask1 & valid]
        if len(a) < 3 or len(b) < 3:
            continue

        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            auc_raw = roc_auc_score(y_arr[valid], vals[valid])
        except Exception:
            continue

        direction = 1.0 if auc_raw >= 0.5 else -1.0
        auc_star = max(auc_raw, 1.0 - auc_raw)
        d = -math.log10(max(float(p), EPS))

        rows.append({
            "biological_process": col,
            "auc_star": float(auc_star),
            "p_value": float(p),
            "D": float(d),
            "direction": float(direction),
        })

    rank = pd.DataFrame(rows)
    if rank.empty:
        raise ValueError("No valid BP observables were ranked.")

    rank = rank.sort_values(
        ["D", "auc_star"],
        ascending=[False, False],
    ).reset_index(drop=True)

    rank["rank"] = np.arange(1, len(rank) + 1)
    rank["q_value"] = multipletests(rank["p_value"], method="fdr_bh")[1]
    return rank


def make_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
        )),
    ])


def nested_repeated_cv(
    bp: pd.DataFrame,
    y: pd.Series,
    k_values: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    fold_rows = []
    repeat_store = {
        repeat_id: {
            k: {"y": [], "prob": [], "selected_sets": []}
            for k in k_values
        }
        for repeat_id in range(N_REPEATS)
    }

    X_all = bp.loc[y.index]
    y_all = y.loc[X_all.index]
    total_folds = N_SPLITS * N_REPEATS

    for global_fold, (train_idx, test_idx) in enumerate(
        splitter.split(X_all, y_all),
        start=1,
    ):
        repeat_id = (global_fold - 1) // N_SPLITS
        fold_id = (global_fold - 1) % N_SPLITS

        X_train = X_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]
        y_train = y_all.iloc[train_idx]
        y_test = y_all.iloc[test_idx]

        ranking = mannwhitney_rank(X_train, y_train)
        ranked_terms = ranking["biological_process"].tolist()

        for k in k_values:
            selected = ranked_terms[:k]

            model = make_model()
            model.fit(X_train[selected], y_train)

            prob = model.predict_proba(X_test[selected])[:, 1]
            pred = (prob >= 0.5).astype(int)

            auc = roc_auc_score(y_test, prob)
            bacc = balanced_accuracy_score(y_test, pred)
            acc = accuracy_score(y_test, pred)

            fold_rows.append({
                "repeat": repeat_id + 1,
                "fold": fold_id + 1,
                "K": k,
                "test_auc": float(auc),
                "balanced_accuracy": float(bacc),
                "accuracy": float(acc),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "selected_terms": "|".join(selected),
            })

            repeat_store[repeat_id][k]["y"].extend(y_test.astype(int).tolist())
            repeat_store[repeat_id][k]["prob"].extend(prob.tolist())
            repeat_store[repeat_id][k]["selected_sets"].append(set(selected))

        if global_fold % 5 == 0 or global_fold == total_folds:
            log(f"Nested CV: completed fold {global_fold}/{total_folds}")

    fold_df = pd.DataFrame(fold_rows)

    repeat_rows = []
    stability_rows = []

    for repeat_id in range(N_REPEATS):
        for k in k_values:
            store = repeat_store[repeat_id][k]
            oof_auc = roc_auc_score(store["y"], store["prob"])

            selected_sets = store["selected_sets"]
            pairwise_jaccard = []

            for i in range(len(selected_sets)):
                for j in range(i + 1, len(selected_sets)):
                    union = selected_sets[i] | selected_sets[j]
                    score = (
                        len(selected_sets[i] & selected_sets[j]) / len(union)
                        if union else 1.0
                    )
                    pairwise_jaccard.append(score)

            jaccard = (
                float(np.mean(pairwise_jaccard))
                if pairwise_jaccard else np.nan
            )

            repeat_rows.append({
                "repeat": repeat_id + 1,
                "K": k,
                "oof_auc": float(oof_auc),
                "selection_jaccard": jaccard,
            })

            stability_rows.append({
                "repeat": repeat_id + 1,
                "K": k,
                "selection_jaccard": jaccard,
            })

    return (
        fold_df,
        pd.DataFrame(repeat_rows),
        pd.DataFrame(stability_rows),
    )


# =============================================================================
# DESCRIPTIVE AGGREGATE ANALYSIS
# =============================================================================

def full_data_aggregate_analysis(
    bp: pd.DataFrame,
    y: pd.Series,
    ranking: pd.DataFrame,
    k_values: Sequence[int],
) -> pd.DataFrame:

    rows = []

    for k in k_values:
        subset = ranking.head(k).copy()
        terms = subset["biological_process"].tolist()

        directions = (
            subset.set_index("biological_process")
            .loc[terms, "direction"]
            .to_numpy(dtype=float)
        )

        X = bp.loc[y.index, terms].apply(pd.to_numeric, errors="coerce")

        means = X.mean(axis=0)
        sds = X.std(axis=0, ddof=1).replace(0, np.nan)
        Xz = (X - means) / sds

        aggregate = np.nanmean(
            Xz.to_numpy(dtype=float) * directions.reshape(1, -1),
            axis=1,
        )

        valid = np.isfinite(aggregate)
        yy = y.to_numpy(dtype=int)[valid]
        ss = aggregate[valid]

        group0 = ss[yy == 0]
        group1 = ss[yy == 1]

        _, p = mannwhitneyu(group0, group1, alternative="two-sided")
        auc = orient_auc(yy, ss)
        d = -math.log10(max(float(p), EPS))

        rows.append({
            "K": k,
            "aggregate_auc": float(auc),
            "nominal_p": float(p),
            "D": float(d),
            "aggregate_definition":
                "signed mean of standardized top-K BP activities",
            "selected_terms": "|".join(terms),
        })

    out = pd.DataFrame(rows)
    out["fdr_q"] = multipletests(
        out["nominal_p"],
        method="fdr_bh",
    )[1]

    return out


# =============================================================================
# OPTIONAL PERMUTATION AUDIT
# =============================================================================

def permutation_audit(
    bp: pd.DataFrame,
    y: pd.Series,
    k_values: Sequence[int],
    observed_auc: Dict[int, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if N_PERMUTATIONS <= 0:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    X = bp.loc[y.index]
    y_array = y.to_numpy(dtype=int)

    for permutation_id in range(1, N_PERMUTATIONS + 1):
        y_perm = pd.Series(
            rng.permutation(y_array),
            index=y.index,
            name="permuted_endpoint",
        )

        splitter = RepeatedStratifiedKFold(
            n_splits=PERMUTATION_CV_SPLITS,
            n_repeats=1,
            random_state=RANDOM_STATE + permutation_id,
        )

        store = {
            k: {"y": [], "prob": []}
            for k in k_values
        }

        for train_idx, test_idx in splitter.split(X, y_perm):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y_perm.iloc[train_idx]
            y_test = y_perm.iloc[test_idx]

            ranking = mannwhitney_rank(X_train, y_train)
            ranked_terms = ranking["biological_process"].tolist()

            for k in k_values:
                selected = ranked_terms[:k]
                model = make_model()
                model.fit(X_train[selected], y_train)

                prob = model.predict_proba(X_test[selected])[:, 1]

                store[k]["y"].extend(y_test.astype(int).tolist())
                store[k]["prob"].extend(prob.tolist())

        for k in k_values:
            auc = roc_auc_score(store[k]["y"], store[k]["prob"])
            rows.append({
                "permutation": permutation_id,
                "K": k,
                "permuted_auc": float(auc),
            })

        if permutation_id % 10 == 0 or permutation_id == N_PERMUTATIONS:
            log(
                f"Permutation audit: completed "
                f"{permutation_id}/{N_PERMUTATIONS}"
            )

    distributions = pd.DataFrame(rows)

    summary_rows = []

    for k in k_values:
        values = distributions.loc[
            distributions["K"] == k,
            "permuted_auc",
        ].to_numpy(dtype=float)

        observed = float(observed_auc[k])
        empirical_p = (
            1 + np.sum(values >= observed)
        ) / (
            1 + len(values)
        )

        summary_rows.append({
            "K": k,
            "permutation_p": float(empirical_p),
            "permutation_auc_mean": float(np.mean(values)),
            "permutation_auc_95th": float(np.quantile(values, 0.95)),
        })

    summary = pd.DataFrame(summary_rows)
    summary["permutation_fdr_q"] = multipletests(
        summary["permutation_p"],
        method="fdr_bh",
    )[1]

    return distributions, summary


# =============================================================================
# SUMMARY TABLE AND FIGURE
# =============================================================================

def summarize_results(
    repeat_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    permutation_summary: pd.DataFrame | None,
) -> pd.DataFrame:

    cv_summary = repeat_df.groupby("K").agg(
        cv_auc_mean=("oof_auc", "mean"),
        cv_auc_sd=("oof_auc", "std"),
        cv_auc_ci_low=("oof_auc", lambda x: np.quantile(x, 0.025)),
        cv_auc_ci_high=("oof_auc", lambda x: np.quantile(x, 0.975)),
        selection_jaccard_mean=("selection_jaccard", "mean"),
    ).reset_index()

    fold_summary = fold_df.groupby("K").agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        accuracy_mean=("accuracy", "mean"),
    ).reset_index()

    table = cv_summary.merge(
        fold_summary,
        on="K",
        how="left",
    )

    table = table.merge(
        aggregate_df[
            [
                "K",
                "aggregate_auc",
                "nominal_p",
                "D",
                "fdr_q",
                "aggregate_definition",
            ]
        ],
        on="K",
        how="left",
    )

    if permutation_summary is not None and not permutation_summary.empty:
        table = table.merge(
            permutation_summary,
            on="K",
            how="left",
        )

    k_star = int(
        table.loc[
            table["cv_auc_mean"].idxmax(),
            "K",
        ]
    )

    best_mean = float(
        table.loc[
            table["K"] == k_star,
            "cv_auc_mean",
        ].iloc[0]
    )

    best_sd = float(
        table.loc[
            table["K"] == k_star,
            "cv_auc_sd",
        ].iloc[0]
    )

    best_se = best_sd / math.sqrt(N_REPEATS)

    eligible = table.loc[
        table["cv_auc_mean"] >= best_mean - best_se,
        "K",
    ]

    k_1se = int(eligible.min())

    table["is_K_star"] = table["K"].eq(k_star)
    table["is_K_1SE"] = table["K"].eq(k_1se)

    ordered = table.sort_values("K")
    differences = np.diff(ordered["cv_auc_mean"].to_numpy())

    if np.all(differences >= -1e-12):
        scale_pattern = "monotonic non-decreasing"
    elif np.all(differences <= 1e-12):
        scale_pattern = "monotonic non-increasing"
    else:
        scale_pattern = "non-monotonic"

    table["observed_scale_pattern"] = scale_pattern

    def interpretation(row: pd.Series) -> str:
        if bool(row["is_K_star"]):
            return "Highest mean nested-CV AUC among tested scales"
        if bool(row["is_K_1SE"]):
            return "Smallest tested scale within one SE of K-star"
        if row["K"] < k_star:
            return "Smaller tested observation scale"
        return "Larger tested observation scale"

    table["interpretation"] = table.apply(
        interpretation,
        axis=1,
    )

    return table


def make_figure(table: pd.DataFrame, output_dir: Path) -> None:
    ordered = table.sort_values("K")

    x = ordered["K"].to_numpy(dtype=int)
    cv_auc = ordered["cv_auc_mean"].to_numpy(dtype=float)
    cv_low = ordered["cv_auc_ci_low"].to_numpy(dtype=float)
    cv_high = ordered["cv_auc_ci_high"].to_numpy(dtype=float)
    aggregate_d = ordered["D"].to_numpy(dtype=float)

    k_star = int(
        ordered.loc[
            ordered["is_K_star"],
            "K",
        ].iloc[0]
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        x,
        cv_auc,
        marker="o",
        linewidth=2,
        label="Nested-CV AUC",
    )

    ax1.fill_between(
        x,
        cv_low,
        cv_high,
        alpha=0.2,
        label="95% repeat interval",
    )

    ax1.axvline(
        k_star,
        linestyle=":",
        linewidth=1.5,
        label=f"K* = {k_star}",
    )

    ax1.set_xlabel(
        "Observation scale K "
        "(top-ranked biological-process observables)"
    )
    ax1.set_ylabel("Held-out AUC")
    ax1.set_xticks(x)

    ax2 = ax1.twinx()

    ax2.plot(
        x,
        aggregate_d,
        marker="s",
        linestyle="--",
        linewidth=1.5,
        label="Aggregate D",
    )

    ax2.set_ylabel("Aggregate discriminability D")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="best",
    )

    ax1.set_title(
        "Supplementary Figure S5. "
        "Observation-scale dependence"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "Figure_S5_observation_scale.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "Figure_S5_observation_scale.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = (
        OUTPUT_ROOT
        / f"D_PHY_ObservationScale_V5_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bp, labels, original_discrim, source_description = (
        load_exact_manuscript_data()
    )

    bp, labels, alignment_report = align_and_validate(
        bp,
        labels,
        original_discrim,
    )

    (output_dir / "source_alignment_report.txt").write_text(
        source_description + "\n\n" + alignment_report + "\n",
        encoding="utf-8",
    )

    config = RunConfig(
        output_root=str(OUTPUT_ROOT),
        k_values=list(K_VALUES),
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_permutations=N_PERMUTATIONS,
        expected_total=EXPECTED_TOTAL,
        expected_early=EXPECTED_EARLY,
        expected_late=EXPECTED_LATE,
        expected_bp_count=EXPECTED_BP_COUNT,
        source_folder=(
            str(AIDO_DPHY_FOLDER)
            if AIDO_DPHY_FOLDER is not None
            else None
        ),
        source_zip=(
            str(AIDO_DPHY_ZIP)
            if AIDO_DPHY_ZIP is not None
            else None
        ),
    )

    (output_dir / "run_config.json").write_text(
        json.dumps(
            asdict(config),
            indent=2,
        ),
        encoding="utf-8",
    )

    matched = pd.DataFrame({
        "sample_id": labels.index,
        "endpoint": labels.to_numpy(dtype=int),
    })

    matched.to_csv(
        output_dir / "matched_endpoint.tsv",
        sep="\t",
        index=False,
    )

    # Save exact BP names used by the scale experiment.
    pd.DataFrame({
        "biological_process": bp.columns.astype(str),
    }).to_csv(
        output_dir / "exact_5065_BP_universe.tsv",
        sep="\t",
        index=False,
    )

    log("Ranking the exact 5,065 BP universe on the full cohort...")
    full_ranking = mannwhitney_rank(bp, labels)

    full_ranking.to_csv(
        output_dir / "full_data_bp_ranking.csv",
        index=False,
    )

    aggregate_df = full_data_aggregate_analysis(
        bp,
        labels,
        full_ranking,
        K_VALUES,
    )

    aggregate_df.to_csv(
        output_dir / "full_data_aggregate_results.csv",
        index=False,
    )

    log("Starting leakage-controlled repeated stratified CV...")
    fold_df, repeat_df, stability_df = nested_repeated_cv(
        bp,
        labels,
        K_VALUES,
    )

    fold_df.to_csv(
        output_dir / "nested_cv_fold_results.csv",
        index=False,
    )

    repeat_df.to_csv(
        output_dir / "nested_cv_repeat_results.csv",
        index=False,
    )

    stability_df.to_csv(
        output_dir / "selection_stability_by_K.csv",
        index=False,
    )

    observed_auc = (
        repeat_df.groupby("K")["oof_auc"]
        .mean()
        .to_dict()
    )

    permutation_summary = pd.DataFrame()

    if N_PERMUTATIONS > 0:
        permutation_distributions, permutation_summary = (
            permutation_audit(
                bp,
                labels,
                K_VALUES,
                observed_auc,
            )
        )

        permutation_distributions.to_csv(
            output_dir / "permutation_auc_distributions.csv",
            index=False,
        )

        permutation_summary.to_csv(
            output_dir / "permutation_summary_by_K.csv",
            index=False,
        )

    table = summarize_results(
        repeat_df,
        fold_df,
        aggregate_df,
        permutation_summary,
    )

    table.to_csv(
        output_dir / "Table_S9_observation_scale_results.csv",
        index=False,
    )

    with pd.ExcelWriter(
        output_dir / "Table_S9_observation_scale_results.xlsx",
        engine="openpyxl",
    ) as writer:

        table.to_excel(
            writer,
            sheet_name="Table_S9",
            index=False,
        )

        full_ranking.to_excel(
            writer,
            sheet_name="Full_BP_ranking",
            index=False,
        )

        aggregate_df.to_excel(
            writer,
            sheet_name="Aggregate_results",
            index=False,
        )

        repeat_df.to_excel(
            writer,
            sheet_name="CV_repeat_results",
            index=False,
        )

        fold_df.to_excel(
            writer,
            sheet_name="CV_fold_results",
            index=False,
        )

        if original_discrim is not None:
            original_discrim.to_excel(
                writer,
                sheet_name="Original_discriminability",
                index=False,
            )

        if not permutation_summary.empty:
            permutation_summary.to_excel(
                writer,
                sheet_name="Permutation_summary",
                index=False,
            )

    make_figure(
        table,
        output_dir,
    )

    k_star = int(
        table.loc[
            table["is_K_star"],
            "K",
        ].iloc[0]
    )

    k_1se = int(
        table.loc[
            table["is_K_1SE"],
            "K",
        ].iloc[0]
    )

    scale_pattern = str(
        table["observed_scale_pattern"].iloc[0]
    )

    summary = f"""D-PHY observation-scale sensitivity experiment V5

Source:
{source_description}

Exact manuscript alignment:
Matched samples: {len(labels)}
Early: {(labels == 0).sum()}
Late: {(labels == 1).sum()}
BP observables: {bp.shape[1]}

Tested K values:
{K_VALUES}

Cross-validation:
{N_SPLITS} folds x {N_REPEATS} repeats

Permutation audit:
{N_PERMUTATIONS} permutations

Observed scale pattern:
{scale_pattern}

K-star:
{k_star}

K-1SE:
{k_1se}

Interpretation boundary:
K-star is the best-performing scale among the tested values in this exact
dataset, endpoint, observable universe, ranking procedure, and model. It is
not a universal biological optimum.

Primary reporting:
Use nested-CV AUC as the main performance evidence.

Secondary reporting:
Use aggregate AUC, nominal P, D, and FDR q as descriptive/inferential
support. Do not treat full-data aggregate performance as held-out prediction.
"""

    (output_dir / "run_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    log(summary)
    log(f"Finished successfully. Output: {output_dir}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )
        raise
