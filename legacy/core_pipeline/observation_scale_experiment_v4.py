
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D-PHY observation-scale sensitivity experiment V4
================================================

Purpose
-------
Evaluate whether progressively increasing the number of top-ranked biological-
process (BP) observables produces monotonic or non-monotonic endpoint information.

Primary analysis
----------------
Leakage-controlled nested repeated stratified cross-validation:
1. Within each training fold, rank BP observables by Mann-Whitney discriminability.
2. Fit a class-balanced logistic-regression model using the top K BPs.
3. Evaluate held-out AUC, balanced accuracy, and accuracy.
4. Repeat across prespecified K values and repeated stratified folds.

Secondary analysis
------------------
Full-cohort signed aggregate score for each K:
    S_p^(K) = mean_j direction_j * z(BP_j,p)
This provides a directly interpretable aggregate AUC, Mann-Whitney P, D=-log10(P),
and BH-FDR q value for Supplementary Table S9.

Outputs
-------
Written under D:/AIDO-Temp/D_PHY_ObservationScale_<timestamp>/
including:
- Table_S9_observation_scale_results.csv/xlsx
- nested_cv_fold_results.csv
- nested_cv_repeat_results.csv
- full_data_bp_ranking.csv
- selection_stability_by_K.csv
- Figure_S5_observation_scale.png/pdf
- run_config.json
- run_summary.txt
- cached BP observable matrix

Dependencies
------------
pip install numpy pandas scipy scikit-learn statsmodels matplotlib openpyxl
"""


import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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


# ---------------------------------------------------------------------------
# USER PATHS
# ---------------------------------------------------------------------------

GE_PATH = Path(r"D:\AIDO-Data\UCSC_XENA\Breast Cancer (BRCA)\GE.tsv")
STAGE_PATH = Path(r"D:\AIDO-Data\UCSC_XENA\Breast Cancer (BRCA)\BRCA_stage_groups_from_survival.tsv")
GMT_PATH = Path(r"D:\AIDO-Data\GSEA\c5.go.bp.v2026.1.Hs.symbols.gmt")
OUTPUT_ROOT = Path(r"D:\AIDO-Temp")

# Optional: set to an existing BP activity matrix to skip raw reconstruction.
# Expected orientation: rows=samples, columns=BP terms.
PRECOMPUTED_BP_MATRIX = None  # e.g. Path(r"D:\...\BP_activity_samples_by_bp.tsv")
FORCE_REBUILD_BP = True  # V4: ignore old 818-sample cache and rebuild from raw GE.tsv


# ---------------------------------------------------------------------------
# ANALYSIS CONFIGURATION
# ---------------------------------------------------------------------------

K_VALUES = [1, 3, 5, 10, 15, 20, 30, 40, 50]
MIN_BP_GENES = 5
MAX_BP_GENES = 500
N_SPLITS = 5
N_REPEATS = 20
RANDOM_STATE = 20260618
MAX_ITER = 5000
N_JOBS = -1

# Aggregate-score inference
EPS = 1e-300

# Optional permutation audit. Set to 0 to skip.
N_PERMUTATIONS = 0
PERMUTATION_CV_SPLITS = 5

# Sample-ID matching: TCGA patient-level truncation is attempted automatically.
TCGA_PATIENT_ID_LENGTH = 12

# Expected manuscript cohort after expression/endpoint matching.
EXPECTED_TOTAL = 1073
EXPECTED_EARLY = 803
EXPECTED_LATE = 270
STRICT_EXPECTED_COUNTS = True


@dataclass
class RunConfig:
    ge_path: str
    stage_path: str
    gmt_path: str
    output_root: str
    precomputed_bp_matrix: str | None
    k_values: List[int]
    min_bp_genes: int
    max_bp_genes: int
    n_splits: int
    n_repeats: int
    random_state: int
    n_permutations: int


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def normalize_sample_id(value: object) -> str:
    s = str(value).strip().replace(".", "-")
    if s.upper().startswith("TCGA-") and len(s) >= TCGA_PATIENT_ID_LENGTH:
        return s[:TCGA_PATIENT_ID_LENGTH].upper()
    return s.upper()


def clean_gene_symbol(value: object) -> str:
    s = str(value).strip()
    # Xena rows occasionally use SYMBOL|ENTREZ or SYMBOL; keep the symbol part.
    for sep in ("|", ";"):
        if sep in s:
            s = s.split(sep)[0]
    return s.upper()


def read_table_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, sep=None, engine="python")


def load_expression(path: Path) -> pd.DataFrame:
    """
    Return expression as samples x genes.
    Handles the usual UCSC-Xena gene x sample format and sample x gene format.
    """
    log(f"Loading expression: {path}")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if df.shape[1] < 3:
        df = read_table_auto(path)

    first_col = df.columns[0]
    first_values = df[first_col].astype(str)

    # Heuristic: if most remaining column names resemble TCGA samples, input is gene x sample.
    remaining_cols = [str(c) for c in df.columns[1:]]
    tcga_col_fraction = np.mean([c.upper().startswith("TCGA") for c in remaining_cols]) if remaining_cols else 0.0
    tcga_row_fraction = np.mean(first_values.str.upper().str.startswith("TCGA")) if len(first_values) else 0.0

    if tcga_col_fraction > tcga_row_fraction:
        genes = [clean_gene_symbol(x) for x in first_values]
        mat = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        mat.index = genes
        mat.columns = [normalize_sample_id(c) for c in mat.columns]
        # Collapse duplicate gene symbols.
        mat = mat.groupby(level=0).mean()
        expr = mat.T
    else:
        samples = [normalize_sample_id(x) for x in first_values]
        mat = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        mat.index = samples
        mat.columns = [clean_gene_symbol(c) for c in mat.columns]
        mat = mat.T.groupby(level=0).mean().T
        expr = mat

    # Collapse duplicate patient/sample rows after TCGA truncation.
    expr = expr.groupby(level=0).mean()
    expr = expr.loc[:, expr.notna().sum(axis=0) >= max(10, int(0.5 * expr.shape[0]))]
    log(f"Expression loaded: {expr.shape[0]:,} samples x {expr.shape[1]:,} genes")
    return expr



def _parse_stage_to_binary(value: object) -> float:
    """
    Map pathological/clinical stage text to early (0) or late (1).

    Early:
      Stage 0, I, IA, IB, IC, II, IIA, IIB, IIC
    Late:
      Stage III, IIIA, IIIB, IIIC, IV, IVA, IVB, IVC
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip().upper()
    if not s or s in {"NA", "N/A", "NAN", "NONE", "UNKNOWN", "NOT REPORTED", "[NOT AVAILABLE]"}:
        return np.nan

    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(PATHOLOGIC|PATHOLOGICAL|CLINICAL|AJCC)\s+", "", s)
    s = re.sub(r"^STAGE\s*", "", s).strip()

    if any(token in s for token in ["EARLY", "I/II", "I II", "1/2"]):
        return 0.0
    if any(token in s for token in ["LATE", "III/IV", "III IV", "3/4"]):
        return 1.0

    roman = re.match(r"^(0|IV|III|II|I)([ABC]?)\b", s)
    if roman:
        base = roman.group(1)
        return 0.0 if base in {"0", "I", "II"} else 1.0

    arabic = re.match(r"^([0-4])([ABC]?)\b", s)
    if arabic:
        base = int(arabic.group(1))
        return 0.0 if base in {0, 1, 2} else 1.0

    return np.nan


def infer_binary_label(series: pd.Series) -> pd.Series:
    """
    Convert an endpoint/group/stage column to nullable integer 0/1.
    """
    raw = series.astype("string").str.strip()
    lowered = raw.str.lower()

    numeric = pd.to_numeric(raw, errors="coerce")
    unique_num = sorted(pd.unique(numeric.dropna()))
    if len(unique_num) == 2 and set(unique_num).issubset({0, 1}):
        return numeric.astype("Int64")

    explicit_map = {
        "early": 0, "early stage": 0, "early-stage": 0,
        "early_stage": 0, "group 0": 0, "class 0": 0,
        "late": 1, "late stage": 1, "late-stage": 1,
        "late_stage": 1, "group 1": 1, "class 1": 1,
    }
    explicit = lowered.map(explicit_map)
    if explicit.notna().sum() > 0 and explicit.dropna().nunique() == 2:
        return explicit.astype("Int64")

    parsed = raw.map(_parse_stage_to_binary)
    return parsed.astype("Int64")


def _score_label_candidate(column_name: str, labels: pd.Series) -> tuple:
    name = str(column_name).lower()
    valid = int(labels.notna().sum())
    classes = int(labels.dropna().nunique())

    score = 0
    if classes == 2:
        score += 1_000_000
    if any(token in name for token in ["early_late", "earlylate", "stage_group", "stagegroup"]):
        score += 500_000
    if any(token in name for token in ["binary", "group", "endpoint", "label"]):
        score += 200_000
    if "stage" in name:
        score += 100_000
    score += valid

    counts = labels.value_counts(dropna=True).to_dict()
    early = int(counts.get(0, 0))
    late = int(counts.get(1, 0))
    if early == EXPECTED_EARLY and late == EXPECTED_LATE:
        score += 5_000_000

    return score, valid, early, late


def load_stage_labels(path: Path) -> pd.Series:
    log(f"Loading endpoint labels: {path}")
    df = read_table_auto(path)
    if df.empty:
        raise ValueError("Stage file is empty.")

    # Identify sample-ID column by both name and TCGA-barcode content.
    best_sample_col = None
    best_sample_score = -1
    for c in df.columns:
        vals = df[c].astype(str).str.upper()
        tcga_fraction = vals.str.startswith("TCGA-").mean()
        score = tcga_fraction * 1_000_000
        if any(token in str(c).lower() for token in ["sample", "patient", "barcode", "submitter", "id"]):
            score += 1000
        if score > best_sample_score:
            best_sample_score = score
            best_sample_col = c

    sample_col = best_sample_col
    if sample_col is None:
        raise ValueError(f"Could not identify sample-ID column. Columns: {list(df.columns)}")

    diagnostic_rows = []
    candidates = []

    for c in df.columns:
        if c == sample_col:
            continue

        labels = infer_binary_label(df[c])
        score, valid, early, late = _score_label_candidate(c, labels)

        preview = " | ".join(
            map(str, pd.Series(df[c].dropna().astype(str).unique()).head(12).tolist())
        )
        diagnostic_rows.append({
            "column": str(c),
            "valid_binary_labels": valid,
            "early_count": early,
            "late_count": late,
            "unique_raw_values_preview": preview,
            "candidate_score": score,
        })

        if labels.dropna().nunique() == 2:
            candidates.append((score, c, labels))

    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["candidate_score", "valid_binary_labels"], ascending=False
    )

    print("\nEndpoint-column diagnostics:")
    print(diagnostics.head(15).to_string(index=False))

    if not candidates:
        raise ValueError(
            "Could not infer a binary endpoint column. "
            f"Columns found: {list(df.columns)}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, label_col, labels = candidates[0]

    sample_ids = df[sample_col].map(normalize_sample_id)
    out_df = pd.DataFrame({
        "sample_id": sample_ids,
        "endpoint": labels,
        "raw_label": df[label_col].astype(str),
    }).dropna(subset=["endpoint"])

    grouped = out_df.groupby("sample_id")["endpoint"]
    inconsistent = grouped.nunique()
    inconsistent_ids = inconsistent[inconsistent > 1].index.tolist()
    if inconsistent_ids:
        log(f"Warning: excluding {len(inconsistent_ids)} patients with conflicting endpoint labels.")
        out_df = out_df[~out_df["sample_id"].isin(inconsistent_ids)]

    out = out_df.groupby("sample_id")["endpoint"].first().astype(int)
    counts = out.value_counts().sort_index().to_dict()

    log(f"Sample-ID column: {sample_col!r}")
    log(f"Endpoint column selected: {label_col!r}")
    log(f"Mapped endpoint counts before expression matching: {counts}")

    try:
        diagnostics.to_csv(
            path.with_name("BRCA_stage_label_column_diagnostics.csv"),
            index=False,
        )
    except Exception:
        pass

    return out

def parse_gmt(path: Path) -> Dict[str, List[str]]:
    log(f"Loading GMT: {path}")
    gene_sets: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0].strip()
            genes = sorted({clean_gene_symbol(g) for g in parts[2:] if str(g).strip()})
            gene_sets[term] = genes
    log(f"GMT terms loaded: {len(gene_sets):,}")
    return gene_sets


def build_bp_activity(expr: pd.DataFrame, gene_sets: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gene-wise z-score across samples, then BP activity = mean z over matched genes.
    """
    log("Constructing BP activity matrix...")
    values = expr.to_numpy(dtype=np.float64)
    means = np.nanmean(values, axis=0)
    sds = np.nanstd(values, axis=0, ddof=1)
    sds[~np.isfinite(sds) | (sds == 0)] = np.nan
    z = (values - means) / sds

    gene_to_idx = {g: i for i, g in enumerate(expr.columns)}
    columns = {}
    sizes = []

    for i, (term, genes) in enumerate(gene_sets.items(), start=1):
        idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        n = len(idx)
        if n < MIN_BP_GENES or n > MAX_BP_GENES:
            continue
        score = np.nanmean(z[:, idx], axis=1)
        if np.isfinite(score).sum() < max(20, int(0.8 * len(score))):
            continue
        columns[term] = score.astype(np.float32)
        sizes.append((term, n))
        if i % 1000 == 0:
            log(f"  processed {i:,}/{len(gene_sets):,} GMT terms")

    bp = pd.DataFrame(columns, index=expr.index)
    size_df = pd.DataFrame(sizes, columns=["biological_process", "matched_gene_count"])
    log(f"BP matrix constructed: {bp.shape[0]:,} samples x {bp.shape[1]:,} processes")
    return bp, size_df


def orient_auc(y: np.ndarray, score: np.ndarray) -> float:
    auc = roc_auc_score(y, score)
    return float(max(auc, 1.0 - auc))


def mannwhitney_rank(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    y_arr = y.to_numpy()
    mask0 = y_arr == 0
    mask1 = y_arr == 1

    for col in X.columns:
        vals = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        a = vals[mask0]
        b = vals[mask1]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            continue
        try:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            auc_raw = roc_auc_score(y_arr[np.isfinite(vals)], vals[np.isfinite(vals)])
        except Exception:
            continue
        direction = 1.0 if auc_raw >= 0.5 else -1.0
        auc_star = max(auc_raw, 1.0 - auc_raw)
        d = -math.log10(max(float(p), EPS))
        rows.append((col, float(auc_star), float(p), float(d), direction))

    rank = pd.DataFrame(
        rows,
        columns=["biological_process", "auc_star", "p_value", "D", "direction"]
    )
    rank = rank.sort_values(["D", "auc_star"], ascending=[False, False]).reset_index(drop=True)
    rank["rank"] = np.arange(1, len(rank) + 1)
    if len(rank):
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


def nested_repeated_cv(bp: pd.DataFrame, y: pd.Series, k_values: Sequence[int]):
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    fold_rows = []
    repeat_predictions = {
        r: {k: {"y": [], "p": [], "selected": []} for k in k_values}
        for r in range(N_REPEATS)
    }

    X_all = bp.loc[y.index]
    y_all = y.loc[X_all.index]

    total_folds = N_SPLITS * N_REPEATS
    for fold_global, (train_idx, test_idx) in enumerate(splitter.split(X_all, y_all), start=1):
        repeat_id = (fold_global - 1) // N_SPLITS
        fold_id = (fold_global - 1) % N_SPLITS

        X_train = X_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]
        y_train = y_all.iloc[train_idx]
        y_test = y_all.iloc[test_idx]

        rank = mannwhitney_rank(X_train, y_train)
        ranked_terms = rank["biological_process"].tolist()

        for k in k_values:
            selected = ranked_terms[: min(k, len(ranked_terms))]
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
                "test_auc": auc,
                "balanced_accuracy": bacc,
                "accuracy": acc,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "selected_terms": "|".join(selected),
            })

            store = repeat_predictions[repeat_id][k]
            store["y"].extend(y_test.astype(int).tolist())
            store["p"].extend(prob.tolist())
            store["selected"].append(set(selected))

        if fold_global % 5 == 0 or fold_global == total_folds:
            log(f"Nested CV: completed fold {fold_global}/{total_folds}")

    fold_df = pd.DataFrame(fold_rows)

    repeat_rows = []
    stability_rows = []
    for repeat_id in range(N_REPEATS):
        for k in k_values:
            store = repeat_predictions[repeat_id][k]
            auc = roc_auc_score(store["y"], store["p"])
            sets = store["selected"]
            pairwise_jaccard = []
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    union = sets[i] | sets[j]
                    pairwise_jaccard.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
            stability = float(np.mean(pairwise_jaccard)) if pairwise_jaccard else np.nan

            repeat_rows.append({
                "repeat": repeat_id + 1,
                "K": k,
                "oof_auc": auc,
                "selection_jaccard": stability,
            })
            stability_rows.append({
                "repeat": repeat_id + 1,
                "K": k,
                "selection_jaccard": stability,
            })

    return fold_df, pd.DataFrame(repeat_rows), pd.DataFrame(stability_rows)


def full_data_aggregate_analysis(bp: pd.DataFrame, y: pd.Series, ranking: pd.DataFrame, k_values: Sequence[int]):
    """
    Signed mean aggregate based on full-data ranking.
    This is descriptive/inferential and not the leakage-controlled predictive estimate.
    """
    rows = []
    for k in k_values:
        subset = ranking.head(k).copy()
        terms = subset["biological_process"].tolist()
        directions = subset.set_index("biological_process")["direction"].reindex(terms).to_numpy()

        X = bp.loc[y.index, terms].apply(pd.to_numeric, errors="coerce")
        # Standardize each BP across the full cohort for an equally weighted aggregate.
        Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1).replace(0, np.nan)
        aggregate = np.nanmean(Xz.to_numpy() * directions.reshape(1, -1), axis=1)

        valid = np.isfinite(aggregate)
        yy = y.to_numpy()[valid]
        ss = aggregate[valid]
        g0 = ss[yy == 0]
        g1 = ss[yy == 1]
        _, p = mannwhitneyu(g0, g1, alternative="two-sided")
        auc = orient_auc(yy, ss)
        d = -math.log10(max(float(p), EPS))

        rows.append({
            "K": k,
            "aggregate_auc": auc,
            "nominal_p": float(p),
            "D": d,
            "aggregate_definition": "signed mean of standardized top-K BP activities",
            "selected_terms": "|".join(terms),
        })

    out = pd.DataFrame(rows)
    out["fdr_q"] = multipletests(out["nominal_p"], method="fdr_bh")[1]
    return out


def permutation_audit(bp: pd.DataFrame, y: pd.Series, k_values: Sequence[int], observed_auc: Dict[int, float]):
    if N_PERMUTATIONS <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    X = bp.loc[y.index]
    y_arr = y.to_numpy()

    for perm in range(1, N_PERMUTATIONS + 1):
        y_perm = pd.Series(rng.permutation(y_arr), index=y.index)
        # A lighter single repeated-CV pass per permutation.
        splitter = RepeatedStratifiedKFold(
            n_splits=PERMUTATION_CV_SPLITS,
            n_repeats=1,
            random_state=RANDOM_STATE + perm,
        )
        preds = {k: {"y": [], "p": []} for k in k_values}

        for train_idx, test_idx in splitter.split(X, y_perm):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_perm.iloc[train_idx], y_perm.iloc[test_idx]
            rank = mannwhitney_rank(X_train, y_train)
            ranked_terms = rank["biological_process"].tolist()

            for k in k_values:
                selected = ranked_terms[: min(k, len(ranked_terms))]
                model = make_model()
                model.fit(X_train[selected], y_train)
                prob = model.predict_proba(X_test[selected])[:, 1]
                preds[k]["y"].extend(y_test.tolist())
                preds[k]["p"].extend(prob.tolist())

        for k in k_values:
            auc = roc_auc_score(preds[k]["y"], preds[k]["p"])
            rows.append({"permutation": perm, "K": k, "permuted_auc": auc})

        if perm % 10 == 0 or perm == N_PERMUTATIONS:
            log(f"Permutation audit: completed {perm}/{N_PERMUTATIONS}")

    perm_df = pd.DataFrame(rows)
    p_rows = []
    for k in k_values:
        vals = perm_df.loc[perm_df["K"] == k, "permuted_auc"].to_numpy()
        obs = observed_auc[k]
        empirical_p = (1 + np.sum(vals >= obs)) / (1 + len(vals))
        p_rows.append({
            "K": k,
            "permutation_p": empirical_p,
            "permutation_auc_mean": float(np.mean(vals)),
            "permutation_auc_95th": float(np.quantile(vals, 0.95)),
        })

    p_df = pd.DataFrame(p_rows)
    p_df["permutation_fdr_q"] = multipletests(p_df["permutation_p"], method="fdr_bh")[1]
    return perm_df, p_df


def summarize_results(
    repeat_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    permutation_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    cv_summary = repeat_df.groupby("K").agg(
        cv_auc_mean=("oof_auc", "mean"),
        cv_auc_sd=("oof_auc", "std"),
        cv_auc_ci_low=("oof_auc", lambda x: np.quantile(x, 0.025)),
        cv_auc_ci_high=("oof_auc", lambda x: np.quantile(x, 0.975)),
        selection_jaccard_mean=("selection_jaccard", "mean"),
    ).reset_index()

    metric_summary = fold_df.groupby("K").agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        accuracy_mean=("accuracy", "mean"),
    ).reset_index()

    out = cv_summary.merge(metric_summary, on="K", how="left")
    out = out.merge(
        aggregate_df[["K", "aggregate_auc", "nominal_p", "D", "fdr_q", "aggregate_definition"]],
        on="K",
        how="left",
    )

    if permutation_summary is not None and not permutation_summary.empty:
        out = out.merge(permutation_summary, on="K", how="left")

    k_star = int(out.loc[out["cv_auc_mean"].idxmax(), "K"])
    best_mean = float(out.loc[out["K"] == k_star, "cv_auc_mean"].iloc[0])
    best_se = float(out.loc[out["K"] == k_star, "cv_auc_sd"].iloc[0] / math.sqrt(N_REPEATS))
    eligible = out.loc[out["cv_auc_mean"] >= best_mean - best_se, "K"]
    k_1se = int(eligible.min())

    out["is_K_star"] = out["K"].eq(k_star)
    out["is_K_1SE"] = out["K"].eq(k_1se)

    # Detect monotonicity descriptively.
    auc_vals = out.sort_values("K")["cv_auc_mean"].to_numpy()
    diffs = np.diff(auc_vals)
    if np.all(diffs >= -1e-12):
        pattern = "monotonic non-decreasing"
    elif np.all(diffs <= 1e-12):
        pattern = "monotonic non-increasing"
    else:
        pattern = "non-monotonic"
    out["observed_scale_pattern"] = pattern

    def interpret(row):
        if row["is_K_star"]:
            return "Highest mean nested-CV AUC among tested scales"
        if row["is_K_1SE"]:
            return "Smallest scale within one SE of the best mean nested-CV AUC"
        if row["K"] < k_star:
            return "Smaller tested observation scale"
        return "Larger tested observation scale"

    out["interpretation"] = out.apply(interpret, axis=1)
    return out


def make_figure(table: pd.DataFrame, output_dir: Path) -> None:
    ordered = table.sort_values("K")
    k = ordered["K"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    y = ordered["cv_auc_mean"].to_numpy()
    lo = ordered["cv_auc_ci_low"].to_numpy()
    hi = ordered["cv_auc_ci_high"].to_numpy()

    ax1.plot(k, y, marker="o", label="Nested-CV AUC")
    ax1.fill_between(k, lo, hi, alpha=0.2, label="95% repeat interval")
    ax1.set_xlabel("Observation scale K (top-ranked biological processes)")
    ax1.set_ylabel("Held-out AUC")
    ax1.set_xticks(k)

    ax2 = ax1.twinx()
    ax2.plot(k, ordered["D"].to_numpy(), marker="s", linestyle="--", label="Aggregate D")
    ax2.set_ylabel("Aggregate discriminability D")

    k_star = int(ordered.loc[ordered["is_K_star"], "K"].iloc[0])
    ax1.axvline(k_star, linestyle=":", label=f"K* = {k_star}")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    ax1.set_title("Supplementary Figure S5. Observation-scale dependence")
    fig.tight_layout()

    fig.savefig(output_dir / "Figure_S5_observation_scale.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "Figure_S5_observation_scale.pdf", bbox_inches="tight")
    plt.close(fig)



def find_latest_cached_bp_matrix(output_root: Path) -> Path | None:
    """
    Reuse the newest previously generated BP activity matrix when available.
    """
    candidates = sorted(
        output_root.glob("D_PHY_ObservationScale_*/BP_activity_samples_by_bp.tsv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"D_PHY_ObservationScale_V4_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        ge_path=str(GE_PATH),
        stage_path=str(STAGE_PATH),
        gmt_path=str(GMT_PATH),
        output_root=str(OUTPUT_ROOT),
        precomputed_bp_matrix=str(PRECOMPUTED_BP_MATRIX) if PRECOMPUTED_BP_MATRIX else None,
        k_values=list(K_VALUES),
        min_bp_genes=MIN_BP_GENES,
        max_bp_genes=MAX_BP_GENES,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_permutations=N_PERMUTATIONS,
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    labels = load_stage_labels(STAGE_PATH)

    label_counts_pre = labels.value_counts().sort_index().to_dict()
    log(
        "Endpoint counts before expression matching: "
        f"early={int(label_counts_pre.get(0, 0))}, "
        f"late={int(label_counts_pre.get(1, 0))}, "
        f"total={len(labels)}"
    )

    cached_bp_matrix = None

    if not FORCE_REBUILD_BP:
        cached_bp_matrix = PRECOMPUTED_BP_MATRIX
        if cached_bp_matrix is None:
            candidate = find_latest_cached_bp_matrix(OUTPUT_ROOT)
            if candidate is not None:
                try:
                    cache_header = pd.read_csv(candidate, sep="\t", index_col=0, nrows=5)
                    cache_index = pd.read_csv(candidate, sep="\t", usecols=[0]).iloc[:, 0]
                    cache_ids = pd.Index([normalize_sample_id(x) for x in cache_index])
                    cache_overlap = len(cache_ids.intersection(labels.index))
                    if cache_overlap >= EXPECTED_TOTAL:
                        cached_bp_matrix = candidate
                        log(
                            f"Validated cached BP matrix: {candidate} "
                            f"with {cache_overlap} endpoint-matched samples."
                        )
                    else:
                        log(
                            f"Ignoring stale/incomplete cached BP matrix: {candidate}; "
                            f"only {cache_overlap} samples overlap the endpoint labels."
                        )
                except Exception as exc:
                    log(f"Ignoring unreadable cached BP matrix {candidate}: {exc}")
    else:
        log("FORCE_REBUILD_BP=True: rebuilding BP matrix from raw GE.tsv.")

    if cached_bp_matrix is not None:
        log(f"Loading precomputed BP matrix: {cached_bp_matrix}")
        bp = pd.read_csv(cached_bp_matrix, sep="\t", index_col=0)
        bp.index = [normalize_sample_id(x) for x in bp.index]
        bp = bp.groupby(level=0).mean()
        size_df = pd.DataFrame()
    else:
        expr = load_expression(GE_PATH)
        shared = expr.index.intersection(labels.index)
        if len(shared) < 100:
            raise ValueError(
                f"Only {len(shared)} shared samples between expression and labels. "
                "Check sample-ID format and stage-file columns."
            )
        expr = expr.loc[shared]
        labels = labels.loc[shared]

        gene_sets = parse_gmt(GMT_PATH)
        bp, size_df = build_bp_activity(expr, gene_sets)
        bp.to_csv(output_dir / "BP_activity_samples_by_bp.tsv", sep="\t")
        size_df.to_csv(output_dir / "BP_gene_set_size.tsv", sep="\t", index=False)

    shared = bp.index.intersection(labels.index)
    bp = bp.loc[shared]
    labels = labels.loc[shared]
    bp = bp.loc[:, bp.notna().sum(axis=0) >= max(20, int(0.8 * len(bp)))]
    bp = bp.loc[:, bp.std(axis=0, ddof=1, skipna=True) > 0]

    if len(shared) < 100:
        raise ValueError(f"Insufficient matched samples after BP/label matching: {len(shared)}")
    if bp.shape[1] < max(K_VALUES):
        raise ValueError(f"Only {bp.shape[1]} valid BP observables; need at least {max(K_VALUES)}.")

    matched = pd.DataFrame({"sample_id": shared, "endpoint": labels.loc[shared].to_numpy()})
    matched.to_csv(output_dir / "matched_endpoint.tsv", sep="\t", index=False)
    log(f"Final analysis matrix: {bp.shape[0]:,} samples x {bp.shape[1]:,} BPs")

    final_counts = labels.value_counts().sort_index().to_dict()
    final_early = int(final_counts.get(0, 0))
    final_late = int(final_counts.get(1, 0))
    final_total = int(len(labels))
    log(f"Final matched endpoint counts: early={final_early}, late={final_late}, total={final_total}")

    if STRICT_EXPECTED_COUNTS:
        if (final_total, final_early, final_late) != (EXPECTED_TOTAL, EXPECTED_EARLY, EXPECTED_LATE):
            raise ValueError(
                "Matched cohort does not reproduce the manuscript cohort. "
                f"Observed total/early/late = {final_total}/{final_early}/{final_late}; "
                f"expected = {EXPECTED_TOTAL}/{EXPECTED_EARLY}/{EXPECTED_LATE}. "
                "The script stopped before nested CV to prevent invalid SI results. "
                "Inspect BRCA_stage_label_column_diagnostics.csv and matched_endpoint.tsv."
            )

    full_rank = mannwhitney_rank(bp, labels)
    full_rank.to_csv(output_dir / "full_data_bp_ranking.csv", index=False)

    aggregate_df = full_data_aggregate_analysis(bp, labels, full_rank, K_VALUES)
    aggregate_df.to_csv(output_dir / "full_data_aggregate_results.csv", index=False)

    fold_df, repeat_df, stability_df = nested_repeated_cv(bp, labels, K_VALUES)
    fold_df.to_csv(output_dir / "nested_cv_fold_results.csv", index=False)
    repeat_df.to_csv(output_dir / "nested_cv_repeat_results.csv", index=False)
    stability_df.to_csv(output_dir / "selection_stability_by_K.csv", index=False)

    observed_auc = repeat_df.groupby("K")["oof_auc"].mean().to_dict()
    permutation_summary = None
    if N_PERMUTATIONS > 0:
        perm_df, permutation_summary = permutation_audit(bp, labels, K_VALUES, observed_auc)
        perm_df.to_csv(output_dir / "permutation_auc_distributions.csv", index=False)
        permutation_summary.to_csv(output_dir / "permutation_summary_by_K.csv", index=False)

    table = summarize_results(
        repeat_df, fold_df, aggregate_df, stability_df, permutation_summary
    )
    table.to_csv(output_dir / "Table_S9_observation_scale_results.csv", index=False)

    with pd.ExcelWriter(output_dir / "Table_S9_observation_scale_results.xlsx", engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Table_S9", index=False)
        full_rank.to_excel(writer, sheet_name="Full_BP_ranking", index=False)
        repeat_df.to_excel(writer, sheet_name="CV_repeat_results", index=False)
        fold_df.to_excel(writer, sheet_name="CV_fold_results", index=False)
        aggregate_df.to_excel(writer, sheet_name="Aggregate_results", index=False)
        if permutation_summary is not None:
            permutation_summary.to_excel(writer, sheet_name="Permutation_summary", index=False)

    make_figure(table, output_dir)

    k_star = int(table.loc[table["is_K_star"], "K"].iloc[0])
    k_1se = int(table.loc[table["is_K_1SE"], "K"].iloc[0])
    pattern = str(table["observed_scale_pattern"].iloc[0])

    summary = f"""D-PHY observation-scale sensitivity experiment

Output directory: {output_dir}
Matched samples: {len(labels)}
Valid BP observables: {bp.shape[1]}
Tested K values: {K_VALUES}
Nested CV: {N_SPLITS} folds x {N_REPEATS} repeats
Observed scale pattern: {pattern}
K*: {k_star}
K_1SE: {k_1se}

Interpretation boundary:
K* is the best-performing scale among the tested values in this dataset and
analysis design. It is not a universal biological optimum.
"""
    (output_dir / "run_summary.txt").write_text(summary, encoding="utf-8")
    log(summary)
    log("Finished successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nERROR:", exc, file=sys.stderr)
        raise
