#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AIDO-D-XOBS core reproducibility pipeline.

This script provides a GitHub-ready implementation of the endpoint-discriminability,
bootstrap-stability, and observation-scale components used in the manuscript:
"Biological-process observability reveals discriminative and endpoint-invariant
information regimes in cancer transcriptomic systems".

Inputs
------
1. BP activity matrix: rows = samples, columns = biological-process observables.
2. Endpoint table: one row per sample with a binary endpoint label.

Main outputs
------------
- process_discriminability.csv
- bootstrap_stability.csv
- observation_scale_summary.csv
- observation_scale_fold_results.csv
- run_config.json

Important reproducibility note
------------------------------
The original internal D-PHY V5 bootstrap implementation ranked processes by
absolute Cohen's d within each stratified bootstrap replicate. The current
manuscript defines D = -log10(Mann-Whitney P). This script supports both modes:

    --bootstrap-ranking abs_cohen_d   # reproduces the original internal code
    --bootstrap-ranking D_score       # follows the current manuscript definition

No claim of external validation is made. Bootstrap and repeated cross-validation
measure within-cohort robustness only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

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

EPS = 1e-300


@dataclass
class RunConfig:
    bp_matrix: str
    endpoint_file: str
    output_dir: str
    sample_column: str
    label_column: str
    positive_label: str
    n_bootstrap: int
    bootstrap_top_k: int
    bootstrap_ranking: str
    k_values: list[int]
    n_splits: int
    n_repeats: int
    random_seed: int


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV/TSV/TXT using extension-aware delimiter handling."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def normalize_sample_id(value: object) -> str:
    """Normalize TCGA-style IDs to patient level while preserving generic IDs."""
    s = str(value).strip().replace(".", "-").upper()
    if s.startswith("TCGA-") and len(s) >= 12:
        return s[:12]
    return s


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.ones_like(p, dtype=float)
    mask = np.isfinite(p)
    if mask.any():
        q[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return q


def orientation_corrected_auc(y: np.ndarray, scores: np.ndarray) -> float:
    auc = roc_auc_score(y, scores)
    return float(max(auc, 1.0 - auc))


def cohen_d(group0: np.ndarray, group1: np.ndarray) -> float:
    x0 = np.asarray(group0, dtype=float)
    x1 = np.asarray(group1, dtype=float)
    x0 = x0[np.isfinite(x0)]
    x1 = x1[np.isfinite(x1)]
    if len(x0) < 2 or len(x1) < 2:
        return np.nan
    v0 = np.var(x0, ddof=1)
    v1 = np.var(x1, ddof=1)
    pooled = math.sqrt(((len(x0) - 1) * v0 + (len(x1) - 1) * v1) / (len(x0) + len(x1) - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(x1) - np.mean(x0)) / pooled)


def load_aligned_data(
    bp_path: Path,
    endpoint_path: Path,
    sample_column: str,
    label_column: str,
    positive_label: str,
) -> tuple[pd.DataFrame, pd.Series]:
    bp = read_table(bp_path)
    if bp.shape[1] < 2:
        raise ValueError("BP matrix must contain a sample-ID column and at least one BP column.")

    bp_sample_col = sample_column if sample_column in bp.columns else bp.columns[0]
    bp[bp_sample_col] = bp[bp_sample_col].map(normalize_sample_id)
    bp = bp.drop_duplicates(bp_sample_col).set_index(bp_sample_col)
    bp = bp.apply(pd.to_numeric, errors="coerce")
    bp = bp.loc[:, bp.notna().mean(axis=0) >= 0.80]
    bp = bp.fillna(bp.median(axis=0))

    endpoint = read_table(endpoint_path)
    if sample_column not in endpoint.columns:
        raise KeyError(f"Endpoint file is missing sample column: {sample_column}")
    if label_column not in endpoint.columns:
        raise KeyError(f"Endpoint file is missing label column: {label_column}")

    endpoint[sample_column] = endpoint[sample_column].map(normalize_sample_id)
    endpoint = endpoint.dropna(subset=[sample_column, label_column]).drop_duplicates(sample_column)
    endpoint = endpoint.set_index(sample_column)[label_column].astype(str)

    common = bp.index.intersection(endpoint.index)
    if len(common) < 20:
        raise ValueError(f"Too few matched samples: {len(common)}")

    bp = bp.loc[common].copy()
    y = (endpoint.loc[common] == str(positive_label)).astype(int)
    if y.nunique() != 2:
        raise ValueError("Endpoint must contain two classes after positive-label mapping.")

    return bp, y


def compute_discriminability(bp: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y_array = y.to_numpy(dtype=int)

    for term in bp.columns:
        values = bp[term].to_numpy(dtype=float)
        x0 = values[y_array == 0]
        x1 = values[y_array == 1]
        mw = mannwhitneyu(x0, x1, alternative="two-sided")
        p_value = float(mw.pvalue)
        d_score = float(-math.log10(max(p_value, EPS)))
        auc = orientation_corrected_auc(y_array, values)
        effect = cohen_d(x0, x1)
        direction = "positive_up" if np.nanmean(x1) > np.nanmean(x0) else "negative_up"

        rows.append(
            {
                "BP_term": term,
                "n_negative": int(len(x0)),
                "n_positive": int(len(x1)),
                "mean_negative": float(np.nanmean(x0)),
                "mean_positive": float(np.nanmean(x1)),
                "direction": direction,
                "cohen_d_positive_minus_negative": effect,
                "abs_cohen_d": abs(effect) if np.isfinite(effect) else np.nan,
                "mannwhitney_p": p_value,
                "D_score": d_score,
                "orientation_corrected_AUC": auc,
            }
        )

    out = pd.DataFrame(rows)
    out["BH_q"] = bh_fdr(out["mannwhitney_p"])
    return out.sort_values(["D_score", "orientation_corrected_AUC"], ascending=False).reset_index(drop=True)


def stratified_bootstrap_stability(
    bp: pd.DataFrame,
    y: pd.Series,
    original: pd.DataFrame,
    n_bootstrap: int,
    top_k: int,
    ranking_metric: str,
    random_seed: int,
) -> pd.DataFrame:
    """Estimate top-K selection and direction stability using class-stratified bootstrap.

    Each replicate samples n0 negative and n1 positive observations with replacement,
    so the total replicate size equals the original sample size and the original class
    counts are preserved exactly.
    """
    if ranking_metric not in {"abs_cohen_d", "D_score"}:
        raise ValueError("ranking_metric must be 'abs_cohen_d' or 'D_score'.")

    idx0 = np.flatnonzero(y.to_numpy() == 0)
    idx1 = np.flatnonzero(y.to_numpy() == 1)
    rng = np.random.default_rng(random_seed)

    selected_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    original_direction = dict(zip(original["BP_term"], original["direction"]))

    X = bp.to_numpy(dtype=float)
    terms = np.asarray(bp.columns, dtype=object)

    for _ in range(n_bootstrap):
        bs0 = rng.choice(idx0, size=len(idx0), replace=True)
        bs1 = rng.choice(idx1, size=len(idx1), replace=True)

        scores = np.zeros(len(terms), dtype=float)
        directions = np.empty(len(terms), dtype=object)

        for j in range(len(terms)):
            x0 = X[bs0, j]
            x1 = X[bs1, j]
            directions[j] = "positive_up" if np.nanmean(x1) > np.nanmean(x0) else "negative_up"

            if ranking_metric == "abs_cohen_d":
                effect = cohen_d(x0, x1)
                scores[j] = abs(effect) if np.isfinite(effect) else 0.0
            else:
                p_value = mannwhitneyu(x0, x1, alternative="two-sided").pvalue
                scores[j] = -math.log10(max(float(p_value), EPS))

        chosen = np.argsort(scores)[::-1][: min(top_k, len(terms))]
        for j in chosen:
            term = str(terms[j])
            selected_counts[term] += 1
            if directions[j] == original_direction.get(term):
                direction_counts[term] += 1

    rows = []
    for term in terms:
        term = str(term)
        n_selected = selected_counts[term]
        rows.append(
            {
                "BP_term": term,
                "bootstrap_topk_stability": n_selected / n_bootstrap,
                "bootstrap_direction_stability": direction_counts[term] / max(n_selected, 1),
                "n_selected": n_selected,
                "n_bootstrap": n_bootstrap,
                "bootstrap_ranking_metric": ranking_metric,
            }
        )
    return pd.DataFrame(rows).sort_values("bootstrap_topk_stability", ascending=False)


def observation_scale_cv(
    bp: pd.DataFrame,
    y: pd.Series,
    k_values: list[int],
    n_splits: int,
    n_repeats: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leakage-controlled repeated stratified CV with within-fold BP ranking."""
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_seed,
    )
    X = bp.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=int)
    terms = np.asarray(bp.columns, dtype=object)

    fold_rows: list[dict[str, object]] = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y_arr), start=1):
        train_scores = []
        for j in range(X.shape[1]):
            x0 = X[train_idx, j][y_arr[train_idx] == 0]
            x1 = X[train_idx, j][y_arr[train_idx] == 1]
            p_value = mannwhitneyu(x0, x1, alternative="two-sided").pvalue
            train_scores.append(-math.log10(max(float(p_value), EPS)))
        ranking = np.argsort(np.asarray(train_scores))[::-1]

        for k in k_values:
            chosen = ranking[: min(k, len(ranking))]
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=random_seed,
                        ),
                    ),
                ]
            )
            model.fit(X[train_idx][:, chosen], y_arr[train_idx])
            probabilities = model.predict_proba(X[test_idx][:, chosen])[:, 1]
            predictions = (probabilities >= 0.5).astype(int)

            fold_rows.append(
                {
                    "split_id": split_id,
                    "K": int(k),
                    "heldout_AUC": float(roc_auc_score(y_arr[test_idx], probabilities)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_arr[test_idx], predictions)),
                    "accuracy": float(accuracy_score(y_arr[test_idx], predictions)),
                    "selected_terms": ";".join(map(str, terms[chosen])),
                }
            )

    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby("K", as_index=False)
        .agg(
            mean_heldout_AUC=("heldout_AUC", "mean"),
            sd_heldout_AUC=("heldout_AUC", "std"),
            median_heldout_AUC=("heldout_AUC", "median"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_accuracy=("accuracy", "mean"),
            n_folds=("heldout_AUC", "size"),
        )
    )
    intervals = folds.groupby("K")["heldout_AUC"].quantile([0.025, 0.975]).unstack()
    intervals.columns = ["AUC_q025", "AUC_q975"]
    summary = summary.merge(intervals.reset_index(), on="K", how="left")
    return summary, folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bp-matrix", required=True, type=Path)
    parser.add_argument("--endpoint-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-column", default="sample_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--positive-label", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--bootstrap-top-k", type=int, default=50)
    parser.add_argument(
        "--bootstrap-ranking",
        choices=["abs_cohen_d", "D_score"],
        default="abs_cohen_d",
    )
    parser.add_argument("--k-values", default="1,3,5,10,15,20,30,40,50")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]

    config = RunConfig(
        bp_matrix=str(args.bp_matrix),
        endpoint_file=str(args.endpoint_file),
        output_dir=str(args.output_dir),
        sample_column=args.sample_column,
        label_column=args.label_column,
        positive_label=str(args.positive_label),
        n_bootstrap=args.n_bootstrap,
        bootstrap_top_k=args.bootstrap_top_k,
        bootstrap_ranking=args.bootstrap_ranking,
        k_values=k_values,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_seed=args.random_seed,
    )

    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    bp, y = load_aligned_data(
        args.bp_matrix,
        args.endpoint_file,
        args.sample_column,
        args.label_column,
        args.positive_label,
    )

    discrim = compute_discriminability(bp, y)
    discrim.to_csv(args.output_dir / "process_discriminability.csv", index=False)

    bootstrap = stratified_bootstrap_stability(
        bp,
        y,
        discrim,
        n_bootstrap=args.n_bootstrap,
        top_k=args.bootstrap_top_k,
        ranking_metric=args.bootstrap_ranking,
        random_seed=args.random_seed,
    )
    bootstrap.to_csv(args.output_dir / "bootstrap_stability.csv", index=False)

    scale_summary, scale_folds = observation_scale_cv(
        bp,
        y,
        k_values=k_values,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_seed=args.random_seed,
    )
    scale_summary.to_csv(args.output_dir / "observation_scale_summary.csv", index=False)
    scale_folds.to_csv(args.output_dir / "observation_scale_fold_results.csv", index=False)

    print(f"Completed. Outputs written to: {args.output_dir}")
    print(f"Matched samples: {len(y)}; negative={int((y == 0).sum())}; positive={int((y == 1).sum())}")
    print(f"BP observables: {bp.shape[1]}")


if __name__ == "__main__":
    main()
