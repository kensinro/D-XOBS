from __future__ import annotations

import numpy as np
import pandas as pd

from aido_d_xobs.core import compute_discriminability, observation_scale_cv


def synthetic_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    n = 40
    y = pd.Series([0] * 20 + [1] * 20, name="label")
    x = pd.DataFrame({
        "BP_signal": np.r_[rng.normal(0, 1, 20), rng.normal(1.2, 1, 20)],
        "BP_noise_1": rng.normal(0, 1, n),
        "BP_noise_2": rng.normal(0, 1, n),
    })
    return x, y


def test_discriminability_and_cv():
    x, y = synthetic_data()
    stats = compute_discriminability(x, y)
    assert set(["BP_term", "D_score", "orientation_corrected_AUC", "BH_q"]).issubset(stats.columns)
    summary, folds = observation_scale_cv(x, y, [1, 2], 2, 1, 42)
    assert set(summary["K"]) == {1, 2}
    assert len(folds) == 4
