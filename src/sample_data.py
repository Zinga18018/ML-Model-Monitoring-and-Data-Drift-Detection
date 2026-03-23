"""
Sample Data Generator
=====================
Produces synthetic reference and drifted datasets for demonstration.

Features
--------
- age          : Customer age
- income       : Annual income ($)
- score        : Internal credit / risk score
- balance      : Account balance ($)
- transactions : Monthly transaction count
"""

import numpy as np
import pandas as pd


# Drift shift configuration 
_DRIFT_SHIFTS: dict[str, dict] = {
    "low": {
        "age":          {"mean_shift": 2, "std_scale": 1.05},
        "income":       {"mean_shift": 3000, "std_scale": 1.05},
        "score":        {"mean_shift": 5, "std_scale": 1.08},
        "balance":      {"mean_shift": 500, "std_scale": 1.05},
        "transactions": {"mean_shift": 2, "std_scale": 1.05},
    },
    "medium": {
        "age":          {"mean_shift": 8, "std_scale": 1.20},
        "income":       {"mean_shift": 12000, "std_scale": 1.25},
        "score":        {"mean_shift": 20, "std_scale": 1.30},
        "balance":      {"mean_shift": 3000, "std_scale": 1.25},
        "transactions": {"mean_shift": 8, "std_scale": 1.20},
    },
    "high": {
        "age":          {"mean_shift": 18, "std_scale": 1.50},
        "income":       {"mean_shift": 30000, "std_scale": 1.60},
        "score":        {"mean_shift": 45, "std_scale": 1.55},
        "balance":      {"mean_shift": 8000, "std_scale": 1.60},
        "transactions": {"mean_shift": 18, "std_scale": 1.50},
    },
}


def generate_reference_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a clean baseline (reference) dataset.

    Parameters
    ----------
    n    : Number of rows.
    seed : Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns [age, income, score, balance, transactions].
    """
    rng = np.random.default_rng(seed)

    return pd.DataFrame({
        "age":          rng.normal(loc=35,    scale=10,    size=n).clip(18, 80).round(0),
        "income":       rng.normal(loc=55000, scale=15000, size=n).clip(15000).round(2),
        "score":        rng.normal(loc=650,   scale=80,    size=n).clip(300, 850).round(0),
        "balance":      rng.normal(loc=12000, scale=5000,  size=n).clip(0).round(2),
        "transactions": rng.poisson(lam=20, size=n).clip(0),
    })


def generate_drifted_data(
    n: int = 2000,
    drift_level: str = "medium",
    seed: int = 123,
) -> pd.DataFrame:
    """
    Generate a dataset whose distributions have shifted from the reference.

    Parameters
    ----------
    n           : Number of rows.
    drift_level : One of ``"low"``, ``"medium"``, ``"high"``.
    seed        : Random seed.

    Returns
    -------
    pd.DataFrame with the same columns as reference, but shifted.
    """
    if drift_level not in _DRIFT_SHIFTS:
        raise ValueError(f"drift_level must be one of {list(_DRIFT_SHIFTS)}")

    shifts = _DRIFT_SHIFTS[drift_level]
    rng = np.random.default_rng(seed)

    age_shift = shifts["age"]
    income_shift = shifts["income"]
    score_shift = shifts["score"]
    balance_shift = shifts["balance"]
    txn_shift = shifts["transactions"]

    return pd.DataFrame({
        "age": rng.normal(
            loc=35 + age_shift["mean_shift"],
            scale=10 * age_shift["std_scale"],
            size=n,
        ).clip(18, 80).round(0),

        "income": rng.normal(
            loc=55000 + income_shift["mean_shift"],
            scale=15000 * income_shift["std_scale"],
            size=n,
        ).clip(15000).round(2),

        "score": rng.normal(
            loc=650 + score_shift["mean_shift"],
            scale=80 * score_shift["std_scale"],
            size=n,
        ).clip(300, 850).round(0),

        "balance": rng.normal(
            loc=12000 + balance_shift["mean_shift"],
            scale=5000 * balance_shift["std_scale"],
            size=n,
        ).clip(0).round(2),

        "transactions": rng.poisson(
            lam=20 + txn_shift["mean_shift"],
            size=n,
        ).clip(0),
    })
