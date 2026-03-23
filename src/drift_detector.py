"""
Statistical Drift Detection Module
===================================
Implements multiple statistical tests for detecting data drift between
reference (training) and production (live) distributions.

Supported Tests:
    - Kolmogorov-Smirnov (KS) Test
    - Population Stability Index (PSI)
    - Jensen-Shannon (JS) Divergence
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon


class DriftDetector:
    """Detects distribution drift between reference and production datasets."""

    # Thresholds 
    KS_P_THRESHOLD = 0.05          # p-value below this → drift detected
    PSI_THRESHOLD = 0.2            # PSI above this → significant drift
    JS_THRESHOLD = 0.1             # JS divergence above this → drift

    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins

    # Kolmogorov-Smirnov Test 
    def ks_test(
        self,
        reference: pd.DataFrame,
        production: pd.DataFrame,
    ) -> dict:
        """
        Run a two-sample KS test on every shared numeric feature.

        Returns
        -------
        dict  {feature: {"statistic", "p_value", "drift"}}
        """
        results = {}
        features = reference.select_dtypes(include=[np.number]).columns
        for feat in features:
            stat, p_val = stats.ks_2samp(
                reference[feat].dropna(),
                production[feat].dropna(),
            )
            results[feat] = {
                "statistic": round(float(stat), 6),
                "p_value": round(float(p_val), 6),
                "drift": p_val < self.KS_P_THRESHOLD,
            }
        return results

    # Population Stability Index 
    def psi(
        self,
        reference: pd.DataFrame,
        production: pd.DataFrame,
    ) -> dict:
        """
        Compute PSI for each numeric feature.

        Returns
        -------
        dict  {feature: {"psi_value", "drift"}}
        """
        results = {}
        features = reference.select_dtypes(include=[np.number]).columns
        for feat in features:
            psi_val = self._compute_psi(
                reference[feat].dropna().values,
                production[feat].dropna().values,
            )
            results[feat] = {
                "psi_value": round(float(psi_val), 6),
                "drift": psi_val > self.PSI_THRESHOLD,
            }
        return results

    def _compute_psi(self, reference: np.ndarray, production: np.ndarray) -> float:
        """Calculate PSI between two 1-D arrays."""
        eps = 1e-4
        # Use reference quantiles so bins are consistent
        breakpoints = np.linspace(0, 100, self.n_bins + 1)
        edges = np.percentile(reference, breakpoints)
        edges[0] = -np.inf
        edges[-1] = np.inf

        ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
        prod_counts = np.histogram(production, bins=edges)[0].astype(float)

        ref_pct = ref_counts / ref_counts.sum() + eps
        prod_pct = prod_counts / prod_counts.sum() + eps

        psi_value = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
        return psi_value

    # Jensen-Shannon Divergence 
    def js_divergence(
        self,
        reference: pd.DataFrame,
        production: pd.DataFrame,
    ) -> dict:
        """
        Compute JS divergence for each numeric feature.

        Returns
        -------
        dict  {feature: {"js_value", "drift"}}
        """
        results = {}
        features = reference.select_dtypes(include=[np.number]).columns
        for feat in features:
            js_val = self._compute_js(
                reference[feat].dropna().values,
                production[feat].dropna().values,
            )
            results[feat] = {
                "js_value": round(float(js_val), 6),
                "drift": js_val > self.JS_THRESHOLD,
            }
        return results

    def _compute_js(self, reference: np.ndarray, production: np.ndarray) -> float:
        """Calculate JS divergence between two 1-D arrays."""
        eps = 1e-10
        all_vals = np.concatenate([reference, production])
        edges = np.linspace(all_vals.min(), all_vals.max(), self.n_bins + 1)

        ref_hist = np.histogram(reference, bins=edges)[0].astype(float) + eps
        prod_hist = np.histogram(production, bins=edges)[0].astype(float) + eps

        ref_hist /= ref_hist.sum()
        prod_hist /= prod_hist.sum()

        return float(jensenshannon(ref_hist, prod_hist) ** 2)

    # Aggregate 
    def detect_all(
        self,
        ref_df: pd.DataFrame,
        prod_df: pd.DataFrame,
    ) -> dict:
        """
        Run every drift test and consolidate results.

        Returns
        -------
        dict  {feature: {"ks": {...}, "psi": {...}, "js": {...}, "overall_drift": bool}}
        """
        ks_results = self.ks_test(ref_df, prod_df)
        psi_results = self.psi(ref_df, prod_df)
        js_results = self.js_divergence(ref_df, prod_df)

        combined: dict = {}
        for feat in ks_results:
            combined[feat] = {
                "ks": ks_results[feat],
                "psi": psi_results[feat],
                "js": js_results[feat],
                "overall_drift": (
                    ks_results[feat]["drift"]
                    or psi_results[feat]["drift"]
                    or js_results[feat]["drift"]
                ),
            }
        return combined
