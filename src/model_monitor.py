"""
Model Performance Monitor
=========================
Trains a lightweight RandomForest on reference data, then evaluates
it on production data to surface concept drift via accuracy degradation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class ModelMonitor:
    """Simulates a trained ML model and monitors its production behavior."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=self.random_state,
        )
        self._is_fitted = False
        self.reference_metrics: dict = {}
        self.production_metrics: dict = {}

    # Training helpers 
    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Synthesise a binary target from the feature set so we can train
        a classifier without requiring real labels.
        """
        np.random.seed(self.random_state)
        numeric = df.select_dtypes(include=[np.number])
        z = numeric.apply(lambda c: (c - c.mean()) / (c.std() + 1e-8))
        score = z.mean(axis=1) + np.random.normal(0, 0.3, len(df))
        return (score > score.median()).astype(int)

    def fit(self, reference_df: pd.DataFrame) -> "ModelMonitor":
        """Train the surrogate model on the reference dataset."""
        X = reference_df.select_dtypes(include=[np.number])
        y = self._create_target(reference_df)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        self._is_fitted = True

        val_preds = self.model.predict(X_val)
        val_proba = self.model.predict_proba(X_val)[:, 1]
        self.reference_metrics = self._evaluate(y_val, val_preds, val_proba)
        return self

    # Tracking methods 
    def track_accuracy(self, production_df: pd.DataFrame) -> dict:
        """Return accuracy on production data vs. reference baseline."""
        self._check_fitted()
        X = production_df.select_dtypes(include=[np.number])
        y = self._create_target(production_df)
        preds = self.model.predict(X)

        prod_acc = accuracy_score(y, preds)
        ref_acc = self.reference_metrics.get("accuracy", 0)
        return {
            "reference_accuracy": round(ref_acc, 4),
            "production_accuracy": round(prod_acc, 4),
            "accuracy_drop": round(ref_acc - prod_acc, 4),
            "concept_drift_detected": (ref_acc - prod_acc) > 0.05,
        }

    def track_predictions(self, production_df: pd.DataFrame) -> dict:
        """Prediction distribution stats on production data."""
        self._check_fitted()
        X = production_df.select_dtypes(include=[np.number])
        proba = self.model.predict_proba(X)[:, 1]
        preds = self.model.predict(X)
        return {
            "mean_probability": round(float(proba.mean()), 4),
            "std_probability": round(float(proba.std()), 4),
            "positive_rate": round(float(preds.mean()), 4),
            "prediction_count": int(len(preds)),
        }

    def generate_report(self, production_df: pd.DataFrame) -> dict:
        """Full comparison report: reference vs. production."""
        self._check_fitted()
        X = production_df.select_dtypes(include=[np.number])
        y = self._create_target(production_df)
        preds = self.model.predict(X)
        proba = self.model.predict_proba(X)[:, 1]

        self.production_metrics = self._evaluate(y, preds, proba)

        degradation = {
            metric: round(
                self.reference_metrics.get(metric, 0)
                - self.production_metrics.get(metric, 0),
                4,
            )
            for metric in self.reference_metrics
        }

        return {
            "reference_metrics": self.reference_metrics,
            "production_metrics": self.production_metrics,
            "metric_degradation": degradation,
            "concept_drift_detected": any(v > 0.05 for v in degradation.values()),
            "feature_importances": dict(
                zip(
                    X.columns.tolist(),
                    [round(fi, 4) for fi in self.model.feature_importances_],
                )
            ),
        }

    # Internals 
    @staticmethod
    def _evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict:
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "auc_roc": round(roc_auc_score(y_true, y_proba), 4),
        }

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call .fit(reference_df) before monitoring.")
