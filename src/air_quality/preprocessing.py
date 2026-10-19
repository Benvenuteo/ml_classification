"""Cloneable transformers fitted independently inside each cross-validation fold."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted

from air_quality.data import FEATURES, mask_invalid_values, validate_features


class DomainCleaner(TransformerMixin, BaseEstimator):
    """Check the full input schema, select a feature set and optionally mask violations."""

    def __init__(self, policy: str = "mask", features: tuple[str, ...] = FEATURES):
        self.policy = policy
        self.features = features

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> DomainCleaner:
        """Validate policy and retain the training schema; no population statistics are learned."""
        frame = validate_features(X)
        if self.policy not in {"mask", "raw"}:
            raise ValueError("Domain policy must be 'mask' or 'raw'.")
        if not self.features or set(self.features) - set(FEATURES):
            raise ValueError("Selected features must be a nonempty subset of the input schema.")
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply deterministic rules without learning from prediction-time records."""
        check_is_fitted(self, "feature_names_in_")
        frame = validate_features(X)
        cleaned = mask_invalid_values(frame) if self.policy == "mask" else frame
        return cleaned.loc[:, self.features]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Expose stable names for feature inspection."""
        return np.asarray(self.features, dtype=object)


class ClusterDistances(TransformerMixin, BaseEstimator):
    """Append distances to two clusters using scaled CO and industrial distance.

    Distances are continuous features, avoiding synthetic fractional cluster labels
    when the transformer is followed by SMOTE. The transformer never sees a CV
    validation fold during fit.
    """

    def __init__(self, columns: tuple[int, int] = (6, 7), random_state: int = 42):
        self.columns = columns
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ClusterDistances:
        """Fit centroids in the current training fold only."""
        values = np.asarray(X, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.kmeans_ = KMeans(n_clusters=2, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(values[:, self.columns])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Append distances to the training centroids without refitting them."""
        check_is_fitted(self, "kmeans_")
        values = np.asarray(X, dtype=float)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected feature count at prediction time.")
        return np.column_stack((values, self.kmeans_.transform(values[:, self.columns])))
