"""Comparable model families and bounded, fold-local hyperparameter searches."""

from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from air_quality.config import StudyConfig
from air_quality.data import FEATURES
from air_quality.preprocessing import ClusterDistances, DomainCleaner

MODEL_NAMES = ("dummy", "logistic", "knn", "svm", "random_forest", "hist_boosting")
DISPLAY_NAMES = {
    "dummy": "Najczęstsza klasa",
    "logistic": "Regresja logistyczna",
    "knn": "kNN",
    "svm": "SVM RBF",
    "random_forest": "Random Forest",
    "hist_boosting": "Histogram Gradient Boosting",
}


class ProbabilitySVC(SVC):
    """Use the probability decision consistently in tuning, testing and inference.

    Standard SVC.predict can disagree with argmax(predict_proba). This subclass
    declares the probability rule explicitly, matching a 0.5 binary threshold.
    The underlying SVC probability calibration is still fitted on training data.
    """

    def predict(self, X: Any) -> np.ndarray:
        """Return the class with the highest fitted probability."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def make_pipeline(
    model_name: str,
    seed: int = 42,
    *,
    policy: str = "mask",
    features: tuple[str, ...] = FEATURES,
    balance: str = "none",
    clusters: bool = False,
    smoke: bool = False,
) -> Pipeline:
    """Build an unfitted pipeline with no resampling during prediction.

    Scaling precedes all distance-based operations, including SMOTE and KMeans.
    A common scaler also keeps the treatment comparisons identical for trees.
    Estimators use one worker; only the inner search is parallelized.
    """
    if balance not in {"none", "weights", "smote"}:
        raise ValueError(f"Unknown balancing strategy: {balance}")
    estimators: dict[str, Any] = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=2500, random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=1),
        "svm": ProbabilitySVC(C=2, probability=True, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=40 if smoke else 160,
            n_jobs=1,
            random_state=seed,
        ),
        "hist_boosting": HistGradientBoostingClassifier(
            max_iter=50 if smoke else 150,
            max_leaf_nodes=15,
            early_stopping=False,
            random_state=seed,
        ),
    }
    if model_name not in estimators:
        raise ValueError(f"Unknown model: {model_name}")
    estimator = estimators[model_name]
    if balance == "weights":
        if "class_weight" not in estimator.get_params():
            raise ValueError(f"{model_name} does not support class weights.")
        estimator.set_params(class_weight="balanced")
    if clusters and features != FEATURES:
        raise ValueError("Cluster distances require the canonical full feature set.")
    return Pipeline(
        [
            ("clean", DomainCleaner(policy=policy, features=features)),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clusters", ClusterDistances(random_state=seed) if clusters else "passthrough"),
            (
                "balance",
                SMOTE(random_state=seed, k_neighbors=3) if balance == "smote" else "passthrough",
            ),
            ("model", estimator),
        ]
    )


def search_space(model_name: str, seed: int) -> list[dict[str, Any]]:
    """Declare balancing as part of inner selection, never before fold splitting."""
    spaces = {
        "logistic": {"model__C": [0.1, 1.0, 10.0]},
        "knn": {"model__n_neighbors": [7, 15, 31], "model__weights": ["uniform", "distance"]},
        "svm": {"model__C": [0.5, 2.0, 8.0], "model__gamma": ["scale", 0.1]},
        "random_forest": {
            "model__min_samples_leaf": [1, 3, 7],
            "model__max_features": ["sqrt", 0.8],
        },
        "hist_boosting": {
            "model__learning_rate": [0.05, 0.12],
            "model__max_leaf_nodes": [7, 15],
            "model__l2_regularization": [0.0, 1.0],
        },
    }
    if model_name not in spaces:
        raise ValueError(f"No tuning space for {model_name}")
    base = spaces[model_name]
    options: list[dict[str, Any]] = []
    for method in ("none", "weights", "smote"):
        if method == "weights" and model_name == "knn":
            continue
        space = dict(base)
        space["balance"] = [
            SMOTE(random_state=seed, k_neighbors=3) if method == "smote" else "passthrough"
        ]
        if model_name != "knn":
            space["model__class_weight"] = ["balanced" if method == "weights" else None]
        options.append(space)
    return options


def make_search(model_name: str, config: StudyConfig) -> RandomizedSearchCV:
    """Return a deterministic inner search optimizing macro-F1, not test accuracy."""
    return RandomizedSearchCV(
        make_pipeline(model_name, config.model_seed, smoke=config.profile == "smoke"),
        param_distributions=search_space(model_name, config.model_seed),
        n_iter=config.search_iterations,
        scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
        refit="macro_f1",
        cv=StratifiedKFold(config.inner_splits, shuffle=True, random_state=config.cv_seed + 1),
        random_state=config.model_seed,
        n_jobs=config.n_jobs,
        error_score="raise",
        return_train_score=False,
    )


def serializable_params(params: dict[str, Any]) -> dict[str, Any]:
    """Represent estimator-valued search parameters safely in JSON."""
    return {
        key: (value if value is None or isinstance(value, (str, int, float, bool)) else repr(value))
        for key, value in params.items()
    }
