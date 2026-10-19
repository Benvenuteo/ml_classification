"""Schema validation, domain diagnostics and immutable target definitions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET = "Air Quality"
CLASS_NAMES = ("Good", "Moderate", "Poor", "Hazardous")
FEATURES = (
    "Temperature",
    "Humidity",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "Proximity_to_Industrial_Areas",
    "Population_Density",
)
POLLUTANTS = ("PM2.5", "PM10", "NO2", "SO2", "CO")
CONTEXT_FEATURES = tuple(name for name in FEATURES if name not in POLLUTANTS)


def file_sha256(path: str | Path) -> str:
    """Return the digest of the original bytes, including line endings."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_features(frame: pd.DataFrame, *, allow_empty: bool = False) -> pd.DataFrame:
    """Validate numeric inputs and return a defensive copy in canonical order.

    Missing numeric values are accepted for fold-local imputation. Infinite values,
    ambiguous columns and nonnumeric strings are rejected rather than silently coerced.
    Domain violations are audited separately and handled by a pipeline transformer.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame with named columns.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    missing = set(FEATURES) - set(frame.columns)
    unexpected = set(frame.columns) - set(FEATURES)
    if missing or unexpected:
        raise ValueError(
            f"Invalid feature schema: missing={sorted(missing)}, extra={sorted(unexpected)}"
        )
    if frame.empty and not allow_empty:
        raise ValueError("The input dataset is empty.")
    result = frame.loc[:, FEATURES].copy()
    for column in FEATURES:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Column {column!r} contains nonnumeric values.") from exc
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Infinite feature values are not allowed.")
    if result.isna().all(axis=1).any():
        raise ValueError("Each record must contain at least one observed numeric feature.")
    return result


def encode_target(labels: pd.Series, task: str) -> np.ndarray:
    """Encode ordered classes or elevated quality labels (Poor/Hazardous = 1).

    The binary outcome is a dataset-label grouping, not a health recommendation.
    Unknown labels must never fall through to a default binary category.
    """
    unknown = set(labels.dropna().unique()) - set(CLASS_NAMES)
    if labels.isna().any() or unknown:
        raise ValueError(f"Missing or unknown target labels: {sorted(unknown)}")
    if task == "multiclass":
        return labels.map(dict(zip(CLASS_NAMES, range(4), strict=True))).to_numpy(dtype=int)
    if task == "binary":
        return labels.isin(("Poor", "Hazardous")).to_numpy(dtype=int)
    raise ValueError(f"Unknown task: {task!r}")


def load_dataset(path: str | Path, expected_sha256: str | None = None) -> pd.DataFrame:
    """Read and validate a labelled dataset without modifying the source file."""
    path = Path(path)
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError("Dataset checksum does not match the configured study snapshot.")
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing.")
    features = validate_features(frame.drop(columns=TARGET))
    encode_target(frame[TARGET], "multiclass")
    if frame[TARGET].nunique() != len(CLASS_NAMES):
        raise ValueError("The study requires all four original classes.")
    features[TARGET] = frame[TARGET]
    if features.duplicated(subset=list(FEATURES)).any():
        raise ValueError(
            "Duplicate feature rows require group-aware splitting; refusing random split."
        )
    return features


def domain_issues(frame: pd.DataFrame) -> pd.DataFrame:
    """Identify hard domain violations and the separate PM fraction consistency flag."""
    issues: dict[str, pd.Series] = {
        "Humidity_above_100": frame["Humidity"] > 100,
        "Humidity_below_0": frame["Humidity"] < 0,
    }
    for column in (*POLLUTANTS, "Proximity_to_Industrial_Areas", "Population_Density"):
        issues[f"{column}_negative"] = frame[column] < 0
    # Measurement periods are unavailable: flag this relation, do not force a correction.
    issues["PM2.5_above_PM10"] = frame["PM2.5"] > frame["PM10"]
    return pd.DataFrame(issues, index=frame.index)


def mask_invalid_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace physically invalid numeric values by NaN without removing rows.

    Large positive pollutant concentrations are retained. Negative temperatures are
    valid Celsius values. PM2.5 > PM10 is flagged only, because metadata is incomplete.
    """
    result = frame.copy()
    if "Humidity" in result:
        result.loc[~result["Humidity"].between(0, 100), "Humidity"] = np.nan
    for column in (*POLLUTANTS, "Proximity_to_Industrial_Areas", "Population_Density"):
        if column in result:
            result.loc[result[column] < 0, column] = np.nan
    return result


def audit_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize quality issues and class-dependent exclusion risks."""
    features = frame.loc[:, FEATURES]
    issues = domain_issues(features)
    hard_issues = issues.drop(columns="PM2.5_above_PM10")
    by_class = []
    for label in CLASS_NAMES:
        selected = frame[TARGET].eq(label)
        by_class.append(
            {
                "class": label,
                "rows": int(selected.sum()),
                "any_issue_rows": int(issues.loc[selected].any(axis=1).sum()),
                "hard_issue_rows": int(hard_issues.loc[selected].any(axis=1).sum()),
                "issue_rate": float(issues.loc[selected].any(axis=1).mean()),
            }
        )
    return {
        "rows": len(frame),
        "feature_count": len(FEATURES),
        "class_counts": frame[TARGET].value_counts().reindex(CLASS_NAMES).astype(int).to_dict(),
        "missing_values": int(features.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "issue_counts": issues.sum().astype(int).to_dict(),
        "any_issue_rows": int(issues.any(axis=1).sum()),
        "hard_issue_rows": int(hard_issues.any(axis=1).sum()),
        "issues_by_class": by_class,
        "ranges": features.agg(["min", "max"]).to_dict(),
    }
