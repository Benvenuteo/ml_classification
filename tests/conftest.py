"""Small fixtures from the immutable study snapshot."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from air_quality.data import FEATURES, TARGET, load_dataset


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    """Load the real schema and all four classes once per test session."""
    return load_dataset(Path(__file__).resolve().parents[1] / "data/raw/pollution_dataset.csv")


@pytest.fixture
def small_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Keep enough minority records for each fold-local SMOTE operation."""
    selected, _ = train_test_split(
        dataset.index, train_size=120, stratify=dataset[TARGET], random_state=7
    )
    return dataset.loc[selected, FEATURES].copy()
