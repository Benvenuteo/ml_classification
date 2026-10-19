"""Target semantics and data quality failures that could invalidate the study."""

import numpy as np
import pandas as pd
import pytest

from air_quality.config import StudyConfig
from air_quality.data import (
    CLASS_NAMES,
    FEATURES,
    TARGET,
    audit_dataset,
    encode_target,
    load_dataset,
    mask_invalid_values,
    validate_features,
)
from air_quality.study import prepare_split


def test_targets_preserve_order_and_binary_orientation():
    labels = pd.Series(CLASS_NAMES)
    np.testing.assert_array_equal(encode_target(labels, "multiclass"), [0, 1, 2, 3])
    np.testing.assert_array_equal(encode_target(labels, "binary"), [0, 0, 1, 1])


@pytest.mark.parametrize("labels", [["Unknown"], [None], ["Good", "poor"]])
def test_unknown_labels_never_silently_become_binary_zero(labels):
    with pytest.raises(ValueError, match="target labels"):
        encode_target(pd.Series(labels), "binary")


def test_quality_snapshot(dataset):
    audit = audit_dataset(dataset)
    assert audit["class_counts"] == dict(zip(CLASS_NAMES, [2000, 1500, 1000, 500], strict=True))
    assert audit["missing_values"] == 0
    assert audit["any_issue_rows"] == 254
    assert audit["issue_counts"]["Humidity_above_100"] == 195
    assert audit["issue_counts"]["SO2_negative"] == 30


def test_domain_mask_preserves_rows_extremes_and_original_input(small_features):
    original = small_features.copy()
    original.iloc[0, original.columns.get_loc("Humidity")] = 120
    original.iloc[1, original.columns.get_loc("SO2")] = -2
    original.iloc[2, original.columns.get_loc("Temperature")] = -15
    original.iloc[3, original.columns.get_loc("PM2.5")] = 1000
    clean = mask_invalid_values(original)
    assert clean.shape == original.shape
    assert pd.isna(clean.iloc[0]["Humidity"])
    assert pd.isna(clean.iloc[1]["SO2"])
    assert clean.iloc[2]["Temperature"] == -15
    assert clean.iloc[3]["PM2.5"] == 1000
    assert original.iloc[0]["Humidity"] == 120


def test_schema_rejects_missing_nonnumeric_and_infinite_features(small_features):
    with pytest.raises(ValueError, match="schema"):
        validate_features(small_features.drop(columns="CO"))
    invalid = small_features.astype(object)
    invalid.iloc[0, 0] = "bad measurement"
    with pytest.raises(ValueError, match="nonnumeric"):
        validate_features(invalid)
    invalid = small_features.copy()
    invalid.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        validate_features(invalid)


def test_dataset_checksum_and_duplicates_are_rejected(dataset, tmp_path):
    path = tmp_path / "changed.csv"
    dataset.to_csv(path, index=False)
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(path, "0" * 64)
    pd.concat([dataset, dataset.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="group-aware"):
        load_dataset(path)


def test_split_is_disjoint_shared_and_stratified(dataset):
    frame, training, test = prepare_split(dataset, StudyConfig())
    assert not set(training) & set(test)
    assert len(training) == 4000 and len(test) == 1000
    assert frame.iloc[test][TARGET].value_counts().to_dict() == {
        "Good": 400,
        "Moderate": 300,
        "Poor": 200,
        "Hazardous": 100,
    }
    binary = encode_target(frame.iloc[test][TARGET], "binary")
    four = encode_target(frame.iloc[test][TARGET], "multiclass")
    np.testing.assert_array_equal(binary, four >= 2)
    assert frame.loc[:, FEATURES].shape[1] == 9


def test_full_profile_cannot_silently_use_a_reduced_dataset():
    with pytest.raises(ValueError, match="entire"):
        StudyConfig(profile="full", sample_size=500)
