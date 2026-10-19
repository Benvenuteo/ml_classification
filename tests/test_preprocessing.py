"""Verify fold isolation of imputation, scaling, clustering and resampling."""

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_validate

from air_quality.data import FEATURES, TARGET, encode_target, mask_invalid_values
from air_quality.models import make_pipeline
from air_quality.preprocessing import ClusterDistances


def test_imputation_and_scaling_do_not_learn_from_prediction_records(small_features, dataset):
    labels = encode_target(dataset.loc[small_features.index, TARGET], "multiclass")
    model = make_pipeline("logistic").fit(small_features, labels)
    expected = mask_invalid_values(small_features).median().to_numpy()
    np.testing.assert_allclose(model.named_steps["impute"].statistics_, expected)
    center = model.named_steps["scale"].mean_.copy()
    validation = small_features.iloc[:4].copy()
    validation["CO"] = 100_000
    assert model.predict(validation).shape == (4,)
    np.testing.assert_array_equal(model.named_steps["impute"].statistics_, expected)
    np.testing.assert_array_equal(model.named_steps["scale"].mean_, center)


def test_smote_runs_only_on_each_training_fold(small_features, dataset, monkeypatch):
    target = encode_target(dataset.loc[small_features.index, TARGET], "multiclass")
    observed_training_sizes = []
    original = SMOTE.fit_resample

    def record_fit(self, X, y, **kwargs):
        observed_training_sizes.append(len(X))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(SMOTE, "fit_resample", record_fit)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(small_features, target))
    cross_validate(
        make_pipeline("logistic", balance="smote"),
        small_features,
        target,
        cv=folds,
        scoring="f1_macro",
        error_score="raise",
        n_jobs=1,
    )
    assert observed_training_sizes == [len(train) for train, _ in folds]
    assert all(size < len(small_features) for size in observed_training_sizes)


def test_clusters_transform_never_refits_and_outputs_continuous_distances():
    training = np.random.default_rng(4).normal(size=(40, len(FEATURES)))
    transformer = ClusterDistances().fit(training)
    centers = transformer.kmeans_.cluster_centers_.copy()
    output = transformer.transform(np.full((7, len(FEATURES)), 10_000))
    assert output.shape == (7, 11)
    np.testing.assert_array_equal(transformer.kmeans_.cluster_centers_, centers)
    assert (output[:, -2:] >= 0).all()


def test_svm_cv_and_inference_use_the_same_probability_decision(small_features, dataset):
    target = encode_target(dataset.loc[small_features.index, TARGET], "multiclass")
    pipeline = make_pipeline("svm").fit(small_features, target)
    np.testing.assert_array_equal(
        pipeline.predict(small_features),
        pipeline.predict_proba(small_features).argmax(axis=1),
    )
