"""End-to-end study with a shared split, nested selection and one final evaluation phase."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_config
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from threadpoolctl import threadpool_limits

from air_quality.config import StudyConfig
from air_quality.data import (
    CLASS_NAMES,
    CONTEXT_FEATURES,
    FEATURES,
    POLLUTANTS,
    TARGET,
    audit_dataset,
    encode_target,
    file_sha256,
    load_dataset,
)
from air_quality.evaluation import (
    aggregate_elevated_probability,
    bootstrap_intervals,
    choose_threshold,
    metrics,
    paired_binary_comparison,
)
from air_quality.models import MODEL_NAMES, make_pipeline, make_search, serializable_params


def save_json(path: Path, value: Any) -> None:
    """Atomically write standards-compliant UTF-8 JSON, rejecting NaN and Infinity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def progress(output: Path, message: str) -> None:
    """Expose a concise progress checkpoint for long local or CI runs."""
    print(message, flush=True)
    save_json(
        output / "progress.json", {"message": message, "updated_at": datetime.now(UTC).isoformat()}
    )


def prepare_split(
    frame: pd.DataFrame, config: StudyConfig
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Use the same four-class-stratified records for both modelling tasks."""
    if config.sample_size is not None:
        if config.sample_size >= len(frame):
            raise ValueError("Smoke sample_size must be smaller than the dataset.")
        selected, _ = train_test_split(
            frame.index,
            train_size=config.sample_size,
            stratify=frame[TARGET],
            random_state=config.split_seed,
        )
        frame = frame.loc[selected].sort_index().copy()
    train_ids, test_ids = train_test_split(
        np.arange(len(frame)),
        test_size=config.test_size,
        stratify=frame[TARGET],
        random_state=config.split_seed,
    )
    smallest_training_class = frame.iloc[train_ids][TARGET].value_counts().min()
    if smallest_training_class < config.outer_splits * 4:
        raise ValueError("Too few minority examples for nested stratification and SMOTE.")
    return frame, train_ids, test_ids


def _scores(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Report scalar metrics consistently in every development fold."""
    return {
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
    }


def tune_task(
    X: pd.DataFrame,
    y: np.ndarray,
    task: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: StudyConfig,
    output: Path,
) -> dict[str, Any]:
    """Evaluate each family with nested CV, then fit the winning family on development data.

    Baselines and tuned procedures use identical outer validation records.
    Outer results estimate each family-specific selection procedure. Picking the
    highest mean across families adds selection optimism; the final holdout is
    the separate assessment of the complete decision.
    """
    records: list[dict] = []
    oof_probabilities: dict[str, np.ndarray] = {}
    classes = 4 if task == "multiclass" else 2
    for model_name in MODEL_NAMES:
        base = make_pipeline(model_name, config.model_seed, smoke=config.profile == "smoke")
        for fold_number, (train, valid) in enumerate(folds, start=1):
            started = perf_counter()
            fitted = clone(base).fit(X.iloc[train], y[train])
            records.append(
                {
                    "task": task,
                    "model": model_name,
                    "procedure": "baseline",
                    "fold": fold_number,
                    **_scores(y[valid], fitted.predict(X.iloc[valid])),
                    "train_macro_f1": float(
                        f1_score(
                            y[train],
                            fitted.predict(X.iloc[train]),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "fit_seconds": perf_counter() - started,
                    "best_inner_macro_f1": None,
                    "parameters": "{}",
                }
            )
        if model_name == "dummy":
            continue
        oof = np.full((len(X), classes), np.nan)
        for fold_number, (train, valid) in enumerate(folds, start=1):
            started = perf_counter()
            search = make_search(model_name, config)
            search.fit(X.iloc[train], y[train])
            predicted = search.predict(X.iloc[valid])
            oof[valid] = search.predict_proba(X.iloc[valid])
            score = _scores(y[valid], predicted)
            records.append(
                {
                    "task": task,
                    "model": model_name,
                    "procedure": "nested_tuned",
                    "fold": fold_number,
                    **score,
                    "train_macro_f1": float(
                        f1_score(
                            y[train],
                            search.predict(X.iloc[train]),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "fit_seconds": perf_counter() - started,
                    "best_inner_macro_f1": float(search.best_score_),
                    "parameters": json.dumps(
                        serializable_params(search.best_params_), sort_keys=True
                    ),
                }
            )
            progress(
                output,
                f"{task}: {model_name}, fold {fold_number}/{len(folds)}, "
                f"macro-F1={score['macro_f1']:.4f}",
            )
        if not np.isfinite(oof).all():
            raise RuntimeError("Out-of-fold coverage is incomplete.")
        oof_probabilities[model_name] = oof
        pd.DataFrame(records).to_csv(output / "tables" / f"{task}_cv_folds.csv", index=False)

    results = pd.DataFrame(records)
    summary = (
        results.groupby(["model", "procedure"], sort=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            train_macro_f1_mean=("train_macro_f1", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output / "tables" / f"{task}_cv_summary.csv", index=False)
    ranking = summary[summary["procedure"] == "nested_tuned"].sort_values(
        ["macro_f1_mean", "model"],
        ascending=[False, True],
    )
    selected_name = str(ranking.iloc[0]["model"])
    final_search = make_search(selected_name, config)
    final_search.fit(X, y)
    cv_search = pd.DataFrame(final_search.cv_results_)
    cv_search["params"] = cv_search["params"].map(
        lambda params: json.dumps(serializable_params(params), sort_keys=True),
    )
    cv_search[
        [
            "params",
            "mean_test_macro_f1",
            "std_test_macro_f1",
            "rank_test_macro_f1",
            "mean_test_balanced_accuracy",
        ]
    ].to_csv(
        output / "tables" / f"{task}_final_inner_search.csv",
        index=False,
    )
    joblib.dump(final_search.best_estimator_, output / "models" / f"{task}.joblib")
    probabilities = oof_probabilities[selected_name]
    oof_table = pd.DataFrame(
        probabilities, columns=[f"probability_{code}" for code in range(classes)]
    )
    oof_table.insert(0, "row_id", X.index)
    oof_table.insert(1, "true_label", y)
    oof_table.to_csv(output / "tables" / f"{task}_nested_oof.csv", index=False)
    progress(output, f"Selected {task}: {selected_name}; final development fit completed")
    return {
        "name": selected_name,
        "estimator": final_search.best_estimator_,
        "params": serializable_params(final_search.best_params_),
        "oof_probabilities": probabilities,
        "nested_macro_f1_mean": float(ranking.iloc[0]["macro_f1_mean"]),
        "nested_macro_f1_std": float(ranking.iloc[0]["macro_f1_std"]),
    }


def development_ablations(
    X: pd.DataFrame,
    y: np.ndarray,
    task: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: StudyConfig,
    output: Path,
) -> None:
    """Run declared sensitivity checks without changing the primary model-selection rule."""
    treatments = [
        ("balancing", "none", "logistic", {"balance": "none"}),
        ("balancing", "class_weights", "logistic", {"balance": "weights"}),
        ("balancing", "smote", "logistic", {"balance": "smote"}),
        ("features", "all_features", "hist_boosting", {}),
        ("features", "pollutants_only", "hist_boosting", {"features": POLLUTANTS}),
        ("features", "context_only", "hist_boosting", {"features": CONTEXT_FEATURES}),
        (
            "features",
            "without_CO",
            "hist_boosting",
            {"features": tuple(name for name in FEATURES if name != "CO")},
        ),
        (
            "features",
            "without_PM2.5",
            "hist_boosting",
            {"features": tuple(name for name in FEATURES if name != "PM2.5")},
        ),
        ("features", "cluster_distances", "hist_boosting", {"clusters": True}),
        ("cleaning", "raw_values", "hist_boosting", {"policy": "raw"}),
        ("cleaning", "domain_mask", "hist_boosting", {"policy": "mask"}),
    ]
    rows = []
    for experiment, treatment, name, settings in treatments:
        for fold_number, (train, valid) in enumerate(folds, start=1):
            pipeline = make_pipeline(
                name,
                config.model_seed,
                smoke=config.profile == "smoke",
                **settings,
            )
            pipeline.fit(X.iloc[train], y[train])
            rows.append(
                {
                    "task": task,
                    "experiment": experiment,
                    "treatment": treatment,
                    "reference_model": name,
                    "fold": fold_number,
                    **_scores(y[valid], pipeline.predict(X.iloc[valid])),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output / "tables" / f"{task}_ablation_folds.csv", index=False)
    table.groupby(["experiment", "treatment", "reference_model"], sort=False).agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
    ).reset_index().to_csv(output / "tables" / f"{task}_ablation_summary.csv", index=False)
    progress(output, f"{task}: development-only sensitivity experiments completed")


def _environment() -> dict:
    """Record the actual numerical stack and source commit when available."""
    packages = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "imbalanced-learn",
        "matplotlib",
        "jinja2",
        "joblib",
        "nbformat",
        "nbclient",
        "ipykernel",
    )
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_base_commit": revision,
        "versions": {name: importlib.metadata.version(name) for name in packages},
    }


def _final_evaluation(
    selected: dict[str, dict],
    X_test: pd.DataFrame,
    labels_test: pd.Series,
    decisions: dict,
    config: StudyConfig,
    output: Path,
) -> dict:
    """Evaluate only the frozen models and thresholds on the untouched holdout."""
    y_four = encode_target(labels_test, "multiclass")
    y_binary = encode_target(labels_test, "binary")
    multi_probability = selected["multiclass"]["estimator"].predict_proba(X_test)
    binary_probability = selected["binary"]["estimator"].predict_proba(X_test)
    aggregated = aggregate_elevated_probability(multi_probability)
    aggregate_probability = np.column_stack((1 - aggregated, aggregated))
    procedures = {
        "multiclass": (y_four, multi_probability.argmax(axis=1), multi_probability, "multiclass"),
        "binary_direct_default": (
            y_binary,
            (binary_probability[:, 1] >= 0.5).astype(int),
            binary_probability,
            "binary",
        ),
        "binary_aggregated_default": (
            y_binary,
            (aggregated >= 0.5).astype(int),
            aggregate_probability,
            "binary",
        ),
        "binary_direct_f2": (
            y_binary,
            (binary_probability[:, 1] >= decisions["direct_threshold"]["threshold"]).astype(int),
            binary_probability,
            "binary",
        ),
        "binary_aggregated_f2": (
            y_binary,
            (aggregated >= decisions["aggregated_threshold"]["threshold"]).astype(int),
            aggregate_probability,
            "binary",
        ),
    }
    results = {}
    predictions = pd.DataFrame(
        {"row_id": X_test.index, "true_multiclass": y_four, "true_binary": y_binary}
    )
    for name, (target, predicted, probability, task) in procedures.items():
        results[name] = metrics(target, predicted, probability, task)
        results[name]["confidence_intervals"] = bootstrap_intervals(
            target,
            predicted,
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
            strata=y_four,
        )
        predictions[f"predicted_{name}"] = predicted
    predictions["probability_binary_direct"] = binary_probability[:, 1]
    predictions["probability_binary_aggregated"] = aggregated
    for code, label in enumerate(CLASS_NAMES):
        predictions[f"probability_{label}"] = multi_probability[:, code]
    predictions.to_csv(output / "tables" / "holdout_predictions.csv", index=False)
    results["paired_binary_comparison"] = paired_binary_comparison(
        y_binary,
        procedures["binary_direct_default"][1],
        procedures["binary_aggregated_default"][1],
        strata=y_four,
        iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
    )
    save_json(output / "holdout_metrics.json", results)
    return results


def run_study(config: StudyConfig) -> Path:
    """Run all development experiments, freeze decisions, then build final deliverables."""
    from air_quality.reporting import build_notebook, build_report, create_figures

    output = Path(config.output_dir).resolve()
    for directory in (output, output / "tables", output / "figures", output / "models"):
        directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    original = load_dataset(config.dataset, config.expected_sha256)
    audit = audit_dataset(original)
    save_json(output / "data_audit.json", audit)
    frame, train, test = prepare_split(original, config)
    X = frame.loc[:, FEATURES]
    X_dev, X_test = X.iloc[train], X.iloc[test]
    labels_dev, labels_test = frame[TARGET].iloc[train], frame[TARGET].iloc[test]
    split = pd.DataFrame(
        {
            "row_id": frame.index,
            "original_class": frame[TARGET].to_numpy(),
            "partition": "development",
        }
    )
    split.loc[test, "partition"] = "holdout"
    split.to_csv(output / "tables" / "split_manifest.csv", index=False)
    y_four_dev = encode_target(labels_dev, "multiclass")
    folds = list(
        StratifiedKFold(
            config.outer_splits,
            shuffle=True,
            random_state=config.cv_seed,
        ).split(X_dev, y_four_dev)
    )
    manifest = {
        "started_at": datetime.now(UTC).isoformat(),
        "config": config.to_dict(),
        "dataset_sha256": file_sha256(config.dataset),
        "development_rows": len(train),
        "holdout_rows": len(test),
        "class_order": list(CLASS_NAMES),
        "binary_positive": ["Poor", "Hazardous"],
        "evaluation_status": "development_only",
        "environment": _environment(),
        "numerical_source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "config.py",
                "data.py",
                "preprocessing.py",
                "models.py",
                "evaluation.py",
                "study.py",
            )
        },
        "historical_limitation": "The same public dataset was explored in a previous project; "
        "this is a fixed procedural holdout, not a prospectively collected external sample.",
    }
    save_json(output / "run_manifest.json", manifest)
    selected = {}
    with threadpool_limits(limits=1), parallel_config(backend="loky", inner_max_num_threads=1):
        for task in ("multiclass", "binary"):
            target = encode_target(labels_dev, task)
            selected[task] = tune_task(X_dev, target, task, folds, config, output)
            development_ablations(X_dev, target, task, folds, config, output)
        if config.profile == "full":
            sizes, training_scores, validation_scores = learning_curve(
                selected["multiclass"]["estimator"],
                X_dev,
                y_four_dev,
                train_sizes=[0.3, 0.5, 0.7, 1.0],
                cv=folds,
                scoring="f1_macro",
                n_jobs=config.n_jobs,
                shuffle=True,
                random_state=config.model_seed,
                error_score="raise",
            )
            pd.DataFrame(
                {
                    "training_rows": sizes,
                    "train_mean": training_scores.mean(axis=1),
                    "validation_mean": validation_scores.mean(axis=1),
                    "validation_std": validation_scores.std(axis=1, ddof=1),
                }
            ).to_csv(
                output / "tables" / "learning_curve.csv",
                index=False,
            )
        decisions = {
            "status": "frozen_before_holdout_evaluation",
            "frozen_at": datetime.now(UTC).isoformat(),
            "model_selection_metric": "mean outer macro-F1",
            "models": {
                task: {
                    key: value
                    for key, value in result.items()
                    if key in {"name", "params", "nested_macro_f1_mean", "nested_macro_f1_std"}
                }
                for task, result in selected.items()
            },
            "direct_threshold": choose_threshold(
                encode_target(labels_dev, "binary"),
                selected["binary"]["oof_probabilities"][:, 1],
            ),
            "aggregated_threshold": choose_threshold(
                encode_target(labels_dev, "binary"),
                aggregate_elevated_probability(
                    selected["multiclass"]["oof_probabilities"],
                ),
            ),
        }
        save_json(output / "frozen_decisions.json", decisions)
        progress(output, "All decisions frozen; starting the final holdout evaluation")
        results = _final_evaluation(selected, X_test, labels_test, decisions, config, output)
        # Holdout inspection is explanatory; it does not trigger feature or model reselection.
        importance = permutation_importance(
            selected["multiclass"]["estimator"],
            X_test,
            encode_target(labels_test, "multiclass"),
            scoring="f1_macro",
            n_repeats=config.permutation_repeats,
            random_state=config.model_seed,
            n_jobs=config.n_jobs,
        )
        pd.DataFrame(
            {
                "feature": FEATURES,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        ).to_csv(
            output / "tables" / "permutation_importance.csv",
            index=False,
        )
        create_figures(frame.iloc[train], audit, results, selected, output)
    manifest["evaluation_status"] = "completed"
    manifest["elapsed_seconds"] = round(perf_counter() - started, 3)
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    save_json(output / "run_manifest.json", manifest)
    build_report(output)
    build_notebook(output, execute=config.execute_notebook)
    progress(
        output, f"Study completed in {manifest['elapsed_seconds']:.1f}s: {output / 'report.html'}"
    )
    return output / "report.html"
