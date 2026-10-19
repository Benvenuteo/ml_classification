"""Independently reconcile committed predictions, metrics, splits and notebook outputs."""

from __future__ import annotations

import argparse
import base64
import json
from collections import Counter
from datetime import datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from air_quality.data import CLASS_NAMES, TARGET, file_sha256, load_dataset
from air_quality.reporting import REQUIRED_FIGURES


def validate_png(payload: bytes) -> None:
    """Check PNG structure and decode pixels, catching broken and truncated images."""
    with Image.open(BytesIO(payload)) as image:
        assert image.format == "PNG", "Report images must use PNG."
        assert min(image.size) > 0
        image.verify()
    with Image.open(BytesIO(payload)) as image:
        image.load()


class ReportParser(HTMLParser):
    """Inspect the generated HTML without external rendering dependencies."""

    def __init__(self) -> None:
        super().__init__()
        self.image_count = 0
        self.images: list[bytes] = []
        self.ids: set[str] = set()
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Require embedded images and collect navigation destinations."""
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(str(attributes["id"]))
        if tag == "a" and str(attributes.get("href", "")).startswith("#"):
            self.anchors.add(str(attributes["href"])[1:])
        if tag == "img":
            source = str(attributes.get("src", ""))
            prefix = "data:image/png;base64,"
            assert source.startswith(prefix), "Charts must be embedded PNG images."
            assert attributes.get("alt"), "Every chart needs alternative text."
            payload = base64.b64decode(source[len(prefix) :], validate=True)
            validate_png(payload)
            self.images.append(payload)
            self.image_count += 1


def verify(output: Path, notebook_path: Path) -> None:
    """Recompute headline metrics and check that data and numerical code match the run."""
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    decisions = json.loads((output / "frozen_decisions.json").read_text(encoding="utf-8"))
    results = json.loads((output / "holdout_metrics.json").read_text(encoding="utf-8"))
    assert manifest["evaluation_status"] == "completed"
    data_path = root / manifest["config"]["dataset"]
    assert file_sha256(data_path) == manifest["dataset_sha256"]
    frame = load_dataset(data_path)
    for name, digest in manifest["numerical_source_sha256"].items():
        assert file_sha256(root / "src/air_quality" / name) == digest, f"Stale source: {name}"
    split = pd.read_csv(output / "tables/split_manifest.csv")
    assert split["row_id"].is_unique
    development = set(split.loc[split["partition"] == "development", "row_id"])
    test = set(split.loc[split["partition"] == "holdout", "row_id"])
    assert not development & test
    assert len(development) == manifest["development_rows"]
    assert len(test) == manifest["holdout_rows"]
    np.testing.assert_array_equal(frame.loc[split["row_id"], TARGET], split["original_class"])

    predictions = pd.read_csv(output / "tables/holdout_predictions.csv")
    assert predictions["row_id"].is_unique
    assert set(predictions["row_id"]) == test
    probabilities = predictions[[f"probability_{name}" for name in CLASS_NAMES]].to_numpy()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(
        probabilities[:, 2:].sum(axis=1), predictions["probability_binary_aggregated"], atol=1e-12
    )
    np.testing.assert_array_equal(predictions["true_multiclass"] >= 2, predictions["true_binary"])
    mapping = dict(zip(CLASS_NAMES, range(4), strict=True))
    expected = frame.loc[predictions["row_id"], TARGET].map(mapping)
    np.testing.assert_array_equal(expected, predictions["true_multiclass"])
    np.testing.assert_array_equal(probabilities.argmax(axis=1), predictions["predicted_multiclass"])
    for method, column, selected_threshold in (
        ("direct", "probability_binary_direct", decisions["direct_threshold"]["threshold"]),
        (
            "aggregated",
            "probability_binary_aggregated",
            decisions["aggregated_threshold"]["threshold"],
        ),
    ):
        for suffix, threshold in (("default", 0.5), ("f2", selected_threshold)):
            np.testing.assert_array_equal(
                (predictions[column] >= threshold).astype(int),
                predictions[f"predicted_binary_{method}_{suffix}"],
            )

    for name in (
        "multiclass",
        "binary_direct_default",
        "binary_aggregated_default",
        "binary_direct_f2",
        "binary_aggregated_f2",
    ):
        task = "multiclass" if name == "multiclass" else "binary"
        truth = predictions[f"true_{task}"].to_numpy()
        predicted = predictions[f"predicted_{name}"].to_numpy()
        result = results[name]
        np.testing.assert_allclose(
            f1_score(truth, predicted, average="macro"), result["macro_f1"], atol=1e-12
        )
        np.testing.assert_allclose(accuracy_score(truth, predicted), result["accuracy"], atol=1e-12)
        np.testing.assert_array_equal(
            confusion_matrix(truth, predicted), result["confusion_matrix"]
        )
        for interval in result["confidence_intervals"].values():
            assert 0 <= interval["low"] <= interval["high"] <= 1

    for task in ("multiclass", "binary"):
        table = pd.read_csv(output / "tables" / f"{task}_cv_summary.csv")
        tuned = table[table["procedure"] == "nested_tuned"].sort_values(
            ["macro_f1_mean", "model"],
            ascending=[False, True],
        )
        assert tuned.iloc[0]["model"] == decisions["models"][task]["name"]
        oof = pd.read_csv(output / "tables" / f"{task}_nested_oof.csv")
        assert oof["row_id"].is_unique and set(oof["row_id"]) == development
    assert datetime.fromisoformat(decisions["frozen_at"]) <= datetime.fromisoformat(
        manifest["finished_at"],
    )
    with notebook_path.open(encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    nbformat.validate(notebook)
    notebook_images = []
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None, "Notebook must contain executed code cells."
            assert not any(item.output_type == "error" for item in cell.outputs)
            for item in cell.outputs:
                if "image/png" in item.get("data", {}):
                    payload = base64.b64decode(item.data["image/png"], validate=True)
                    validate_png(payload)
                    notebook_images.append(payload)
    parser = ReportParser()
    parser.feed((output / "report.html").read_text(encoding="utf-8"))
    names = list(REQUIRED_FIGURES)
    if (output / "figures/learning_curve.png").exists():
        names.append("learning_curve")
    expected_images = [(output / "figures" / f"{name}.png").read_bytes() for name in names]
    assert Counter(parser.images) == Counter(expected_images), "HTML charts differ from exports."
    assert Counter(notebook_images) == Counter(expected_images), (
        "Notebook charts differ from exports."
    )
    assert parser.anchors <= parser.ids
    print(
        f"Verified: {len(split)} split rows, {len(predictions)} held-out predictions, "
        f"{parser.image_count} decoded PNG charts in HTML and notebook, matching source hashes."
    )


def main() -> None:
    """Verify saved results without fitting a model or recalculating the test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--notebook", type=Path, default=Path("notebooks/air_quality_study.ipynb"))
    args = parser.parse_args()
    verify(args.output, args.notebook)


if __name__ == "__main__":
    main()
