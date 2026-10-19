"""Protect portable chart exports against missing and undecodable report images."""

import base64
import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import pandas as pd
import pytest
from PIL import Image

from air_quality.reporting import _save_figure, build_notebook, build_report, execute_notebook_cells
from scripts.verify_outputs import ReportParser


@pytest.fixture
def legacy_text_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduce Windows text reads when UTF-8 mode is disabled."""
    original_read_text = Path.read_text

    def read_text(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        return original_read_text(path, encoding=encoding or "cp1250", errors=errors)

    monkeypatch.setattr(Path, "read_text", read_text)


def test_exported_chart_survives_html_and_notebook_embedding(
    tmp_path: Path, legacy_text_locale: None
) -> None:
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    axis.set(xlabel="Wilgotność", ylabel="Czułość", title="Jakość powietrza")
    _save_figure(figure, tmp_path, "example")
    path = tmp_path / "figures/example.png"
    payload = path.read_bytes()
    with Image.open(BytesIO(payload)) as image:
        assert image.format == "PNG"
        assert image.width > 500 and image.height > 300
    assert "Jakość powietrza" in (tmp_path / "figures/example.svg").read_text(encoding="utf-8")
    encoded = base64.b64encode(payload).decode()
    parser = ReportParser()
    parser.feed(f'<img src="data:image/png;base64,{encoded}" alt="Jakość powietrza">')
    assert parser.images == [payload]
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "from IPython.display import Image, display\n"
                f"display(Image(filename={str(path)!r}))"
            )
        ]
    )
    execute_notebook_cells(notebook)
    image_output = notebook.cells[0].outputs[0].data["image/png"]
    assert base64.b64decode(image_output) == payload


def test_generated_notebook_preserves_unicode_and_embeds_png_in_legacy_locale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_text_locale: None,
    dataset: pd.DataFrame,
) -> None:
    """Exercise generated cells with Polish UTF-8 data under a legacy system encoding."""
    (tmp_path / "pyproject.toml").touch()
    data_dir = tmp_path / "data/raw"
    data_dir.mkdir(parents=True)
    dataset.to_csv(data_dir / "pollution_dataset.csv", index=False, encoding="utf-8")
    output = tmp_path / "reports"
    (output / "figures").mkdir(parents=True)
    phrase = "Zażółć gęślą jaźń — ŁÓDŹ"
    manifest = {
        "config": {"profile": "smoke", "note": phrase},
        "evaluation_status": "completed",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    Image.new("RGB", (20, 10), "white").save(output / "figures/example.png")
    monkeypatch.chdir(tmp_path)
    notebook_path = build_notebook(output, execute=False)
    with notebook_path.open(encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    assert "Klasyfikacja jakości powietrza" in notebook.cells[0].source
    # Run the actual generated setup and image helper without training any models.
    notebook.cells = [
        notebook.cells[1],
        nbformat.v4.new_code_cell("display(manifest['config']['note'])\nshow_figure('example')"),
    ]
    execute_notebook_cells(notebook)
    outputs = notebook.cells[1].outputs
    assert phrase in outputs[0].data["text/plain"]
    assert outputs[1].metadata["image/png"]["width"] == 1000
    assert "image/svg+xml" not in outputs[1].data
    assert (
        base64.b64decode(outputs[1].data["image/png"], validate=True)
        == (output / "figures/example.png").read_bytes()
    )


@pytest.mark.parametrize(
    "image",
    [
        '<img src="figures/missing.png" alt="Wykres">',
        '<img src="data:image/svg+xml;base64,PHN2Zy8+" alt="Wykres">',
        '<img src="data:image/png;base64,aW52YWxpZA==" alt="Wykres">',
        '<img src="data:image/png;base64,invalid!" alt="Wykres">',
    ],
)
def test_report_rejects_missing_or_broken_images(image):
    with pytest.raises((AssertionError, ValueError, OSError)):
        ReportParser().feed(image)


def test_report_generation_stops_when_a_required_figure_is_missing(tmp_path):
    for name in ("data_audit", "holdout_metrics", "frozen_decisions", "run_manifest"):
        (tmp_path / f"{name}.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="Missing PNG figures"):
        build_report(tmp_path)
