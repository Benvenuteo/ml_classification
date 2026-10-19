"""Standalone scientific figures, a self-contained HTML report and an executed notebook."""

from __future__ import annotations

import base64
import json
import os
import platform
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from sklearn.calibration import calibration_curve
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import precision_recall_curve

from air_quality.data import CLASS_NAMES, FEATURES
from air_quality.models import DISPLAY_NAMES

COLORS = ("#17846b", "#c09a21", "#d27935", "#bd4453")
BLUE, ORANGE, INK = "#286a93", "#c07035", "#172b41"
CLASS_LABELS = ("Dobra", "Umiarkowana", "Zła", "Niebezpieczna")
FEATURE_LABELS = {
    "Temperature": "Temperatura",
    "Humidity": "Wilgotność",
    "PM2.5": "PM2,5",
    "PM10": "PM10",
    "NO2": "NO₂",
    "SO2": "SO₂",
    "CO": "CO",
    "Proximity_to_Industrial_Areas": "Odległość od przemysłu",
    "Population_Density": "Gęstość zaludnienia",
}
MODEL_LABELS = {
    **DISPLAY_NAMES,
    "knn": "k najbliższych sąsiadów",
    "svm": "SVM z jądrem RBF",
    "random_forest": "Las losowy",
    "hist_boosting": "Boosting histogramowy",
}
TREATMENT_LABELS = {
    "none": "Bez równoważenia",
    "class_weights": "Wagi klas",
    "smote": "Próbkowanie SMOTE",
    "all_features": "Wszystkie cechy",
    "pollutants_only": "Tylko zanieczyszczenia",
    "context_only": "Tylko cechy kontekstowe",
    "without_CO": "Bez CO",
    "without_PM2.5": "Bez PM2,5",
    "cluster_distances": "Cechy i odległości do klastrów",
    "raw_values": "Dane surowe",
    "domain_mask": "Maskowanie błędnych wartości",
}
REQUIRED_FIGURES = (
    "data_quality",
    "development_distributions",
    "correlation",
    "nested_cv",
    "multiclass_confusion",
    "binary_confusion",
    "probability_quality",
    "ablations",
    "permutation_importance",
    "partial_dependence",
)


def _save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    """Export raster previews with fixed typography and editable vector originals."""
    (output / "figures").mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output / "figures" / f"{name}.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    path = output / "figures" / f"{name}.svg"
    figure.savefig(
        path,
        bbox_inches="tight",
        metadata={"Date": None},
        facecolor="white",
    )
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(figure)


def _confusion_axis(axis: plt.Axes, matrix: list[list[int]], labels: tuple[str, ...]) -> None:
    """Annotate counts with row-normalized shading for unequal class sizes."""
    values = np.asarray(matrix)
    normalized = values / np.maximum(values.sum(axis=1, keepdims=True), 1)
    axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(
                column,
                row,
                f"{values[row, column]}\n{normalized[row, column]:.1%}",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if normalized[row, column] > 0.55 else INK,
            )
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Klasa przewidziana")
    axis.set_ylabel("Klasa rzeczywista")
    axis.grid(False)


def create_figures(
    development: pd.DataFrame,
    audit: dict,
    results: dict,
    selected: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    """Create all views from computed data; EDA uses development records only."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "air-quality-study-v1",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 3.8), layout="constrained")
    counts = audit["class_counts"]
    axes[0].bar(CLASS_LABELS, [counts[name] for name in CLASS_NAMES], color=COLORS, width=0.65)
    for index, name in enumerate(CLASS_NAMES):
        axes[0].text(index, counts[name] + 45, f"{counts[name]:,}".replace(",", " "), ha="center")
    axes[0].set_ylim(0, max(counts.values()) * 1.16)
    axes[0].set_title("Oryginalny rozkład klas")
    axes[0].set_ylabel("Liczba rekordów")
    rates = [row["issue_rate"] * 100 for row in audit["issues_by_class"]]
    axes[1].bar(CLASS_LABELS, rates, color=COLORS, width=0.65)
    for index, rate in enumerate(rates):
        axes[1].text(index, rate + 0.5, f"{rate:.1f}%", ha="center")
    axes[1].set_ylim(0, max(rates) * 1.2)
    axes[1].set_ylabel("Rekordy z flagą jakości (%)")
    axes[1].set_title("Ryzyko selektywnego usuwania danych")
    _save_figure(figure, output, "data_quality")

    units = ("°C", "%", "µg/m³", "µg/m³", "ppb", "ppb", "ppm", "km", "osób/km²")
    figure, axes = plt.subplots(3, 3, figsize=(11.5, 8.2), layout="constrained")
    for axis, column, unit in zip(axes.flat, FEATURES, units, strict=True):
        axis.hist(development[column], bins=35, color=BLUE, edgecolor="white", linewidth=0.4)
        axis.set_title(FEATURE_LABELS[column], fontsize=11)
        axis.set_xlabel(unit)
        axis.set_ylabel("Liczba rekordów")
    _save_figure(figure, output, "development_distributions")

    correlation = development.loc[:, FEATURES].corr(method="spearman")
    short_names = (
        "Temp.",
        "Wilg.",
        "PM2,5",
        "PM10",
        "NO₂",
        "SO₂",
        "CO",
        "Odległość",
        "Zaludnienie",
    )
    figure, axis = plt.subplots(figsize=(8.8, 6.5), layout="constrained")
    artist = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(9), short_names, rotation=35, ha="right")
    axis.set_yticks(range(9), short_names)
    for row in range(9):
        for column in range(9):
            value = correlation.iloc[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.65 else INK,
            )
    figure.colorbar(artist, ax=axis, shrink=0.8, label="Korelacja Spearmana")
    axis.set_title("Zależności między cechami — część treningowa")
    _save_figure(figure, output, "correlation")

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), layout="constrained")
    for axis, task, title in zip(
        axes, ("multiclass", "binary"), ("Cztery klasy", "Dwie klasy"), strict=True
    ):
        table = pd.read_csv(output / "tables" / f"{task}_cv_summary.csv")
        table = table[table["procedure"] == "nested_tuned"].sort_values("macro_f1_mean")
        y = np.arange(len(table))
        axis.errorbar(
            table["macro_f1_mean"], y, xerr=table["macro_f1_std"], fmt="o", color=BLUE, capsize=4
        )
        axis.set_yticks(y, [MODEL_LABELS[name] for name in table["model"]])
        left = max(0, float((table["macro_f1_mean"] - table["macro_f1_std"]).min()) - 0.025)
        axis.set_xlim(left, 1.005)
        axis.set_xlabel("F1 macro (średnia ± odchylenie standardowe)", fontsize=9)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    _save_figure(figure, output, "nested_cv")

    figure, axis = plt.subplots(figsize=(6.8, 5.5), layout="constrained")
    _confusion_axis(axis, results["multiclass"]["confusion_matrix"], CLASS_LABELS)
    axis.set_title("Cztery klasy — końcowy zbiór testowy")
    _save_figure(figure, output, "multiclass_confusion")

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), layout="constrained")
    for axis, key, title in zip(
        axes,
        ("binary_direct_default", "binary_aggregated_default"),
        ("Model binarny", "Agregacja modelu czteroklasowego"),
        strict=True,
    ):
        _confusion_axis(
            axis, results[key]["confusion_matrix"], ("Dobra / umiarkowana", "Zła / niebezpieczna")
        )
        axis.set_title(title, fontsize=12)
    _save_figure(figure, output, "binary_confusion")

    predictions = pd.read_csv(output / "tables" / "holdout_predictions.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), layout="constrained")
    for column, label, color in (
        ("probability_binary_direct", "Model binarny", BLUE),
        ("probability_binary_aggregated", "Agregacja czterech klas", ORANGE),
    ):
        values = predictions[column].to_numpy()
        precision, recall, _ = precision_recall_curve(predictions["true_binary"], values)
        axes[0].plot(recall, precision, label=label, color=color, linewidth=2)
        fraction, average = calibration_curve(
            predictions["true_binary"], values, n_bins=8, strategy="quantile"
        )
        axes[1].plot(average, fraction, "o-", label=label, color=color, linewidth=1.6)
    axes[0].axhline(
        predictions["true_binary"].mean(),
        color="#8291a0",
        linestyle="--",
        label="Częstość klasy dodatniej",
    )
    axes[0].set(
        xlabel="Czułość (recall)", ylabel="Precyzja (precision)", title="Precyzja i czułość"
    )
    axes[1].plot([0, 1], [0, 1], color="#8291a0", linestyle="--")
    axes[1].set(
        xlabel="Średnie przewidywane prawdopodobieństwo",
        ylabel="Obserwowany udział klasy dodatniej",
        title="Kalibracja prawdopodobieństw",
    )
    for axis in axes:
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.03)
        axis.legend(fontsize=8, loc="lower left")
        axis.grid(alpha=0.15)
    _save_figure(figure, output, "probability_quality")

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), layout="constrained")
    for axis, task, title in zip(
        axes, ("multiclass", "binary"), ("Cztery klasy", "Dwie klasy"), strict=True
    ):
        table = pd.read_csv(output / "tables" / f"{task}_ablation_summary.csv")
        table["label"] = table["treatment"].map(TREATMENT_LABELS)
        y = np.arange(len(table))
        axis.errorbar(
            table["macro_f1_mean"], y, xerr=table["macro_f1_std"], fmt="o", color=BLUE, capsize=3
        )
        axis.set_yticks(y, table["label"], fontsize=9)
        for position in (2.5, 8.5):
            axis.axhline(position, color="#dce3eb", linewidth=1)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("F1 macro (średnia ± odchylenie standardowe)", fontsize=9)
        axis.grid(axis="x", alpha=0.2)
    _save_figure(figure, output, "ablations")

    table = pd.read_csv(output / "tables" / "permutation_importance.csv").sort_values(
        "importance_mean"
    )
    figure, axis = plt.subplots(figsize=(10.5, 4.8), layout="constrained")
    axis.barh(
        table["feature"].map(FEATURE_LABELS),
        table["importance_mean"],
        color=BLUE,
        height=0.65,
        xerr=table["importance_std"],
        error_kw={"ecolor": INK, "capsize": 3},
    )
    axis.set_xlabel("Spadek F1 macro po permutacji (średnia ± odchylenie standardowe)")
    axis.set_title("Znaczenie cech dla wybranego modelu czteroklasowego")
    _save_figure(figure, output, "permutation_importance")

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), layout="constrained")
    PartialDependenceDisplay.from_estimator(
        selected["multiclass"]["estimator"],
        development.loc[:, FEATURES],
        features=["CO", "Proximity_to_Industrial_Areas"],
        target=3,
        grid_resolution=25,
        method="brute",
        ax=axes,
        line_kw={"color": BLUE, "linewidth": 2},
    )
    for axis in axes:
        axis.set_ylabel("Średnie prawdopodobieństwo\nklasy niebezpiecznej")
    axes[0].set(xlabel="CO (ppm)", title="Zależność od stężenia CO")
    axes[1].set(xlabel="Odległość od przemysłu (km)", title="Zależność od odległości")
    _save_figure(figure, output, "partial_dependence")

    curve_path = output / "tables" / "learning_curve.csv"
    if curve_path.exists():
        curve = pd.read_csv(curve_path)
        figure, axis = plt.subplots(figsize=(8, 3.9), layout="constrained")
        axis.plot(curve["training_rows"], curve["train_mean"], "o-", color=ORANGE, label="Trening")
        axis.plot(
            curve["training_rows"], curve["validation_mean"], "o-", color=BLUE, label="Walidacja"
        )
        axis.fill_between(
            curve["training_rows"],
            curve["validation_mean"] - curve["validation_std"],
            curve["validation_mean"] + curve["validation_std"],
            color=BLUE,
            alpha=0.14,
        )
        axis.set(
            xlabel="Liczba rekordów w części treningowej podziału walidacyjnego",
            ylabel="F1 macro",
            title="Krzywa uczenia wybranej konfiguracji",
        )
        axis.legend()
        axis.grid(alpha=0.15)
        _save_figure(figure, output, "learning_curve")


def _html_table(frame: pd.DataFrame, *, digits: int = 4) -> str:
    """Format report tables while escaping source labels and preserving numeric precision."""
    return frame.to_html(
        index=False,
        border=0,
        classes="data-table",
        na_rep="—",
        float_format=lambda value: f"{value:.{digits}f}",
        escape=True,
    )


def empirical_findings(output: Path, results: dict) -> list[str]:
    """Summarize actual development comparisons without overstating their scope."""
    ablations = pd.read_csv(output / "tables/multiclass_ablation_summary.csv")

    def score(experiment: str, treatment: str) -> float:
        selected = ablations[
            ablations["experiment"].eq(experiment) & ablations["treatment"].eq(treatment)
        ]
        return float(selected.iloc[0]["macro_f1_mean"])

    plain = score("balancing", "none")
    smote = score("balancing", "smote")
    weights = score("balancing", "class_weights")
    all_features = score("features", "all_features")
    context = score("features", "context_only")
    pollutants = score("features", "pollutants_only")
    without_co = score("features", "without_CO")
    raw = score("cleaning", "raw_values")
    cleaned = score("cleaning", "domain_mask")
    clusters = score("features", "cluster_distances")
    low, high = results["paired_binary_comparison"]["paired_95ci"]
    if low <= 0 <= high:
        binary_conclusion = (
            "Przedział różnicy binarnego macro-F1 obejmuje zero. "
            "Ten test nie rozstrzyga przewagi modelu bezpośredniego lub agregacji "
            "i nie dowodzi ich równoważności."
        )
    else:
        better = "model bezpośredni" if low > 0 else "agregacja czterech klas"
        binary_conclusion = (
            f"Przedział różnicy binarnego macro-F1 nie obejmuje zera; lepszy wynik "
            f"na tym teście uzyskuje {better}. Wniosek dotyczy ustalonych modeli i tego zbioru."
        )
    return [
        f"Cztery klasy: testowe macro-F1 = {results['multiclass']['macro_f1']:.4f}. "
        "Nie ma empirycznej konieczności redukowania celu do dwóch klas w tym zbiorze.",
        "Recall klasy Hazardous wynosi "
        f"{results['multiclass']['classification_report']['3']['recall']:.1%} "
        f"przy {int(results['multiclass']['classification_report']['3']['support'])} "
        "obserwacjach testowych tej klasy. Wynik globalny nie zastępuje analizy "
        "pomyłek dotyczących najrzadszej kategorii.",
        f"Równoważenie w regresji logistycznej: bez równoważenia {plain:.4f}, "
        f"SMOTE {smote:.4f} (różnica {smote - plain:+.4f}), "
        f"wagi klas {weights:.4f} (różnica {weights - plain:+.4f}). "
        "To średnie development, zależne od modelu referencyjnego.",
        f"Model referencyjny z samym kontekstem uzyskuje {context:.4f}, "
        f"z samymi zanieczyszczeniami {pollutants:.4f}, ze wszystkimi cechami {all_features:.4f}. "
        f"Usunięcie CO zmienia wynik o {without_co - all_features:+.4f}. "
        "Sygnał kontekstu wymaga ostrożności wobec nieudokumentowanego sposobu tworzenia etykiet.",
        f"Maskowanie wartości zamiast zachowania surowych zmienia macro-F1 "
        f"modelu referencyjnego o {cleaned - raw:+.4f}. "
        f"Dodanie odległości do klastrów zmienia je o {clusters - all_features:+.4f}. "
        "Małe różnice opisowe nie stanowią samodzielnego dowodu istotnej poprawy.",
        binary_conclusion,
    ]


def build_report(output: Path) -> Path:
    """Render a self-contained report entirely from saved experiment results."""
    audit = json.loads((output / "data_audit.json").read_text(encoding="utf-8"))
    results = json.loads((output / "holdout_metrics.json").read_text(encoding="utf-8"))
    decisions = json.loads((output / "frozen_decisions.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    figures = {
        path.stem: "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
        for path in sorted((output / "figures").glob("*.png"))
    }
    missing = set(REQUIRED_FIGURES) - figures.keys()
    if missing:
        raise FileNotFoundError(f"Missing PNG figures: {', '.join(sorted(missing))}")
    model_tables, ablation_tables = {}, {}
    for task in ("multiclass", "binary"):
        cv = pd.read_csv(output / "tables" / f"{task}_cv_summary.csv")
        cv["model"] = cv["model"].map(MODEL_LABELS)
        cv["procedure"] = cv["procedure"].map(
            {"baseline": "Ustawienia bazowe", "nested_tuned": "Strojenie zagnieżdżone"}
        )
        model_tables[task] = _html_table(
            cv.rename(
                columns={
                    "model": "Model",
                    "procedure": "Procedura",
                    "macro_f1_mean": "F1 macro",
                    "macro_f1_std": "Odchylenie standardowe",
                    "accuracy_mean": "Trafność",
                    "balanced_accuracy_mean": "Trafność zrównoważona",
                    "train_macro_f1_mean": "F1 trening",
                    "fit_seconds_mean": "Czas / podział (s)",
                }
            )
        )
        ablation = pd.read_csv(output / "tables" / f"{task}_ablation_summary.csv")
        ablation["experiment"] = ablation["experiment"].map(
            {"balancing": "Równoważenie", "features": "Cechy", "cleaning": "Czyszczenie"}
        )
        ablation["treatment"] = ablation["treatment"].map(TREATMENT_LABELS)
        ablation["reference_model"] = ablation["reference_model"].map(MODEL_LABELS)
        ablation_tables[task] = _html_table(
            ablation.rename(
                columns={
                    "experiment": "Eksperyment",
                    "treatment": "Wariant",
                    "reference_model": "Model referencyjny",
                    "macro_f1_mean": "F1 macro",
                    "macro_f1_std": "Odchylenie standardowe",
                    "accuracy_mean": "Trafność",
                    "balanced_accuracy_mean": "Trafność zrównoważona",
                    "positive_recall_mean": "Czułość klasy dodatniej",
                }
            )
        )
    test_rows = []
    descriptions = {
        "multiclass": "Cztery klasy",
        "binary_direct_default": "Binarny, próg 0,50",
        "binary_aggregated_default": "Agregacja czterech klas, próg 0,50",
        "binary_direct_f2": f"Binarny, próg F2 = {decisions['direct_threshold']['threshold']:.2f}",
        "binary_aggregated_f2": (
            f"Agregacja, próg F2 = {decisions['aggregated_threshold']['threshold']:.2f}"
        ),
    }
    for key, description in descriptions.items():
        result = results[key]
        interval = result["confidence_intervals"]["macro_f1"]
        test_rows.append(
            {
                "Zadanie / procedura": description,
                "Macro-F1": result["macro_f1"],
                "95% CI macro-F1": f"[{interval['low']:.4f}; {interval['high']:.4f}]",
                "Accuracy": result["accuracy"],
                "Balanced accuracy": result["balanced_accuracy"],
                "Recall Poor/Hazardous": result.get("positive_recall", np.nan),
            }
        )
    class_report = pd.DataFrame(results["multiclass"]["classification_report"]).T.iloc[:4].copy()
    class_report["support"] = class_report["support"].astype(int)
    class_report.insert(0, "Klasa źródłowa", CLASS_NAMES)
    class_report.insert(1, "Nazwa na wykresie", CLASS_LABELS)
    class_report.rename(
        columns={
            "precision": "Precyzja",
            "recall": "Czułość",
            "f1-score": "F1",
            "support": "Liczba rekordów",
        },
        inplace=True,
    )
    binary_rows = []
    for key in ("binary_direct_default", "binary_aggregated_default"):
        result = results[key]
        binary_rows.append(
            {
                "Procedura": descriptions[key],
                **{
                    name: result[name]
                    for name in (
                        "average_precision",
                        "roc_auc",
                        "log_loss",
                        "brier",
                        "positive_precision",
                        "positive_recall",
                    )
                },
            }
        )
    issues = pd.DataFrame(audit["issues_by_class"])
    issues["issue_rate"] = issues["issue_rate"].map(lambda value: f"{value:.1%}")
    issues.rename(
        columns={
            "class": "Klasa źródłowa",
            "rows": "Liczba rekordów",
            "any_issue_rows": "Rekordy z dowolną flagą",
            "hard_issue_rows": "Rekordy z błędną wartością",
            "issue_rate": "Udział z dowolną flagą",
        },
        inplace=True,
    )
    issue_labels = {
        **{f"{name}_negative": f"{label} < 0" for name, label in FEATURE_LABELS.items()},
        "Humidity_above_100": "Wilgotność powyżej 100%",
        "Humidity_below_0": "Wilgotność poniżej 0%",
        "PM2.5_above_PM10": "PM2,5 powyżej PM10",
    }
    environment = Environment(
        loader=PackageLoader("air_quality", "templates"),
        autoescape=select_autoescape(["html"]),
        undefined=StrictUndefined,
    )
    template = environment.get_template("report.html")
    content = template.render(
        audit=audit,
        results=results,
        decisions=decisions,
        manifest=manifest,
        figures=figures,
        model_names=MODEL_LABELS,
        model_tables=model_tables,
        ablation_tables=ablation_tables,
        test_table=_html_table(pd.DataFrame(test_rows)),
        class_table=_html_table(class_report),
        probability_table=_html_table(
            pd.DataFrame(binary_rows).rename(
                columns={
                    "average_precision": "Średnia precyzja (AP)",
                    "roc_auc": "Pole pod krzywą ROC",
                    "log_loss": "Strata logarytmiczna",
                    "brier": "Wynik Briera",
                    "positive_precision": "Precyzja klasy dodatniej",
                    "positive_recall": "Czułość klasy dodatniej",
                }
            )
        ),
        issue_table=_html_table(issues, digits=0),
        issue_counts=_html_table(
            pd.DataFrame(
                {
                    "Reguła": [issue_labels[name] for name in audit["issue_counts"]],
                    "Liczba": audit["issue_counts"].values(),
                }
            )
        ),
        versions=_html_table(
            pd.DataFrame(manifest["environment"]["versions"].items(), columns=["Pakiet", "Wersja"])
        ),
        config_json=json.dumps(manifest["config"], indent=2, ensure_ascii=False),
        decision_json=json.dumps(decisions["models"], indent=2, ensure_ascii=False),
        findings=empirical_findings(output, results),
    )
    path = output / "report.html"
    path.write_text(
        "\n".join(line.rstrip() for line in content.splitlines()) + "\n", encoding="utf-8"
    )
    findings = empirical_findings(output, results)
    (output / "study_summary.md").write_text(
        "# Wyniki i wnioski\n\n"
        + "\n\n".join(f"- {item}" for item in findings)
        + "\n\nSzczegóły: report.html, holdout_metrics.json oraz tables/. "
        "Wyniki dotyczą klasyfikacji etykiet w tym zbiorze, bez walidacji zewnętrznej.\n",
        encoding="utf-8",
    )
    return path


def build_notebook(output: Path, *, execute: bool) -> Path:
    """Build and optionally execute a portable notebook reading the saved study outputs."""
    root = Path.cwd().resolve()
    relative_output = Path(os.path.relpath(output, root)).as_posix()
    full_profile = (
        json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["config"]["profile"]
        == "full"
    )
    path = (
        root / "notebooks" / "air_quality_study.ipynb" if full_profile else output / "smoke.ipynb"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        markdown(
            "# Klasyfikacja jakości powietrza\n\n"
            "Analiza czterech klas oraz binarnego grupowania etykiet. "
            "Notebook korzysta z wyników zapisanych przez kompletny eksperyment.\n\n"
            "Wykresy PNG są osadzone w pliku i zachowują polskie znaki również "
            "w podglądzie GitHuba, bez uruchamiania komórek.\n\n"
            "Pełne przeliczenie: `python -m air_quality run --config configs/study.json`. "
            "Nie dobieramy modeli na zbiorze testowym."
        ),
        code(
            "import json\nfrom pathlib import Path\nimport pandas as pd\n"
            "from IPython.display import Image, display\n"
            "from air_quality.data import load_dataset, audit_dataset\n\n"
            "ROOT = next(path for path in (Path.cwd(), *Path.cwd().parents) "
            "if (path / 'pyproject.toml').exists())\n"
            f"OUTPUT = ROOT / {relative_output!r}\n"
            "data = load_dataset(ROOT / 'data/raw/pollution_dataset.csv')\n"
            "audit = audit_dataset(data)\n"
            "manifest = json.loads((OUTPUT / 'run_manifest.json').read_text(encoding='utf-8'))\n"
            "assert manifest['evaluation_status'] == 'completed'\n"
            "\n"
            "def show_figure(name: str) -> None:\n"
            '    """Embed a PNG with fixed typography and a readable notebook width."""\n'
            "    display(Image(filename=str(OUTPUT / 'figures' / f'{name}.png'),\n"
            "                  embed=True, width=1000))\n"
            "\n"
            "display(data.head())"
        ),
        markdown(
            "## Dane i zakres wnioskowania\n\n"
            "Źródło: Mujtaba Mateen, Kaggle, wersja 2, Apache 2.0. "
            "Dokumentacja nie podaje stacji, dat, procedury zbierania danych "
            "ani reguły etykietowania. "
            "Nie potwierdza pochodzenia z WHO/Banku Światowego ani rzeczywistego lub "
            "syntetycznego charakteru obserwacji."
        ),
        code("display(pd.DataFrame(audit['issues_by_class']))\nshow_figure('data_quality')"),
        markdown(
            "## Protokół oceny\n\n"
            "Wspólny podział stratyfikowany według czterech klas, 80% development / 20% test. "
            "Pipeline: reguły domenowe, imputacja medianą, skalowanie, opcjonalny SMOTE, model. "
            "Przetwarzanie jest dopasowywane osobno w każdym foldzie. "
            "Klasa dodatnia w zadaniu binarnym: Poor lub Hazardous."
        ),
        code(
            "display(pd.DataFrame(manifest['config'].items(), columns=['Setting', 'Value']))\n"
            "split = pd.read_csv(OUTPUT / 'tables/split_manifest.csv')\n"
            "display(pd.crosstab(split['original_class'], split['partition']))"
        ),
        markdown(
            "## EDA części treningowej\n\n"
            "Pełny zbiór służy wyłącznie audytowi schematu i jakości. "
            "Poniższe rozkłady i korelacje dotyczą części development."
        ),
        code("show_figure('development_distributions')\nshow_figure('correlation')"),
        markdown(
            "## Porównanie modeli\n\n"
            "Wyniki `nested_tuned` pochodzą z zewnętrznych foldów, a hiperparametry "
            "wybierano w wewnętrznej walidacji. SD foldów nie jest przedziałem ufności. "
            "Wybór najlepszej rodziny na podstawie tych wyników wprowadza dodatkową selekcję."
        ),
        code(
            "for task in ('multiclass', 'binary'):\n"
            "    print(task)\n"
            "    display(pd.read_csv(OUTPUT / 'tables' / f'{task}_cv_summary.csv'))\n"
            "show_figure('nested_cv')"
        ),
        markdown(
            "## Eksperymenty wrażliwości\n\n"
            "Równoważenie porównano dla regresji logistycznej. Grupy cech, czyszczenie "
            "i odległości do klastrów porównano dla ustalonego Histogram Gradient Boosting. "
            "Eksperymenty opisowe nie zmieniają głównej procedury wyboru modelu."
        ),
        code(
            "for task in ('multiclass', 'binary'):\n"
            "    display(pd.read_csv(OUTPUT / 'tables' / f'{task}_ablation_summary.csv'))\n"
            "show_figure('ablations')"
        ),
        markdown(
            "## Zamrożone decyzje i ocena końcowa\n\n"
            "Modele i progi F2 wybrano przed oceną testową. Porównujemy model binarny "
            "z sumą P(Poor) + P(Hazardous), stosując wspólny cel binarny. "
            "Przedziały bootstrap dotyczą niepewności próby testowej przy stałych modelach."
        ),
        code(
            "decisions = json.loads(\n"
            "    (OUTPUT / 'frozen_decisions.json').read_text(encoding='utf-8'))\n"
            "results = json.loads((OUTPUT / 'holdout_metrics.json').read_text(encoding='utf-8'))\n"
            "display(decisions['models'])\n"
            "rows = [{'procedure': name, **{key: value for key, value in result.items() "
            "if isinstance(value, (float, int))}} for name, result in results.items() "
            "if name != 'paired_binary_comparison']\n"
            "display(pd.DataFrame(rows))\n"
            "show_figure('multiclass_confusion')\n"
            "show_figure('binary_confusion')\n"
            "show_figure('probability_quality')\n"
            "display(results['paired_binary_comparison'])"
        ),
        markdown(
            "## Interpretacja i ograniczenia\n\n"
            "Permutation importance opisuje wrażliwość predykcji, a nie przyczynowość. "
            "Skorelowane cechy mogą dzielić znaczenie. PDP może tworzyć nierealistyczne "
            "kombinacje cech. Jakość na tym zbiorze nie dowodzi skuteczności w innych "
            "miastach, okresach ani zastosowaniach zdrowotnych."
        ),
        code(
            "display(pd.read_csv(OUTPUT / 'tables/permutation_importance.csv'))\n"
            "show_figure('permutation_importance')\n"
            "show_figure('partial_dependence')\n"
            "curve = OUTPUT / 'figures/learning_curve.png'\n"
            "if curve.exists():\n    show_figure('learning_curve')"
        ),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": platform.python_version()},
        },
    )
    if execute:
        execute_notebook_cells(notebook)
    nbformat.validate(notebook)
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return path


def execute_notebook_cells(notebook: Any) -> None:
    """Execute real IPython cells and capture their rich outputs without socket access.

    The CLI creates a fresh process, so this shell has no inherited notebook state.
    Generated notebooks retain a normal Python kernelspec and can also be rerun
    in Jupyter or VS Code. Execution errors abort delivery instead of producing
    a notebook with invented output or execution counts.
    """
    shell = InteractiveShell.instance(user_ns={})
    notebook.metadata["execution_backend"] = "IPython in-process"
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
        if result.error_before_exec is not None:
            raise result.error_before_exec
        if result.error_in_exec is not None:
            raise result.error_in_exec
        cell.execution_count = result.execution_count
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for item in captured.outputs:
            outputs.append(
                nbformat.v4.new_output(
                    "display_data",
                    data=item.data,
                    metadata=item.metadata,
                )
            )
        cell.outputs = outputs
