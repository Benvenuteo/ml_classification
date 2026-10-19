"""Command-line entry points for analysis, report rebuilding and inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import joblib
import pandas as pd

from air_quality.config import load_config
from air_quality.data import CLASS_NAMES, validate_features
from air_quality.evaluation import aggregate_elevated_probability


def main() -> None:
    """Dispatch explicit commands; report-only never retrains or reselects a model."""
    parser = argparse.ArgumentParser(description="Reproducible air quality classification")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the complete fixed study")
    run.add_argument("--config", default="configs/study.json")
    run.add_argument("--output", help="Override the output directory")
    report = commands.add_parser("report", help="Rebuild the HTML and notebook from saved results")
    report.add_argument("--output", default="reports")
    report.add_argument("--execute-notebook", action="store_true")
    predict = commands.add_parser("predict", help="Predict from nine named features after training")
    predict.add_argument("--input", required=True)
    predict.add_argument("--model", default="reports/models/multiclass.joblib")
    predict.add_argument("--output", help="Optional output CSV; otherwise print JSON")
    args = parser.parse_args()
    if args.command == "run":
        from air_quality.study import run_study

        config = load_config(args.config)
        if args.output:
            config = replace(config, output_dir=args.output)
        run_study(config)
    elif args.command == "report":
        from air_quality.reporting import build_notebook, build_report

        output = Path(args.output).resolve()
        print(build_report(output))
        print(build_notebook(output, execute=args.execute_notebook))
    else:
        frame = validate_features(pd.read_csv(args.input))
        estimator = joblib.load(args.model)
        probabilities = estimator.predict_proba(frame)
        if probabilities.shape[1] != 4:
            parser.error("Inference expects a four-class model in canonical class order.")
        result = pd.DataFrame(
            probabilities, columns=[f"probability_{name}" for name in CLASS_NAMES]
        )
        result.insert(
            0, "predicted_class", [CLASS_NAMES[index] for index in probabilities.argmax(axis=1)]
        )
        result["probability_elevated"] = aggregate_elevated_probability(probabilities)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(path, index=False)
        else:
            print(json.dumps(result.to_dict(orient="records"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
