from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTRA_DEPS = ROOT / ".codex_competition_pydeps"

if EXTRA_DEPS.exists():
    sys.path.insert(0, str(EXTRA_DEPS))

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


COMPETITIONS = {
    "linear": {
        "folder": ROOT / "13_linear_regression" / "competition",
        "baseline_notebook": "simple_baseline.ipynb",
        "author_notebook": "author_solution.ipynb",
        "baseline_submission": "submission.csv",
        "author_submission": "author_submission.csv",
        "main_metric": "MAE",
        "metric_direction": "меньше — лучше",
    },
    "logistic": {
        "folder": ROOT / "14_logistic_regression" / "competition",
        "baseline_notebook": "simple_baseline.ipynb",
        "author_notebook": "author_solution.ipynb",
        "baseline_submission": "submission.csv",
        "author_submission": "author_submission.csv",
        "main_metric": "F1",
        "metric_direction": "больше — лучше",
    },
    "ensemble": {
        "folder": ROOT / "final_ensemble_competition",
        "baseline_notebook": "simple_baseline.ipynb",
        "author_notebook": "author_solution.ipynb",
        "baseline_submission": "submission.csv",
        "author_submission": "author_submission.csv",
        "main_metric": "ROC_AUC",
        "metric_direction": "больше — лучше",
    },
}


def display(*objects, **_: object) -> None:
    for obj in objects:
        print(obj)


def run_notebook_in_temp(folder: Path, notebook_name: str, submission_name: str) -> pd.DataFrame:
    notebook_path = folder / notebook_name
    if not notebook_path.exists():
        raise FileNotFoundError(notebook_path)

    with tempfile.TemporaryDirectory(prefix="ml_speckurs_score_") as tmp_name:
        tmp = Path(tmp_name)
        shutil.copytree(folder / "data", tmp / "data")

        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        global_ns: dict[str, object] = {
            "__name__": "__main__",
            "display": display,
        }

        old_cwd = Path.cwd()
        try:
            # Notebook code uses relative DATA_DIR = Path("data").
            import os

            os.chdir(tmp)
            for index, cell in enumerate(nb["cells"]):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                if not source.strip():
                    continue
                try:
                    exec(compile(source, f"{notebook_path}:{index}", "exec"), global_ns)
                except Exception as exc:
                    raise RuntimeError(f"Ошибка в {notebook_path}, code cell {index}") from exc

            submission_path = tmp / submission_name
            if not submission_path.exists():
                raise FileNotFoundError(f"{notebook_path} did not create {submission_name}")
            return pd.read_csv(submission_path)
        finally:
            import os

            os.chdir(old_cwd)


def score_linear(folder: Path, submission: pd.DataFrame) -> dict[str, float]:
    private = pd.read_csv(folder / "data" / "private_target.csv")
    merged = private.merge(submission, on="id", how="inner", suffixes=("_true", "_pred"))
    if len(merged) != len(private):
        raise ValueError(f"linear submission has {len(merged)} ids, expected {len(private)}")

    y_true = merged["price_mln_true"]
    y_pred = merged["price_mln_pred"]
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def score_logistic(folder: Path, submission: pd.DataFrame) -> dict[str, float]:
    private = pd.read_csv(folder / "data" / "private_target.csv")
    merged = private.merge(submission, on="id", how="inner", suffixes=("_true", "_pred"))
    if len(merged) != len(private):
        raise ValueError(f"logistic submission has {len(merged)} ids, expected {len(private)}")

    y_true = merged["will_finish_true"]
    if "will_finish_pred" in merged.columns:
        y_pred = merged["will_finish_pred"].astype(int)
    elif "probability" in merged.columns:
        y_pred = (merged["probability"] >= 0.5).astype(int)
    else:
        raise ValueError("logistic submission must contain will_finish or probability")

    result = {
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if "probability" in merged.columns:
        result["ROC_AUC"] = roc_auc_score(y_true, merged["probability"])
    return result


def score_ensemble(folder: Path, submission: pd.DataFrame) -> dict[str, float]:
    private = pd.read_csv(folder / "data" / "private_target.csv")
    merged = private.merge(submission, on="id", how="inner")
    if len(merged) != len(private):
        raise ValueError(f"ensemble submission has {len(merged)} ids, expected {len(private)}")
    if "churn_probability" not in merged.columns:
        raise ValueError("ensemble submission must contain churn_probability")

    y_true = merged["churn"]
    proba = merged["churn_probability"].clip(0, 1)
    pred = (proba >= 0.5).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
        "F1_at_0_5": f1_score(y_true, pred, zero_division=0),
        "precision_at_0_5": precision_score(y_true, pred, zero_division=0),
        "recall_at_0_5": recall_score(y_true, pred, zero_division=0),
    }


def score_competition(key: str) -> dict[str, dict[str, float]]:
    cfg = COMPETITIONS[key]
    folder = cfg["folder"]
    if key == "linear":
        score_fn = score_linear
    elif key == "logistic":
        score_fn = score_logistic
    elif key == "ensemble":
        score_fn = score_ensemble
    else:
        raise ValueError(key)

    baseline_submission = run_notebook_in_temp(
        folder, cfg["baseline_notebook"], cfg["baseline_submission"]
    )
    author_submission = run_notebook_in_temp(folder, cfg["author_notebook"], cfg["author_submission"])

    return {
        "baseline": score_fn(folder, baseline_submission),
        "author": score_fn(folder, author_submission),
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def readme_scores_section(key: str, scores: dict[str, dict[str, float]]) -> str:
    cfg = COMPETITIONS[key]
    main = cfg["main_metric"]
    direction = cfg["metric_direction"]

    if key == "linear":
        rows = [
            (
                "`simple_baseline.ipynb`",
                f"MAE = {fmt(scores['baseline']['MAE'])}",
                f"RMSE = {fmt(scores['baseline']['RMSE'])}, R² = {fmt(scores['baseline']['R2'])}",
            ),
            (
                "`author_solution.ipynb`",
                f"MAE = {fmt(scores['author']['MAE'])}",
                f"RMSE = {fmt(scores['author']['RMSE'])}, R² = {fmt(scores['author']['R2'])}",
            ),
        ]
    elif key == "logistic":
        rows = [
            (
                "`simple_baseline.ipynb`",
                f"F1 = {fmt(scores['baseline']['F1'])}",
                (
                    f"precision = {fmt(scores['baseline']['precision'])}, "
                    f"recall = {fmt(scores['baseline']['recall'])}, "
                    f"ROC-AUC = {fmt(scores['baseline']['ROC_AUC'])}"
                ),
            ),
            (
                "`author_solution.ipynb`",
                f"F1 = {fmt(scores['author']['F1'])}",
                (
                    f"precision = {fmt(scores['author']['precision'])}, "
                    f"recall = {fmt(scores['author']['recall'])}, "
                    f"ROC-AUC = {fmt(scores['author']['ROC_AUC'])}"
                ),
            ),
        ]
    elif key == "ensemble":
        rows = [
            (
                "`simple_baseline.ipynb`",
                f"ROC-AUC = {fmt(scores['baseline']['ROC_AUC'])}",
                (
                    f"AP = {fmt(scores['baseline']['average_precision'])}, "
                    f"F1@0.5 = {fmt(scores['baseline']['F1_at_0_5'])}"
                ),
            ),
            (
                "`author_solution.ipynb`",
                f"ROC-AUC = {fmt(scores['author']['ROC_AUC'])}",
                (
                    f"AP = {fmt(scores['author']['average_precision'])}, "
                    f"F1@0.5 = {fmt(scores['author']['F1_at_0_5'])}"
                ),
            ),
        ]
    else:
        raise ValueError(key)

    table = "\n".join(f"| {name} | {score} | {extra} |" for name, score, extra in rows)
    return f"""
## Результаты на private

Эти числа посчитаны по скрытому `data/private_target.csv`. До проверки их лучше не показывать ученикам.

Главная метрика: **{main}** ({direction}).

| Решение | Private score | Дополнительные метрики |
|---|---:|---|
{table}
""".strip()


def update_readme(key: str, scores: dict[str, dict[str, float]]) -> None:
    folder = COMPETITIONS[key]["folder"]
    readme_path = folder / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    marker = "## Результаты на private"
    base = text.split(marker)[0].rstrip()
    readme_path.write_text(base + "\n\n" + readme_scores_section(key, scores) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-readme", action="store_true")
    parser.add_argument(
        "--only",
        choices=list(COMPETITIONS),
        help="Посчитать только один комплект.",
    )
    args = parser.parse_args()

    all_scores = {}
    keys = [args.only] if args.only else list(COMPETITIONS)
    for key in keys:
        print(f"\n=== {key} ===")
        scores = score_competition(key)
        all_scores[key] = scores
        print(json.dumps(scores, ensure_ascii=False, indent=2))
        if args.write_readme:
            update_readme(key, scores)
            print(f"README updated: {COMPETITIONS[key]['folder'] / 'README.md'}")

    print("\nSUMMARY_JSON")
    print(json.dumps(all_scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
