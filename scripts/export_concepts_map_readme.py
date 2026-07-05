#!/usr/bin/env python3
"""Export canvas concept map (placements-v2) to README.md in repo root."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANVAS_DATA = ROOT / "canvases/ml-theory-concepts-map.canvas.data.json"

BIG_CONCEPTS: list[tuple[str, str]] = [
    ("workflow", "Постановка задачи и ML-workflow"),
    ("metrics", "Метрики качества и baseline"),
    ("train-split", "Train / validation / test"),
    ("cv", "Кросс-валидация, curves и схемы разбиения"),
    ("leakage", "Утечка данных"),
    ("features", "Конструирование и отбор признаков"),
    ("missing", "Пропуски"),
    ("categorical", "Категориальные признаки"),
    ("outliers", "Выбросы"),
    ("multicollinearity", "Мультиколлинеарность"),
    ("scaling", "Масштабирование признаков"),
    ("imbalance", "Дисбаланс классов"),
    ("models", "Модели (kNN, регрессии, деревья, ансамбли)"),
    ("regularization", "Регуляризация"),
    ("hyperparams", "Подбор гиперпараметров"),
    ("overfitting", "Переобучение, недообучение, bias–variance"),
    ("importance", "Важность признаков и интерпретация модели"),
]

LESSONS: list[tuple[int, str]] = [
    (25, "kNN / workflow"),
    (27, "признаки (FE)"),
    (29, "лин. регрессия"),
    (31, "логрег"),
    (33, "валидация"),
    (35, "дерево"),
    (37, "bagging / RF"),
    (39, "бoostинг"),
]


def cell_labels(placements: dict[str, list[str]], row_key: str, lesson: int) -> list[str]:
    return placements.get(f"{row_key}:{lesson}", [])


def format_cell(labels: list[str]) -> str:
    if not labels:
        return "—"
    return "<br>".join(f"- {label.replace('|', '\\|')}" for label in labels)


def export_readme() -> None:
    data = json.loads(CANVAS_DATA.read_text(encoding="utf-8"))
    placements = data["placements-v2"]

    visible = [
        bc
        for bc in BIG_CONCEPTS
        if any(cell_labels(placements, bc[0], lesson) for lesson, _ in LESSONS)
    ]

    headers = ["Концепт"] + [f"{n}<br>{short}" for n, short in LESSONS]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for _key, title in visible:
        row = [title.replace("|", "\\|")]
        for lesson_num, _short in LESSONS:
            row.append(format_cell(cell_labels(placements, _key, lesson_num)))
        lines.append("| " + " | ".join(row) + " |")

    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote README.md ({len(visible)} rows)")


if __name__ == "__main__":
    export_readme()
