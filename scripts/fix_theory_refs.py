#!/usr/bin/env python3
"""Fix broken п./пп. references after section splits."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPLACEMENTS: list[tuple[str, str, str]] = [
    # (relative path glob prefix folder/file, old, new)
    ("11_workflow/workflow_theory.ipynb", "п. 13–14–14", "пп. 13–14"),
    ("11_workflow/workflow_theory.ipynb", "пп. 13–15", "пп. 13–14"),
    ("11_workflow/workflow_theory.ipynb", "п. 19–18", "пп. 17–18"),
    ("11_workflow/workflow_theory.ipynb", "сравнение «на равных» — п. 17–18", "сравнение «на равных» — п. 16"),
    ("11_workflow/workflow_theory.ipynb", "4. Признаки, предобработка, модель — параметры преобразований только на train (пп. 12–15).", "4. Признаки, предобработка, модель — параметры преобразований только на train (пп. 12–14)."),
    ("11_workflow/workflow_theory.ipynb", "5. Сравнение идей и подбор `k` на validation, журнал экспериментов (пп. 16–18).", "5. Сравнение идей и подбор `k` на validation, журнал экспериментов (пп. 15–16)."),
    ("11_workflow/workflow_theory.ipynb", "6. Анализ ошибок (пп. 19–20).", "6. Анализ ошибок (пп. 17–18)."),
    ("11_workflow/workflow_theory.ipynb", "7. Один раз test и мониторинг после внедрения (п. 19).\n", "7. Один раз test и мониторинг после внедрения (п. 19).\n"),
    ("11_workflow/workflow_theory.ipynb", "Краткая памятка по циклу из п. 2:", "## 20. Чек-лист эксперимента\n\nКраткая памятка по циклу из п. 2:"),
    ("11_workflow/workflow_practice.ipynb", "п. 17–18):", "п. 17):"),
    ("11_workflow/workflow_practice.ipynb", "пп. 13–16)", "пп. 13–14)"),
    ("14_logistic_regression/logistic_regression_theory.ipynb", "Порог подбирают отдельно на validation (п. 14–15).", "Порог подбирают отдельно на validation (п. 13)."),
    ("14_logistic_regression/logistic_regression_theory.ipynb", "permutation importance (п. 17).", "permutation importance (п. 16)."),
    ("16_decision_tree/decision_tree_theory.ipynb", "permutation importance (п. 17).", "permutation importance (п. 16)."),
    ("16_decision_tree/decision_tree_exercises.ipynb", "## 7. Pruning (п. 17)", "## 7. Pruning (п. 11)"),
    ("16_decision_tree/decision_tree_exercises.ipynb", "## 8. Масштаб (п. 10)", "## 8. Масштаб (п. 12)"),
    ("17_bagging_random_forest/bagging_random_forest_theory.ipynb", "смена порога вероятности (п. 14).", "смена порога вероятности (п. 13)."),
    ("17_bagging_random_forest/bagging_random_forest_exercises.ipynb", "## 1. Bootstrap (п. 3)", "## 1. Bootstrap (п. 1)"),
    ("17_bagging_random_forest/bagging_random_forest_exercises.ipynb", "## 6. Случайный лес (п. 10)", "## 6. Случайный лес (п. 7)"),
    ("17_bagging_random_forest/bagging_random_forest_exercises.ipynb", "## 7. Много деревьев (п. 8)", "## 7. Много деревьев (п. 9)"),
    ("18_gradient_boosting/gradient_boosting_theory.ipynb", "занятие 29, п. 13).", "занятие 29, п. 12)."),
    ("18_gradient_boosting/gradient_boosting_exercises.ipynb", "## 4. Bagging и boosting (п. 7)", "## 4. Bagging и boosting (п. 12)"),
    ("18_gradient_boosting/gradient_boosting_exercises.ipynb", "## 6. Early stopping (п. 13)", "## 6. Early stopping (п. 8)"),
    ("18_gradient_boosting/gradient_boosting_exercises.ipynb", "## 7. Глубина (п. 8–9)", "## 7. Глубина (п. 5)"),
    ("18_gradient_boosting/gradient_boosting_exercises.ipynb", "## 8. Масштаб (п. 10)", "## 8. Масштаб (п. 14)"),
    ("18_gradient_boosting/gradient_boosting_practice.ipynb", "пп. 4–9 теории", "пп. 4–9 теории"),
    ("18_gradient_boosting/gradient_boosting_practice_solution.ipynb", "По п. 7–8.", "По п. 7."),
    ("18_gradient_boosting/gradient_boosting_practice_solution.ipynb", "По п. 8–9.", "По п. 8–9."),
]


def patch_file(rel: str, old: str, new: str) -> bool:
    path = ROOT / rel
    if not path.exists():
        return False
    if path.suffix == ".ipynb":
        text = path.read_text(encoding="utf-8")
        if old not in text:
            return False
        path.write_text(text.replace(old, new), encoding="utf-8")
        return True
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return False
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True


def main() -> None:
    for rel, old, new in REPLACEMENTS:
        if patch_file(rel, old, new):
            print(f"fixed: {rel}")


if __name__ == "__main__":
    main()
