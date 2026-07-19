#!/usr/bin/env python3
"""Fix broken п./пп. references after section splits."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPLACEMENTS: list[tuple[str, str, str]] = [
    # (relative path glob prefix folder/file, old, new)
    ("Урок_21_ML_workflow_и_kNN_Теория/Урок_21_Workflow_kNN.ipynb", "п. 13–14–14", "пп. 13–14"),
    ("Урок_21_ML_workflow_и_kNN_Теория/Урок_21_Workflow_kNN.ipynb", "пп. 13–15", "пп. 13–14"),
    ("Урок_21_ML_workflow_и_kNN_Теория/Урок_21_Workflow_kNN.ipynb", "п. 19–18", "пп. 17–18"),
    ("Урок_21_ML_workflow_и_kNN_Теория/Урок_21_Workflow_kNN.ipynb", "сравнение «на равных» — п. 17–18", "сравнение «на равных» — п. 16"),
    ("Урок_21_ML_workflow_и_kNN_Теория/Урок_21_Workflow_kNN.ipynb", "4. Признаки, предобработка, модель — параметры преобразований только на train (пп. 12–15).", "4. Признаки, предобработка, модель — параметры преобразований только на train (пп. 12–14)."),
    ("Урок_21_ML_workflow_и_kNN_Теория/Урок_21_Workflow_kNN.ipynb", "5. Сравнение идей и подбор `k` на validation, журнал экспериментов (пп. 16–18).", "5. Сравнение идей и подбор `k` на validation, журнал экспериментов (пп. 15–16)."),
    ("Урок_21_ML_workflow_и_kNN_Теория/Урок_21_Workflow_kNN.ipynb", "6. Анализ ошибок (пп. 19–20).", "6. Анализ ошибок (пп. 17–18)."),
    ("Урок_21_ML_workflow_и_kNN_Теория/Урок_21_Workflow_kNN.ipynb", "7. Один раз test и мониторинг после внедрения (п. 19).\n", "7. Один раз test и мониторинг после внедрения (п. 19).\n"),
    ("Урок_21_ML_workflow_и_kNN_Теория/Урок_21_Workflow_kNN.ipynb", "Краткая памятка по циклу из п. 2:", "## 20. Чек-лист эксперимента\n\nКраткая памятка по циклу из п. 2:"),
    ("Урок_22_ML_workflow_и_kNN_Практика/knn_imba.ipynb", "п. 17–18):", "п. 17):"),
    ("Урок_22_ML_workflow_и_kNN_Практика/knn_imba.ipynb", "пп. 13–16)", "пп. 13–14)"),
    ("Урок_27_Логистическая_регрессия_Теория/Урок_27_Логистическая_регрессия.ipynb", "Порог подбирают отдельно на validation (п. 14–15).", "Порог подбирают отдельно на validation (п. 13)."),
    ("Урок_27_Логистическая_регрессия_Теория/Урок_27_Логистическая_регрессия.ipynb", "permutation importance (п. 17).", "permutation importance (п. 16)."),
    ("Урок_31_Решающее_дерево_Теория/Урок_31_Решающее_дерево.ipynb", "permutation importance (п. 17).", "permutation importance (п. 16)."),
    ("Урок_31_Решающее_дерево_Теория/Тест_к_теории.ipynb", "## 7. Pruning (п. 17)", "## 7. Pruning (п. 11)"),
    ("Урок_31_Решающее_дерево_Теория/Тест_к_теории.ipynb", "## 8. Масштаб (п. 10)", "## 8. Масштаб (п. 12)"),
    ("Урок_33_Ансамбли_Bagging_Случайный_лес_Теория/Урок_33_Ансамбли_Bagging_Случайный_лес.ipynb", "смена порога вероятности (п. 14).", "смена порога вероятности (п. 13)."),
    ("Урок_33_Ансамбли_Bagging_Случайный_лес_Теория/Тест_к_теории.ipynb", "## 1. Bootstrap (п. 3)", "## 1. Bootstrap (п. 1)"),
    ("Урок_33_Ансамбли_Bagging_Случайный_лес_Теория/Тест_к_теории.ipynb", "## 6. Случайный лес (п. 10)", "## 6. Случайный лес (п. 7)"),
    ("Урок_33_Ансамбли_Bagging_Случайный_лес_Теория/Тест_к_теории.ipynb", "## 7. Много деревьев (п. 8)", "## 7. Много деревьев (п. 9)"),
    ("Урок_35_Boosting_Теория/35_Градиентный_бустинг.ipynb", "занятие 29, п. 13).", "занятие 29, п. 12)."),
    ("Урок_35_Boosting_Теория/Тест_к_теории.ipynb", "## 4. Bagging и boosting (п. 7)", "## 4. Bagging и boosting (п. 12)"),
    ("Урок_35_Boosting_Теория/Тест_к_теории.ipynb", "## 6. Early stopping (п. 13)", "## 6. Early stopping (п. 8)"),
    ("Урок_35_Boosting_Теория/Тест_к_теории.ipynb", "## 7. Глубина (п. 8–9)", "## 7. Глубина (п. 5)"),
    ("Урок_35_Boosting_Теория/Тест_к_теории.ipynb", "## 8. Масштаб (п. 10)", "## 8. Масштаб (п. 14)"),
    ("Урок_36_Boosting_Практика/Урок_36_Boosting_Практика.ipynb", "пп. 4–9 теории", "пп. 4–9 теории"),
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
