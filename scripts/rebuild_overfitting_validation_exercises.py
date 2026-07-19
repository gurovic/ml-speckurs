#!/usr/bin/env python3
"""Rebuild Тест_к_теории.ipynb synced with 15-section theory."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "Урок_29_Переобучение_и_валидация_Теория/Тест_к_теории.ipynb"
)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [src],
    }


def build_cells() -> list[dict]:
    return [
        md(
            """# Занятие 29. Упражнения: переобучение и валидация

Короткая проверка теории (`Урок_29_Переобучение_и_валидация.ipynb`, занятие 29). Сквозной код — в `Урок_30_Переобучение_и_валидация_Практика.ipynb` (занятие 30)."""
        ),
        code("import numpy as np\nimport pandas as pd\n"),
        md("## 1. Validation error (п. 1)\n\nНа каких данных считают **validation error**?"),
        code(
            """where='...'
assert where == 'на данных, которых модель не видела при fit'
print('Верно')
"""
        ),
        md(
            """## 2. Диагноз (п. 2)

Train MSE 0,01, validation MSE 0,33. Назовите состояние модели одним словом (как в теории)."""
        ),
        code(
            """diagnosis='...'
assert diagnosis == 'переобучение'
print('Верно')
"""
        ),
        md("## 3. Параметры (п. 5)\n\nРаспределите коэффициенты регрессии и степень полинома."),
        code(
            """parameter='...'
hyperparameter='...'
assert parameter == 'коэффициенты' and hyperparameter == 'степень полинома'
print('Верно')
"""
        ),
        md("## 4. Test (п. 6)\n\nСколько раз открывают test для финальной оценки?"),
        code(
            """times=...
assert times == 1
print('Верно')
"""
        ),
        md("## 5. K-fold (п. 10)\n\nСколько обучений (`fit`) для одного варианта при 5-fold?"),
        code(
            """fits=...
assert fits == 5
print('Верно')
"""
        ),
        md("## 6. Группы (п. 11)\n\nКакое разбиение выбрать для нескольких фото одного пациента?"),
        code(
            """splitter='...'
assert splitter == 'GroupKFold'
print('Верно')
"""
        ),
        md("## 7. Время (п. 11)\n\nМожно ли перемешать будущие и прошлые продажи при прогнозе времени?"),
        code(
            """answer=...
assert answer is False
print('Верно')
"""
        ),
        md("## 8. Fold preprocessing (п. 12)\n\nГде обучать `StandardScaler` текущего fold?"),
        code(
            """part='...'
assert part == 'fold-train'
print('Верно')
"""
        ),
        md("## 9. Validation curve (п. 8)\n\nПо какой кривой выбирают степень полинома?"),
        code(
            """choice='...'
assert choice == 'минимум validation-ошибки'
print('Верно')
"""
        ),
        md("## 10. Финальный fit (п. 15)\n\nНа каких данных переобучают выбранный процесс перед test?"),
        code(
            """data='...'
assert data == 'train+validation'
print('Верно')
"""
        ),
    ]


def main() -> None:
    old = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    nb = {
        "cells": build_cells(),
        "metadata": old.get("metadata", {}),
        "nbformat": old.get("nbformat", 4),
        "nbformat_minor": old.get("nbformat_minor", 5),
    }
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {NOTEBOOK} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
