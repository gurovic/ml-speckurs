#!/usr/bin/env python3
"""УСТАРЕЛ: блокнот теперь редактируется руками, точечными правками.

Запуск этого скрипта ПЕРЕЗАПИШЕТ ручные правки. Не запускать.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "Урок_29_Переобучение_и_валидация_Теория/Урок_29_Переобучение_и_валидация.ipynb"
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
            """# Занятие 29. Переобучение и валидация

**Обобщение** (generalization) — способность модели давать хорошие прогнозы на **новых** объектах, которых не было при обучении.

Модель нужна не для запоминания таблицы, а для прогноза в будущем. Сегодня разберём, как отличить настоящую закономерность от удачного совпадения на известных данных и как **честно** сравнивать варианты моделей.

**Сквозной пример:** синтетическая регрессия $y \\approx x^2$ с шумом; подбираем степень полинома (`PolynomialFeatures` + `LinearRegression`).

К концу занятия вы сможете:

- объяснить разницу между train-, validation- и test-ошибкой;
- распознать недообучение и переобучение по двум кривым ошибки;
- выбрать гиперпараметр по validation curve или k-fold CV;
- назвать типичную утечку при предобработке и описать финальный протокол оценки."""
        ),
        md(
            """## 1. Ошибка на train и на новых данных

**Train error** — ошибка на данных, на которых модель **обучалась** (вызывался `fit`).

**Validation error** — ошибка на данных, которые модель **не видела** при обучении коэффициентов. Эту же выборку используют для выбора гиперпараметров (п. 6).

Если train-ошибка низкая, а validation высокая, модель, скорее всего, **запомнила** train, а не выучила закономерность.

Ошибку измеряем знакомой метрикой **MSE** (занятие 25, пп. 4 и 6): чем меньше, тем лучше."""
        ),
        md(
            """## 2. Три состояния модели

| Состояние | Train error | Validation error | Интуиция |
|-----------|-------------|------------------|----------|
| **Недообучение** (underfitting) | высокая | высокая | модель слишком проста |
| **Хорошее обобщение** | умеренная | близка к train | разумная сложность |
| **Переобучение** (overfitting) | очень низкая | заметно выше train | модель запомнила шум |

**Пример (условные числа, MSE):**

| модель | train error | validation error |
|--------|-------------|------------------|
| слишком простая | 0.35 | 0.38 |
| разумная | 0.12 | 0.15 |
| слишком сложная | 0.01 | 0.33 |

**Диагноз:** сравнивайте train и validation **на одном протоколе** (одинаковый split, одна метрика). Низкая train-ошибка сама по себе ничего не гарантирует.

На сквозном примере: линейный полином (степень 1) не описывает $y \\approx x^2$ → недообучение; полином 15-й степени «обвивает» каждую точку train → переобучение."""
        ),
        md(
            """## 3. Откуда берётся переобучение

Модель учит и **закономерность**, и **случайный шум** в train. Риск переобучения растёт, если:

- **мало данных**, а модель гибкая;
- **много признаков** или высокая степень полинома;
- долго **перебирали** варианты по одной validation-выборке (п. 7);
- есть **утечка данных** — validation «подсмотрела» в обучение (определение — занятие 21, п. 10; варианты для CV — пп. 12–13 этого занятия).

Регуляризация (занятие 25, п. 13) и ограничение сложности помогают, но **не заменяют** честную проверку на отложенных данных."""
        ),
        md(
            """## 4. Смещение и разброс *

**Смещение (bias)** — систематическая ошибка: модель слишком проста и **всегда** промахивается в одну сторону (недообучение).

**Разброс (variance)** — чувствительность к составу train: при малом изменении выборки модель сильно меняется (переобучение).

| | Высокое смещение | Высокий разброс |
|---|------------------|-----------------|
| Пример на $y \\approx x^2$ | линейная модель (степень 1) | полином 15-й степени на 20 точках |
| Симптом | плохо и на train, и на val | отлично на train, плохо на val |

Validation помогает найти компромисс. Формальный bias–variance decomposition на курсе не требуется."""
        ),
        md(
            """## 5. Параметры и гиперпараметры

**Параметры** — числа, которые модель **учит из train** при вызове `fit`. Пример: коэффициенты `LinearRegression`.

**Гиперпараметры** — настройки, которые задаёт **исследователь до** `fit`. Пример: степень полинома в `PolynomialFeatures(degree=...)`, `max_depth` дерева, сила регуляризации.

| В сквозном примере | Параметр или гиперпараметр? |
|--------------------|----------------------------|
| коэффициенты `LinearRegression` | **параметр** (учатся в `fit`) |
| `degree` в `PolynomialFeatures` | **гиперпараметр** (задаём до `fit`) |

**Правило:** параметры учат на **train**; гиперпараметры выбирают по **validation** или **кросс-валидации** (пп. 8–10). **Test** не используют для подбора (п. 6)."""
        ),
        md(
            """## 6. Train, validation и test

Роли трёх выборок вы уже знаете (занятие 21, п. 4): train — обучение параметров (`fit`), validation — сравнение идей и подбор настроек, test — финальная проверка **один раз**.

Новое на этом занятии — чем именно грозит нарушение протокола: если посмотреть test, изменить степень полинома и снова посмотреть test, вы фактически **обучаетесь на test**, и его оценка становится оптимистичной.

На практике (занятие 30) для validation curve достаточно **train + validation**; test вводим как принцип финальной проверки (п. 15)."""
        ),
        code(
            """import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 180).reshape(-1, 1)
y = X[:, 0] ** 2 + rng.normal(0, 1.5, len(X))

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

plt.figure(figsize=(7, 4))
plt.scatter(X_tr, y_tr, s=25, alpha=0.7, label='train')
plt.scatter(X_val, y_val, s=25, alpha=0.7, label='validation')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Сквозной датасет: y ≈ x² + шум')
plt.grid(alpha=0.3); plt.legend(); plt.show()
print('Train:', len(X_tr), '| validation:', len(X_val))
"""
        ),
        md(
            """## 7. Перебор гиперпараметров и «переобучение на validation»

Если проверить **сотни** комбинаций на **одной** validation-выборке, случайно найдётся удачная — исследователь подстраивается под validation так же, как модель под train.

**Что делать:**
- ограничить хаотичный перебор;
- использовать **k-fold CV** (п. 10) вместо одного случайного split;
- сохранить **независимый test** для финала (п. 15).

Одна validation-оценка также зависит от **случайного** состава объектов. Повторные разбиения или CV показывают **устойчивость** вывода. Смотрите не только среднее, но и **разброс** по fold."""
        ),
        md(
            """## 8. Validation curve

**Validation curve** — график качества (например, MSE) при разных значениях **одного** гиперпараметра.

**Как читать** (степень полинома слева направо):
- **слева** — модель слишком проста (недообучение): train и validation оба плохи;
- **середина** — разумная сложность: validation минимальна;
- **справа** — переобучение: train MSE падает, validation MSE **растёт**.

Выбирают гиперпараметр по **минимуму validation** (или CV), не по train."""
        ),
        code(
            """from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

degrees = range(1, 16)
train_scores, val_scores = [], []
for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xtr_poly = poly.fit_transform(X_tr)
    Xval_poly = poly.transform(X_val)
    m = LinearRegression().fit(Xtr_poly, y_tr)
    train_scores.append(mean_squared_error(y_tr, m.predict(Xtr_poly)))
    val_scores.append(mean_squared_error(y_val, m.predict(Xval_poly)))

plt.plot(degrees, train_scores, label='train MSE')
plt.plot(degrees, val_scores, label='validation MSE')
plt.xlabel('Степень полинома')
plt.ylabel('MSE, меньше — лучше')
plt.title('Validation curve')
plt.grid(alpha=0.3); plt.legend(); plt.show()
"""
        ),
        md(
            """## 9. Learning curve

**Learning curve** — график качества при **росте размера train** (при фиксированных гиперпараметрах).

**Три типичных картины:**
1. Обе кривые плохи и близки → модель, вероятно, **слишком проста** (нужна сложнее модель или признаки).
2. Train хорош, validation заметно хуже → **переобучение**; больше данных может помочь.
3. Кривые почти сошлись на низком уровне → добавление данных в этом диапазоне **мало поможет**; нужна другая модель или признаки.

На графике ниже validation оценивается **внутри k-fold CV** по мере роста train."""
        ),
        code(
            """from sklearn.model_selection import learning_curve, KFold

poly5 = PolynomialFeatures(degree=5, include_bias=False)
X_poly5 = poly5.fit_transform(X)
learning_cv = KFold(n_splits=5, shuffle=True, random_state=42)
sizes, tr, va = learning_curve(
    LinearRegression(), X_poly5, y, cv=learning_cv,
    train_sizes=np.linspace(0.15, 1, 6),
    scoring='neg_mean_squared_error',
    shuffle=True, random_state=42,
)
tr, va = -tr, -va
plt.plot(sizes, tr.mean(axis=1), marker='o', label='train')
plt.plot(sizes, va.mean(axis=1), marker='o', label='cross-validation')
plt.xlabel('Размер train'); plt.ylabel('MSE, меньше — лучше')
plt.title('Learning curve (полином 5-й степени)')
plt.grid(alpha=0.3); plt.legend(); plt.show()
"""
        ),
        md(
            """## 10. K-fold cross-validation

**K-fold CV:** данные делят на $k$ **fold** (частей). Модель обучают $k$ раз:
- в каждом запуске **один** fold — validation, остальные $k-1$ — train;
- метрики усредняют по $k$ запускам.

При **5-fold** и одном варианте гиперпараметров выполняется **5 обучений** (`fit`).

**Плюсы:** оценка устойчивее одного split. **Минусы:** дороже по времени; неправильная схема разбиения (п. 11) даст неверный ответ.

В `cross_validate` поле `test_score` — метрика на validation-fold (не путать с финальным test из п. 6)."""
        ),
        code(
            """from sklearn.model_selection import cross_validate, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)
result = cross_validate(
    LinearRegression(), X_poly5, y, cv=cv,
    scoring='neg_mean_squared_error', return_train_score=True,
)
print('Validation scores (neg MSE):', result['test_score'].round(3))
print('Средняя MSE:', round(-result['test_score'].mean(), 3))
print('Разброс по fold (std neg MSE):', round(result['test_score'].std(), 3))
"""
        ),
        md(
            """## 11. Стратификация, группы и время

Принцип «схема разбиения должна отражать независимость объектов» знаком по занятию 25 (п. 9). Здесь — готовые инструменты sklearn для CV:

| Ситуация | Инструмент | Зачем |
|----------|------------|-------|
| Классификация, редкий класс | `StratifiedKFold` | в каждом fold примерно те же доли классов |
| Несколько записей на одного пациента / пользователя | `GroupKFold` | один объект не попадает и в train, и в val |
| Прогноз будущего по прошлому | `TimeSeriesSplit` | train — прошлое, validation — будущее |"""
        ),
        code(
            """# Схемы разбиения: синий — train, оранжевый — validation.
fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharey=True)
for fold in range(4):
    colors = ['tab:blue'] * 12
    for j in range(fold * 3, (fold + 1) * 3):
        colors[j] = 'tab:orange'
    axes[0].scatter(range(12), [fold] * 12, c=colors, marker='s', s=90)
axes[0].set_title('Обычный K-fold')

groups = np.repeat(range(4), 3)
for fold in range(4):
    axes[1].scatter(
        range(12), [fold] * 12,
        c=np.where(groups == fold, 'tab:orange', 'tab:blue'), marker='s', s=90,
    )
axes[1].set_title('GroupKFold: группа целиком')

for fold, end in enumerate([4, 6, 8, 10]):
    axes[2].scatter(
        range(end), [fold] * end,
        c=['tab:blue'] * (end - 2) + ['tab:orange'] * 2, marker='s', s=90,
    )
axes[2].set_title('Время: прошлое → будущее')
for ax in axes:
    ax.set_xlabel('Порядок объектов'); ax.grid(alpha=0.15)
plt.tight_layout(); plt.show()
"""
        ),
        md(
            """## 12. Предобработка внутри каждого fold

Правило «`fit` преобразований только на train» вы знаете по занятию 25 (пп. 10 и 13). Новое при кросс-валидации: train меняется **от fold к fold**, поэтому и предобработку обучают заново **на train-части каждого fold**.

**Безопасная последовательность внутри одного fold:**

`fit preprocessing на fold-train` → `transform` fold-train и fold-validation → `fit` модель → `score` на fold-validation.

**Пример утечки:** `poly.fit_transform(X_вся_таблица)` **до** split — validation повлияла на выбор признаков. Правильно: `fit` только на train, `transform` на train и val (как в п. 8)."""
        ),
        md(
            """## 13. Утечки в CV и grid search

**Типичные утечки:**
- масштабирование или imputation на **всей** таблице до CV;
- отбор признаков по корреляции с целью на **всей** таблице;
- подгонка степени полинома по **test**.

**Проверочный вопрос:** «Узнаёт ли шаг что-нибудь из данных при `fit`?» Если да — `fit` только внутри train (или внутри fold-train).

**Grid search** (`GridSearchCV`) перебирает сетку гиперпараметров и оценивает каждый вариант CV. Чем больше вариантов, тем выше риск случайно «попасть» в удачный fold. Поэтому **test** остаётся отдельно."""
        ),
        md(
            """## 14. Вложенная валидация *

**Вложенная CV (nested CV):**
- **внутренний** цикл — подбор гиперпараметров;
- **внешний** цикл — оценка всего процесса выбора.

Защищает от оптимистичной оценки при большом grid search. Метод дорогой; в учебном проекте достаточно понимать различие **настройки** (validation/CV) и **независимой оценки** (test)."""
        ),
        md(
            """## 15. Финальный протокол

1. **Отделить test** подходящим способом (п. 11).
2. На train+validation сравнивать идеи **одним** CV-протоколом.
3. Выбрать модель и гиперпараметры по validation/CV.
4. **Переобучить** выбранный процесс на всех train+validation данных.
5. **Один раз** оценить на test.
6. Сообщить среднее, разброс по fold и ограничения данных.

> **Главная мысль.** Валидация проверяет не только алгоритм, но и **честность** всего способа получения модели."""
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
    nb["metadata"]["name"] = "overfitting_validation_theory"
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {NOTEBOOK} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    raise SystemExit(
        "Скрипт устарел: блокнот редактируется руками. "
        "Запуск перезаписал бы ручные правки."
    )
