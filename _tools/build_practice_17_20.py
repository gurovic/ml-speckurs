"""Generate practice + practice_solution for topics 17-20."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    lines = text.split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(text: str) -> dict:
    lines = text.rstrip("\n").split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def save_nb(path: Path, cells: list, name: str):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "name": name,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def practice_header(n_theory: int, n_practice: int, title: str, model: str, theory_file: str, intro: str):
    return md(
        f"""# Занятие {n_practice}. Практика: {title} (~90 мин)

Вы **пишете весь код сами**. Ячейку **«Дано»** не меняйте.

Главная модель — **{model}** (теория — занятие {n_theory}, `{theory_file}`).

{intro}

Термины и определения — в теоретическом ноутбуке. Здесь только практика."""
    )


def task(n: int, title: str, body: str, minutes: int):
    return md(f"---\n## Задание {n}. {title} (~{minutes} мин)\n\n{body}")


def empty_code():
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": []}


def build_practice_cells(spec: dict, filled: bool) -> list:
    cells = [practice_header(**spec["header"])]
    cells.append(md(spec["given_md"]))
    cells.append(code(spec["given_code"]) if filled else code(spec["given_code"]))
    for t in spec["tasks"]:
        cells.append(task(t["n"], t["title"], t["body"], t["min"]))
        if filled and "solution" in t:
            cells.append(code(t["solution"]))
        else:
            cells.append(empty_code())
    if spec.get("final_md"):
        cells.append(md(spec["final_md"]))
    return cells


# --- 17 Overfitting ---
OV_DIR = ROOT / "17_overfitting_validation"
ov_spec = {
    "header": {
        "n_theory": 33,
        "n_practice": 34,
        "title": "переобучение и validation curve",
        "model": "LinearRegression + PolynomialFeatures",
        "theory_file": "overfitting_validation_theory.ipynb",
        "intro": "Построим validation curve по степени полинома, learning curve и k-fold CV на синтетических данных (как в п. 7–9 теории).",
    },
    "given_md": """---
## Дано: синтетическая регрессия

Объект — одна точка на оси X. Цель — предсказать `y ≈ x²` с шумом. Это тот же пример, что в теории: видно недообучение, хорошее обобщение и переобучение.""",
    "given_code": """import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 180).reshape(-1, 1)
y = X[:, 0] ** 2 + rng.normal(0, 1.5, len(X))
print('Объектов:', len(X))""",
    "tasks": [
        {
            "n": 0,
            "title": "Импорты и split",
            "min": 8,
            "body": """Подключите `train_test_split`, `PolynomialFeatures`, `LinearRegression`, `mean_squared_error`, `learning_curve`, `cross_validate`, `KFold`.

Константы: `RANDOM_STATE = 42`, разбиение **70% train / 30% validation** (как в теории, п. 4).

**Критерий:** получены `X_train`, `X_val`, `y_train`, `y_val`.""",
            "solution": """from sklearn.model_selection import train_test_split, learning_curve, cross_validate, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

RANDOM_STATE = 42
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)
print('train / val:', len(X_train), len(X_val))""",
        },
        {
            "n": 1,
            "title": "Validation curve",
            "min": 20,
            "body": """По мотивам п. 7. Для степеней полинома **1–15**:

1. `PolynomialFeatures` → `fit` только на train, `transform` на train и val.
2. Обучите `LinearRegression`, посчитайте **MSE** на train и validation.
3. Постройте график «степень → MSE».

**Критерий:** виден U-образный (или похожий) validation MSE; train MSE падает монотонно.""",
            "solution": """degrees = range(1, 16)
train_scores, val_scores = [], []
for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xtr = poly.fit_transform(X_train)
    Xva = poly.transform(X_val)
    m = LinearRegression().fit(Xtr, y_train)
    train_scores.append(mean_squared_error(y_train, m.predict(Xtr)))
    val_scores.append(mean_squared_error(y_val, m.predict(Xva)))

plt.plot(degrees, train_scores, marker='o', label='train MSE')
plt.plot(degrees, val_scores, marker='o', label='validation MSE')
plt.xlabel('Степень полинома')
plt.ylabel('MSE')
plt.title('Validation curve')
plt.grid(alpha=0.3)
plt.legend()
plt.show()
best_degree = degrees[int(np.argmin(val_scores))]
print('Лучшая степень по validation:', best_degree)""",
        },
        {
            "n": 2,
            "title": "Диагноз переобучения",
            "min": 10,
            "body": """Выберите степень с **минимальным validation MSE** (`best_degree`) и степень **14** (или 15).

Сравните train vs validation MSE. В markdown ниже: где недообучение, где переобучение? (п. 2)

**Критерий:** для большой степени train MSE << validation MSE.""",
            "solution": """for d in [best_degree, 14]:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    Xtr = poly.fit_transform(X_train)
    Xva = poly.transform(X_val)
    m = LinearRegression().fit(Xtr, y_train)
    print(
        f'degree={d}: train MSE={mean_squared_error(y_train, m.predict(Xtr)):.2f}, '
        f'val MSE={mean_squared_error(y_val, m.predict(Xva)):.2f}'
    )""",
        },
        {
            "n": 3,
            "title": "Learning curve",
            "min": 15,
            "body": """По мотивам п. 8. Для `best_degree` постройте **learning curve** (`learning_curve`, 5-fold CV, `neg_mean_squared_error`).

**Критерий:** график train vs CV MSE от размера train.""",
            "solution": """poly_best = PolynomialFeatures(degree=best_degree, include_bias=False)
X_poly = poly_best.fit_transform(X)
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
sizes, tr, va = learning_curve(
    LinearRegression(), X_poly, y, cv=cv,
    train_sizes=np.linspace(0.15, 1.0, 6),
    scoring='neg_mean_squared_error', shuffle=True, random_state=RANDOM_STATE,
)
plt.plot(sizes, -tr.mean(axis=1), marker='o', label='train')
plt.plot(sizes, -va.mean(axis=1), marker='o', label='CV')
plt.xlabel('Размер train')
plt.ylabel('MSE')
plt.title('Learning curve')
plt.grid(alpha=0.3)
plt.legend()
plt.show()""",
        },
        {
            "n": 4,
            "title": "K-fold CV",
            "min": 12,
            "body": """По мотивам п. 9. `cross_validate` с 5-fold для модели с `best_degree`. Выведите среднюю validation MSE и std.

**Критерий:** напечатаны `test_score` (отрицательный MSE) и среднее.""",
            "solution": """result = cross_validate(
    LinearRegression(), X_poly, y, cv=cv,
    scoring='neg_mean_squared_error', return_train_score=True,
)
print('CV MSE scores:', (-result['test_score']).round(3))
print('Средняя validation MSE:', round(-result['test_score'].mean(), 3))""",
        },
        {
            "n": 5,
            "title": "Параметры vs гиперпараметры",
            "min": 8,
            "body": """В markdown: что в этом эксперименте **параметр** модели, а что **гиперпараметр**? (п. 6)

**Критерий:** степень полинома — гиперпараметр; коэффициенты регрессии — параметры.""",
            "solution": "",
        },
        {
            "n": 6,
            "title": "Утечка в preprocessing",
            "min": 10,
            "body": """По мотивам п. 14. Объясните в markdown: почему `poly.fit_transform` на **всей** таблице до split — утечка?

**Критерий:** validation/test «видят» статистику из своих строк.""",
            "solution": "",
        },
        {
            "n": 7,
            "title": "Финальная модель",
            "min": 12,
            "body": """Обучите финальную модель: `best_degree`, `fit` poly на **train**, оцените MSE на **validation** ещё раз.

**Критерий:** validation MSE близка к минимуму из задания 1.""",
            "solution": """poly_final = PolynomialFeatures(degree=best_degree, include_bias=False)
Xtr = poly_final.fit_transform(X_train)
Xva = poly_final.transform(X_val)
final = LinearRegression().fit(Xtr, y_train)
print('Final validation MSE:', round(mean_squared_error(y_val, final.predict(Xva)), 3))""",
        },
        {
            "n": 8,
            "title": "Итог",
            "min": 5,
            "body": """В markdown: 3 вывода — validation curve, learning curve, честный fit poly только на train.

**Критерий:** три коротких пункта.""",
            "solution": "",
        },
    ],
    "final_md": None,
}

# --- 18 Decision Tree ---
DT_DIR = ROOT / "18_decision_tree"
dt_spec = {
    "header": {
        "n_theory": 35,
        "n_practice": 36,
        "title": "решающее дерево",
        "model": "DecisionTreeClassifier",
        "theory_file": "decision_tree_theory.ipynb",
        "intro": "На `make_moons` сравним глубокие и ограниченные деревья, подберём `max_depth` и посмотрим обрезку (пп. 10–11 теории).",
    },
    "given_md": """---
## Дано: классификация make_moons

Два полумесяца — нелинейная граница. Масштабирование для дерева не обязательно (п. 12).""",
    "given_code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=350, noise=0.3, random_state=42)
print('Объектов:', len(X), '| классы:', np.bincount(y))""",
    "tasks": [
        {
            "n": 0,
            "title": "Split и импорты",
            "min": 8,
            "body": """`train_test_split`, `DecisionTreeClassifier`, `accuracy_score`, `plot_tree`, `confusion_matrix`, `RANDOM_STATE=42`, split 65/35 со `stratify`.

**Критерий:** `X_train`, `X_val`, `y_train`, `y_val`.""",
            "solution": """from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

RANDOM_STATE = 42
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.35, stratify=y, random_state=RANDOM_STATE
)""",
        },
        {
            "n": 1,
            "title": "Глубина и переобучение",
            "min": 15,
            "body": """По п. 10. Обучите деревья с `max_depth` in `[1, 3, 6, None]`. Таблица train/validation accuracy.

**Критерий:** при `None` train ≈ 1.0, validation хуже.""",
            "solution": """for d in [1, 3, 6, None]:
    m = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE).fit(X_train, y_train)
    tr = accuracy_score(y_train, m.predict(X_train))
    va = accuracy_score(y_val, m.predict(X_val))
    print(f'max_depth={d!s:4} train={tr:.3f} val={va:.3f}')""",
        },
        {
            "n": 2,
            "title": "Подбор max_depth",
            "min": 15,
            "body": """Переберите `max_depth` от 1 до 12. Выберите лучший по **validation accuracy**.

**Критерий:** переменная `best_depth`.""",
            "solution": """rows = []
for d in range(1, 13):
    m = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE).fit(X_train, y_train)
    va = accuracy_score(y_val, m.predict(X_val))
    rows.append((d, va))
best_depth = max(rows, key=lambda x: x[1])[0]
print('best_depth:', best_depth, 'val acc:', round(max(rows, key=lambda x: x[1])[1], 3))""",
        },
        {
            "n": 3,
            "title": "Визуализация дерева",
            "min": 12,
            "body": """По п. 16. `plot_tree` для модели с `best_depth` (figsize ≥ 10).

**Критerий:** график отображается.""",
            "solution": """best = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE).fit(X_train, y_train)
plt.figure(figsize=(12, 6))
plot_tree(best, feature_names=['x1', 'x2'], class_names=['0', '1'], filled=True)
plt.show()""",
        },
        {
            "n": 4,
            "title": "min_samples_leaf",
            "min": 12,
            "body": """При фиксированном `best_depth` попробуйте `min_samples_leaf` in `[1, 5, 15]`. Сравните validation accuracy (п. 11).

**Критерий:** таблица из 3 строк.""",
            "solution": """for leaf in [1, 5, 15]:
    m = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=leaf, random_state=RANDOM_STATE).fit(X_train, y_train)
    print(f'min_samples_leaf={leaf} val acc={accuracy_score(y_val, m.predict(X_val)):.3f}')""",
        },
        {
            "n": 5,
            "title": "ccp_alpha",
            "min": 12,
            "body": """По п. 11 (post-pruning). Для полного дерева (`max_depth=None`) выведите число листьев при `ccp_alpha` in `[0, 0.005, 0.02, 0.08]`.

**Критерий:** листьев становится меньше при росте alpha.""",
            "solution": """for alpha in [0.0, 0.005, 0.02, 0.08]:
    m = DecisionTreeClassifier(ccp_alpha=alpha, random_state=RANDOM_STATE).fit(X_train, y_train)
    print(f'ccp_alpha={alpha} leaves={m.get_n_leaves()} depth={m.get_depth()}')""",
        },
        {
            "n": 6,
            "title": "Confusion matrix",
            "min": 10,
            "body": """Матрица ошибок лучшего дерева на validation.

**Критерий:** `confusion_matrix(y_val, y_pred)`.""",
            "solution": """final = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE).fit(X_train, y_train)
y_pred = final.predict(X_val)
print(confusion_matrix(y_val, y_pred))""",
        },
        {
            "n": 7,
            "title": "Feature importance",
            "min": 10,
            "body": """По п. 15. Выведите `feature_importances_`. Кратко в markdown: почему importance не доказывает причинность?

**Критерий:** два числа importance.""",
            "solution": """print('importance:', final.feature_importances_)""",
        },
        {
            "n": 8,
            "title": "Итог",
            "min": 5,
            "body": """Три вывода: переобучение глубокого дерева, польза ограничений, интерпретируемость.

**Критерий:** markdown с тремя пунктами.""",
            "solution": "",
        },
    ],
    "final_md": None,
}

# Fix typo in task 3 criterion
dt_spec["tasks"][3]["body"] = dt_spec["tasks"][3]["body"].replace("Критerий", "Критерий")

# --- 19 Random Forest ---
RF_DIR = ROOT / "19_bagging_random_forest"
rf_spec = {
    "header": {
        "n_theory": 37,
        "n_practice": 38,
        "title": "bagging и случайный лес",
        "model": "RandomForestClassifier",
        "theory_file": "bagging_random_forest_theory.ipynb",
        "intro": "Сравним одно дерево, bagging и random forest; настроим `n_estimators`, посмотрим OOB (пп. 7–10 теории).",
    },
    "given_md": """---
## Дано: make_classification

20 признаков, бинарная классификация — как в теории (п. 8).""",
    "given_code": """import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=7, flip_y=0.08, random_state=42
)
print('Объектов:', len(X))""",
    "tasks": [
        {
            "n": 0,
            "title": "Split",
            "min": 8,
            "body": """Импорты: `train_test_split`, `DecisionTreeClassifier`, `BaggingClassifier`, `RandomForestClassifier`, `accuracy_score`, `permutation_importance`. Split 70/30, stratify, `RANDOM_STATE=42`.

**Критерий:** train/val готовы.""",
            "solution": """from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)""",
        },
        {
            "n": 1,
            "title": "Три модели",
            "min": 15,
            "body": """По п. 8. Одно дерево, bagging (150 estimators), random forest (150). Train и validation accuracy.

**Критерий:** лес ≥ bagging ≥ одно дерево на validation (обычно).""",
            "solution": """models = {
    'одно дерево': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'bagging': BaggingClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
    'random forest': RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
}
for name, m in models.items():
    m.fit(X_train, y_train)
    tr = accuracy_score(y_train, m.predict(X_train))
    va = accuracy_score(y_val, m.predict(X_val))
    print(f'{name}: train={tr:.3f} val={va:.3f}')""",
        },
        {
            "n": 2,
            "title": "n_estimators",
            "min": 15,
            "body": """По п. 9–10. График validation accuracy vs `n_estimators` in `[1,5,20,60,150,300]`.

**Критерий:** кривая выходит на плато.""",
            "solution": """import matplotlib.pyplot as plt

counts = [1, 5, 20, 60, 150, 300]
scores = []
for n in counts:
    m = RandomForestClassifier(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
    scores.append(accuracy_score(y_val, m.predict(X_val)))
plt.plot(counts, scores, marker='o')
plt.xlabel('n_estimators')
plt.ylabel('validation accuracy')
plt.grid(alpha=0.3)
plt.show()""",
        },
        {
            "n": 3,
            "title": "OOB score",
            "min": 10,
            "body": """По п. 7. `RandomForestClassifier(oob_score=True, n_estimators=300)`. Сравните OOB и validation accuracy.

**Критерий:** оба числа напечатаны.""",
            "solution": """forest_oob = RandomForestClassifier(
    n_estimators=300, oob_score=True, random_state=RANDOM_STATE, n_jobs=-1
).fit(X_train, y_train)
print('OOB:', round(forest_oob.oob_score_, 3))
print('Val:', round(accuracy_score(y_val, forest_oob.predict(X_val)), 3))""",
        },
        {
            "n": 4,
            "title": "max_depth",
            "min": 12,
            "body": """Подберите `max_depth` in `[None, 5, 10, 20]` при `n_estimators=150`. Лучший по validation.

**Критерий:** `best_depth` выбран.""",
            "solution": """best = (-1, None)
for d in [None, 5, 10, 20]:
    m = RandomForestClassifier(n_estimators=150, max_depth=d, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
    va = accuracy_score(y_val, m.predict(X_val))
    if va > best[0]:
        best = (va, d)
print('best max_depth:', best[1], 'val acc:', round(best[0], 3))""",
        },
        {
            "n": 5,
            "title": "Bootstrap вручную",
            "min": 12,
            "body": """По п. 3. Сгенерируйте bootstrap-индексы длины n из `len(X_train)` с `replace=True`. Сколько **уникальных** объектов в среднем попадёт в выборку? (100 повторов)

**Критерий:** ~632 при n=1000 (63.2%).""",
            "solution": """rng = np.random.default_rng(RANDOM_STATE)
n = len(X_train)
uniq = [len(np.unique(rng.choice(n, n, replace=True))) for _ in range(100)]
print('Среднее уникальных:', round(np.mean(uniq), 1))""",
        },
        {
            "n": 6,
            "title": "Permutation importance",
            "min": 15,
            "body": """По п. 14. `permutation_importance` для леса на validation, top-5 признаков.

**Критерий:** barh или таблица top-5.""",
            "solution": """rf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
perm = permutation_importance(rf, X_val, y_val, n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1)
top = np.argsort(perm.importances_mean)[-5:]
for i in top:
    print(f'x{i}: {perm.importances_mean[i]:.4f}')""",
        },
        {
            "n": 7,
            "title": "Train vs val gap",
            "min": 8,
            "body": """В markdown: почему у леса большой train acc не всегда означает переобучение? (п. 9)

**Критерий:** 2–3 предложения.""",
            "solution": "",
        },
        {
            "n": 8,
            "title": "Итог",
            "min": 5,
            "body": """Сравните bagging и RF в одном предложении каждый.

**Критерий:** markdown.""",
            "solution": "",
        },
    ],
    "final_md": None,
}

# --- 20 Gradient Boosting ---
GB_DIR = ROOT / "20_gradient_boosting"
gb_spec = {
    "header": {
        "n_theory": 39,
        "n_practice": 40,
        "title": "градиентный бустинг",
        "model": "GradientBoostingRegressor",
        "theory_file": "gradient_boosting_theory.ipynb",
        "intro": "На `make_regression` настроим глубину, построим staged-кривую MAE и попробуем early stopping (пп. 4–8 теории).",
    },
    "given_md": """---
## Дано: синтетическая регрессия

900 объектов, 10 признаков — как в теории.""",
    "given_code": """import numpy as np
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=900, n_features=10, n_informative=6, noise=25, random_state=42
)
print('Объектов:', len(X))""",
    "tasks": [
        {
            "n": 0,
            "title": "Split",
            "min": 8,
            "body": """`train_test_split`, `GradientBoostingRegressor`, `RandomForestRegressor`, `mean_absolute_error`, `RANDOM_STATE=42`, 70/30.

**Критерий:** train/val.""",
            "solution": """from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

RANDOM_STATE = 42
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)""",
        },
        {
            "n": 1,
            "title": "max_depth",
            "min": 15,
            "body": """По п. 5. `max_depth` in `[1,2,4,7]`, `n_estimators=180`, `learning_rate=0.05`. Train и validation MAE.

**Критерий:** таблица 4 строк.""",
            "solution": """for depth in [1, 2, 4, 7]:
    m = GradientBoostingRegressor(
        n_estimators=180, learning_rate=0.05, max_depth=depth, random_state=RANDOM_STATE
    ).fit(X_train, y_train)
    tr = mean_absolute_error(y_train, m.predict(X_train))
    va = mean_absolute_error(y_val, m.predict(X_val))
    print(f'depth={depth} train MAE={tr:.1f} val MAE={va:.1f}')""",
        },
        {
            "n": 2,
            "title": "Staged curve",
            "min": 18,
            "body": """По п. 7–8. Модель `n_estimators=350`, `max_depth=3`. Кривая train/validation MAE по `staged_predict`. Отметьте лучшее число деревьев.

**Критерий:** график + `best_n`.""",
            "solution": """import matplotlib.pyplot as plt

model = GradientBoostingRegressor(
    n_estimators=350, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
).fit(X_train, y_train)
train_mae, val_mae = [], []
for pt, pv in zip(model.staged_predict(X_train), model.staged_predict(X_val)):
    train_mae.append(mean_absolute_error(y_train, pt))
    val_mae.append(mean_absolute_error(y_val, pv))
best_n = int(np.argmin(val_mae)) + 1
plt.plot(train_mae, label='train')
plt.plot(val_mae, label='validation')
plt.axvline(best_n, color='crimson', ls='--', label=f'best={best_n}')
plt.xlabel('Число деревьев')
plt.ylabel('MAE')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
print('best_n:', best_n)""",
        },
        {
            "n": 3,
            "title": "learning_rate",
            "min": 12,
            "body": """Сравните `learning_rate` 0.2 vs 0.05 при `n_estimators=150`, `max_depth=3`. Validation MAE.

**Критерий:** два числа.""",
            "solution": """for lr in [0.2, 0.05]:
    m = GradientBoostingRegressor(
        n_estimators=150, learning_rate=lr, max_depth=3, random_state=RANDOM_STATE
    ).fit(X_train, y_train)
    print(f'lr={lr} val MAE={mean_absolute_error(y_val, m.predict(X_val)):.1f}')""",
        },
        {
            "n": 4,
            "title": "Early stopping",
            "min": 12,
            "body": """По п. 8. `validation_fraction=0.2`, `n_iter_no_change=15`. Сколько деревьев реально обучилось?

**Критерий:** `early.n_estimators_` < 350.""",
            "solution": """early = GradientBoostingRegressor(
    n_estimators=350, learning_rate=0.05, max_depth=3,
    validation_fraction=0.2, n_iter_no_change=15, tol=1e-4, random_state=RANDOM_STATE,
).fit(X_train, y_train)
print('Обучено деревьев:', early.n_estimators_)""",
        },
        {
            "n": 5,
            "title": "vs Random Forest",
            "min": 12,
            "body": """По п. 17. `RandomForestRegressor(n_estimators=150)` — validation MAE vs лучший бустинг.

**Критерий:** два MAE.""",
            "solution": """rf = RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
gb = GradientBoostingRegressor(
    n_estimators=best_n, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
).fit(X_train, y_train)
print('RF val MAE:', round(mean_absolute_error(y_val, rf.predict(X_val)), 1))
print('GB val MAE:', round(mean_absolute_error(y_val, gb.predict(X_val)), 1))""",
        },
        {
            "n": 6,
            "title": "Остатки вручную",
            "min": 12,
            "body": """По п. 3. На 4 точках из теории (`x=[1,2,5,8]`, `y=[10,14,18,30]`) посчитайте остатки после константного прогноза (среднее).

**Критерий:** массив residual.""",
            "solution": """x_demo = np.array([1., 2., 5., 8.])
y_demo = np.array([10., 14., 18., 30.])
pred0 = np.full_like(y_demo, y_demo.mean())
residual = y_demo - pred0
print('остатки:', residual)""",
        },
        {
            "n": 7,
            "title": "Boosting vs bagging",
            "min": 8,
            "body": """Markdown: одно предложение — чем boosting отличается от bagging (п. 2).

**Критерий:** последовательное исправление ошибок vs параллельные модели.""",
            "solution": "",
        },
        {
            "n": 8,
            "title": "Итог",
            "min": 5,
            "body": """Чек-лист из п. 16 — отметьте 4 пункта, которые выполнили сегодня.

**Критерий:** markdown список.""",
            "solution": "",
        },
    ],
    "final_md": None,
}


def patch_exercises(path: Path, n_theory: int, n_practice: int, topic: str):
    nb = json.loads(path.read_text(encoding="utf-8"))
    intro = (
        f"# Занятие {n_theory}. Упражнения: {topic}\n\n"
        f"Короткая проверка `{path.name.replace('_exercises', '_theory')}`). "
        f"Сквозной код — в `{path.name.replace('_exercises', '_practice')}` (занятие {n_practice}).\n"
    )
    nb["cells"][0]["source"] = [intro]
    nb.setdefault("metadata", {})["name"] = f"Занятие {n_theory}. Упражнения"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


SPECS = [
    (OV_DIR, "overfitting_validation", ov_spec, 33, 34, "валидация"),
    (DT_DIR, "decision_tree", dt_spec, 35, 36, "дерево"),
    (RF_DIR, "bagging_random_forest", rf_spec, 37, 38, "лес"),
    (GB_DIR, "gradient_boosting", gb_spec, 39, 40, "бустинг"),
]

for folder, prefix, spec, n_th, n_pr, topic in SPECS:
    practice_path = folder / f"{prefix}_practice.ipynb"
    solution_path = folder / f"{prefix}_practice_solution.ipynb"
    exercises_path = folder / f"{prefix}_exercises.ipynb"

    practice_cells = build_practice_cells(spec, filled=False)
    solution_cells = build_practice_cells(
        {**spec, "header": {**spec["header"]}},
        filled=True,
    )
    # Solution header note
    sol_header = solution_cells[0]
    sol_text = "".join(sol_header["source"])
    sol_text = sol_text.replace(
        "Вы **пишете весь код сами**.",
        "**Только для преподавателя. Не выдавать студентам.**\n\nЭталон практики.",
    )
    sol_lines = sol_text.split("\n")
    solution_cells[0]["source"] = [ln + "\n" for ln in sol_lines[:-1]] + [sol_lines[-1]]

    save_nb(practice_path, practice_cells, f"Занятие {n_pr}. Практика")
    save_nb(solution_path, solution_cells, f"Занятие {n_pr}. Решение (преподаватель)")

    patch_exercises(exercises_path, n_th, n_pr, topic)
    print("Wrote", practice_path.name, solution_path.name)

print("Done")
