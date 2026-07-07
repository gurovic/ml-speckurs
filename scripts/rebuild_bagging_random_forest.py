#!/usr/bin/env python3
"""Rebuild lesson 19_bagging_random_forest: theory, exercises, practice, solution."""

from __future__ import annotations

import json
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1] / "19_bagging_random_forest"
RANDOM_STATE = 42


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


SETUP_CODE = f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_STATE = {RANDOM_STATE}
"""


DATA_CODE = f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = {RANDOM_STATE}
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=7,
    flip_y=0.08, random_state=RANDOM_STATE,
)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE,
)
print('Объектов:', len(X), '| train:', len(X_train), '| validation:', len(X_val))
"""


def theory_cells() -> list[dict]:
    return [
        md(
            """# Занятие 37. Bagging и случайный лес

На занятии 35 одно дерево было **понятным**, но **нестабильным**: маленькое изменение train меняло верхние вопросы. Логичный вопрос: а что, если взять **много** деревьев и объединить их прогнозы?

Сначала — **математика**: почему голосование или усреднение **слабых** моделей иногда обгоняет одну «сильную». Потом — **виды ансамблей** в целом. И только затем — **bagging** и **случайный лес** как главная модель занятия.

**Сквозной пример:** синтетическая бинарная классификация (`make_classification`, 1000 объектов, 20 признаков). Данные и split — в следующем блоке с кодом; все демо после него — на них."""
        ),
        code(SETUP_CODE),
        # --- A. Математическая основа ---
        md(
            """## 1. Одна сильная модель или много слабых?

**Сильная модель** — одна сложная (глубокое дерево, большая нейросеть): гибкая, может подогнаться под train.

**Слабые модели** в ансамбле — проще по отдельности (мелкие деревья, «эксперты» с accuracy 55–65 %), но их **много**, и итог берётся **усреднением** (регрессия) или **голосованием** (классификация).

| Подход | Плюс | Риск |
|--------|------|------|
| Одна сильная | Просто обучить и деплоить | Высокий разброс, переобучение |
| Много слабых + объединение | Стабильнее, ошибки частично гасятся | Нужно разнообразие моделей; дороже по ресурсам |

Дальше — **почему** объединение работает: разложение ошибки на смещение и разброс и арифметика усреднения."""
        ),
        md(
            """## 2. Смещение, разброс и шум (bias–variance)

Ошибка модели на новых данных раскладывается (упрощённо):

$$\\mathbb{E}[(y - \\hat f)^2] = \\text{Bias}^2 + \\text{Var}(\\hat f) + \\text{шум}.$$

| Член | Смысл |
|------|--------|
| **Смещение (bias)** | Систематическая ошибка: модель «не той формы» (недообучение) |
| **Разброс (variance)** | Разброс прогноза при разных train: модель **нервная** |
| **Шум** | Случайность в $y$, которую не предсказать |

**Одна сильная гибкая модель** часто даёт **малый bias**, но **большой variance** (занятие 33). **Ансамбль** обычно **снижает variance**, почти не увеличивая bias — отсюда выигрыш над одним деревом."""
        ),
        md(
            """## 3. Почему усреднение уменьшает разброс

Пусть $M$ моделей выдают прогнозы $\\hat f_1, \\ldots, \\hat f_M$ с одинаковым разбросом $\\sigma^2$, а их **ошибки не коррелируют**. Тогда усреднение

$$\\bar f = \\frac{1}{M}\\sum_{j=1}^M \\hat f_j$$

имеет разброс

$$\\text{Var}(\\bar f) = \\frac{\\sigma^2}{M}.$$

**Чем больше независимых моделей, тем уже «облако» прогнозов.** Bias усреднения при одинаковых моделях почти не растёт; выигрываем за счёт **разброса**.

На практике модели **связаны** (одни и те же данные, похожие деревья) — выигрыш меньше, чем $1/M$, но всё равно заметен. Нужно **разнообразие** моделей."""
        ),
        code(
            """# Истинное значение 100; каждый «эксперт» шумит с σ=15
rng = np.random.default_rng(RANDOM_STATE)
true_value = 100
simulated = true_value + rng.normal(0, 15, size=(2000, 30))
single = simulated[:, 0]
ensemble = simulated.mean(axis=1)
print('σ одного прогноза:', round(single.std(), 1))
print('σ среднего 30 прогнозов:', round(ensemble.std(), 1))
plt.hist(single, bins=30, alpha=0.5, label='один эксперт')
plt.hist(ensemble, bins=30, alpha=0.7, label='среднее 30')
plt.axvline(true_value, color='black', ls='--')
plt.legend()
plt.title('Усреднение сужает разброс прогноза')
plt.xlabel('прогноз')
plt.show()
"""
        ),
        md(
            """## 4. Голосование классификаторов

В **классификации** вместо среднего числа — **голосование**: каждая модель голосует за класс, побеждает **большинство** (majority vote).

**Интуиция (парадокс Кондорсе).** Если каждый «эксперт» прав с вероятностью $p > 0.5$ и ошибается **независимо**, то при большом числе голосующих вероятность правильного **большинства** стремится к 1.

Для нечётного числа голосов вероятность правильного решения большинства равна сумме хвоста биномиального распределения:

$$P(\\text{majority correct}) = \\sum_{k=\\lfloor m/2 \\rfloor + 1}^{m} C_m^k p^k (1-p)^{m-k}.$$

Для $m=3$ можно посчитать вручную: большинство право, если правильно ответили ровно 2 или все 3 эксперта.

$$P = C_3^2 p^2(1-p) + C_3^3 p^3 = 3p^2(1-p)+p^3 = 3p^2-2p^3.$$

При $p=0.6$ получаем $P=0.648$, то есть группа из трёх уже лучше одного эксперта (0.6).

Пример: $p = 0.6$, 11 независимых голосов — вероятность, что большинство ошибётся, уже **меньше 0.3**. Отдельный эксперт слабый (60 %), коллектив — заметно сильнее.

Ниже — график: для каждого числа голосующих $m$ видно, как растёт $P(\\text{majority correct})$ при фиксированном $p=0.6$. Чем больше $m$, тем выше кривая."""
        ),
        code(
            """from math import comb

p_correct = 0.6  # accuracy одного слабого классификатора
n_voters = 11
# P(большинство право) = P(число правильных > n/2)
p_majority = sum(
    comb(n_voters, k) * p_correct**k * (1 - p_correct) ** (n_voters - k)
    for k in range(n_voters // 2 + 1, n_voters + 1)
)
print(f'Один эксперт: {p_correct:.0%} правильных')
print(f'Большинство из {n_voters}: {p_majority:.1%} правильных')

# Симуляция: 5000 «экспертов» по 11 голосов
rng = np.random.default_rng(RANDOM_STATE)
votes = rng.random((5000, n_voters)) < p_correct
majority_ok = votes.sum(axis=1) > n_voters // 2
print(f'Симуляция: {majority_ok.mean():.1%} правильных решений')

# Кривые Кондорсе: P(большинство право) от p для разного числа голосующих
p_grid = np.linspace(0.5, 1.0, 100)
plt.figure(figsize=(7, 4))
for m in [3, 7, 11, 21]:
    probs = [
        sum(comb(m, k) * p**k * (1 - p) ** (m - k) for k in range(m // 2 + 1, m + 1))
        for p in p_grid
    ]
    plt.plot(p_grid, probs, label=f'm={m}')
plt.axvline(0.6, color='gray', ls=':', lw=1)
plt.xlabel('p — вероятность, что один эксперт прав')
plt.ylabel('P(большинство право)')
plt.title('Парадокс Кондорсе: больше голосующих → выше P(majority)')
plt.legend()
plt.ylim(0.5, 1.0)
plt.show()
"""
        ),
        md(
            """## 5. Разнообразие — главное условие ансамбля

Усреднение и голосование помогают, только если модели **ошибаются не одинаково**.

| Ситуация | Эффект ансамбля |
|----------|-----------------|
| Ошибки **независимы** | Сильное снижение разброса ($\\approx 1/M$) |
| Ошибки **сильно коррелируют** | Почти нет выигрыша: все промахиваются вместе |
| Одинаковый сильный признак в каждом дереве | Деревья похожи → мало разнообразия |

**Bagging** создаёт разнообразие через **разные bootstrap-выборки**: каждая базовая модель видит слегка другой train. Другие способы (случайные признаки в узлах и т.п.) — в блоке про random forest ниже."""
        ),
        code(
            """# Два независимых шумных прогноза → среднее стабильнее
# Два одинаковых (корреляция 1) → среднее = тому же шуму
rng = np.random.default_rng(RANDOM_STATE)
pred1 = 100 + rng.normal(0, 10, 5000)
pred2_indep = 100 + rng.normal(0, 10, 5000)
pred2_same_error = pred1.copy()
print('σ одного:', round(pred1.std(), 1))
print('σ среднего независимых:', round(((pred1 + pred2_indep) / 2).std(), 1))
print('σ среднего одинаковых ошибок:', round(((pred1 + pred2_same_error) / 2).std(), 1))
"""
        ),
        # --- B. Виды ансамблей ---
        md(
            """## 6. Bagging, boosting и stacking — обзор

| Метод | Как обучают | Что улучшают | Пример sklearn |
|-------|-------------|--------------|----------------|
| **Bagging** | Параллельно, на разных bootstrap-выборках | **Разброс** (variance) | `BaggingClassifier`, `RandomForestClassifier` |
| **Boosting** | Последовательно, каждая модель исправляет ошибки предыдущих | **Смещение** (bias) | `GradientBoostingClassifier` (занятие 40) |
| **Stacking** | Базовые модели + **мета-модель** над их прогнозами | Комбинация разных типов моделей | `StackingClassifier` (вне курса) |

**Это занятие** — про **bagging** и **случайный лес**. Boosting — следующая тема; stacking упоминаем как идею «модель над моделями».

Общее правило: ансамбль выигрывает, когда базовые модели **достаточно хороши**, но **разные**."""
        ),
        # --- C. Основная тема ---
        md(
            """## 7. Нестабильность одиночного дерева

На занятии 35 глубокое дерево легко **переобучается**. Ещё одна проблема — **нестабильность**: два дерева на слегка разных bootstrap-выборках часто дают **разные** верхние вопросы.

На сквозном `make_classification` ниже сначала сравним **одно дерево** и **bagging**. Идея: если отдельное дерево нервное, много разных деревьев можно усреднить."""
        ),
        md(
            """## 8. Сквозной датасет и разбиение train/validation

**Датасет:** `make_classification` — 1000 объектов, 20 признаков, 7 информативных, лёгкий шум (`flip_y=0.08`). Бинарная классификация.

**Разбиение:** 70 % train, 30 % validation, `stratify=y`, `random_state=42`. Как на занятии 33: гиперпараметры и сравнение моделей — по **validation**; отдельный test здесь не выделяем."""
        ),
        code(DATA_CODE),
        md(
            """## 9. Bootstrap-выборка

**Bootstrap** — выборка длины $n$ из $n$ объектов **с возвращением**: один объект может попасть 0, 1 или несколько раз.

«С возвращением» — как **мешок с номерками**: вытащили один номер, записали, **положили обратно** и тянем снова. Поэтому один объект может выпасть несколько раз, а часть — ни разу.

В bagging **каждая** базовая модель обучается на своей bootstrap-выборке из train. Разные выборки → разные модели → разнообразие."""
        ),
        code(
            """rng = np.random.default_rng(RANDOM_STATE)
objects = np.arange(10)
bootstrap = rng.choice(objects, size=len(objects), replace=True)
oob = np.setdiff1d(objects, np.unique(bootstrap))
print('Bootstrap:', bootstrap)
print('Не попали (OOB для этого дерева):', oob)
"""
        ),
        md(
            """## 10. Доля уникальных объектов в bootstrap *

При большом $n$ в bootstrap попадает примерно **63,2 %** уникальных объектов train; около **36,8 %** не попадают ни разу — это **out-of-bag (OOB)** объекты **для данного дерева**.

Формула: $P(\\text{объект не выбран за } n \\text{ попыток}) \\approx (1 - 1/n)^n \\to 1/e \\approx 0.368$.

Это **средние** доли; в конкретной выборке из 10 объектов числа могут сильно отличаться."""
        ),
        md(
            """## 11. Алгоритм bagging

**Bagging** (bootstrap aggregating):

1. Повторить $B$ раз: взять bootstrap-выборку из train.
2. Обучить базовую модель (здесь — дерево) на каждой выборке **независимо**.
3. **Регрессия:** усреднить числовые прогнозы.
4. **Классификация:** голосование или среднее вероятностей.

В sklearn: `BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=...)`. Деревья можно строить **параллельно** (`n_jobs`).
Классический bagging можно применять и к другим нестабильным базовым моделям, но в курсе фокус на деревьях, потому что на них эффект наиболее наглядный."""
        ),
        md(
            """## 12. Усреднение, голосование и сравнение дерева с bagging

**Регрессия (ручной пример):** прогнозы деревьев `[72, 80, 75, 77, 71]` → среднее **75**.

**Классификация (голосование):** голоса `[1,1,0,1,0]` → класс **1** (три из пяти).

Ниже — одно дерево и bagging на сквозном train/validation."""
        ),
        code(
            """tree_predictions = np.array([72.0, 80, 75, 77, 71])
print('Средний прогноз регрессии:', tree_predictions.mean())
votes = np.array([1, 1, 0, 1, 0])
print('Голоса класса 1:', votes.mean(), '→ класс', int(votes.mean() >= 0.5))

models = {
    'одно дерево': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'bagging': BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1,
    ),
}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    tr = accuracy_score(y_train, clf.predict(X_train))
    va = accuracy_score(y_val, clf.predict(X_val))
    print(f'{name:14s}  train={tr:.3f}  validation={va:.3f}')
"""
        ),
        md(
            """## 13. Связь ошибок и выигрыш ансамбля

Усреднение сильнее всего, когда прогнозы базовых моделей **шумные**, но **не совпадают по знаку ошибки**. Если все модели систематически завышают прогноз — среднее тоже завышено (bias не исчезает).

Bagging снижает корреляцию ошибок за счёт разных bootstrap-выборок. Дополнительные приёмы для деревьев — в блоке random forest."""
        ),
        md(
            """## 14. Out-of-bag оценка

Объекты, **не попавшие** в bootstrap конкретного дерева, для этого дерева — **OOB**. Для каждого объекта train можно собрать прогнозы **только тех** деревьев, которые его не видели, и сравнить с истинным классом.

**OOB score** — accuracy по таким «честным» прогнозам на train. Удобен как **быстрая** дополнительная оценка, но **не заменяет** validation при подборе гиперпараметров."""
        ),
        md(
            """## 15. Как собирается OOB-прогноз

Для объекта $i$:

1. Найти все деревья, в чьей bootstrap-выборке **не было** $i$.
2. Усреднить их `predict_proba` (или взять голосование).
3. Сравнить с $y_i$.

У разных объектов разное число OOB-деревьев → при **малом** `n_estimators` OOB-оценка шумная."""
        ),
        code(
            """rf_oob = RandomForestClassifier(
    n_estimators=300, oob_score=True, random_state=RANDOM_STATE, n_jobs=-1,
)
rf_oob.fit(X_train, y_train)
val_acc = accuracy_score(y_val, rf_oob.predict(X_val))
print('OOB score (train):', round(rf_oob.oob_score_, 3))
print('Validation accuracy:', round(val_acc, 3))
"""
        ),
        md(
            """## 16. Случайный лес

**Random Forest** = bagging деревьев **+** в каждом узле при поиске лучшего разбиения рассматривается **случайное подмножество** признаков (`max_features`).

Обычное дерево на всех признаках часто снова и снова выбирает **один сильный** столбец → деревья похожи. Случайные признаки **разводят** деревья; отдельное дерево чуть слабее, ансамбль — сильнее.

Сравним bagging и random forest на том же train/validation:"""
        ),
        code(
            """compare = {
    'bagging': BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1,
    ),
    'random forest': RandomForestClassifier(
        n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1,
    ),
}
for name, clf in compare.items():
    clf.fit(X_train, y_train)
    tr = accuracy_score(y_train, clf.predict(X_train))
    va = accuracy_score(y_val, clf.predict(X_val))
    print(f'{name:14s}  train={tr:.3f}  validation={va:.3f}')
"""
        ),
        md(
            """## 17. Случайные признаки и разнообразие деревьев

В обычном дереве в каждом узле перебирают все признаки и выбирают лучший сплит. Если в данных есть 1-2 очень сильных признака, многие деревья снова выбирают их в верхних узлах, и деревья становятся похожими.

В random forest в каждом узле сначала случайно выбирают подмножество признаков, и лучший сплит ищут только внутри этого подмножества. Из-за этого:

- деревья чаще расходятся по структуре;
- корреляция ошибок между деревьями падает;
- усреднение работает сильнее.

Цена: отдельное дерево обычно немного слабее, чем дерево на всех признаках. Выгода: весь ансамбль чаще сильнее за счёт разнообразия.

| `max_features` | Эффект |
|----------------|--------|
| `'sqrt'` (классификация по умолчанию) | Умеренное разнообразие |
| Меньше признаков в узле | Больше разнообразие, слабее отдельные деревья |
| Все признаки | Ближе к bagging без RF-рандомизации |

Цель — **баланс**: деревья должны быть **разными**, но не **случайными угадывателями**."""
        ),
        md(
            """## 18. Классификация и регрессия в лесу

**Регрессия** (`RandomForestRegressor`): среднее прогнозов деревьев.

**Классификация:** sklearn усредняет **вероятности** классов по деревьям и берёт класс с максимальной средней вероятностью. Это может отличаться от простого подсчёта голосов «жёстких» классов, если деревья по-разному уверены."""
        ),
        md(
            """## 19. Основные гиперпараметры

| Параметр | Смысл |
|----------|--------|
| `n_estimators` | Число деревьев; больше — стабильнее, но медленнее |
| `max_features` | Сколько признаков в узле (ключ RF) |
| `max_depth`, `min_samples_leaf` | Сложность **одного** дерева |
| `max_samples` | Размер выборки для каждого дерева при `bootstrap=True`: по умолчанию берут столько же объектов, сколько в train; можно задать меньше для ускорения |
| `bootstrap` | Включить выборку с возвращением (`True`) или обучать каждое дерево на случайном поднаборе без возвращения (`False`) |

Добавление деревьев **редко** вызывает классическое переобучение: validation часто выходит на **плато**, а train продолжает расти."""
        ),
        code(
            """scores = []
for n in [1, 5, 20, 60, 150, 300]:
    m = RandomForestClassifier(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X_train, y_train)
    scores.append(accuracy_score(y_val, m.predict(X_val)))
print(dict(zip([1, 5, 20, 60, 150, 300], [round(s, 3) for s in scores])))
"""
        ),
        md(
            """## 20. Что настраивать первым

1. **`n_estimators`** — достаточно большое, пока validation не **стабилизируется** (кривая плато).
2. **`max_depth` / `min_samples_leaf`** — ограничить переобучение **отдельного** дерева.
3. **`max_features`** — для RF обычно оставляют default или слегка варьируют.

Сравнение — **один протокол** (тот же split, те же метрики). OOB — быстрый ориентир, финальный выбор — по validation."""
        ),
        md(
            """## 21. Дисбаланс: метрики, class_weight и порог

При редком классе **accuracy** обманчива. Смотрят precision, recall, F1, PR-кривую на validation.

`class_weight='balanced'` меняет веса классов **при обучении** деревьев. **Порог вероятности** — другой рычаг: после `predict_proba` вручную задают правило вида `y_pred = (p >= threshold)`; в sklearn это обычно настраивают отдельным шагом кода, а не параметром `RandomForestClassifier`."""
        ),
        md(
            """## 22. Impurity importance в лесу

`feature_importances_` — усреднённый по деревьям **impurity gain** (занятие 35). Удобно для первого взгляда, но:

- не доказывает причинность;
- завышает признаки с множеством уникальных значений;
- делит важность между коррелирующими признаками."""
        ),
        code(
            """rf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
imp = pd.Series(rf.feature_importances_, index=[f'f{i}' for i in range(X.shape[1])])
imp.sort_values(ascending=False).head(5)
"""
        ),
        md(
            """## 23. Permutation importance

На **validation** перемешивают один столбец и смотрят **падение** accuracy (или другой метрики). Показывает, использовал ли **готовый** лес признак на новых данных — честнее impurity importance с train.

Если важность получилась отрицательной, это значит: после перемешивания метрика чуть выросла (обычно в пределах статистического шума). Практически такой признак считают неинформативным для модели."""
        ),
        code(
            """from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf, X_val, y_val, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
)
perm = pd.Series(result.importances_mean, index=[f'f{i}' for i in range(X.shape[1])])
perm.sort_values(ascending=False).head(5)
"""
        ),
        md(
            """## 24. OOB, validation и test

**Random forest** — частный случай bagging для деревьев: те же bootstrap-выборки, плюс случайный поднабор признаков в узлах. Поэтому сравнение bagging и RF на validation — сравнение **двух конкретных моделей**, а не разных типов задач.

| Оценка | Когда |
|--------|--------|
| **OOB** | Быстрая проверка на train: для каждого объекта прогноз только от деревьев, которые его не видели при обучении |
| **Validation** | Подбор гиперпараметров и сравнение моделей (bagging, random forest и др.) |
| **Test** | Один раз в конце проекта (занятие 33) |

Не подбирайте `max_depth` по OOB, а потом снова по validation на тех же данных без протокола — это **переобучение на метрике**."""
        ),
        md(
            """## 25. Скорость, память и параллельное обучение

Деревья в bagging/RF **независимы** → `n_jobs=-1` использует все ядра. Стоимость: память под $B$ деревьев, время прогноза растёт с `n_estimators`.

На больших данных сначала уменьшают `max_depth` или `n_estimators`, включают параллельность и при необходимости обучают на подвыборке (`max_samples`) — это обычно даёт основной выигрыш по времени."""
        ),
        md(
            """## 26. Когда лес полезен и где ограничен

**Полезен:** табличные данные, нелинейности, взаимодействия признаков, мало предобработки, нужен сильный baseline.

**Ограничен:** экстраполяция за пределами train, очень разреженные высокоразмерные данные, нужна максимальная интерпретируемость одного правила, жёсткие лимиты памяти/латентности.

**Сравнение с нейросетями (практически):**
- на небольших и средних табличных данных лес часто даёт сильный baseline быстрее и с меньшей настройкой;
- на изображениях, тексте и аудио обычно выигрывают нейросети, потому что лучше извлекают сложные представления из сырых данных;
- в проде лес используют в задачах скоринга, риска, оттока, антифрода и как надёжную модель сравнения при запуске новых подходов.

> **Главная мысль.** Сначала — **зачем** объединять модели (снижение разброса при разнообразии). Затем — **bagging** и **RF** как рабочая реализация для деревьев. На занятии 40 — **boosting**, другой тип ансамбля."""
        ),
        md(
            """## 27. * Bagging, RF и Extra Trees

**Extra Trees** (Extremely Randomized Trees, в sklearn — `ExtraTreesClassifier`) — ансамбль деревьев, похожий на random forest, но с **ещё большей случайностью** при построении:

1. В каждом узле, как в RF, рассматривают случайное подмножество признаков (`max_features`).
2. Для каждого признака порог **не подбирают оптимально** по impurity — берут **случайное** значение между min и max признака на выборке узла.
3. По умолчанию деревья обучают на **всём train** (`bootstrap=False`), а не на bootstrap-выборке.

Из-за этого отдельные деревья слабее, но строятся **быстрее**; разнообразие ансамбля часто высокое. Итоговое качество сравнивают с RF на validation — универсального победителя нет.

| Метод | Bootstrap | Случайные признаки в узле | Пороги в узле | sklearn |
|-------|-----------|---------------------------|---------------|---------|
| **Bagging** | да | нет (как у базового дерева) | оптимальные по impurity | `BaggingClassifier` |
| **Random Forest** | да | да | оптимальные по impurity | `RandomForestClassifier` |
| **Extra Trees** | по умолчанию нет | да | **случайные** | `ExtraTreesClassifier` |

Для курса достаточно bagging и random forest; Extra Trees — опциональное расширение для тех, кто хочет ещё один быстрый baseline на табличных данных."""
        ),
    ]


def exercises_cells() -> list[dict]:
    exercises = [
        (
            "## 1. Разброс среднего\n\n30 независимых шумных прогноза вокруг 100 с σ=10. Найдите σ **среднего** (теория: σ/√30).",
            "rng = np.random.default_rng(42)\n"
            "parts = 100 + rng.normal(0, 10, size=(5000, 30))\n"
            "single_std = ...\n"
            "mean_std = ...\n"
            "assert mean_std < single_std / 3\n"
            "print('Верно')",
        ),
        (
            "## 2. Majority vote\n\n7 голосов: [1,0,1,1,0,1,0]. Какой класс победил?",
            "votes = np.array([1, 0, 1, 1, 0, 1, 0])\n"
            "winner = ...\n"
            "assert winner == 1\n"
            "print('Верно')",
        ),
        (
            "## 3. Виды ансамблей\n\nКакой метод обучает модели **последовательно**?",
            "sequential_method = ...  # 'bagging', 'boosting' или 'stacking'\n"
            "assert sequential_method == 'boosting'\n"
            "print('Верно')",
        ),
        (
            "## 4. Bootstrap\n\nВыборка длины 5 с возвращением из [0,1,2,3,4]. Может ли индекс 2 встретиться 3 раза?",
            "can_repeat = ...\n"
            "assert can_repeat is True\n"
            "print('Верно')",
        ),
        (
            "## 5. Доля уникальных\n\nПри n=1000 bootstrap ожидаемая доля уникальных ≈ 0.632. Для n=10 посчитайте среднюю долю уникальных за 2000 повторов.",
            "rng = np.random.default_rng(42)\n"
            "n = 10\n"
            "uniq_fracs = []\n"
            "for _ in range(2000):\n"
            "    b = rng.choice(n, size=n, replace=True)\n"
            "    uniq_fracs.append(len(np.unique(b)) / n)\n"
            "mean_unique = ...\n"
            "assert 0.6 < mean_unique < 0.95\n"
            "print('Верно:', round(mean_unique, 3))",
        ),
        (
            "## 6. Усреднение регрессии\n\nПрогнозы деревьев [10, 12, 11]. Средний прогноз?",
            "preds = np.array([10.0, 12, 11])\n"
            "ensemble_pred = ...\n"
            "assert np.isclose(ensemble_pred, 11.0)\n"
            "print('Верно')",
        ),
        (
            "## 7. Голосование\n\nГолоса [0,0,1,1,1]. Класс большинства?",
            "votes = np.array([0, 0, 1, 1, 1])\n"
            "majority_class = ...\n"
            "assert majority_class == 1\n"
            "print('Верно')",
        ),
        (
            "## 8. OOB-доля\n\nПри большом n доля OOB ≈ 1/e ≈ 0.368. Вычислите 1/np.e.",
            "oob_fraction = ...\n"
            "assert np.isclose(oob_fraction, 1/np.e, atol=0.01)\n"
            "print('Верно:', round(oob_fraction, 3))",
        ),
        (
            "## 9. Плато n_estimators\n\nПри росте числа деревьев validation обычно…",
            "trend = ...  # 'растёт без остановки', 'выходит на плато' или 'падает'\n"
            "assert trend == 'выходит на плато'\n"
            "print('Верно')",
        ),
        (
            "## 10. Bagging vs RF\n\nЧто добавляет RF поверх bagging?",
            "extra = ...  # строка: 'случайные признаки в узлах' или 'бустинг'\n"
            "assert extra == 'случайные признаки в узлах'\n"
            "print('Верно')",
        ),
    ]
    cells = [
        md(
            "# Занятие 37. Упражнения: bagging и случайный лес\n\n"
            "Короткая проверка теории (`bagging_random_forest_theory.ipynb`, занятие 37). "
            "Сквозной код — в `bagging_random_forest_practice.ipynb` (занятие 38).\n"
        ),
        code("import numpy as np\nimport pandas as pd\n"),
    ]
    for title, src in exercises:
        cells.append(md(title + "\n"))
        cells.append(code(src))
    return cells


def practice_cells(solution: bool = False) -> list[dict]:
    if solution:
        header = (
            "# Занятие 38. Решение практики (только преподаватель)\n\n"
            "**Не выдавать студентам.**\n\n"
            "Код заполнен для проверки преподавателем. Главная модель — **RandomForestClassifier** "
            "(теория — занятие 37).\n\n"
            "Решение сравнивает одно дерево, bagging и random forest; настраивает `n_estimators`, "
            "проверяет OOB и permutation importance.\n"
        )
    else:
        header = (
            "# Занятие 38. Практика: bagging и случайный лес (~90 мин)\n\n"
            "Вы **пишете весь код сами**. Ячейку **«Дано»** не меняйте.\n\n"
            "Главная модель — **RandomForestClassifier** (теория — занятие 37).\n\n"
            "Сравним одно дерево, bagging и random forest; настроим `n_estimators`, посмотрим OOB.\n"
        )

    cells: list[dict] = [md(header)]
    cells.append(md("---\n## Дано: make_classification\n\n20 признаков — как в теории."))
    cells.append(
        code(
            """import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=7, flip_y=0.08, random_state=42
)
print('Объектов:', len(X))"""
        )
    )

    tasks: list[tuple[str, str | None]] = [
        (
            "---\n## Задание 0. Split (~8 мин)\n\n"
            "Импорты: `train_test_split`, `DecisionTreeClassifier`, `BaggingClassifier`, "
            "`RandomForestClassifier`, `accuracy_score`, `permutation_importance`. "
            "Split 70/30, stratify, `RANDOM_STATE=42`.\n\n**Критерий:** train/val готовы.",
            """RANDOM_STATE = 42
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE,
)
print(len(X_train), len(X_val))""",
        ),
        (
            "---\n## Задание 1. Три модели (~15 мин)\n\n"
            "Одно дерево, bagging (150 estimators), random forest (150). Train и validation accuracy.\n\n"
            "**Критерий:** bagging и/или RF заметно выше одного дерева на validation.",
            """models = {
    'tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'bagging': BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1,
    ),
    'rf': RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
}
for name, m in models.items():
    m.fit(X_train, y_train)
    print(name, accuracy_score(y_train, m.predict(X_train)),
          accuracy_score(y_val, m.predict(X_val)))""",
        ),
        (
            "---\n## Задание 2. n_estimators (~15 мин)\n\n"
            "График validation accuracy vs `n_estimators` in `[1,5,20,60,150,300]`.\n\n"
            "**Критерий:** кривая выходит на плато.",
            """ns = [1, 5, 20, 60, 150, 300]
val_scores = []
for n in ns:
    m = RandomForestClassifier(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X_train, y_train)
    val_scores.append(accuracy_score(y_val, m.predict(X_val)))
plt.plot(ns, val_scores, 'o-')
plt.xlabel('n_estimators')
plt.ylabel('validation accuracy')
plt.title('Плато качества леса')
plt.show()""",
        ),
        (
            "---\n## Задание 3. OOB score (~10 мин)\n\n"
            "`RandomForestClassifier(oob_score=True, n_estimators=300)`. Сравните OOB и validation.\n\n"
            "**Критерий:** оба числа напечатаны.",
            """rf = RandomForestClassifier(
    n_estimators=300, oob_score=True, random_state=RANDOM_STATE, n_jobs=-1,
)
rf.fit(X_train, y_train)
print('OOB:', rf.oob_score_)
print('val:', accuracy_score(y_val, rf.predict(X_val)))""",
        ),
        (
            "---\n## Задание 4. max_depth (~12 мин)\n\n"
            "`max_depth` in `[None, 5, 10, 20]` при `n_estimators=150`. Лучший по validation.\n\n"
            "**Критерий:** `best_depth` выбран.",
            """best_depth, best_acc = None, -1
for d in [None, 5, 10, 20]:
    m = RandomForestClassifier(
        n_estimators=150, max_depth=d, random_state=RANDOM_STATE, n_jobs=-1,
    )
    m.fit(X_train, y_train)
    acc = accuracy_score(y_val, m.predict(X_val))
    print(d, acc)
    if acc > best_acc:
        best_acc, best_depth = acc, d
print('best_depth:', best_depth)""",
        ),
        (
            "---\n## Задание 5. Bootstrap вручную (~12 мин)\n\n"
            "100 повторов bootstrap длины `len(X_train)`. Средняя доля уникальных.\n\n"
            "**Критерий:** ~0.63 при большом n.",
            """rng = np.random.default_rng(RANDOM_STATE)
n = len(X_train)
fracs = [len(np.unique(rng.choice(n, n, replace=True))) / n for _ in range(100)]
print('средняя доля уникальных:', round(np.mean(fracs), 3))""",
        ),
        (
            "---\n## Задание 6. Permutation importance (~15 мин)\n\n"
            "Top-5 признаков на validation.\n\n"
            "**Критерий:** barh или таблица top-5.",
            """rf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
r = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=RANDOM_STATE)
imp = r.importances_mean
top = np.argsort(imp)[-5:]
for i in top[::-1]:
    print(f'f{i}', round(imp[i], 4))""",
        ),
        (
            "---\n## Задание 7. Train vs val gap (~8 мин)\n\n"
            "В markdown: почему высокий train acc у леса не всегда переобучение?\n\n"
            "**Критерий:** 2–3 предложения.",
            """# Лес усредняет много деревьев: на train каждое дерево может подгоняться,
# но ансамбль стабильнее одного дерева. Большой разрыв train–val бывает,
# но добавление деревьев реже ухудшает validation — в отличие от одной глубокой модели.""",
        ),
        (
            "---\n## Задание 8. Итог (~5 мин)\n\n"
            "Сравните bagging и RF в одном предложении каждый.\n\n"
            "**Критерий:** markdown.",
            """# Bagging: много деревьев на разных bootstrap-выборках, усреднение снижает разброс.
# RF: то же + случайный поднабор признаков в узлах → деревья разнообразнее.""",
        ),
    ]

    for md_text, sol in tasks:
        cells.append(md(md_text))
        if solution and sol is not None:
            if sol.strip().startswith("#"):
                cells.append(md(sol))
            else:
                cells.append(code(sol))
        else:
            cells.append(code("" if sol is None else ""))

    return cells


def notebook(cells: list[dict], name: str) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "name": name,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(path: Path, cells: list[dict], name: str) -> None:
    path.write_text(
        json.dumps(notebook(cells, name), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Wrote {path.name}: {len(cells)} cells")


def main() -> None:
    write_nb(
        FOLDER / "bagging_random_forest_theory.ipynb",
        theory_cells(),
        "Занятие 37. Bagging и случайный лес",
    )
    write_nb(
        FOLDER / "bagging_random_forest_exercises.ipynb",
        exercises_cells(),
        "Занятие 37. Упражнения",
    )
    write_nb(
        FOLDER / "bagging_random_forest_practice.ipynb",
        practice_cells(solution=False),
        "Занятие 38. Практика",
    )
    write_nb(
        FOLDER / "bagging_random_forest_practice_solution.ipynb",
        practice_cells(solution=True),
        "Занятие 38. Решение",
    )


if __name__ == "__main__":
    main()
