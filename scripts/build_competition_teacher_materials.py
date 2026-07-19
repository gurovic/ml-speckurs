from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def lines(text: str) -> list[str]:
    return (text.strip("\n") + "\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def add_teacher_section(readme_path: Path, section: str) -> None:
    marker = "## Дополнительные материалы"
    old = readme_path.read_text(encoding="utf-8")
    base = old.split(marker)[0].rstrip()
    readme_path.write_text(base + "\n\n" + marker + "\n\n" + section.strip() + "\n", encoding="utf-8")


LINEAR_NOTES = """
# Разбор решения: линейная регрессия и цена квартиры

Этот файл — для преподавателя. Его удобно использовать после занятия: показать не один “правильный ответ”, а набор идей, из которых складывается хорошее решение.

## Что важно проверить у учеников

- Поняли ли они объект и цель: одна строка — квартира, цель — `price_mln`.
- Не используют ли они скрытый ответ из `private_target.csv`.
- Делают ли validation-разбиение до выбора признаков и модели.
- Сравнивают ли идеи одной и той же метрикой: MAE.
- Создают ли одинаковые признаки для train и test.
- Не забывают ли обрабатывать пропуски и категориальные признаки.

## Минимальный путь к рабочему решению

1. Прочитать `train.csv`, `test.csv`, `sample_submission.csv`.
2. Убрать из признаков `id` и `price_mln`.
3. Разделить `train.csv` на train/validation.
4. Числовые признаки заполнить медианой и масштабировать.
5. Категориальные признаки заполнить частым значением и закодировать one-hot.
6. Обучить `LinearRegression` или `Ridge`.
7. Посчитать MAE на validation.
8. Обучить финальную модель на всём train и сохранить `submission.csv`.

## Идеи признаков

- `area_per_room = area_m2 / rooms`: площадь на комнату.
- `kitchen_share = kitchen_m2 / area_m2`: доля кухни в квартире.
- `relative_floor = floor / floors_total`: насколько высоко находится квартира.
- `is_first_floor`, `is_last_floor`: первый и последний этаж часто оцениваются иначе.
- `no_metro`: отдельный флаг, если расстояние до метро неизвестно.
- `log_views = log1p(views_30d)`: просмотры имеют длинный хвост, логарифм делает признак спокойнее.
- `age_x_distance = house_age * distance_to_center_km`: старый дом далеко от центра может влиять иначе, чем новый.
- `area_x_center = area_m2 / (distance_to_center_km + 1)`: простое взаимодействие площади и близости к центру.

## Идеи моделей

- Константный baseline: всегда предсказывать среднюю цену.
- `LinearRegression`: первая честная линейная модель.
- `Ridge`: обычно устойчивее, чем обычная линейная регрессия, особенно после one-hot.
- `Lasso`: можно показать как модель, которая зануляет часть коэффициентов, но она чувствительна к масштабу.
- `ElasticNet`: компромисс L1 и L2, но для 9 класса лучше оставить как дополнительную идею.

## Частые ошибки

- Считать качество на test с помощью `private_target.csv` во время подбора.
- Создать признак только в train и забыть test.
- Закодировать категории вручную разными числами без проверки смысла.
- Сравнивать модели на разных validation-разбиениях.
- Ориентироваться только на R², хотя в задаче главным критерием является MAE.

## Что можно обсудить после занятия

- Почему MAE измеряется в миллионах рублей и поэтому легко объясняется.
- Почему RMSE сильнее наказывает большие промахи.
- Какие признаки оказались понятными человеку, а какие просто помогли модели.
- Почему `Ridge` часто выигрывает у `LinearRegression` на табличных данных с one-hot.
"""


LOGISTIC_NOTES = """
# Разбор решения: логистическая регрессия и завершение курса

Этот файл — для преподавателя. Его можно использовать после занятия для разбора решений и ошибок команд.

## Что важно проверить у учеников

- Поняли ли они, что это бинарная классификация: `will_finish` равен 0 или 1.
- Не путают ли вероятность класса 1 и итоговый класс.
- Используют ли стратификацию при разбиении.
- Сравнивают ли модели на validation.
- Подбирают ли порог вероятности, если главная метрика — F1.
- Не подбирают ли порог по скрытому test.

## Минимальный путь к рабочему решению

1. Прочитать данные.
2. Убрать `id` и целевой столбец из признаков.
3. Сделать `train_test_split(..., stratify=y)`.
4. Числовые признаки заполнить медианой и масштабировать.
5. Категориальные признаки закодировать one-hot.
6. Обучить `LogisticRegression(max_iter=1000)`.
7. Получить `predict_proba`.
8. Подобрать порог по F1 на validation.
9. Обучить модель на всём train и сохранить `submission.csv`.

## Идеи признаков

- `activity_total = practice_hours_week + 0.5 * club_visits`: общий учебный ритм.
- `score_per_missed_class = last_score / (missed_classes + 1)`: оценка с учётом пропусков.
- `teacher_messages_per_week = messages_to_teacher / 4`: частота обращений.
- `is_many_missed = missed_classes >= 4`: явный флаг частых пропусков.
- `strong_homework = homework_rate >= 0.8`: привычка сдавать задания.
- `low_last_score = last_score < 50`: риск по последней работе.

## Идеи моделей и параметров

- `LogisticRegression(C=1.0)`: базовая версия.
- `C < 1`: сильнее регуляризация, модель спокойнее.
- `C > 1`: слабее регуляризация, модель может лучше подстроиться, но риск переобучения выше.
- `class_weight="balanced"`: полезно проверить при дисбалансе классов.
- Порог не обязан быть 0.5: для F1 его почти всегда стоит перебрать.

## Частые ошибки

- Сохранить только вероятность и забыть итоговый столбец `will_finish`, если проверка ждёт класс.
- Считать F1 по вероятностям, а не по классам.
- Подобрать порог на test.
- Сравнивать accuracy вместо F1 и сделать неправильный вывод.
- Не использовать `stratify=y`, из-за чего validation может случайно получить другую долю классов.

## Что можно обсудить после занятия

- Почему ROC-AUC может быть хорошим, но F1 при пороге 0.5 — средним.
- Как меняются precision и recall при повышении порога.
- Почему вероятность 0.7 не означает “модель уверена на 70%” без разговора о калибровке.
"""


ENSEMBLE_NOTES = """
# Разбор решения: итоговое соревнование по модулю, классификация ансамблями

Этот файл — для преподавателя. Он помогает разобрать не только победившее решение, но и разные разумные стратегии.

## Что важно проверить у учеников

- Поняли ли они цель: предсказать вероятность ухода пользователя `churn_probability`.
- Используют ли ROC-AUC как главную метрику.
- Не превращают ли задачу только в 0/1-классификацию: для ROC-AUC нужны вероятности.
- Сравнивают ли несколько ансамблей по одной и той же validation-выборке.
- Пробуют ли внешние бустинги: XGBoost, LightGBM, CatBoost.
- Не используют ли `private_target.csv` во время подбора.

## Минимальный путь к рабочему решению

1. Прочитать данные.
2. Разделить train на train/validation со стратификацией.
3. Обработать числовые пропуски медианой.
4. Обработать категориальные пропуски и применить one-hot.
5. Сравнить несколько моделей: RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost.
6. Выбрать лучшую по ROC-AUC.
7. Обучить финальную модель на всём train.
8. Сохранить вероятности класса 1 в `submission.csv`.

## Идеи признаков

- `sessions_per_100_days`: активность с учётом возраста аккаунта.
- `price_after_discount`: реальная цена после скидки.
- `is_inactive`: давно не было активности.
- `support_per_session`: много обращений в поддержку при малой активности — сигнал риска.
- `failed_payment_flag`: были ли ошибки оплаты.
- `high_price_no_discount`: дорогой тариф без скидки.
- `low_homework_many_tickets`: мало выполненных заданий и много обращений.

## Идеи моделей

- `RandomForestClassifier`: устойчивый понятный baseline среди ансамблей.
- `ExtraTreesClassifier`: похож на случайный лес, но деревья более случайные.
- `GradientBoostingClassifier`: последовательное исправление ошибок.
- `HistGradientBoostingClassifier`: быстрый sklearn-бустинг.
- `XGBClassifier`: сильный внешний бустинг, много параметров.
- `LGBMClassifier`: часто быстрый и сильный на таблицах.
- `CatBoostClassifier`: хорошо работает с табличными задачами; в этом материале можно использовать через one-hot или отдельно обсудить нативные категории.

## Идеи настройки

- Для лесов: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.
- Для бустинга: `n_estimators`/`iterations`, `learning_rate`, `max_depth`/`depth`, `num_leaves`, `subsample`.
- Для всех моделей: сравнивать не один запуск на глаз, а таблицу метрик.
- Для финала: можно усреднить вероятности 2–3 сильных моделей.

## Частые ошибки

- Сдать 0/1 вместо вероятности для ROC-AUC.
- Выбрать модель по F1, хотя рейтинг считается по ROC-AUC.
- Сравнивать модели после разных разбиений.
- Слишком сильно подбирать параметры на одной validation-выборке.
- Забыть одинаково обработать train и test.

## Что можно обсудить после модуля

- Почему ансамбли часто выигрывают у одной линейной модели на нелинейных табличных зависимостях.
- Чем bagging отличается от boosting на практике.
- Почему усреднение вероятностей иногда улучшает ROC-AUC.
- Почему “лучшая модель на validation” не гарантирует победу на скрытом test.
"""


LINEAR_BASELINE = [
    md(
        """
# Простой baseline: линейная регрессия

Этот блокнот можно выдать ученикам как стартовый код, если цель занятия — проверять не загрузку данных и базовую предобработку, а улучшение модели.

Baseline делает минимум:

- читает train/test;
- кодирует категории one-hot;
- заполняет пропуски;
- обучает `LinearRegression`;
- сохраняет `submission.csv`.
"""
    ),
    code(
        """
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_DIR = Path("data")
"""
    ),
    code(
        """
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

target = "price_mln"
id_col = "id"

features = [c for c in train.columns if c not in [target, id_col]]
X = train[features]
y = train[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)
"""
    ),
    code(
        """
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_ohe()),
                ]
            ),
            categorical_features,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", LinearRegression()),
    ]
)

model.fit(X_train, y_train)
val_pred = model.predict(X_val)
print("Validation MAE:", round(mean_absolute_error(y_val, val_pred), 3))
"""
    ),
    code(
        """
model.fit(X, y)
test_pred = model.predict(test[features])

submission = pd.DataFrame(
    {
        "id": test[id_col],
        "price_mln": np.maximum(test_pred, 0).round(3),
    }
)

submission.to_csv("submission.csv", index=False)
submission.head()
"""
    ),
]


LINEAR_AUTHOR = [
    md(
        """
# Авторское решение: линейная регрессия и цена квартиры

Это не единственный правильный путь. Блокнот показывает аккуратный ход: validation, несколько признаков, сравнение линейных моделей и финальный `submission.csv`.
"""
    ),
    code(
        """
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_DIR = Path("data")
"""
    ),
    code(
        """
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

target = "price_mln"
id_col = "id"
"""
    ),
    code(
        """
def add_features(df):
    df = df.copy()
    df["area_per_room"] = df["area_m2"] / df["rooms"]
    df["kitchen_share"] = df["kitchen_m2"] / df["area_m2"]
    df["relative_floor"] = df["floor"] / df["floors_total"]
    df["is_first_floor"] = (df["floor"] == 1).astype(int)
    df["is_last_floor"] = (df["floor"] == df["floors_total"]).astype(int)
    df["no_metro"] = df["metro_min"].isna().astype(int)
    df["log_views"] = np.log1p(df["views_30d"])
    df["distance_squared"] = df["distance_to_center_km"] ** 2
    df["area_x_center"] = df["area_m2"] / (df["distance_to_center_km"] + 1)
    df["age_x_distance"] = df["house_age"] * df["distance_to_center_km"]
    return df


train_fe = add_features(train)
test_fe = add_features(test)

features = [c for c in train_fe.columns if c not in [target, id_col]]
X = train_fe[features]
y = train_fe[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)
"""
    ),
    code(
        """
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocess(X_part):
    numeric_features = X_part.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_part.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_ohe()),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def evaluate_model(name, estimator):
    model = Pipeline(
        steps=[
            ("preprocess", make_preprocess(X_train)),
            ("model", estimator),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    return {
        "name": name,
        "MAE": mean_absolute_error(y_val, pred),
        "RMSE": np.sqrt(mean_squared_error(y_val, pred)),
        "R2": r2_score(y_val, pred),
        "model": model,
    }
"""
    ),
    code(
        """
candidates = [
    ("LinearRegression", LinearRegression()),
    ("Ridge alpha=0.3", Ridge(alpha=0.3)),
    ("Ridge alpha=1", Ridge(alpha=1.0)),
    ("Ridge alpha=3", Ridge(alpha=3.0)),
    ("Ridge alpha=10", Ridge(alpha=10.0)),
    ("Lasso alpha=0.001", Lasso(alpha=0.001, max_iter=10000)),
    ("ElasticNet alpha=0.001", ElasticNet(alpha=0.001, l1_ratio=0.3, max_iter=10000)),
]

rows = []
models = {}
estimators_by_name = {name: estimator for name, estimator in candidates}
for name, estimator in candidates:
    result = evaluate_model(name, estimator)
    models[name] = result.pop("model")
    rows.append(result)

leaderboard = pd.DataFrame(rows).sort_values("MAE")
leaderboard
"""
    ),
    code(
        """
best_name = leaderboard.iloc[0]["name"]
best_name
"""
    ),
    code(
        """
final_model = Pipeline(
    steps=[
        ("preprocess", make_preprocess(X)),
        ("model", clone(estimators_by_name[best_name])),
    ]
)

final_model.fit(X, y)
test_pred = final_model.predict(test_fe[features])

submission = pd.DataFrame(
    {
        "id": test_fe[id_col],
        "price_mln": np.maximum(test_pred, 0).round(3),
    }
)

submission.to_csv("author_submission.csv", index=False)
submission.head()
"""
    ),
]


LOGISTIC_BASELINE = [
    md(
        """
# Простой baseline: логистическая регрессия

Этот блокнот можно выдать ученикам как стартовый код. Он делает базовую предобработку, обучает логистическую регрессию и сохраняет `submission.csv`.
"""
    ),
    code(
        """
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_DIR = Path("data")
"""
    ),
    code(
        """
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

target = "will_finish"
id_col = "id"
features = [c for c in train.columns if c not in [target, id_col]]

X = train[features]
y = train[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
"""
    ),
    code(
        """
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_ohe()),
                ]
            ),
            categorical_features,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)

model.fit(X_train, y_train)
val_proba = model.predict_proba(X_val)[:, 1]
val_pred = (val_proba >= 0.5).astype(int)
print("Validation F1:", round(f1_score(y_val, val_pred), 3))
"""
    ),
    code(
        """
model.fit(X, y)
test_proba = model.predict_proba(test[features])[:, 1]
test_pred = (test_proba >= 0.5).astype(int)

submission = pd.DataFrame(
    {
        "id": test[id_col],
        "probability": test_proba.round(5),
        "will_finish": test_pred,
    }
)

submission.to_csv("submission.csv", index=False)
submission.head()
"""
    ),
]


LOGISTIC_AUTHOR = [
    md(
        """
# Авторское решение: логистическая регрессия и завершение курса

Блокнот показывает полный ход: признаки, сравнение параметров `C` и `class_weight`, подбор порога по F1 и финальный файл.
"""
    ),
    code(
        """
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_DIR = Path("data")
"""
    ),
    code(
        """
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

target = "will_finish"
id_col = "id"
"""
    ),
    code(
        """
def add_features(df):
    df = df.copy()
    df["activity_total"] = df["practice_hours_week"] + 0.5 * df["club_visits"]
    df["score_per_missed_class"] = df["last_score"] / (df["missed_classes"] + 1)
    df["teacher_messages_per_week"] = df["messages_to_teacher"] / 4
    df["is_many_missed"] = (df["missed_classes"] >= 4).astype(int)
    df["strong_homework"] = (df["homework_rate"] >= 0.8).astype(int)
    df["low_last_score"] = (df["last_score"] < 50).astype(int)
    return df


train_fe = add_features(train)
test_fe = add_features(test)

features = [c for c in train_fe.columns if c not in [target, id_col]]
X = train_fe[features]
y = train_fe[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
"""
    ),
    code(
        """
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocess(X_part):
    numeric_features = X_part.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_part.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_ohe()),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def best_threshold_for_f1(y_true, proba):
    thresholds = np.linspace(0.15, 0.85, 71)
    scores = []
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        scores.append(f1_score(y_true, pred, zero_division=0))
    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def evaluate_model(name, estimator):
    model = Pipeline(
        steps=[
            ("preprocess", make_preprocess(X_train)),
            ("model", estimator),
        ]
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    threshold, best_f1 = best_threshold_for_f1(y_val, proba)
    pred = (proba >= threshold).astype(int)
    return {
        "name": name,
        "threshold": threshold,
        "F1": best_f1,
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "accuracy": accuracy_score(y_val, pred),
        "ROC_AUC": roc_auc_score(y_val, proba),
        "model": model,
    }
"""
    ),
    code(
        """
candidates = []
for C in [0.3, 1.0, 3.0, 10.0]:
    candidates.append((f"LogReg C={C}", LogisticRegression(max_iter=1000, C=C)))
    candidates.append(
        (
            f"LogReg C={C}, balanced",
            LogisticRegression(max_iter=1000, C=C, class_weight="balanced"),
        )
    )

rows = []
models = {}
estimators_by_name = {name: estimator for name, estimator in candidates}
for name, estimator in candidates:
    result = evaluate_model(name, estimator)
    models[name] = result.pop("model")
    rows.append(result)

leaderboard = pd.DataFrame(rows).sort_values("F1", ascending=False)
leaderboard
"""
    ),
    code(
        """
best_row = leaderboard.iloc[0]
best_name = best_row["name"]
best_threshold = float(best_row["threshold"])

print("Лучшая модель:", best_name)
print("Порог:", round(best_threshold, 3))
"""
    ),
    code(
        """
final_model = Pipeline(
    steps=[
        ("preprocess", make_preprocess(X)),
        ("model", clone(estimators_by_name[best_name])),
    ]
)

final_model.fit(X, y)
test_proba = final_model.predict_proba(test_fe[features])[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

submission = pd.DataFrame(
    {
        "id": test_fe[id_col],
        "probability": test_proba.round(5),
        "will_finish": test_pred,
    }
)

submission.to_csv("author_submission.csv", index=False)
submission.head()
"""
    ),
]


ENSEMBLE_BASELINE = [
    md(
        """
# Простой baseline: классификация ансамблем

Этот блокнот можно выдать как стартовый код для итогового соревнования. Он обучает один `RandomForestClassifier` и сохраняет вероятности.
"""
    ),
    code(
        """
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
DATA_DIR = Path("data")
"""
    ),
    code(
        """
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

target = "churn"
id_col = "id"
features = [c for c in train.columns if c not in [target, id_col]]

X = train[features]
y = train[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
"""
    ),
    code(
        """
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_ohe()),
                ]
            ),
            categorical_features,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

model.fit(X_train, y_train)
val_proba = model.predict_proba(X_val)[:, 1]
print("Validation ROC-AUC:", round(roc_auc_score(y_val, val_proba), 3))
"""
    ),
    code(
        """
model.fit(X, y)
test_proba = model.predict_proba(test[features])[:, 1]

submission = pd.DataFrame(
    {
        "id": test[id_col],
        "churn_probability": test_proba.round(5),
    }
)

submission.to_csv("submission.csv", index=False)
submission.head()
"""
    ),
]


ENSEMBLE_AUTHOR = [
    md(
        """
# Авторское решение: итоговое соревнование по модулю

Блокнот сравнивает несколько ансамблей, включая XGBoost, LightGBM и CatBoost, а затем строит финальный файл с вероятностями.
"""
    ),
    code(
        """
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42
DATA_DIR = Path("data")
"""
    ),
    code(
        """
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

target = "churn"
id_col = "id"
"""
    ),
    code(
        """
def add_features(df):
    df = df.copy()
    df["sessions_per_100_days"] = df["sessions_last_30"] / (df["account_age_days"] / 100 + 1)
    df["price_after_discount"] = df["plan_price"] * (1 - df["discount_percent"] / 100)
    df["is_inactive"] = (df["days_since_last_activity"] >= 14).astype(int)
    df["support_per_session"] = df["support_tickets"] / (df["sessions_last_30"] + 1)
    df["failed_payment_flag"] = (df["failed_payments"] > 0).astype(int)
    df["high_price_no_discount"] = ((df["plan_price"] >= 1590) & (df["discount_percent"] <= 5)).astype(int)
    df["low_homework_many_tickets"] = ((df["homework_completion"] < 0.45) & (df["support_tickets"] >= 3)).astype(int)
    return df


train_fe = add_features(train)
test_fe = add_features(test)

features = [c for c in train_fe.columns if c not in [target, id_col]]
X = train_fe[features]
y = train_fe[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
"""
    ),
    code(
        """
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocess(X_part):
    numeric_features = X_part.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_part.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_ohe()),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def make_pipeline(estimator, X_part):
    return Pipeline(
        steps=[
            ("preprocess", make_preprocess(X_part)),
            ("model", estimator),
        ]
    )


def evaluate(name, estimator):
    model = make_pipeline(estimator, X_train)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "name": name,
        "ROC_AUC": roc_auc_score(y_val, proba),
        "average_precision": average_precision_score(y_val, proba),
        "F1_at_0_5": f1_score(y_val, pred, zero_division=0),
        "model": model,
        "val_proba": proba,
    }
"""
    ),
    code(
        """
estimators = {
    "RandomForest": RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=350,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=220,
        learning_rate=0.04,
        max_depth=3,
        random_state=RANDOM_STATE,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.04,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=260,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=260,
        learning_rate=0.04,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    ),
    "CatBoost": CatBoostClassifier(
        iterations=260,
        learning_rate=0.04,
        depth=4,
        loss_function="Logloss",
        random_seed=RANDOM_STATE,
        verbose=False,
    ),
}

rows = []
trained = {}
val_predictions = {}

for name, estimator in estimators.items():
    result = evaluate(name, estimator)
    trained[name] = result.pop("model")
    val_predictions[name] = result.pop("val_proba")
    rows.append(result)

leaderboard = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)
leaderboard
"""
    ),
    md(
        """
## Финальная смесь моделей

Для финального файла берём не просто первые строки одной validation-таблицы, а смесь разных семейств моделей: случайный лес, extremely randomized trees и CatBoost. Такая смесь показывает идею ансамблирования решений: модели ошибаются по-разному, и усреднение вероятностей иногда даёт более устойчивый результат.
"""
    ),
    code(
        """
final_names = ["ExtraTrees", "CatBoost", "RandomForest"]
avg_val_proba = np.mean([val_predictions[name] for name in final_names], axis=0)

print("Усредняем модели:", final_names)
print("ROC-AUC среднего:", round(roc_auc_score(y_val, avg_val_proba), 4))
print("Average precision среднего:", round(average_precision_score(y_val, avg_val_proba), 4))
"""
    ),
    code(
        """
final_probas = []
for name in final_names:
    estimator = clone(estimators[name])
    model = make_pipeline(estimator, X)
    model.fit(X, y)
    final_probas.append(model.predict_proba(test_fe[features])[:, 1])

test_proba = np.mean(final_probas, axis=0)

submission = pd.DataFrame(
    {
        "id": test_fe[id_col],
        "churn_probability": test_proba.round(5),
    }
)

submission.to_csv("author_submission.csv", index=False)
submission.head()
"""
    ),
]


def main() -> None:
    materials = [
        (
            ROOT / "15_linear_regression" / "competition",
            LINEAR_NOTES,
            LINEAR_BASELINE,
            LINEAR_AUTHOR,
            """
Для преподавателя:

- `solution_notes.md` — разбор решения и список идей.
- `author_solution.ipynb` — авторское решение.
- `simple_baseline.ipynb` — простой baseline, который можно выдать ученикам по желанию.

Если цель — проверить полный цикл работы, baseline лучше не выдавать. Если цель — сосредоточиться на улучшении признаков и модели, baseline можно дать как старт.
""",
        ),
        (
            ROOT / "16_logistic_regression" / "competition",
            LOGISTIC_NOTES,
            LOGISTIC_BASELINE,
            LOGISTIC_AUTHOR,
            """
Для преподавателя:

- `solution_notes.md` — разбор решения и список идей.
- `author_solution.ipynb` — авторское решение.
- `simple_baseline.ipynb` — простой baseline, который можно выдать ученикам по желанию.

Если важно проверить, умеют ли ученики сами строить pipeline и подбирать порог, baseline не выдавайте. Если важнее сравнение идей, baseline можно дать.
""",
        ),
        (
            ROOT / "final_ensemble_competition",
            ENSEMBLE_NOTES,
            ENSEMBLE_BASELINE,
            ENSEMBLE_AUTHOR,
            """
Для преподавателя:

- `solution_notes.md` — разбор решения и список идей.
- `author_solution.ipynb` — авторское решение с RandomForest, ExtraTrees, sklearn-бустингом, XGBoost, LightGBM и CatBoost.
- `simple_baseline.ipynb` — простой baseline на RandomForest, который можно выдать ученикам по желанию.

Для итогового соревнования baseline удобно выдавать слабым группам, а сильным оставить только условие и данные.
""",
        ),
    ]

    for folder, notes, baseline, author, readme_section in materials:
        write_text(folder / "solution_notes.md", notes)
        write_notebook(folder / "simple_baseline.ipynb", baseline)
        write_notebook(folder / "author_solution.ipynb", author)
        add_teacher_section(folder / "README.md", readme_section)

    print("Competition teacher materials generated.")


if __name__ == "__main__":
    main()
