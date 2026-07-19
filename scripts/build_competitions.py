from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def lines(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


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
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=2), encoding="utf-8")


def replaced_cells(cells: list[dict], replacements: dict[str, str]) -> list[dict]:
    result = json.loads(json.dumps(cells, ensure_ascii=False))
    for cell in result:
        source = cell.get("source", [])
        new_source = []
        for line in source:
            for old, new in replacements.items():
                line = line.replace(old, new)
            new_source.append(line)
        cell["source"] = new_source
    return result


def write_readme(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def ensure_submission_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".gitkeep").write_text("", encoding="utf-8")


def make_linear_data(n_train: int = 700, n_test: int = 300) -> None:
    rng = np.random.default_rng(RANDOM_STATE + 15)
    n = n_train + n_test

    rooms = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.23, 0.34, 0.25, 0.13, 0.05])
    area = np.clip(rng.normal(22 + rooms * 18, 9, n), 24, 140).round(1)
    kitchen_area = np.clip(rng.normal(6 + area * 0.12, 2.0, n), 4, 28).round(1)
    districts = rng.choice(["центр", "спальный", "новостройки", "пригород"], size=n, p=[0.22, 0.43, 0.22, 0.13])
    condition = rng.choice(["требует ремонта", "среднее", "хорошее", "евроремонт"], size=n, p=[0.15, 0.42, 0.32, 0.11])
    total_floors = rng.integers(5, 26, n)
    floor = np.array([rng.integers(1, tf + 1) for tf in total_floors])
    house_age = np.clip(rng.normal(24, 17, n), 0, 80).round(0).astype(int)
    has_balcony = rng.binomial(1, 0.58, n)
    ceiling_height = np.clip(rng.normal(2.68, 0.18, n), 2.45, 3.4).round(2)

    district_distance = pd.Series(districts).map(
        {"центр": 2.5, "спальный": 9.5, "новостройки": 13.0, "пригород": 22.0}
    ).to_numpy()
    distance_to_center_km = np.clip(rng.normal(district_distance, 3.0), 0.3, 35).round(1)

    has_metro = rng.binomial(1, np.where(districts == "пригород", 0.35, 0.78), n)
    metro_min = np.where(has_metro == 1, np.clip(rng.normal(8 + distance_to_center_km * 0.25, 4, n), 2, 30), np.nan)
    metro_min = np.round(metro_min, 1)

    views_30d = np.expm1(rng.normal(4.0, 0.75, n)).round(0).astype(int)
    listing_month = rng.choice(["январь", "февраль", "март", "апрель", "май", "июнь"], size=n)

    district_effect = pd.Series(districts).map(
        {"центр": 3.7, "спальный": 1.2, "новостройки": 1.7, "пригород": -0.2}
    ).to_numpy()
    condition_effect = pd.Series(condition).map(
        {"требует ремонта": -1.1, "среднее": 0.0, "хорошее": 0.8, "евроремонт": 1.5}
    ).to_numpy()
    month_effect = pd.Series(listing_month).map(
        {"январь": -0.15, "февраль": -0.05, "март": 0.05, "апрель": 0.1, "май": 0.12, "июнь": 0.08}
    ).to_numpy()

    first_or_last_floor_penalty = np.where((floor == 1) | (floor == total_floors), -0.35, 0.0)
    price = (
        0.7
        + 0.122 * area
        + 0.055 * kitchen_area
        + 0.18 * rooms
        - 0.085 * distance_to_center_km
        - 0.018 * house_age
        - 0.035 * np.nan_to_num(metro_min, nan=18)
        + 0.55 * has_balcony
        + 0.65 * (ceiling_height - 2.6)
        + 0.13 * np.log1p(views_30d)
        + district_effect
        + condition_effect
        + month_effect
        + first_or_last_floor_penalty
        + rng.normal(0, 0.75, n)
    )
    price = np.clip(price, 2.0, None).round(2)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "area_m2": area,
            "rooms": rooms,
            "kitchen_m2": kitchen_area,
            "floor": floor,
            "floors_total": total_floors,
            "district": districts,
            "condition": condition,
            "distance_to_center_km": distance_to_center_km,
            "metro_min": metro_min,
            "house_age": house_age,
            "has_balcony": has_balcony,
            "ceiling_height": ceiling_height,
            "views_30d": views_30d,
            "listing_month": listing_month,
            "price_mln": price,
        }
    )
    train = df.iloc[:n_train].copy()
    test_full = df.iloc[n_train:].copy()
    test = test_full.drop(columns=["price_mln"])

    out = ROOT / "15_linear_regression" / "competition" / "data"
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False, encoding="utf-8")
    test.to_csv(out / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"id": test["id"], "price_mln": train["price_mln"].mean().round(2)}).to_csv(
        out / "sample_submission.csv", index=False, encoding="utf-8"
    )
    test_full[["id", "price_mln"]].to_csv(out / "private_target.csv", index=False, encoding="utf-8")


def make_logistic_data(n_train: int = 800, n_test: int = 350) -> None:
    rng = np.random.default_rng(RANDOM_STATE + 16)
    n = n_train + n_test

    age = rng.choice([13, 14, 15, 16, 17], size=n, p=[0.12, 0.22, 0.27, 0.24, 0.15])
    practice_hours_week = np.clip(rng.gamma(2.3, 1.4, n), 0, 12).round(1)
    homework_rate = np.clip(rng.beta(4, 2.2, n), 0, 1).round(2)
    last_score = np.clip(rng.normal(58 + 5.5 * practice_hours_week + 12 * homework_rate, 14, n), 10, 100).round(0).astype(int)
    missed_classes = np.clip(rng.poisson(2.0, n), 0, 12)
    club_visits = np.clip(rng.poisson(3.0 + practice_hours_week * 0.6, n), 0, 18)
    distance_min = np.clip(rng.normal(28, 16, n), 3, 95).round(0).astype(int)
    internet_quality = rng.choice(["низкое", "среднее", "высокое"], size=n, p=[0.18, 0.5, 0.32])
    parent_support = rng.choice(["редко", "иногда", "часто"], size=n, p=[0.3, 0.45, 0.25])
    course_level = rng.choice(["стартовый", "средний", "продвинутый"], size=n, p=[0.38, 0.44, 0.18])
    favorite_subject = rng.choice(["математика", "информатика", "физика", "гуманитарные"], size=n, p=[0.34, 0.31, 0.18, 0.17])
    has_friend_in_group = rng.binomial(1, 0.42, n)
    joined_trial = rng.binomial(1, 0.55, n)
    messages_to_teacher = np.clip(rng.poisson(1.2 + homework_rate * 2.5, n), 0, 12)

    internet_effect = pd.Series(internet_quality).map({"низкое": -0.55, "среднее": 0.0, "высокое": 0.35}).to_numpy()
    parent_effect = pd.Series(parent_support).map({"редко": -0.35, "иногда": 0.05, "часто": 0.38}).to_numpy()
    level_effect = pd.Series(course_level).map({"стартовый": 0.18, "средний": 0.0, "продвинутый": -0.2}).to_numpy()
    subject_effect = pd.Series(favorite_subject).map(
        {"математика": 0.22, "информатика": 0.32, "физика": 0.1, "гуманитарные": -0.18}
    ).to_numpy()

    logit = (
        -2.2
        + 0.045 * (last_score - 60)
        + 1.65 * homework_rate
        + 0.20 * practice_hours_week
        - 0.17 * missed_classes
        + 0.06 * club_visits
        - 0.015 * distance_min
        + 0.06 * messages_to_teacher
        + 0.33 * has_friend_in_group
        + 0.24 * joined_trial
        + 0.04 * (age - 15)
        + internet_effect
        + parent_effect
        + level_effect
        + subject_effect
        + rng.normal(0, 0.45, n)
    )
    probability = sigmoid(logit)
    will_finish = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "age": age,
            "practice_hours_week": practice_hours_week,
            "homework_rate": homework_rate,
            "last_score": last_score,
            "missed_classes": missed_classes,
            "club_visits": club_visits,
            "distance_min": distance_min,
            "internet_quality": internet_quality,
            "parent_support": parent_support,
            "course_level": course_level,
            "favorite_subject": favorite_subject,
            "has_friend_in_group": has_friend_in_group,
            "joined_trial": joined_trial,
            "messages_to_teacher": messages_to_teacher,
            "will_finish": will_finish,
        }
    )
    train = df.iloc[:n_train].copy()
    test_full = df.iloc[n_train:].copy()
    test = test_full.drop(columns=["will_finish"])

    out = ROOT / "16_logistic_regression" / "competition" / "data"
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False, encoding="utf-8")
    test.to_csv(out / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"id": test["id"], "probability": 0.5, "will_finish": 0}).to_csv(
        out / "sample_submission.csv", index=False, encoding="utf-8"
    )
    test_full[["id", "will_finish"]].to_csv(out / "private_target.csv", index=False, encoding="utf-8")


def make_ensemble_data(n_train: int = 1000, n_test: int = 500) -> None:
    rng = np.random.default_rng(RANDOM_STATE + 20)
    n = n_train + n_test

    account_age_days = np.clip(rng.gamma(3.0, 80, n), 7, 900).round(0).astype(int)
    sessions_last_30 = np.clip(rng.poisson(9, n), 0, 35)
    minutes_per_session = np.clip(rng.normal(32, 14, n), 3, 120).round(1)
    days_since_last_activity = np.clip(rng.exponential(9, n), 0, 60).round(0).astype(int)
    support_tickets = np.clip(rng.poisson(0.9, n), 0, 8)
    failed_payments = rng.binomial(3, 0.08, n)
    discount_percent = rng.choice([0, 5, 10, 15, 20, 30], size=n, p=[0.24, 0.14, 0.22, 0.17, 0.14, 0.09])
    plan_price = rng.choice([490, 790, 1190, 1590, 1990], size=n, p=[0.25, 0.3, 0.24, 0.14, 0.07])
    homework_completion = np.clip(rng.beta(3.2, 2.4, n), 0, 1).round(2)
    previous_courses = np.clip(rng.poisson(1.0, n), 0, 7)
    mobile_share = np.clip(rng.beta(2.0, 2.0, n), 0, 1).round(2)
    avg_gap_days = np.clip(rng.normal(5 + days_since_last_activity * 0.35, 4.0, n), 0, 35).round(1)

    plan_type = rng.choice(["basic", "plus", "premium"], size=n, p=[0.42, 0.4, 0.18])
    device = rng.choice(["phone", "laptop", "tablet"], size=n, p=[0.48, 0.39, 0.13])
    region = rng.choice(["центр", "север", "юг", "восток", "запад"], size=n, p=[0.24, 0.2, 0.18, 0.2, 0.18])
    acquisition_channel = rng.choice(["реклама", "друг", "поиск", "школа", "соцсети"], size=n, p=[0.24, 0.2, 0.25, 0.13, 0.18])
    favorite_topic = rng.choice(["python", "data", "games", "math", "web"], size=n, p=[0.27, 0.24, 0.2, 0.16, 0.13])

    channel_effect = pd.Series(acquisition_channel).map(
        {"реклама": 0.25, "друг": -0.25, "поиск": -0.05, "школа": -0.15, "соцсети": 0.1}
    ).to_numpy()
    plan_effect = pd.Series(plan_type).map({"basic": 0.08, "plus": -0.08, "premium": 0.18}).to_numpy()
    device_effect = pd.Series(device).map({"phone": 0.12, "laptop": -0.1, "tablet": 0.08}).to_numpy()

    low_activity = sessions_last_30 < 5
    expensive_without_discount = (plan_price >= 1590) & (discount_percent <= 5)
    many_tickets_and_low_homework = (support_tickets >= 3) & (homework_completion < 0.45)
    premium_low_activity = (plan_type == "premium") & low_activity
    loyal_friend_channel = (acquisition_channel == "друг") & (previous_courses >= 2)

    logit = (
        -1.05
        - 0.055 * sessions_last_30
        - 1.15 * homework_completion
        + 0.048 * days_since_last_activity
        + 0.04 * avg_gap_days
        + 0.22 * support_tickets
        + 0.58 * failed_payments
        + 0.00022 * plan_price
        - 0.018 * discount_percent
        - 0.0015 * account_age_days
        - 0.13 * previous_courses
        + 0.35 * low_activity.astype(int)
        + 0.45 * expensive_without_discount.astype(int)
        + 0.58 * many_tickets_and_low_homework.astype(int)
        + 0.5 * premium_low_activity.astype(int)
        - 0.45 * loyal_friend_channel.astype(int)
        + channel_effect
        + plan_effect
        + device_effect
        + rng.normal(0, 0.55, n)
    )
    churn_probability = sigmoid(logit)
    churn = rng.binomial(1, churn_probability)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "account_age_days": account_age_days,
            "sessions_last_30": sessions_last_30,
            "minutes_per_session": minutes_per_session,
            "days_since_last_activity": days_since_last_activity,
            "support_tickets": support_tickets,
            "failed_payments": failed_payments,
            "discount_percent": discount_percent,
            "plan_price": plan_price,
            "homework_completion": homework_completion,
            "previous_courses": previous_courses,
            "mobile_share": mobile_share,
            "avg_gap_days": avg_gap_days,
            "plan_type": plan_type,
            "device": device,
            "region": region,
            "acquisition_channel": acquisition_channel,
            "favorite_topic": favorite_topic,
            "churn": churn,
        }
    )

    for col, share in {"minutes_per_session": 0.04, "mobile_share": 0.05, "favorite_topic": 0.03}.items():
        mask = rng.random(n) < share
        df.loc[mask, col] = np.nan

    train = df.iloc[:n_train].copy()
    test_full = df.iloc[n_train:].copy()
    test = test_full.drop(columns=["churn"])

    out = ROOT / "final_ensemble_competition" / "data"
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False, encoding="utf-8")
    test.to_csv(out / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"id": test["id"], "churn_probability": train["churn"].mean().round(4)}).to_csv(
        out / "sample_submission.csv", index=False, encoding="utf-8"
    )
    test_full[["id", "churn"]].to_csv(out / "private_target.csv", index=False, encoding="utf-8")


LINEAR_STUDENT = [
    md(
        """
# Занятие 30. Практика: линейная регрессия и цена квартиры

Это второе занятие по теме линейной регрессии: практическая работа в формате учебного leaderboard. Команды по 1–2 человека строят модель, сохраняют `submission.csv`, а преподаватель считает результат на скрытой test-выборке.

**Задача:** предсказать `price_mln` — цену квартиры в миллионах рублей.

**Главная метрика занятия:** MAE, то есть средняя абсолютная ошибка в млн рублей. Меньше — лучше.

Файл с ответами для test специально не дан в этом блокноте. Для честной проверки не открывайте `data/private_target.csv`, если он лежит рядом в учительской версии материалов.
"""
    ),
    md(
        """
## Правила и ориентир на 90 минут

1. 0–10 мин — понять данные, метрику и baseline.
2. 10–30 мин — сделать первый `submission.csv`.
3. 30–70 мин — улучшать признаки и модель на validation.
4. 70–85 мин — выбрать финальную версию и сохранить submission.
5. 85–90 мин — обсудить, какие признаки помогли и почему.

Можно использовать только `train.csv` для обучения и выбора идей. `test.csv` нужен только для финального прогноза.
"""
    ),
    code(
        """
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

print(train.shape, test.shape)
display(train.head())
display(sample_submission.head())
"""
    ),
    md(
        """
## Что есть в данных

Одна строка — одна квартира из объявления.

Цель `price_mln` есть только в `train.csv`. В `test.csv` её нет: её нужно предсказать.

Подумайте перед моделированием:

- какие признаки похожи на числовые;
- какие признаки категориальные;
- какие признаки нельзя считать “расстоянием” между числами, даже если они записаны числами;
- что делать с пропусками в `metro_min`.
"""
    ),
    code(
        """
train.info()
"""
    ),
    code(
        """
train.describe(include="all").T
"""
    ),
    md(
        """
## Validation: честная проверка своих идей

Мы отделяем часть `train.csv` и делаем вид, что ответов к ней нет. Так можно сравнивать идеи до финальной отправки.
"""
    ),
    code(
        """
target = "price_mln"
id_col = "id"

features = [c for c in train.columns if c not in [target, id_col]]
X = train[features]
y = train[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)

print("train:", X_train.shape, "validation:", X_val.shape)
"""
    ),
    code(
        """
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

print("Числовые:", numeric_features)
print("Категориальные:", categorical_features)
"""
    ),
    code(
        """
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


numeric_preprocess = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_preprocess = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe()),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess, numeric_features),
        ("cat", categorical_preprocess, categorical_features),
    ]
)
"""
    ),
    md(
        """
## Первый baseline

Baseline — это первая простая модель. Её задача не победить, а дать точку отсчёта: стало ли лучше после ваших идей.
"""
    ),
    code(
        """
baseline_model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", LinearRegression()),
    ]
)

baseline_model.fit(X_train, y_train)
val_pred = baseline_model.predict(X_val)

mae = mean_absolute_error(y_val, val_pred)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
r2 = r2_score(y_val, val_pred)

print(f"MAE:  {mae:.3f} млн руб.")
print(f"RMSE: {rmse:.3f} млн руб.")
print(f"R²:   {r2:.3f}")
"""
    ),
    md(
        """
## Идеи для улучшения

Попробуйте 2–4 идеи и сравнивайте их только по validation:

- добавить признаки `area_per_room`, `kitchen_share`, `is_first_floor`, `is_last_floor`;
- заменить `views_30d` на `log_views = log1p(views_30d)`;
- сравнить `LinearRegression`, `Ridge`, `Lasso`;
- убрать явно бесполезные признаки;
- проверить, не стала ли модель лучше на MAE, но хуже на RMSE.

Важно: если вы создаёте признак по train, такой же признак нужно создать и в test.
"""
    ),
    code(
        """
def add_features(df):
    df = df.copy()
    df["area_per_room"] = df["area_m2"] / df["rooms"]
    df["kitchen_share"] = df["kitchen_m2"] / df["area_m2"]
    df["is_first_floor"] = (df["floor"] == 1).astype(int)
    df["is_last_floor"] = (df["floor"] == df["floors_total"]).astype(int)
    df["log_views"] = np.log1p(df["views_30d"])
    return df


train_fe = add_features(train)
test_fe = add_features(test)

features_fe = [c for c in train_fe.columns if c not in [target, id_col]]
X = train_fe[features_fe]
y = train_fe[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)

numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

preprocess_fe = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess, numeric_features),
        ("cat", categorical_preprocess, categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess_fe),
        ("model", Ridge(alpha=1.0)),
    ]
)

model.fit(X_train, y_train)
val_pred = model.predict(X_val)

print(f"MAE:  {mean_absolute_error(y_val, val_pred):.3f} млн руб.")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, val_pred)):.3f} млн руб.")
print(f"R²:   {r2_score(y_val, val_pred):.3f}")
"""
    ),
    md(
        """
## Финальное обучение и submission

Когда выбрали признаки и модель, обучите её на всём `train.csv`, затем сделайте прогноз для `test.csv`.
"""
    ),
    code(
        """
final_X = train_fe[features_fe]
final_y = train_fe[target]

final_model = model
final_model.fit(final_X, final_y)

test_pred = final_model.predict(test_fe[features_fe])
test_pred = np.maximum(test_pred, 0)  # цена не может быть отрицательной

submission = pd.DataFrame(
    {
        "id": test_fe["id"],
        "price_mln": np.round(test_pred, 3),
    }
)

submission.to_csv("submission.csv", index=False)
submission.head()
"""
    ),
    md(
        """
## Что сдать

Сдайте файл `submission.csv`.

В нём должны быть ровно два столбца:

- `id`;
- `price_mln`.

Если работаете в команде, переименуйте файл в понятное имя, например `team_ivan_masha.csv`.
"""
    ),
]


LOGISTIC_STUDENT = [
    md(
        """
# Занятие 32. Практика: логистическая регрессия и завершение курса

Это второе занятие по теме логистической регрессии: практическая работа в формате учебного leaderboard. Команды по 1–2 человека строят модель, сохраняют `submission.csv`, а преподаватель считает результат на скрытой test-выборке.

**Задача:** предсказать, завершит ли ученик курс. Целевой столбец в train — `will_finish`.

**Главная метрика занятия:** F1-score. Больше — лучше.

F1 полезна здесь потому, что нам важны и найденные будущие “завершившие курс”, и количество ложных срабатываний.
"""
    ),
    md(
        """
## Правила и ориентир на 90 минут

1. 0–10 мин — понять данные, целевой класс и метрику.
2. 10–30 мин — сделать первый прогноз логистической регрессией.
3. 30–60 мин — улучшить признаки и регуляризацию.
4. 60–80 мин — подобрать порог вероятности для F1.
5. 80–90 мин — сохранить submission и обсудить ошибки.

Используйте `test.csv` только для финального прогноза. Ответы к test не должны участвовать в выборе модели.
"""
    ),
    code(
        """
from pathlib import Path

import numpy as np
import pandas as pd

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
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

print(train.shape, test.shape)
display(train.head())
display(sample_submission.head())
"""
    ),
    md(
        """
## Validation и стратификация

Стратификация нужна, чтобы доля классов 0 и 1 была похожей в train и validation.
"""
    ),
    code(
        """
target = "will_finish"
id_col = "id"

features = [c for c in train.columns if c not in [target, id_col]]
X = train[features]
y = train[target]

print("Доля класса 1:", y.mean().round(3))

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

numeric_preprocess = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_preprocess = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe()),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess, numeric_features),
        ("cat", categorical_preprocess, categorical_features),
    ]
)
"""
    ),
    md(
        """
## Первый baseline: вероятность → класс

Логистическая регрессия выдаёт вероятность класса 1. Чтобы получить класс, выбираем порог.

Если порог 0.5, то:

- вероятность 0.49 даёт класс 0;
- вероятность 0.51 даёт класс 1.
"""
    ),
    code(
        """
model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)

model.fit(X_train, y_train)
val_proba = model.predict_proba(X_val)[:, 1]
val_pred = (val_proba >= 0.5).astype(int)

print("accuracy:", round(accuracy_score(y_val, val_pred), 3))
print("precision:", round(precision_score(y_val, val_pred), 3))
print("recall:", round(recall_score(y_val, val_pred), 3))
print("F1:", round(f1_score(y_val, val_pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_val, val_proba), 3))
"""
    ),
    md(
        """
## Подбор порога

Модель может хорошо расставлять вероятности, но порог 0.5 не всегда лучший для F1.

Ниже мы перебираем пороги и выбираем тот, где F1 на validation максимальна.
"""
    ),
    code(
        """
thresholds = np.linspace(0.15, 0.85, 71)
rows = []

for threshold in thresholds:
    pred = (val_proba >= threshold).astype(int)
    rows.append(
        {
            "threshold": threshold,
            "precision": precision_score(y_val, pred, zero_division=0),
            "recall": recall_score(y_val, pred, zero_division=0),
            "f1": f1_score(y_val, pred, zero_division=0),
        }
    )

threshold_report = pd.DataFrame(rows).sort_values("f1", ascending=False)
threshold_report.head(10)
"""
    ),
    code(
        """
best_threshold = float(threshold_report.iloc[0]["threshold"])
print("Лучший порог на validation:", round(best_threshold, 3))
"""
    ),
    md(
        """
## Идеи для улучшения

Попробуйте несколько вариантов и сравнивайте их на validation:

- `LogisticRegression(C=0.3)`, `C=1`, `C=3`;
- `class_weight="balanced"`;
- новые признаки вроде `score_per_missed_class`, `activity_total`;
- удаление слабых или дублирующих признаков;
- другой порог вероятности.

Не подбирайте порог по test: там нет честных ответов.
"""
    ),
    code(
        """
def add_features(df):
    df = df.copy()
    df["activity_total"] = df["practice_hours_week"] + 0.5 * df["club_visits"]
    df["score_per_missed_class"] = df["last_score"] / (df["missed_classes"] + 1)
    df["teacher_messages_per_week"] = df["messages_to_teacher"] / 4
    return df


train_fe = add_features(train)
test_fe = add_features(test)

features_fe = [c for c in train_fe.columns if c not in [target, id_col]]
X = train_fe[features_fe]
y = train_fe[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

preprocess_fe = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess, numeric_features),
        ("cat", categorical_preprocess, categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess_fe),
        ("model", LogisticRegression(max_iter=1000, C=1.0, class_weight=None)),
    ]
)

model.fit(X_train, y_train)
val_proba = model.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.15, 0.85, 71)
best_threshold = max(
    thresholds,
    key=lambda t: f1_score(y_val, (val_proba >= t).astype(int), zero_division=0),
)
val_pred = (val_proba >= best_threshold).astype(int)

print("threshold:", round(float(best_threshold), 3))
print("precision:", round(precision_score(y_val, val_pred, zero_division=0), 3))
print("recall:", round(recall_score(y_val, val_pred, zero_division=0), 3))
print("F1:", round(f1_score(y_val, val_pred, zero_division=0), 3))
print("ROC-AUC:", round(roc_auc_score(y_val, val_proba), 3))
"""
    ),
    md(
        """
## Финальное обучение и submission

После выбора модели обучаем её на всём `train.csv` и делаем прогноз для `test.csv`.
"""
    ),
    code(
        """
final_X = train_fe[features_fe]
final_y = train_fe[target]

final_model = model
final_model.fit(final_X, final_y)

test_proba = final_model.predict_proba(test_fe[features_fe])[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

submission = pd.DataFrame(
    {
        "id": test_fe["id"],
        "probability": np.round(test_proba, 5),
        "will_finish": test_pred,
    }
)

submission.to_csv("submission.csv", index=False)
submission.head()
"""
    ),
    md(
        """
## Что сдать

Сдайте файл `submission.csv`.

В нём должны быть столбцы:

- `id`;
- `probability` — вероятность класса 1;
- `will_finish` — итоговый прогноз 0 или 1.

Главная метрика считается по `will_finish`.
"""
    ),
]


ENSEMBLE_STUDENT = [
    md(
        """
# Итоговое соревнование по модулю. Классификация ансамблями: риск ухода пользователя

Это итоговая задача в формате учебного leaderboard после деревьев, случайного леса и градиентного бустинга.

**Задача:** предсказать `churn` — уйдёт ли пользователь из образовательного сервиса в ближайший месяц.

**Главная метрика соревнования:** ROC-AUC. Больше — лучше.

Интуитивно ROC-AUC проверяет, насколько хорошо модель ставит более высокий риск тем пользователям, которые действительно ушли.
"""
    ),
    md(
        """
## Правила и ориентир на 90 минут

1. 0–10 мин — понять данные, метрику и baseline.
2. 10–30 мин — обучить первый ансамбль.
3. 30–65 мин — сравнить несколько ансамблей.
4. 65–80 мин — улучшить признаки и параметры.
5. 80–90 мин — сохранить submission и обсудить победившие идеи.

Обязательный набор моделей для сравнения:

- `RandomForestClassifier`;
- `ExtraTreesClassifier`;
- `GradientBoostingClassifier`;
- `HistGradientBoostingClassifier`;
- `XGBClassifier`;
- `LGBMClassifier`;
- `CatBoostClassifier`.
"""
    ),
    code(
        """
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
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
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

print(train.shape, test.shape)
display(train.head())
display(sample_submission.head())
"""
    ),
    md(
        """
## Validation

Доля класса 1 может быть не 50/50, поэтому используем `stratify=y`.
"""
    ),
    code(
        """
target = "churn"
id_col = "id"

features = [c for c in train.columns if c not in [target, id_col]]
X = train[features]
y = train[target]

print("Доля ушедших пользователей:", round(y.mean(), 3))

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
"""
    ),
    code(
        """
def make_ohe():
    # Для HistGradientBoosting удобнее получить плотную матрицу, а не sparse.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

numeric_preprocess = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_preprocess = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe()),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess, numeric_features),
        ("cat", categorical_preprocess, categorical_features),
    ]
)
"""
    ),
    md(
        """
## Baseline и функция оценки

Baseline здесь почти ничего не знает об объектах. Хорошая модель должна заметно обогнать его.
"""
    ),
    code(
        """
def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_val, proba),
        "average_precision": average_precision_score(y_val, proba),
        "f1_at_0_5": f1_score(y_val, pred, zero_division=0),
        "precision_at_0_5": precision_score(y_val, pred, zero_division=0),
        "recall_at_0_5": recall_score(y_val, pred, zero_division=0),
    }


dummy = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", DummyClassifier(strategy="prior")),
    ]
)

pd.DataFrame([evaluate_model("Dummy prior", dummy, X_train, y_train, X_val, y_val)])
"""
    ),
    md(
        """
## Сравнение ансамблей

Сначала сравните несколько моделей “как есть”, а потом улучшайте самую перспективную.
"""
    ),
    code(
        """
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=160,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=220,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=220,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    ),
    "CatBoost": CatBoostClassifier(
        iterations=220,
        learning_rate=0.05,
        depth=4,
        loss_function="Logloss",
        random_seed=RANDOM_STATE,
        verbose=False,
    ),
}

rows = []
trained_models = {}

for name, estimator in models.items():
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )
    rows.append(evaluate_model(name, pipe, X_train, y_train, X_val, y_val))
    trained_models[name] = pipe

leaderboard = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
leaderboard
"""
    ),
    md(
        """
## Идеи для улучшения

Что можно пробовать:

- `n_estimators`: больше деревьев часто лучше, но медленнее;
- `max_depth` или `max_leaf_nodes`: ограничивают сложность деревьев;
- `min_samples_leaf`: делает деревья спокойнее и иногда улучшает качество;
- `learning_rate` и `n_estimators` для бустинга;
- признаки вроде `sessions_per_age_day`, `price_after_discount`, `is_inactive`;
- усреднение вероятностей двух сильных моделей.

Главное правило: сравнивайте варианты по одной и той же validation-выборке.
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
    return df


train_fe = add_features(train)
test_fe = add_features(test)

features_fe = [c for c in train_fe.columns if c not in [target, id_col]]
X = train_fe[features_fe]
y = train_fe[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

preprocess_fe = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess, numeric_features),
        ("cat", categorical_preprocess, categorical_features),
    ]
)

best_model = Pipeline(
    steps=[
        ("preprocess", preprocess_fe),
        (
            "model",
            GradientBoostingClassifier(
                n_estimators=220,
                learning_rate=0.04,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

report = evaluate_model("Tuned GradientBoosting", best_model, X_train, y_train, X_val, y_val)
pd.DataFrame([report])
"""
    ),
    md(
        """
## Финальное обучение и submission

Для ROC-AUC нужно сдавать вероятность класса 1, а не только 0/1.
"""
    ),
    code(
        """
final_X = train_fe[features_fe]
final_y = train_fe[target]

final_model = best_model
final_model.fit(final_X, final_y)

test_proba = final_model.predict_proba(test_fe[features_fe])[:, 1]

submission = pd.DataFrame(
    {
        "id": test_fe["id"],
        "churn_probability": np.round(test_proba, 5),
    }
)

submission.to_csv("submission.csv", index=False)
submission.head()
"""
    ),
    md(
        """
## Что сдать

Сдайте файл `submission.csv`.

В нём должны быть ровно два столбца:

- `id`;
- `churn_probability` — вероятность ухода пользователя.

Если работаете в команде, переименуйте файл в понятное имя, например `team_ivan_masha.csv`.
"""
    ),
]


def score_notebook(metric_kind: str) -> list[dict]:
    if metric_kind == "linear":
        title = "# Учительская проверка submission: линейная регрессия"
        code_text = """
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("data")
SUBMISSIONS_DIR = Path("submissions")
TARGET = "price_mln"

private = pd.read_csv(DATA_DIR / "private_target.csv")
files = sorted(SUBMISSIONS_DIR.glob("*.csv"))

rows = []
for path in files:
    sub = pd.read_csv(path)
    merged = private.merge(sub, on="id", how="inner", suffixes=("_true", "_pred"))
    if len(merged) != len(private):
        print(f"Пропускаю {path.name}: найдено {len(merged)} id из {len(private)}")
        continue
    y_true = merged[f"{TARGET}_true"]
    y_pred = merged[f"{TARGET}_pred"]
    rows.append(
        {
            "team_file": path.name,
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
        }
    )

if rows:
    leaderboard = pd.DataFrame(rows).sort_values(["MAE", "RMSE"], ascending=[True, True])
    display(leaderboard)
else:
    print("Положите CSV-файлы команд в папку submissions и запустите ячейку снова.")
"""
    elif metric_kind == "logistic":
        title = "# Учительская проверка submission: логистическая регрессия"
        code_text = """
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

DATA_DIR = Path("data")
SUBMISSIONS_DIR = Path("submissions")

private = pd.read_csv(DATA_DIR / "private_target.csv")
files = sorted(SUBMISSIONS_DIR.glob("*.csv"))

rows = []
for path in files:
    sub = pd.read_csv(path)
    merged = private.merge(sub, on="id", how="inner", suffixes=("_true", "_pred"))
    if len(merged) != len(private):
        print(f"Пропускаю {path.name}: найдено {len(merged)} id из {len(private)}")
        continue
    if "will_finish_pred" in merged.columns:
        y_pred = merged["will_finish_pred"].astype(int)
    elif "probability" in merged.columns:
        y_pred = (merged["probability"] >= 0.5).astype(int)
    else:
        print(f"Пропускаю {path.name}: нужен столбец will_finish или probability")
        continue
    y_true_col = "will_finish_true" if "will_finish_true" in merged.columns else "will_finish"
    y_true = merged[y_true_col]
    row = {
        "team_file": path.name,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if "probability" in merged.columns:
        row["ROC_AUC"] = roc_auc_score(y_true, merged["probability"])
    rows.append(row)

if rows:
    leaderboard = pd.DataFrame(rows).sort_values(["F1", "recall"], ascending=[False, False])
    display(leaderboard)
else:
    print("Положите CSV-файлы команд в папку submissions и запустите ячейку снова.")
"""
    elif metric_kind == "ensemble":
        title = "# Учительская проверка submission: итоговое соревнование ансамблей"
        code_text = """
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

DATA_DIR = Path("data")
SUBMISSIONS_DIR = Path("submissions")

private = pd.read_csv(DATA_DIR / "private_target.csv")
files = sorted(SUBMISSIONS_DIR.glob("*.csv"))

rows = []
for path in files:
    sub = pd.read_csv(path)
    merged = private.merge(sub, on="id", how="inner")
    if len(merged) != len(private):
        print(f"Пропускаю {path.name}: найдено {len(merged)} id из {len(private)}")
        continue
    if "churn_probability" not in merged.columns:
        print(f"Пропускаю {path.name}: нужен столбец churn_probability")
        continue
    y_true = merged["churn"]
    proba = merged["churn_probability"].clip(0, 1)
    pred = (proba >= 0.5).astype(int)
    rows.append(
        {
            "team_file": path.name,
            "ROC_AUC": roc_auc_score(y_true, proba),
            "average_precision": average_precision_score(y_true, proba),
            "F1_at_0_5": f1_score(y_true, pred, zero_division=0),
            "precision_at_0_5": precision_score(y_true, pred, zero_division=0),
            "recall_at_0_5": recall_score(y_true, pred, zero_division=0),
        }
    )

if rows:
    leaderboard = pd.DataFrame(rows).sort_values(["ROC_AUC", "average_precision"], ascending=[False, False])
    display(leaderboard)
else:
    print("Положите CSV-файлы команд в папку submissions и запустите ячейку снова.")
"""
    else:
        raise ValueError(metric_kind)

    return [
        md(
            f"""
{title}

Этот блокнот не нужно раздавать участникам вместе с ответами. Он читает файлы команд из папки `submissions` и сравнивает их со скрытым `data/private_target.csv`.
"""
        ),
        code(code_text),
    ]


def write_materials() -> None:
    make_linear_data()
    make_logistic_data()
    make_ensemble_data()

    linear_root = ROOT / "15_linear_regression"
    logistic_root = ROOT / "16_logistic_regression"
    linear_dir = linear_root / "competition"
    logistic_dir = logistic_root / "competition"
    ensemble_dir = ROOT / "final_ensemble_competition"

    practice_data_replacement = {"DATA_DIR = Path(\"data\")": "DATA_DIR = Path(\"competition\") / \"data\""}
    write_notebook(
        linear_root / "linear_regression_practice.ipynb",
        replaced_cells(LINEAR_STUDENT, practice_data_replacement),
    )
    write_notebook(
        logistic_root / "logistic_regression_practice.ipynb",
        replaced_cells(LOGISTIC_STUDENT, practice_data_replacement),
    )
    write_notebook(ensemble_dir / "ensemble_classification_competition.ipynb", ENSEMBLE_STUDENT)

    write_notebook(linear_dir / "score_submissions.ipynb", score_notebook("linear"))
    write_notebook(logistic_dir / "score_submissions.ipynb", score_notebook("logistic"))
    write_notebook(ensemble_dir / "score_submissions.ipynb", score_notebook("ensemble"))

    ensure_submission_dir(linear_dir / "submissions")
    ensure_submission_dir(logistic_dir / "submissions")
    ensure_submission_dir(ensemble_dir / "submissions")

    write_readme(
        linear_dir / "README.md",
        """
# Занятие 30. Практика: линейная регрессия

Участникам раздать:

- `../linear_regression_practice.ipynb`
- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

Не раздавать участникам до конца занятия:

- `data/private_target.csv`
- `score_submissions.ipynb`

Главная метрика: MAE по `price_mln`, меньше — лучше.

Для проверки соберите CSV команд в папку `submissions` и запустите `score_submissions.ipynb`.
""",
    )
    write_readme(
        logistic_dir / "README.md",
        """
# Занятие 32. Практика: логистическая регрессия

Участникам раздать:

- `../logistic_regression_practice.ipynb`
- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

Не раздавать участникам до конца занятия:

- `data/private_target.csv`
- `score_submissions.ipynb`

Главная метрика: F1 по столбцу `will_finish`, больше — лучше.

Для проверки соберите CSV команд в папку `submissions` и запустите `score_submissions.ipynb`.
""",
    )
    write_readme(
        ensemble_dir / "README.md",
        """
# Итоговое соревнование по модулю: классификация ансамблями

Участникам раздать:

- `ensemble_classification_competition.ipynb`
- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

Не раздавать участникам до конца соревнования:

- `data/private_target.csv`
- `score_submissions.ipynb`

Главная метрика: ROC-AUC по столбцу `churn_probability`, больше — лучше.

Для проверки соберите CSV команд в папку `submissions` и запустите `score_submissions.ipynb`.
""",
    )


if __name__ == "__main__":
    write_materials()
    print("Competition materials generated.")
