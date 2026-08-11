# -*- coding: utf-8 -*-
"""Build practice 34: Orbital Yard — ablation + Gradio panel + protocol + detective."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "Урок_34_Ансамбли_Bagging_Случайный_лес_Практика.ipynb"


def md(text: str) -> dict:
    body = text.strip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in body.split("\n")],
    }


def code(text: str) -> dict:
    body = text.strip("\n")
    lines = body.split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


DATA_CODE = r'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n = 900

# Полезные приборы центра Orbital Yard
radar_rcs = rng.normal(0, 1, n)          # радиолокационный RCS
optical_mag = rng.normal(0, 1, n)        # оптическая яркость
ir_delta = rng.normal(0, 1, n)           # ИК-сигнатура
doppler_shift = rng.normal(0, 1, n)      # доплеровский сдвиг
spin_period = rng.normal(0, 1, n)        # период вращения

# Три класса контактов: satellite / debris / glitch
score_sat = 1.6 * radar_rcs - 1.2 * optical_mag + 0.8 * doppler_shift
score_deb = -1.1 * radar_rcs + 1.8 * ir_delta + 1.0 * spin_period
score_gli = 0.3 * radar_rcs + 0.4 * optical_mag - 1.7 * ir_delta + 1.5 * doppler_shift
logits = np.column_stack([score_sat, score_deb, score_gli])
ex = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = ex / ex.sum(axis=1, keepdims=True)
y = np.array([rng.choice(3, p=p) for p in probs])

# Небольшой шум меток (ошибки операторов)
flip = rng.random(n) < 0.06
y[flip] = rng.integers(0, 3, size=int(flip.sum()))

# Шумовые каналы приборов — не несут сигнала о классе
noise = rng.normal(0, 1, size=(n, 4))

FEATURE_NAMES = [
    "radar_rcs",
    "optical_mag",
    "ir_delta",
    "doppler_shift",
    "spin_period",
    "noise_0",
    "noise_1",
    "noise_2",
    "noise_3",
]
CLASS_NAMES = ["satellite", "debris", "glitch"]

X = np.column_stack([radar_rcs, optical_mag, ir_delta, doppler_shift, spin_period, noise])
df = pd.DataFrame(X, columns=FEATURE_NAMES)
df["contact_type"] = [CLASS_NAMES[i] for i in y]

print("Журнал контактов Orbital Yard:")
print(df.head())
print()
print("Размер:", df.shape)
print("Классы:")
print(df["contact_type"].value_counts())
'''


TASK1_SOL = r'''
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print("train:", X_train.shape, "validation:", X_val.shape)
print("доля классов в train:", np.bincount(y_train) / len(y_train))
print("доля классов в val:  ", np.bincount(y_val) / len(y_val))
'''


TASK2_SOL = r'''
def fit_score(model):
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    return train_acc, val_acc


configs = []

# 1) одно дерево
tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
tr, va = fit_score(tree)
configs.append({"режим": "одно дерево", "n_estimators": 1, "max_features": "все",
                "bootstrap": "—", "train_acc": tr, "val_acc": va})

# 2) bagging без random features
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
    n_estimators=150,
    random_state=RANDOM_STATE,
)
tr, va = fit_score(bag)
configs.append({"режим": "bagging", "n_estimators": 150, "max_features": "все",
                "bootstrap": True, "train_acc": tr, "val_acc": va})

# 3) RF с random features
rf = RandomForestClassifier(
    n_estimators=150, max_features="sqrt", random_state=RANDOM_STATE
)
tr, va = fit_score(rf)
configs.append({"режим": "RF (sqrt)", "n_estimators": 150, "max_features": "sqrt",
                "bootstrap": True, "train_acc": tr, "val_acc": va})

# 4) RF малый vs большой n_estimators
rf_small = RandomForestClassifier(
    n_estimators=10, max_features="sqrt", random_state=RANDOM_STATE
)
tr, va = fit_score(rf_small)
configs.append({"режим": "RF малый", "n_estimators": 10, "max_features": "sqrt",
                "bootstrap": True, "train_acc": tr, "val_acc": va})

rf_big = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", random_state=RANDOM_STATE
)
tr, va = fit_score(rf_big)
configs.append({"режим": "RF большой", "n_estimators": 300, "max_features": "sqrt",
                "bootstrap": True, "train_acc": tr, "val_acc": va})

# 5) RF без bootstrap
rf_noboot = RandomForestClassifier(
    n_estimators=150, max_features="sqrt", bootstrap=False, random_state=RANDOM_STATE
)
tr, va = fit_score(rf_noboot)
configs.append({"режим": "RF без bootstrap", "n_estimators": 150, "max_features": "sqrt",
                "bootstrap": False, "train_acc": tr, "val_acc": va})

ablation_df = pd.DataFrame(configs)
print(ablation_df.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4))
x_pos = np.arange(len(ablation_df))
ax.bar(x_pos - 0.18, ablation_df["train_acc"], width=0.36, label="train")
ax.bar(x_pos + 0.18, ablation_df["val_acc"], width=0.36, label="validation")
ax.set_xticks(x_pos)
ax.set_xticklabels(ablation_df["режим"], rotation=20, ha="right")
ax.set_ylabel("accuracy")
ax.set_title("Абляция Orbital Yard: что даёт прирост на validation")
ax.legend()
ax.set_ylim(0.4, 1.05)
plt.tight_layout()
plt.show()

# кривая n_estimators для RF
n_grid = [5, 10, 25, 50, 100, 150, 300]
val_curve = []
for n_est in n_grid:
    m = RandomForestClassifier(
        n_estimators=n_est, max_features="sqrt", random_state=RANDOM_STATE
    )
    _, va = fit_score(m)
    val_curve.append(va)

plt.figure(figsize=(7, 4))
plt.plot(n_grid, val_curve, marker="o")
plt.xlabel("n_estimators")
plt.ylabel("validation accuracy")
plt.title("RF: рост числа деревьев и стабильность на validation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''


TASK3_SOL = r'''
# Если Gradio ещё нет в окружении:
# !pip install gradio

import gradio as gr

# Глобальный протокол экспериментов (пополняется из пульта)
experiments_log = pd.DataFrame(
    columns=[
        "гипотеза",
        "режим",
        "n_estimators",
        "max_depth",
        "max_features",
        "bootstrap",
        "train_acc",
        "val_acc",
        "вывод",
    ]
)


def _parse_max_features(mf: str):
    if mf == "sqrt":
        return "sqrt"
    if mf == "log2":
        return "log2"
    if mf == "все (None)":
        return None
    if mf.startswith("доля "):
        return float(mf.split()[1])
    return "sqrt"


def _parse_max_depth(depth_mode: str, depth_value: int):
    if depth_mode == "без ограничения (None)":
        return None
    return int(depth_value)


def train_ablation_config(mode, n_estimators, depth_mode, depth_value, max_features_ui, bootstrap):
    max_depth = _parse_max_depth(depth_mode, depth_value)
    max_features = _parse_max_features(max_features_ui)
    n_estimators = int(n_estimators)

    if mode == "одно дерево":
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
        settings = (
            f"режим=одно дерево | max_depth={max_depth} | "
            f"(n_estimators/max_features/bootstrap не применяются)"
        )
    elif mode == "bagging":
        model = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE),
            n_estimators=n_estimators,
            bootstrap=bool(bootstrap),
            random_state=RANDOM_STATE,
        )
        settings = (
            f"режим=bagging | n_estimators={n_estimators} | max_depth={max_depth} | "
            f"bootstrap={bool(bootstrap)} | max_features=все у каждого дерева"
        )
    else:  # random forest
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            bootstrap=bool(bootstrap),
            random_state=RANDOM_STATE,
        )
        settings = (
            f"режим=RF | n_estimators={n_estimators} | max_depth={max_depth} | "
            f"max_features={max_features} | bootstrap={bool(bootstrap)}"
        )

    model.fit(X_train, y_train)
    train_acc = float(accuracy_score(y_train, model.predict(X_train)))
    val_acc = float(accuracy_score(y_val, model.predict(X_val)))

    # короткий bar для UI
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(["train", "validation"], [train_acc, val_acc], color=["#4C78A8", "#F58518"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Качество текущей конфигурации")
    for i, v in enumerate([train_acc, val_acc]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()

    metrics = f"train accuracy = {train_acc:.3f}\nvalidation accuracy = {val_acc:.3f}"
    return settings, metrics, fig, train_acc, val_acc, mode, n_estimators, max_depth, max_features, bool(bootstrap)


def add_to_protocol(
    hypothesis,
    conclusion,
    train_acc,
    val_acc,
    mode,
    n_estimators,
    max_depth,
    max_features,
    bootstrap,
):
    global experiments_log
    if train_acc is None or val_acc is None:
        return experiments_log, "Сначала нажмите «Пересчитать», потом «Добавить в протокол»."

    row = {
        "гипотеза": hypothesis.strip() or "(без гипотезы)",
        "режим": mode,
        "n_estimators": int(n_estimators) if mode != "одно дерево" else 1,
        "max_depth": max_depth if max_depth is not None else "None",
        "max_features": max_features if mode == "random forest" else "—",
        "bootstrap": bootstrap if mode != "одно дерево" else "—",
        "train_acc": round(float(train_acc), 4),
        "val_acc": round(float(val_acc), 4),
        "вывод": conclusion.strip() or "(нет вывода)",
    }
    experiments_log = pd.concat([experiments_log, pd.DataFrame([row])], ignore_index=True)
    msg = f"В протокол добавлена строка №{len(experiments_log)}."
    return experiments_log, msg


with gr.Blocks(title="Orbital Yard — абляционный пульт") as demo:
    gr.Markdown(
        """
        ## Абляционный пульт Orbital Yard
        Включайте и выключайте ингредиенты ансамбля. Сравнивайте **validation** accuracy.
        После каждого эксперимента нажмите **«Добавить в протокол»**.
        """
    )
    with gr.Row():
        with gr.Column():
            mode = gr.Radio(
                ["одно дерево", "bagging", "random forest"],
                value="random forest",
                label="Режим модели",
            )
            n_estimators = gr.Slider(5, 300, value=150, step=5, label="n_estimators")
            depth_mode = gr.Radio(
                ["без ограничения (None)", "ограничить"],
                value="без ограничения (None)",
                label="max_depth",
            )
            depth_value = gr.Slider(1, 30, value=8, step=1, label="Значение max_depth (если ограничить)")
            max_features_ui = gr.Dropdown(
                ["sqrt", "log2", "все (None)", "доля 0.3", "доля 0.5", "доля 0.8"],
                value="sqrt",
                label="max_features (для RF)",
            )
            bootstrap = gr.Checkbox(value=True, label="bootstrap (для bagging / RF)")
            btn_run = gr.Button("Пересчитать", variant="primary")
        with gr.Column():
            settings_out = gr.Textbox(label="Что сейчас включено", lines=3)
            metrics_out = gr.Textbox(label="Метрики", lines=2)
            plot_out = gr.Plot(label="Train vs validation")
            # скрытые state для протокола
            st_train = gr.State(None)
            st_val = gr.State(None)
            st_mode = gr.State(None)
            st_n = gr.State(None)
            st_depth = gr.State(None)
            st_mf = gr.State(None)
            st_boot = gr.State(None)

    with gr.Row():
        hypothesis = gr.Textbox(
            label="Гипотеза эксперимента",
            placeholder="Например: random features дадут прирост относительно bagging",
        )
        conclusion = gr.Textbox(
            label="Краткий вывод",
            placeholder="Например: прирост маленький, главная выгода уже от bagging",
        )
    btn_log = gr.Button("Добавить в протокол")
    log_status = gr.Textbox(label="Статус протокола")
    log_table = gr.Dataframe(label="Протокол experiments_log", interactive=False)

    btn_run.click(
        train_ablation_config,
        inputs=[mode, n_estimators, depth_mode, depth_value, max_features_ui, bootstrap],
        outputs=[
            settings_out, metrics_out, plot_out,
            st_train, st_val, st_mode, st_n, st_depth, st_mf, st_boot,
        ],
    )
    btn_log.click(
        add_to_protocol,
        inputs=[hypothesis, conclusion, st_train, st_val, st_mode, st_n, st_depth, st_mf, st_boot],
        outputs=[log_table, log_status],
    )

# share=False — локальный интерфейс в классе
demo.launch(share=False)
'''


TASK4_SOL = r'''
# Авторский пример заполненного протокола (если пульт не запускали в этой сессии —
# воспроизводим те же конфигурации программно и пишем выводы).

def run_config(mode, n_estimators=150, max_depth=None, max_features="sqrt", bootstrap=True):
    if mode == "одно дерево":
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    elif mode == "bagging":
        model = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE),
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            random_state=RANDOM_STATE,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=RANDOM_STATE,
        )
    model.fit(X_train, y_train)
    return (
        accuracy_score(y_train, model.predict(X_train)),
        accuracy_score(y_val, model.predict(X_val)),
    )


protocol_rows = [
    {
        "гипотеза": "Одно глубокое дерево нестабильно на validation",
        "режим": "одно дерево",
        "n_estimators": 1,
        "max_depth": "None",
        "max_features": "—",
        "bootstrap": "—",
        "вывод": "Базовый ориентир: высокий train, заметно ниже val — большая дисперсия.",
        "_cfg": ("одно дерево", 1, None, None, True),
    },
    {
        "гипотеза": "Bootstrap-усреднение снизит разброс относительно одного дерева",
        "режим": "bagging",
        "n_estimators": 150,
        "max_depth": "None",
        "max_features": "—",
        "bootstrap": True,
        "вывод": "Главный скачок качества: bagging сильно поднимает validation accuracy.",
        "_cfg": ("bagging", 150, None, None, True),
    },
    {
        "гипотеза": "Случайные признаки (RF) дадут ещё прирост к bagging",
        "режим": "random forest",
        "n_estimators": 150,
        "max_depth": "None",
        "max_features": "sqrt",
        "bootstrap": True,
        "вывод": "Прирост к bagging небольшой; random features добавляют разнообразие, но не всегда огромный скачок.",
        "_cfg": ("random forest", 150, None, "sqrt", True),
    },
    {
        "гипотеза": "Мало деревьев (n_estimators=10) даст шумный/слабый лес",
        "режим": "random forest",
        "n_estimators": 10,
        "max_depth": "None",
        "max_features": "sqrt",
        "bootstrap": True,
        "вывод": "Малый лес хуже большого: усреднение ещё не успело стабилизироваться.",
        "_cfg": ("random forest", 10, None, "sqrt", True),
    },
    {
        "гипотеза": "Большой лес (n_estimators=300) стабилизирует метрику",
        "режим": "random forest",
        "n_estimators": 300,
        "max_depth": "None",
        "max_features": "sqrt",
        "bootstrap": True,
        "вывод": "Метрика выходит на плато: после ~100–150 деревьев прирост почти исчезает.",
        "_cfg": ("random forest", 300, None, "sqrt", True),
    },
    {
        "гипотеза": "Выключение bootstrap ослабит разнообразие RF",
        "режим": "random forest",
        "n_estimators": 150,
        "max_depth": "None",
        "max_features": "sqrt",
        "bootstrap": False,
        "вывод": "Без bootstrap качество близко к обычному RF: на этих данных эффект слабее, чем от bagging vs дерево.",
        "_cfg": ("random forest", 150, None, "sqrt", False),
    },
]

filled = []
for row in protocol_rows:
    mode, n_est, md_, mf, boot = row["_cfg"]
    tr, va = run_config(mode, n_est, md_, mf if mode == "random forest" else "sqrt", boot)
    item = {k: v for k, v in row.items() if k != "_cfg"}
    item["train_acc"] = round(float(tr), 4)
    item["val_acc"] = round(float(va), 4)
    filled.append(item)

experiments_log = pd.DataFrame(filled)
print("Протокол экспериментов (experiments_log):")
print(experiments_log.to_string(index=False))

print()
print("Сводка абляции по validation:")
print(experiments_log[["режим", "n_estimators", "bootstrap", "val_acc", "вывод"]].to_string(index=False))
'''


TASK5_SOL = r'''
rf_oob = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    oob_score=True,
    random_state=RANDOM_STATE,
)
rf_oob.fit(X_train, y_train)

oob = rf_oob.oob_score_
val = accuracy_score(y_val, rf_oob.predict(X_val))
print(f"OOB score (контакты, которые смена/дерево не видело): {oob:.3f}")
print(f"Validation accuracy:                                    {val:.3f}")
print()
print(
    "OOB — быстрый ориентир на train-объектах вне bootstrap-мешка. "
    "Это не финальный скрытый test: гиперпараметры по OOB всё равно можно переподогнать."
)
'''


TASK6_SOL = r'''
rf_det = RandomForestClassifier(
    n_estimators=200, max_features="sqrt", random_state=RANDOM_STATE
)
rf_det.fit(X_train, y_train)

perm = permutation_importance(
    rf_det, X_val, y_val, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1
)
imp_df = (
    pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    )
    .sort_values("importance_mean", ascending=False)
    .reset_index(drop=True)
)
print(imp_df.round(4).to_string(index=False))

plt.figure(figsize=(8, 4))
plt.barh(imp_df["feature"][::-1], imp_df["importance_mean"][::-1])
plt.xlabel("permutation importance (падение accuracy на validation)")
plt.title("Какие приборы Orbital Yard реально полезны")
plt.tight_layout()
plt.show()

useful = imp_df[~imp_df["feature"].str.startswith("noise_")]
noise_rows = imp_df[imp_df["feature"].str.startswith("noise_")]
print()
print("Топ полезных приборов:", list(useful.head(3)["feature"]))
print(
    "Средняя важность noise_*:",
    round(float(noise_rows["importance_mean"].mean()), 4),
    "— около нуля / чуть отрицательная → бесполезны.",
)
'''


TASK7_SOL = r'''
# Странный контакт: смешиваем сигнатуры satellite и debris + шум
strange = X_val[0].copy()
# усиливаем конфликт приборов
strange[FEATURE_NAMES.index("radar_rcs")] = 1.8      # тянет к satellite
strange[FEATURE_NAMES.index("ir_delta")] = 1.9        # тянет к debris
strange[FEATURE_NAMES.index("optical_mag")] = -0.4
strange[FEATURE_NAMES.index("noise_0")] = 2.5         # большой, но бесполезный шум

proba0 = rf_det.predict_proba(strange.reshape(1, -1))[0]
pred0 = CLASS_NAMES[int(np.argmax(proba0))]
print("Исходный прогноз странного контакта:", pred0)
print("Вероятности:", dict(zip(CLASS_NAMES, np.round(proba0, 3))))

# Какой признак сильнее всего тянет решение: заменяем каждый на медиану train
# и смотрим изменение вероятности предсказанного класса
base_p = float(proba0[np.argmax(proba0)])
medians = np.median(X_train, axis=0)
deltas = []
for j, name in enumerate(FEATURE_NAMES):
    x2 = strange.copy()
    x2[j] = medians[j]
    p2 = rf_det.predict_proba(x2.reshape(1, -1))[0]
    new_p = float(p2[np.argmax(proba0)])
    deltas.append({"feature": name, "delta_proba_pred": base_p - new_p})

delta_df = pd.DataFrame(deltas).sort_values("delta_proba_pred", ascending=False)
print()
print("Насколько падает уверенность в исходном классе, если обнулить прибор (→ медиана train):")
print(delta_df.round(4).to_string(index=False))

top_pull = delta_df.iloc[0]["feature"]
print()
print(f"Сильнее всего тянет решение признак: {top_pull}")
print("Шумовые каналы noise_* почти не двигают вероятность — как и в permutation importance.")

# Дополнительно: последовательное отключение групп признаков
def val_acc_without(drop_names):
    keep = [i for i, f in enumerate(FEATURE_NAMES) if f not in drop_names]
    m = RandomForestClassifier(
        n_estimators=200, max_features="sqrt", random_state=RANDOM_STATE
    )
    m.fit(X_train[:, keep], y_train)
    return accuracy_score(y_val, m.predict(X_val[:, keep]))


print()
print("Validation accuracy при отключении групп:")
print(" все признаки:     ", round(val_acc_without([]), 3))
print(" без noise_*:      ", round(val_acc_without([f for f in FEATURE_NAMES if f.startswith("noise_")]), 3))
print(" без ir_delta:     ", round(val_acc_without(["ir_delta"]), 3))
print(" только noise_*:   ", round(val_acc_without([f for f in FEATURE_NAMES if not f.startswith("noise_")]), 3))
'''


TASK8_SOL = r'''
print("Краткая сводка для итога:")
print(ablation_df[["режим", "val_acc"]].round(3).to_string(index=False))
print()
print("OOB vs val:", round(float(oob), 3), round(float(val), 3))
print("Топ приборов:", list(imp_df.head(3)["feature"]))
'''


def build() -> None:
    cells = [
        md(
            """
# Занятие 34. Практика: bagging и случайный лес

Вы исследуете **ансамбль деревьев** не как «гонку за accuracy», а как **абляцию**:
какие ингредиенты (bootstrap, random features, число деревьев) реально дают прирост и стабильность.

Теория — занятие 33, ноутбук `Урок_33_Ансамбли_Bagging_Случайный_лес.ipynb`.
Главная модель: **RandomForestClassifier** (+ сравнение с одним деревом и bagging).

### Оценивание (30 баллов)

| № | Тема | Баллы |
|---|------|------:|
| 1 | Импорты и split журнала контактов | 2 |
| 2 | Кодовая абляция: дерево / bagging / RF / n_estimators / bootstrap | 5 |
| 3 | Gradio-пульт абляции | 4 |
| 4 | Протокол экспериментов (`experiments_log`) | 5 |
| 5 | OOB: контакты, которые смена не видела | 3 |
| 6 | Детектив: permutation importance приборов | 4 |
| 7 | Детектив: странный контакт и отключение признаков | 5 |
| 8 | Итоговые выводы | 2 |
| | **Итого** | **30** |

**Часть A — абляция + пульт + протокол** (задания 1–4).
**Часть B — детектив по приборам** (задания 5–7).
"""
        ),
        md(
            """
---
## Легенда: центр сопровождения «Orbital Yard»

Вы — аналитик в центре **Orbital Yard**. По ночному небу летят контакты трёх типов:

| Класс | Что это |
|-------|---------|
| `satellite` | рабочий спутник |
| `debris` | обломок / мусор |
| `glitch` | ложное срабатывание приборов |

По каждому контакту пишут показания приборов: `radar_rcs`, `optical_mag`, `ir_delta`,
`doppler_shift`, `spin_period`, а также четыре «шумовых» канала `noise_0`…`noise_3`
(калибровочный мусор, который в журнал попал по ошибке).

Ваша смена делает две вещи:

1. **Абляционный пульт** — включает/выключает ингредиенты леса и ведёт **протокол экспериментов**.
2. **Детектив по приборам** — выясняет, какие датчики реально помогают, а какие — пустышки.
"""
        ),
        md(
            """
---
## Дано: журнал контактов

Ячейку ниже **не меняйте**. Она создаёт синтетический журнал Orbital Yard
с именованными приборами и шумовыми каналами `noise_*`.

После запуска будут:
- `df` — таблица контактов;
- `X`, `y` — признаки и метки (`0=satellite`, `1=debris`, `2=glitch`);
- `FEATURE_NAMES`, `CLASS_NAMES`.

> Если позже не импортируется Gradio: `pip install gradio` (или `!pip install gradio` в ячейке).
"""
        ),
        code(DATA_CODE),
        # ---- Task 1 ----
        md(
            """
---
## Задание 1. Импорты и split — **2 балла**

Подготовьте лабораторию абляции.

**Шаг 1.** Импортируйте:
`train_test_split`, `DecisionTreeClassifier`, `BaggingClassifier`,
`RandomForestClassifier`, `accuracy_score`, `permutation_importance`.

**Шаг 2.** Задайте `RANDOM_STATE = 42`.

**Шаг 3.** Разделите `X`, `y` на train / validation (**70 / 30**),
`stratify=y`, `random_state=RANDOM_STATE`.
Сохраните `X_train`, `X_val`, `y_train`, `y_val`.

**Шаг 4.** Выведите размеры выборок и доли классов.

### Подробные критерии (для проверки LLM)

- **0.5 балла** — импортированы нужные классы/функции.
- **0.5 балла** — задан `RANDOM_STATE = 42`.
- **0.5 балла** — split 70/30 со `stratify=y`.
- **0.5 балла** — выведены размеры и/или доли классов.

### Снижение баллов

- Нет `stratify` → минус **0.5**.
- Validation используется до обучения как «вторая train» → минус **1.0**.
"""
        ),
        code(TASK1_SOL),
        # ---- Task 2 ----
        md(
            """
---
## Задание 2. Кодовая абляция — **5 баллов**

Соберите **таблицу абляции** одного семейства моделей. Это не лидерборд «кто круче»,
а ответ на вопрос: **что именно** даёт прирост?

Обучите и сравните на **одном и том же** validation:

1. одно `DecisionTreeClassifier`;
2. `BaggingClassifier` из деревьев (`n_estimators=150`) — bootstrap, **без** random features;
3. `RandomForestClassifier` (`n_estimators=150`, `max_features="sqrt"`);
4. RF с малым и большим `n_estimators` (например 10 и 300);
5. RF с `bootstrap=False` (если API позволяет).

Для каждой строки сохраните train/validation accuracy.
Постройте **bar**-график конфигураций и **line**-график `n_estimators → val accuracy`.

Графики: заголовок, подписи осей, легенда где нужна.

### Подробные критерии (для проверки LLM)

- **1.0 балл** — обучено одно дерево и посчитаны train/val accuracy.
- **1.0 балл** — обучен bagging (bootstrap, без random features) и сравнён с деревом.
- **1.0 балл** — обучен RF с `max_features="sqrt"` и сравнён с bagging.
- **1.0 балл** — сравнены малый и большой `n_estimators` у RF (+ опционально `bootstrap=False`).
- **1.0 балл** — есть таблица абляции и bar/line-графики с заголовком и подписями осей.

### Снижение баллов

- Сравнивают только train accuracy → минус **1.5**.
- Нет графика → минус **1.0**.
- Разные `random_state`/разные split между моделями без фиксации → минус **0.5**.
"""
        ),
        code(TASK2_SOL),
        # ---- Task 3 ----
        md(
            """
---
## Задание 3. Gradio-пульт абляции — **4 балла**

Соберите **интерактивный веб-пульт** (предпочтительно **Gradio** `gr.Blocks`).

Тумблеры / контроли:

- режим: `одно дерево` / `bagging` / `random forest`;
- слайдер `n_estimators`;
- `max_depth` (None или число);
- `max_features` для RF: `sqrt` / `log2` / `None` / доля;
- `bootstrap` on/off (для bagging/RF);
- кнопка **«Пересчитать»**;
- кнопка **«Добавить в протокол»**.

Вывод пульта: текст «что включено», train/val accuracy, короткий bar-график.
Запуск: `demo.launch(share=False)` — выполните ячейку, откроется **локальный** интерфейс.

Если `import gradio` не работает: `pip install gradio`.

### Подробные критерии (для проверки LLM)

- **1.0 балл** — есть Gradio UI (`gr.Blocks` / Interface) с режимом модели.
- **1.0 балл** — есть слайдеры/контроли `n_estimators`, `max_depth`, `max_features`, `bootstrap`.
- **1.0 балл** — кнопка пересчёта обучает модель на train и показывает train/val accuracy.
- **1.0 балл** — UI запускается через `demo.launch(share=False)` (или эквивалент) из ячейки.

### Снижение баллов

- Нет интерактивных контролов (только статичный код) → минус **2.0**.
- Модель учится с подглядыванием в validation при `fit` → минус **1.0**.
- Fallback на `ipywidgets` допустим, если Gradio недоступен, но контроли должны быть.
"""
        ),
        code(TASK3_SOL),
        # ---- Task 4 ----
        md(
            """
---
## Задание 4. Протокол экспериментов — **5 баллов**

Ведите **лабораторный протокол** `experiments_log` (DataFrame и/или markdown-таблица).

Каждая строка — один эксперимент:

| Поле | Смысл |
|------|--------|
| гипотеза | что проверяете тумблерами |
| режим / n_estimators / max_depth / max_features / bootstrap | настройки пульта |
| train_acc / val_acc | метрики |
| вывод | что показали числа |

**Минимум 5 строк**, и среди них должны быть **разные** конфигурации
(дерево vs bagging vs RF, разный `n_estimators`, желательно bootstrap on/off).

Используйте пульт (кнопка «Добавить в протокол») или заполните таблицу кодом после серии запусков.
В конце выведите итоговый `experiments_log` и коротко ответьте: **какой ингредиент дал главный прирост?**

### Подробные критерии (для проверки LLM)

- **1.0 балл** — есть таблица/DataFrame протокола с полями настроек и метрик.
- **1.5 балла** — не меньше **5** строк экспериментов.
- **1.0 балл** — конфигурации реально разные (не копипаста одной строки).
- **1.0 балл** — у каждой (или почти каждой) строки есть краткий вывод/гипотеза.
- **0.5 балла** — итоговый вывод: что дало основной прирост (обычно bagging vs одно дерево).

### Снижение баллов

- Меньше 5 экспериментов → минус **1.5**.
- В протоколе только train без validation → минус **1.0**.
- Выводы не связаны с числами протокола → минус **0.5**.
"""
        ),
        code(TASK4_SOL),
        md(
            """
**Вывод по протоколу (пример):** главный скачок validation accuracy даёт переход
**одно дерево → bagging** (bootstrap-усреднение). Random features у RF и рост
`n_estimators` после ~100 добавляют стабильность/небольшой прирост, но не такой большой,
как сам переход к ансамблю. `bootstrap=False` на этих данных меняет картину слабо.
"""
        ),
        # ---- Task 5 ----
        md(
            """
---
## Задание 5. OOB: контакты вне смены — **3 балла**

Обучите `RandomForestClassifier(..., oob_score=True)` на **train**.
Сравните `oob_score_` с validation accuracy.

Смысл OOB простыми словами: для каждого дерева есть контакты,
**которые эта смена (bootstrap-мешок) не видела**. Их ответы — честная быстрая проверка.
Это **не** финальный скрытый test всего проекта.

### Подробные критерии (для проверки LLM)

- **1.0 балл** — RF обучен с `oob_score=True` на train.
- **1.0 балл** — выведены OOB score и validation accuracy.
- **1.0 балл** — в markdown/print есть пояснение, что OOB ≠ финальный test.

### Снижение баллов

- OOB считают на validation напрямую «вручную» вместо `oob_score_` → минус **0.5**
  (если идея верная, но API не использован).
- Путают OOB с test и предлагают подбирать всё только по OOB без оговорок → минус **1.0**.
"""
        ),
        code(TASK5_SOL),
        # ---- Task 6 ----
        md(
            """
---
## Задание 6. Детектив: permutation importance — **4 балла**

Обучите RF на train. На **validation** посчитайте `permutation_importance`.

Покажите таблицу и barh-график. Ответьте:

1. какие приборы реально полезны;
2. что каналы `noise_*` бесполезны (важность около нуля).

### Подробные критерии (для проверки LLM)

- **1.0 балл** — RF обучен на train.
- **1.5 балла** — `permutation_importance` посчитан на validation.
- **0.5 балла** — есть таблица/сортировка важностей.
- **1.0 балл** — явно отмечены полезные приборы и бесполезность `noise_*`.

### Снижение баллов

- Importance считают на train и выдают за «боевую» полезность без оговорки → минус **0.5**.
- Нет вывода про `noise_*` → минус **1.0**.
"""
        ),
        code(TASK6_SOL),
        # ---- Task 7 ----
        md(
            """
---
## Задание 7. Детектив: странный контакт — **5 баллов**

Возьмите (или сконструируйте) **странный контакт**: конфликт показаний приборов
и большой `noise_*`.

1. Получите `predict_proba` обученного RF.
2. По очереди «выключайте» признаки (замена на медиану train) и смотрите,
   как падает вероятность исходного класса — какой прибор сильнее тянет решение.
3. Дополнительно сравните validation accuracy при отключении групп:
   без `noise_*`, без ключевого прибора, только на `noise_*`.

### Подробные критерии (для проверки LLM)

- **1.0 балл** — построен/выбран странный контакт и получен `predict_proba`.
- **1.5 балла** — есть анализ влияния признаков через замену значения / importance на кейсе.
- **1.5 балла** — есть абляция групп признаков (noise / ключевой прибор) с метрикой на validation.
- **1.0 балл** — сформулирован вывод: какой признак тянет кейс; `noise_*` не тянут.

### Снижение баллов

- Нет работы с вероятностями/`predict_proba` → минус **1.0**.
- Вывод противоречит числам (например, объявляют `noise_*` главными) → минус **1.5**.
"""
        ),
        code(TASK7_SOL),
        # ---- Task 8 ----
        md(
            """
---
## Задание 8. Итоговые выводы — **2 балла**

Напишите **три** коротких вывода по своим числам и протоколу:

1. какой ингредиент абляции дал главный прирост;
2. зачем нужен протокол экспериментов (а не одна цифра accuracy);
3. что показал детектив по приборам (`noise_*` vs реальные датчики).

### Подробные критерии (для проверки LLM)

- **0.7 балла** — вывод про главный ингредиент абляции.
- **0.6 балла** — вывод про ценность протокола.
- **0.7 балла** — вывод про приборы / `noise_*`.

### Снижение баллов

- Общие фразы без опоры на таблицу/протокол → минус **0.5**.
"""
        ),
        code(TASK8_SOL),
        md(
            """
**Итоговые выводы (пример ответа):**

1. **Абляция:** главный прирост validation accuracy даёт **bagging** (усреднение по bootstrap)
   относительно одного дерева. Random features и большой `n_estimators` добавляют сверху,
   но обычно меньший шаг.
2. **Протокол:** одна цифра «RF = 0.69» не объясняет *почему*. Строки гипотеза → настройки →
   метрики → вывод показывают, какой тумблер реально сработал.
3. **Детектив:** полезны именованные приборы (`ir_delta`, `radar_rcs`, …);
   каналы `noise_*` в permutation importance и в кейсе странного контакта почти не влияют.
"""
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
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
            "name": "Урок_34_Ансамбли_Bagging_Случайный_лес_Практика",
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
