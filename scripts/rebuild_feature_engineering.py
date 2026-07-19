#!/usr/bin/env python3
"""Rebuild lesson 23/24 feature engineering: theory, exercises, practice.

Сквозной датасет: таблица объявлений о продаже квартир (flats), цель — цена.
"""

from __future__ import annotations

import json
from pathlib import Path

THEORY_FOLDER = Path(__file__).resolve().parents[1] / "Урок_23_Feature_Engineering_Теория"
PRACTICE_FOLDER = Path(__file__).resolve().parents[1] / "Урок_24_Feature_Engineering_Практика"
CRITERION_MARKER = "@@CRITERION@@"


def expand_short_criteria(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(CRITERION_MARKER):
            criterion = stripped.removeprefix(CRITERION_MARKER).strip().rstrip(".")
            if out and out[-1] != "":
                out.append("")
            out.extend(
                [
                    "### Подробные критерии (для проверки LLM)",
                    "",
                    "- Выполнены все действия, перечисленные в условии задания.",
                    f"- Проверочный ориентир: {criterion}.",
                    "- Если задание требует код, код запускается без ошибок и создаёт названные в условии переменные, таблицы или графики.",
                    "- Если задание требует текстовый вывод, вывод опирается на полученные числа, таблицы или графики, а не на общие рассуждения.",
                    "",
                    "### Снижение баллов",
                    "",
                    "- Отсутствует ключевой объект из условия (переменная, таблица, график или текстовый вывод) → существенное снижение.",
                    "- Использована не та выборка для обучения, подбора или оценки качества → существенное снижение.",
                    "- Код не запускается из-за ошибки или результат не связан с условием задания → существенное снижение.",
                    "- Текстовый вывод не объясняет полученный результат или противоречит числам/графикам → снижение.",
                ]
            )
        else:
            out.append(line.rstrip())
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [expand_short_criteria(text)]}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [src],
    }


FLATS_CODE = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

flats = pd.DataFrame({
    'площадь': [35, 50, 80, 120, 42, 28, 95, 60, 33, 75, 55, 110, 38, 68, 47],
    'комнаты': [1, 2, 3, 4, 1, 1, 3, 2, 1, 3, 2, 4, 1, 2, 2],
    'район': ['центр', 'спальный', 'центр', 'пригород', 'спальный',
              'центр', 'пригород', 'спальный', 'центр', 'спальный',
              'пригород', 'центр', 'спальный', 'исторический', 'спальный'],
    'состояние': ['хорошее', 'среднее', 'хорошее', 'требует ремонта', 'среднее',
                  'хорошее', 'среднее', 'хорошее', 'требует ремонта', 'среднее',
                  'хорошее', 'хорошее', 'среднее', 'хорошее', 'требует ремонта'],
    'балкон': [0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    'дата': pd.to_datetime(['2026-01-10', '2026-02-15', '2026-06-01', '2026-03-20',
                            '2026-01-25', '2026-04-05', '2026-02-08', '2026-05-12',
                            '2026-03-03', '2026-06-18', '2026-04-22', '2026-05-30',
                            '2026-01-17', '2026-02-27', '2026-06-09']),
    'просмотры_за_месяц': [120, 45, 300, 15, 60, 800, 25, 90, 40, 150,
                           55, 2500, 35, 70, 5000],
    'доход_района': [95, 60, np.nan, 45, 62, 90, np.nan, 58, 88, 61,
                     47, 97, np.nan, 85, 59],
    'цена': [7.2, 9.8, 17.5, 18.0, 8.1, 6.5, 15.2, 11.0, 6.9, 14.0,
             9.0, 24.0, 7.6, 13.5, 8.8],
})

flats_train, flats_val = train_test_split(flats, test_size=0.3, random_state=42)
print('Объектов:', len(flats), '| train:', len(flats_train), '| validation:', len(flats_val))
flats.head()
"""


def theory_cells() -> list[dict]:
    return [
        md(
            """# Занятие 23. Конструирование признаков

Модель не видит квартиру, клиента или дату — она видит **столбцы чисел**. **Конструирование признаков** (feature engineering) — превращение исходных данных в такие столбцы, из которых модели легче извлечь закономерность.

К концу занятия вы сможете:

- отличать безопасный признак от утечки и «признака из будущего»;
- строить числовые, категориальные и временные признаки;
- обрабатывать пропуски и неизвестные категории без утечки;
- честно проверять пользу нового признака на validation.

**Сквозной пример.** Все разделы работают с одной таблицей: объявления о продаже квартир, цель — предсказать **цену** (млн руб.). Один объект — одно объявление. Сразу делим на train и validation (70 % / 30 %), как на занятии 21: всё, что «обучается» по данным, будем считать только на train."""
        ),
        code(FLATS_CODE),
        md(
            """## 1. Что делает признак хорошим

**Признак** — столбец таблицы, который модель использует как вход. **Момент прогноза** — момент, когда модель должна дать ответ; для нашей задачи это момент публикации объявления.

Хороший признак:

1. **известен в момент прогноза** — «дата продажи» не годится: её ещё нет;
2. **не содержит цель** — «цена за метр» = цена / площадь, а цену мы и предсказываем;
3. **одинаково вычисляется** для train и новых объявлений;
4. **связан с задачей по смыслу** — есть гипотеза, почему он влияет на цену;
5. **устойчив** — не ломается от одного нового объекта.

Нарушение пп. 1–2 называется **утечкой** (занятие 21): модель получает информацию, которой в реальности не будет, и validation-оценка становится обманчиво хорошей."""
        ),
        md(
            """## 2. Отношения и разности

Комбинация двух столбцов часто информативнее каждого по отдельности:

- **отношение** — `площадь / комнаты`: просторность жилья; 80 м² на 3 комнаты и 80 м² на 1 комнату — разные квартиры;
- **разность** — например, «доход района − средний доход по городу»: насколько район богаче типичного.

Оба признака известны в момент публикации — безопасны."""
        ),
        code(
            """flats['площадь_на_комнату'] = flats['площадь'] / flats['комнаты']
flats[['площадь', 'комнаты', 'площадь_на_комнату', 'цена']].head()
"""
        ),
        md(
            """## 3. Логарифмирование скошенных величин

`просмотры_за_месяц` распределены **скошенно**: у большинства объявлений десятки просмотров, у пары «вирусных» — тысячи. Модели, чувствительные к масштабу (kNN, линейные), будут ориентироваться в основном на выбросы.

`np.log1p(x)` = log(1 + x) сжимает большие значения и не ломается на нуле. Порядок объектов сохраняется, а разрывы уменьшаются."""
        ),
        code(
            """flats['log_просмотры'] = np.log1p(flats['просмотры_за_месяц'])

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].hist(flats['просмотры_за_месяц'], bins=10)
axes[0].set_title('просмотры_за_месяц: хвост до 5000')
axes[1].hist(flats['log_просмотры'], bins=10, color='tab:orange')
axes[1].set_title('log1p(просмотры): компактнее')
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""
        ),
        md(
            """## 4. Интервалы и ограничение выбросов

Иногда важна не точная площадь, а **класс** жилья: малогабаритное / стандартное / просторное. `pd.cut` превращает число в интервал — линейной модели проще выучить пороговые эффекты.

Экстремальные значения (те же 5000 просмотров) можно **ограничить** разумной границей (`clip`). Но сначала проверьте смысл: настоящий редкий объект — не ошибка, и обрезать его стоит только если он мешает модели."""
        ),
        code(
            """flats['класс_площади'] = pd.cut(
    flats['площадь'], bins=[0, 40, 80, 200],
    labels=['малогабаритная', 'стандартная', 'просторная'],
)
flats['просмотры_clip'] = flats['просмотры_за_месяц'].clip(upper=500)
flats[['площадь', 'класс_площади', 'просмотры_за_месяц', 'просмотры_clip']].head()
"""
        ),
        md(
            """## 5. Нелинейность и взаимодействия

Линейная модель (занятие 25) складывает вклады признаков по отдельности. Новые столбцы позволяют ей описывать более сложные связи:

- $x^2$ — изгиб зависимости;
- $x_1 \\cdot x_2$ — **взаимодействие**: влияние одного признака зависит от другого;
- интервалы (п. 4) — разные «режимы».

Гипотеза для нашей таблицы: балкон добавляет к цене тем больше, чем больше квартира. Тогда полезен признак `площадь × балкон`.

Чем больше сгенерированных преобразований, тем выше риск переобучения — каждое добавление проверяем на validation (п. 16)."""
        ),
        code(
            """flats['площадь_x_балкон'] = flats['площадь'] * flats['балкон']
flats[['площадь', 'балкон', 'площадь_x_балкон']].head()
"""
        ),
        md(
            """## 6. Как увидеть пользу взаимодействия

Признак-взаимодействие нужен, когда на графике «признак → цель» **наклон зависит от группы**. Нарисуем цену против площади отдельно для квартир с балконом и без."""
        ),
        code(
            """plt.figure(figsize=(7, 4))
for balcony_value, color in [(0, 'tab:blue'), (1, 'tab:orange')]:
    subset = flats[flats['балкон'] == balcony_value]
    plt.scatter(subset['площадь'], subset['цена'], color=color,
                label=f'балкон={balcony_value}', s=60)
    k, b = np.polyfit(subset['площадь'], subset['цена'], deg=1)
    xs = np.linspace(25, 125, 50)
    plt.plot(xs, k * xs + b, color=color, alpha=0.6)
plt.xlabel('Площадь, м²'); plt.ylabel('Цена, млн руб.')
plt.title('Разные наклоны — сигнал в пользу признака площадь × балкон')
plt.grid(alpha=0.3); plt.legend(); plt.show()
"""
        ),
        md(
            """## 7. Даты и цикличность *

Из `дата` объявления можно получить компоненты: год, **месяц**, день недели, признак выходного.

У циклических величин есть подвох: месяц 12 и месяц 1 — соседи, но числа 12 и 1 далеки. Пара признаков

$$\\sin(2\\pi m / 12), \\qquad \\cos(2\\pi m / 12)$$

размещает месяцы на окружности: декабрь и январь оказываются рядом.

Раздел со звёздочкой: для начала достаточно компонентов даты и самой идеи цикличности."""
        ),
        code(
            """flats['месяц'] = flats['дата'].dt.month
flats['день_недели'] = flats['дата'].dt.dayofweek
flats['месяц_sin'] = np.sin(2 * np.pi * flats['месяц'] / 12)
flats['месяц_cos'] = np.cos(2 * np.pi * flats['месяц'] / 12)
flats[['дата', 'месяц', 'день_недели', 'месяц_sin', 'месяц_cos']].head().round(2)
"""
        ),
        md(
            """## 8. Время с события

Часто полезнее не сама дата, а **сколько времени прошло**: дней с публикации объявления — «залежавшиеся» квартиры могут быть переоценены.

Обе даты в разности должны быть доступны в **момент прогноза**. Если «последнее событие» случилось позже момента прогноза — это утечка (п. 1)."""
        ),
        code(
            """prediction_date = pd.Timestamp('2026-07-01')
flats['дней_с_публикации'] = (prediction_date - flats['дата']).dt.days
flats[['дата', 'дней_с_публикации']].head()
"""
        ),
        md(
            """## 9. Категориальные признаки и one-hot

`район` — **категориальный** признак: значения «центр», «спальный», «пригород» нельзя заменить числами 1, 2, 3 — модель увидит несуществующий порядок и «расстояния» между районами.

**One-hot encoding** создаёт для каждой категории столбец 0/1: объект получает 1 в «своём» столбце. Кодировщик обучаем (`fit`) на train — как любое преобразование."""
        ),
        code(
            """from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
train_encoded = encoder.fit_transform(flats_train[['район']])
print('Столбцы:', list(encoder.get_feature_names_out()))
pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out()).head()
"""
        ),
        md(
            """## 10. Порядковые категории

У `состояние` порядок настоящий: «требует ремонта» < «среднее» < «хорошее». **Ordinal encoding** сохраняет его числами 0, 1, 2 — важно задать порядок явно, а не доверять алфавиту.

Нюанс: линейная модель после такого кодирования считает шаг между уровнями **одинаковым** (ремонт → среднее = среднее → хорошее). Если это не так по смыслу — сравните с one-hot на validation. Для района и цвета порядка нет вовсе — там ordinal вреден."""
        ),
        code(
            """from sklearn.preprocessing import OrdinalEncoder

condition_order = [['требует ремонта', 'среднее', 'хорошее']]
ord_encoder = OrdinalEncoder(categories=condition_order)
flats['состояние_код'] = ord_encoder.fit_transform(flats[['состояние']])
flats[['состояние', 'состояние_код']].drop_duplicates().sort_values('состояние_код')
"""
        ),
        md(
            """## 11. Редкие и неизвестные категории

В нашей таблице район «исторический» встречается **один раз** — и после split он может оказаться только в validation. Кодировщик, обученный на train, такой категории не знает.

- `handle_unknown='ignore'` в `OneHotEncoder` кодирует неизвестную категорию **нулями во всех столбцах** — без ошибки.
- Очень редкие категории часто объединяют в группу «другое»; правило объединения выбирают по **train** и неизменно применяют к validation и test."""
        ),
        code(
            """val_encoded = encoder.transform(flats_val[['район']])
print('Районы в validation:', list(flats_val['район']))
pd.DataFrame(val_encoded, columns=encoder.get_feature_names_out())
"""
        ),
        md(
            """## 12. Пропуски и индикатор

В `доход_района` есть `NaN` — пропуски. Пропуск не означает ноль. Стратегии:

- заполнить **медианой**, посчитанной **по train** (не по всей таблице — иначе утечка);
- для категорий — отдельное значение «неизвестно»;
- добавить **индикатор** `доход_пропущен` (0/1): сам факт пропуска бывает информативен."""
        ),
        code(
            """flats['доход_пропущен'] = flats['доход_района'].isna().astype(int)
median_income = flats_train['доход_района'].median()  # только train!
flats['доход_заполнен'] = flats['доход_района'].fillna(median_income)
print('Медиана по train:', median_income)
flats[['доход_района', 'доход_заполнен', 'доход_пропущен']].head(8)
"""
        ),
        md(
            """## 13. Почему пропуски появились

Пропуск бывает трёх видов:

- **случайная техническая ошибка** — форма не сохранилась;
- **следствие процесса** — для новых районов доход ещё не посчитан;
- **сигнал о группе объектов** — владельцы дорогих квартир реже указывают детали.

В третьем случае пропуск сам по себе — информация, и индикатор из п. 12 может улучшить модель. Проверка стандартная: сравнить распределение **цены** для строк с пропуском и без."""
        ),
        md(
            """## 14. Агрегаты по прошлому и утечка *

Признаки-**агрегаты** считаются по другим строкам: «средняя цена квартир этого района, проданных **раньше**». Ключевое слово — раньше: если в среднее попали продажи **после** момента прогноза, модель заглянула в будущее.

**Target encoding (*)** — крайний случай: категорию заменяют средним значением **цели** по этой категории. Наивный расчёт по всей таблице сообщает объекту часть его собственного ответа — сильная утечка. Безопасные варианты требуют out-of-fold вычислений; пока достаточно запомнить: **среднее цели по всей таблице считать нельзя**.

Проверочный вопрос к любому агрегату: «мог ли я посчитать это число в момент прогноза, не зная ответа?»"""
        ),
        md(
            """## 15. Масштабирование

После пп. 2–12 признаки живут в разных масштабах: площадь — десятки, просмотры — тысячи, one-hot — нули и единицы. Моделям, которые сравнивают **расстояния** (kNN) или штрафуют большие коэффициенты, нужен сопоставимый масштаб.

`StandardScaler` вычитает среднее и делит на стандартное отклонение. Правило прежнее: **`fit` на train, `transform` — на train и validation**. Деревьям и лесам масштабирование обычно не нужно — они сравнивают значения с порогом."""
        ),
        code(
            """from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(flats_train[['площадь', 'комнаты']])
scaled_val = scaler.transform(flats_val[['площадь', 'комнаты']])
print('Средние train (для вычитания):', scaler.mean_.round(1))
pd.DataFrame(scaled_val, columns=['площадь_std', 'комнаты_std']).round(2)
"""
        ),
        md(
            """## 16. Проверка признака на validation

Больше столбцов — не значит лучше: лишние признаки добавляют шум и риск утечки. Пользу проверяют **экспериментом**:

1. зафиксировать базовый набор признаков;
2. добавить **одну** группу новых;
3. обучить одну и ту же модель на том же train;
4. сравнить метрику на том же validation;
5. test не трогать — он для финала (занятие 29).

Проверим гипотезу из п. 2: помогает ли `площадь_на_комнату` предсказывать цену. Модель — знакомый kNN (занятие 21), метрика — MAE (средняя абсолютная ошибка, млн руб.).

Спойлер: на нашей маленькой таблице признак MAE **не улучшит** (2.52 → 2.63). Это нормальный исход честного эксперимента: гипотеза не подтвердилась — признак в финальный набор не берём. Отрицательный результат — тоже результат."""
        ),
        code(
            """from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

def validation_mae(features):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(n_neighbors=3)),
    ])
    model.fit(flats_train[features], flats_train['цена'])
    predictions = model.predict(flats_val[features])
    return mean_absolute_error(flats_val['цена'], predictions)

base_features = ['площадь', 'комнаты']
extended_features = base_features + ['площадь_на_комнату']

# доп. признаки уже посчитаны в flats, поэтому пересчитаем их и в split-таблицах
for table in (flats_train, flats_val):
    table['площадь_на_комнату'] = table['площадь'] / table['комнаты']

print('MAE базовый набор:      ', round(validation_mae(base_features), 2), 'млн')
print('MAE + площадь_на_комнату:', round(validation_mae(extended_features), 2), 'млн')
"""
        ),
        md(
            """## 17. Стоимость признака и чек-лист

Признак может улучшать метрику, но быть **дорогим**: медленно считается, требует внешнего источника, недоступен в реальном времени. Финальный набор — компромисс качества, риска утечки и цены поддержки, а не максимум столбцов.

**Чек-лист перед добавлением признака:**

1. Доступен в момент прогноза? (п. 1)
2. Не содержит цель или будущее? (пп. 1, 14)
3. Одинаково вычисляется для train и новых данных? (пп. 9–12)
4. Обработаны пропуски и неизвестные категории? (пп. 11–12)
5. Нужен ли масштаб? (п. 15)
6. Есть понятная гипотеза о пользе? (пп. 5–6)
7. Устойчиво улучшает validation? (п. 16)

> **Главная мысль.** Признаки — способ рассказать модели о задаче, не сообщая ей ответы из будущего."""
        ),
    ]


def exercises_cells() -> list[dict]:
    return [
        md(
            """# Занятие 23. Упражнения: признаки

Короткая проверка теории (`Урок_23_Feature_Engineering_Теория.ipynb`, занятие 23). Полный пайплайн — на практике (занятие 24).

Формат: замените `...` своим ответом и запустите ячейку."""
        ),
        code("import numpy as np\nimport pandas as pd\n"),
        md(
            """## 1. Безопасный признак (п. 1)

Мы предсказываем **цену** квартиры. Какой признак безопасен: `'площадь на комнату'` или `'цена за метр'`?"""
        ),
        code(
            """feature = ...
assert feature == 'площадь на комнату'
print('Верно')
"""
        ),
        md("## 2. Отношение (п. 2)\n\nВычислите площадь на комнату."),
        code(
            """area = np.array([30, 60, 90])
rooms = np.array([1, 2, 3])
answer = ...
assert np.array_equal(answer, [30, 30, 30])
print('Верно')
"""
        ),
        md("## 3. Логарифм (п. 3)\n\nПримените `np.log1p` к просмотрам."),
        code(
            """views = np.array([0, 100, 10000])
answer = ...
assert np.allclose(answer, np.log1p(views))
print('Верно')
"""
        ),
        md("## 4. Компоненты даты (п. 7)\n\nПолучите номера месяцев."),
        code(
            """dates = pd.Series(pd.to_datetime(['2026-01-05', '2026-07-02']))
months = ...
assert list(months) == [1, 7]
print('Верно')
"""
        ),
        md(
            """## 5. Неизвестная категория (п. 11)

Каким параметром `OneHotEncoder` безопасно обрабатывает район, которого не было в train? Запишите строкой вида `параметр='значение'`."""
        ),
        code(
            """parameter = ...
assert parameter == "handle_unknown='ignore'"
print('Верно')
"""
        ),
        md(
            """## 6. Ordinal или one-hot (пп. 9–10)

Для какого признака подходит **ordinal encoding**: `'состояние'` (требует ремонта / среднее / хорошее) или `'район'`?"""
        ),
        code(
            """ordinal_feature = ...
assert ordinal_feature == 'состояние'
print('Верно')
"""
        ),
        md("## 7. Индикатор пропуска (п. 12)\n\nСоздайте индикатор: 1, если значение пропущено."),
        code(
            """values = pd.Series([2, np.nan, 5, np.nan])
flag = ...
assert np.array_equal(flag, [0, 1, 0, 1])
print('Верно')
"""
        ),
        md(
            """## 8. Утечка в агрегате (п. 14)

Момент прогноза — **1 июня**. Какую продажу нельзя включать в признак «средняя цена прошлых продаж района»: `'продажа 20 мая'` или `'продажа 3 июня'`?"""
        ),
        code(
            """forbidden = ...
assert forbidden == 'продажа 3 июня'
print('Верно')
"""
        ),
        md(
            """## 9. Взаимодействие (п. 5)

Гипотеза: балкон добавляет к цене тем больше, чем больше площадь. Постройте признак-взаимодействие."""
        ),
        code(
            """area = np.array([30, 60, 100])
balcony = np.array([0, 1, 1])
interaction = ...
assert np.array_equal(interaction, [0, 60, 100])
print('Верно')
"""
        ),
        md(
            """## 10. Проверка признака (п. 16)

На какой части данных сравнивают наборы признаков: `'train'`, `'validation'` или `'test'`?"""
        ),
        code(
            """part = ...
assert part == 'validation'
print('Верно')
"""
        ),
    ]


PRACTICE_GIVEN = """import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 300

districts = rng.choice(['центр', 'спальный', 'пригород'], size=n, p=[0.3, 0.5, 0.2])
condition = rng.choice(['требует ремонта', 'среднее', 'хорошее'], size=n, p=[0.2, 0.5, 0.3])
area = rng.uniform(25, 130, n).round(0)
rooms = np.clip((area / 30).astype(int) + rng.integers(0, 2, n), 1, 5)
balcony = rng.integers(0, 2, n)
listing_date = pd.Timestamp('2026-01-01') + pd.to_timedelta(rng.integers(0, 180, n), unit='D')
views = np.expm1(rng.normal(4.0, 1.2, n)).round(0)

district_bonus = pd.Series(districts).map({'центр': 4.0, 'спальный': 1.0, 'пригород': 0.0}).values
condition_bonus = pd.Series(condition).map(
    {'требует ремонта': -1.5, 'среднее': 0.0, 'хорошее': 1.5}
).values
price = (0.13 * area + 0.02 * area * balcony + district_bonus
         + condition_bonus + rng.normal(0, 1.0, n)).round(1)

district_income = pd.Series(districts).map({'центр': 95.0, 'спальный': 60.0, 'пригород': 45.0}).values
district_income = district_income + rng.normal(0, 5, n).round(0)
missing_mask = rng.random(n) < 0.15
district_income[missing_mask] = np.nan

flats = pd.DataFrame({
    'площадь': area,
    'комнаты': rooms,
    'район': districts,
    'состояние': condition,
    'балкон': balcony,
    'дата': listing_date,
    'просмотры_за_месяц': views,
    'доход_района': district_income,
    'цена': price,
})

PREDICTION_DATE = pd.Timestamp('2026-07-01')
RANDOM_STATE = 42
print('Объектов:', len(flats))
flats.head()
"""


def practice_cells(with_solutions: bool) -> list[dict]:
    def task_code(solution: str) -> dict:
        return code(solution) if with_solutions else code("")

    header = (
        """# Занятие 24. Практика: признаки для цены квартиры (~90 мин)

Вы **пишете весь код сами**. Ячейку **«Дано»** не меняйте.

Задача — регрессия: предсказать **цену** квартиры (млн руб.) по объявлению. Модель для проверки идей — kNN-регрессия (занятие 21), метрика — **MAE**. Теория — `Урок_23_Feature_Engineering_Теория.ipynb` (занятие 23).

Ориентир по времени указан у каждого задания. Если застряли — идите дальше и вернитесь позже."""
    )
    if with_solutions:
        header = (
            "# Занятие 24. Практика: признаки для цены квартиры (~90 мин)\n\n"
            "Авторский вариант практики: в блокноте есть условия, ответы и подробные критерии для LLM-проверки."
        )

    return [
        md(header),
        md(
            """---
## Дано: датасет

Синтетическая таблица объявлений (300 строк): площадь, комнаты, район, состояние, балкон, дата публикации, просмотры за месяц, доход района (есть пропуски) и **цена** — целевая переменная. Момент прогноза — `PREDICTION_DATE`."""
        ),
        code(PRACTICE_GIVEN),
        md(
            """---
## Задание 0. Постановка (~5 мин)

По мотивам п. 1 теории.

1. Создайте словарь `task_spec` с ключами `объект`, `цель`, `тип_задачи`, `момент_прогноза`.
2. Назовите в комментарии один признак, который **нельзя** строить (утечка), и почему.

@@CRITERION@@ словарь заполнен; пример утечки назван."""
        ),
        task_code(
            """task_spec = {
    'объект': 'объявление о продаже квартиры',
    'цель': 'цена, млн руб.',
    'тип_задачи': 'регрессия',
    'момент_прогноза': 'публикация объявления (PREDICTION_DATE для агрегатов)',
}
# Утечка: 'цена за метр' = цена / площадь — содержит целевую переменную.
task_spec
"""
        ),
        md(
            """---
## Задание 1. Split (~7 мин)

По мотивам п. 0 теории (и занятия 21, п. 4).

1. Разбейте `flats` на `flats_train` и `flats_val` (70/30, `random_state=RANDOM_STATE`).
2. Выведите размеры.

@@CRITERION@@ 210 / 90 объектов; дальше всё «обучаемое» — только по train."""
        ),
        task_code(
            """from sklearn.model_selection import train_test_split

flats_train, flats_val = train_test_split(flats, test_size=0.3, random_state=RANDOM_STATE)
print('train:', len(flats_train), '| validation:', len(flats_val))
"""
        ),
        md(
            """---
## Задание 2. Числовые признаки (~10 мин)

По мотивам пп. 2–3 теории. В **обеих** таблицах (train и val):

1. `площадь_на_комнату` = площадь / комнаты.
2. `log_просмотры` = `np.log1p(просмотры_за_месяц)`.

@@CRITERION@@ новые столбцы есть в train и val; значения без NaN."""
        ),
        task_code(
            """for table in (flats_train, flats_val):
    table['площадь_на_комнату'] = table['площадь'] / table['комнаты']
    table['log_просмотры'] = np.log1p(table['просмотры_за_месяц'])
flats_train[['площадь_на_комнату', 'log_просмотры']].describe().round(2)
"""
        ),
        md(
            """---
## Задание 3. Признаки из даты (~10 мин)

По мотивам пп. 7–8 теории. В обеих таблицах:

1. `месяц` из даты; `месяц_sin`, `месяц_cos` (цикличность).
2. `дней_с_публикации` = `PREDICTION_DATE` − дата.

@@CRITERION@@ `дней_с_публикации` ≥ 0 для всех строк."""
        ),
        task_code(
            """for table in (flats_train, flats_val):
    table['месяц'] = table['дата'].dt.month
    table['месяц_sin'] = np.sin(2 * np.pi * table['месяц'] / 12)
    table['месяц_cos'] = np.cos(2 * np.pi * table['месяц'] / 12)
    table['дней_с_публикации'] = (PREDICTION_DATE - table['дата']).dt.days
assert (flats_train['дней_с_публикации'] >= 0).all()
flats_train[['месяц', 'месяц_sin', 'месяц_cos', 'дней_с_публикации']].head().round(2)
"""
        ),
        md(
            """---
## Задание 4. Категории (~12 мин)

По мотивам пп. 9–11 теории.

1. `OneHotEncoder(handle_unknown='ignore')` для `район`: `fit` на train, `transform` на train и val.
2. `OrdinalEncoder` для `состояние` с явным порядком: требует ремонта < среднее < хорошее; столбец `состояние_код` в обеих таблицах.

@@CRITERION@@ кодировщики обучены только на train; в val нет ошибок."""
        ),
        task_code(
            """from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

district_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
district_train = district_encoder.fit_transform(flats_train[['район']])
district_val = district_encoder.transform(flats_val[['район']])
district_cols = list(district_encoder.get_feature_names_out())

cond_encoder = OrdinalEncoder(categories=[['требует ремонта', 'среднее', 'хорошее']])
flats_train['состояние_код'] = cond_encoder.fit_transform(flats_train[['состояние']])
flats_val['состояние_код'] = cond_encoder.transform(flats_val[['состояние']])

for table, encoded in ((flats_train, district_train), (flats_val, district_val)):
    for j, col in enumerate(district_cols):
        table[col] = encoded[:, j]
print('One-hot столбцы:', district_cols)
"""
        ),
        md(
            """---
## Задание 5. Пропуски (~8 мин)

По мотивам п. 12 теории.

1. `доход_пропущен` — индикатор 0/1 в обеих таблицах.
2. `доход_заполнен` — NaN заменены **медианой train**.

@@CRITERION@@ медиана посчитана только по train; NaN не осталось."""
        ),
        task_code(
            """median_income = flats_train['доход_района'].median()
for table in (flats_train, flats_val):
    table['доход_пропущен'] = table['доход_района'].isna().astype(int)
    table['доход_заполнен'] = table['доход_района'].fillna(median_income)
assert flats_val['доход_заполнен'].notna().all()
print('Медиана train:', round(median_income, 1))
"""
        ),
        md(
            """---
## Задание 6. Baseline-набор и kNN (~10 мин)

По мотивам пп. 15–16 теории.

1. Базовый набор: `['площадь', 'комнаты']`.
2. Pipeline: `StandardScaler` + `KNeighborsRegressor(n_neighbors=5)`.
3. Обучите на train, посчитайте **MAE на validation** → `mae_base`.

@@CRITERION@@ scaler обучается внутри pipeline только на train."""
        ),
        task_code(
            """from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def validation_mae(features):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(n_neighbors=5)),
    ])
    model.fit(flats_train[features], flats_train['цена'])
    return mean_absolute_error(flats_val['цена'], model.predict(flats_val[features]))

base_features = ['площадь', 'комнаты']
mae_base = validation_mae(base_features)
print('MAE базовый набор:', round(mae_base, 3), 'млн')
"""
        ),
        md(
            """---
## Задание 7. Добавляем группы признаков (~12 мин)

По мотивам п. 16 теории. Сравните на validation наборы (каждый = базовый + одна группа):

- `+ ['площадь_на_комнату', 'log_просмотры']`
- `+ ['состояние_код', 'балкон']`
- `+ one-hot районов`

@@CRITERION@@ таблица «набор → MAE»; выводы о том, какая группа помогла."""
        ),
        task_code(
            """candidate_groups = {
    'числовые': ['площадь_на_комнату', 'log_просмотры'],
    'состояние+балкон': ['состояние_код', 'балкон'],
    'район (one-hot)': district_cols,
}
results = {'база': mae_base}
for name, group in candidate_groups.items():
    results[name] = validation_mae(base_features + group)
pd.Series(results).round(3).to_frame('MAE, млн')
"""
        ),
        md(
            """---
## Задание 8. Комбинированный набор (~8 мин)

1. Соберите набор из базового + всех групп, которые **улучшили** MAE в задании 7.
2. Посчитайте `mae_final` на validation и сравните с `mae_base`.

@@CRITERION@@ `mae_final` ≤ `mae_base`; состав набора обоснован результатами задания 7."""
        ),
        task_code(
            """final_features = base_features + candidate_groups['состояние+балкон'] + candidate_groups['район (one-hot)']
mae_final = validation_mae(final_features)
print('Финальный набор:', final_features)
print('MAE база:', round(mae_base, 3), '→ MAE финал:', round(mae_final, 3))
"""
        ),
        md(
            """---
## Задание 9. Итог (~8 мин)

В markdown-ячейке ниже ответьте:

1. Какие группы признаков улучшили MAE, какие нет — и почему (гипотеза)?
2. Какой признак вы **не стали** строить из-за утечки?
3. Что нужно не забыть при применении модели к завтрашним объявлениям (кодировщики, медиана, масштаб)?

@@CRITERION@@ ответы ссылаются на числа из заданий 7–8, а не «кажется»."""
        ),
        md(
            "*Ответ студента.*"
            if not with_solutions
            else """**Эталонные тезисы:**

1. Состояние+балкон и район дают наибольший вклад — они напрямую входят в формулу цены; log-просмотры почти не помогают (просмотры не влияют на цену в этих данных).
2. «Цена за метр» — содержит цель; агрегат «средняя цена района по всем данным» — заглядывает в будущее и в validation.
3. Сохранить и применять: обученные на train кодировщики (`OneHotEncoder`, `OrdinalEncoder`), медиану дохода train, `StandardScaler` внутри pipeline; новые категории районов обрабатываются `handle_unknown='ignore'`."""
        ),
    ]


def write_nb(path: Path, cells: list[dict], name: str) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "name": name,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {path.name} ({len(cells)} cells)")


def main() -> None:
    write_nb(THEORY_FOLDER / "Урок_23_Feature_Engineering_Теория.ipynb", theory_cells(), "feature_engineering_theory")
    write_nb(THEORY_FOLDER / "Тест_к_теории.ipynb", exercises_cells(), "feature_engineering_exercises")
    write_nb(PRACTICE_FOLDER / "Урок_24_Feature_Engineering_Практика.ipynb", practice_cells(True), "feature_engineering_practice")


if __name__ == "__main__":
    main()
