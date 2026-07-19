from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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


def lines(text: str) -> list[str]:
    return (text.strip("\n") + "\n").splitlines(keepends=True)


def set_cell(nb: dict, index: int, text: str) -> None:
    nb["cells"][index]["source"] = lines(expand_short_criteria(text))


def patch_notebook(path: str, patches: dict[int, str]) -> None:
    p = ROOT / path
    nb = json.loads(p.read_text(encoding="utf-8"))
    for index, text in patches.items():
        set_cell(nb, index, text)
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"patched {path}: {len(patches)} cells")


def main() -> None:
    patch_notebook(
        "11_workflow/workflow_practice.ipynb",
        {
            0: """
# Занятие 22. Практика: классификация иконок методом k ближайших соседей (kNN)

Вы **пишете код сами** в пустых ячейках. Блоки **«Легенда»** и **«Дано»** не меняйте.

Теория — в `workflow_theory.ipynb`. Проверочные критерии прямо указаны в заданиях; решение преподаватель может хранить отдельно.

### Оценивание (30 баллов)

| № | Тема | Баллы |
|---|------|------:|
| 1 | Мини-симуляция релиза иконок | 2 |
| 2 | Импорты и разбиение данных | 4 |
| 3 | Осмотр train (числа) | 2 |
| 4 | Диаграммы по train | 3 |
| 5 | Baseline | 3 |
| 6 | kNN и сравнение с baseline | 3 |
| 7 | Масштабирование и kNN | 3 |
| 8 | Подбор `k` и график | 4 |
| 9 | Анализ ошибок и диаграммы | 5 |
| 10 | Финальная проверка на test | 2 |
| | **Итого** | **30** |
""",
            4: """
---
## Задание 1. Мини-симуляция релиза иконок — **2 балла**

Представьте, что в игру пришла новая иконка. Нужно решить:
- показать её автоматически;
- или отправить художнику на ручную проверку.

В этом задании модель ещё не обучена, поэтому `model_confidence` — **условная уверенность прототипа**, которую вы задаёте сами числами от 0 до 1. Это маленькая разминка про постановку задачи и правила принятия решения.

### Что сделать

**Шаг 1.** Создайте `release_rule` с ключами:
`auto_accept_threshold`, `auto_action`, `manual_action`.

**Шаг 2.** Создайте `incoming_icons` (4 строки) с колонками:
`icon_id`, `predicted_class`, `model_confidence`.

**Шаг 3.** Добавьте `decision` по порогу уверенности:
- если `model_confidence >= auto_accept_threshold`, иконка проходит автоматически;
- иначе иконка уходит на ручную проверку.

**Шаг 4.** Задайте `PRIMARY_METRIC = 'accuracy'` и в markdown объясните, зачем нужна ручная проверка для неуверенных иконок.

### Подробные критерии (для проверки LLM)

- **0.5 балла** — в `release_rule` есть все 3 ключа и `auto_accept_threshold` в диапазоне `(0, 1)`.
- **0.5 балла** — `incoming_icons` содержит минимум 4 строки и нужные 3 колонки.
- **0.5 балла** — `decision` заполнена по правилу порога (`>=` → авто, `<` → ручная проверка).
- **0.5 балла** — задан `PRIMARY_METRIC = 'accuracy'` и есть прикладное объяснение в markdown.

### Снижение баллов
- Нет `decision` или она не зависит от порога → не выше **1.0**.
- Нет `PRIMARY_METRIC` или нет объяснения → минус **0.5**.
""",
            9: """
### Шаг 2. Код разбиения train / validation / test
""",
            15: """
---
## Задание 5. Baseline — **3 балла**

### Что сделать

**Шаг 1.** Обучите `DummyClassifier(strategy='most_frequent')` на train.

**Шаг 2.** Посчитайте `dummy_acc` на validation.

**Шаг 3.** Найдите самый частый класс в train и его долю.

**Шаг 4.** Сравните `dummy_acc` с долей самого частого класса. Они должны быть близки по смыслу: baseline всегда выбирает самый частый класс.

### Подробные критерии (для проверки LLM)

- **1.0 балл** — baseline обучен.
- **1.0 балл** — `dummy_acc` посчитан на validation.
- **1.0 балл** — есть сравнение с долей самого частого класса в train.

### Снижение баллов
- Метрика не на validation → минус **1.0**.
- Нет сравнения с самым частым классом → минус **0.5**.
""",
            16: """
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
dummy_acc = accuracy_score(y_val, dummy.predict(X_val))

most_frequent_class = y_train.value_counts().idxmax()
most_frequent_share = y_train.value_counts(normalize=True).max()

print('dummy_acc', round(dummy_acc, 3))
print('самый частый класс в train:', most_frequent_class)
print('его доля в train:', round(most_frequent_share, 3))
""",
            23: """
---
## Задание 8. Подбор `k` и график — **4 балла**

### Что сделать

**Шаг 1.** Сначала посчитайте accuracy для `k=3` как проверочный пример: так вы убедитесь, что обучение kNN на масштабированных данных работает.

**Шаг 2.** Переберите `k` из `[1, 3, 5, 7, 11, 15, 21]`.

**Шаг 3.** Соберите таблицу `k_results` (`k`, `val_acc`).

**Шаг 4.** Найдите `best_k` по validation accuracy.

**Шаг 5.** Постройте линейный график `k`→`val_acc` и отметьте `best_k`.

### Подробные критерии (для проверки LLM)

- **1.0 балл** — отдельно посчитана accuracy для `k=3` как sanity-check.
- **1.0 балл** — есть перебор всех `k`.
- **1.0 балл** — есть `k_results` и `best_k`.
- **1.0 балл** — график с отмеченной лучшей точкой.

### Снижение баллов
- Подбор `k` по test → минус **1.0**.
- Нет отметки `best_k` на графике → минус **0.5**.
""",
            27: """
---
## Задание 9. Анализ ошибок и диаграммы — **5 баллов**

Это большое задание, поэтому делайте его по частям.

### Часть A. Какие классы модель путает?

**Шаг 1.** Обучите kNN с `best_k` на `X_train_s`, получите `y_pred` на validation.

**Шаг 2.** Посчитайте `confusion_matrix` с фиксированным порядком классов:
`['castle', 'dragon', 'knight']`.

**Шаг 3.** Постройте тепловую карту матрицы ошибок.

**Шаг 4.** Посчитайте recall по классам (`average=None`) и отдельно `recall_dragon`.

### Часть B. На каких иконках ошибок больше?

**Шаг 5.** Разделите validation на группы «разреженные/плотные» по медиане `pixel_density`.

**Шаг 6.** Посчитайте долю ошибок в каждой группе и постройте график.

**Шаг 7.** В markdown дайте 2 вывода:
1. какая путаница классов чаще всего видна в матрице ошибок;
2. в какой группе плотности ошибок больше.

### Подробные критерии (для проверки LLM)

- **1.0 балл** — `confusion_matrix` с фиксированным порядком классов.
- **1.0 балл** — есть тепловая карта.
- **1.0 балл** — есть recall по классам и `recall_dragon`.
- **1.0 балл** — есть расчёт и график ошибок по плотности.
- **1.0 балл** — есть оба текстовых вывода в markdown.

### Снижение баллов
- Нет фиксированного порядка классов → минус **0.5**.
- Нет одной из диаграмм → минус **0.5** за каждую.
- Неполный текстовый вывод → минус **0.5**.
""",
        },
    )

    patch_notebook(
        "12_feature_engineering/feature_engineering_practice.ipynb",
        {
            5: """
---
## Задание 1. Split (~7 мин)

По мотивам занятия 21 о train/validation/test.

1. Разбейте `flats` на `flats_train` и `flats_val` (70/30, `random_state=RANDOM_STATE`).
2. Выведите размеры.

@@CRITERION@@ 210 / 90 объектов; дальше всё «обучаемое» — только по train.
""",
            11: """
---
## Задание 4. Категории (~12 мин)

По мотивам пп. 9–11 теории.

1. `OneHotEncoder(handle_unknown='ignore')` для `район`: `fit` на train, `transform` на train и val.
2. Превратите результат one-hot в таблицы с понятными именами столбцов, например `район_центр`, `район_спальный`.
3. `OrdinalEncoder` для `состояние` с явным порядком: требует ремонта < среднее < хорошее; столбец `состояние_код` в обеих таблицах.

Когда будете обучать kNN, one-hot столбцы нужно объединить с числовыми признаками через `pd.concat` или другим явным способом.

@@CRITERION@@ кодировщики обучены только на train; в val нет ошибок; понятно, какие столбцы one-hot добавлены.
""",
            17: """
---
## Задание 7. Добавляем группы признаков (~12 мин)

По мотивам п. 16 теории. Сравните на validation наборы (каждый = базовый + одна группа):

- `+ ['площадь_на_комнату', 'log_просмотры']`
- `+ ['состояние_код', 'балкон']`
- `+ one-hot столбцы районов`

Для каждого набора явно соберите таблицу признаков train и val. Если используете one-hot районы, присоедините one-hot столбцы к числовым через `pd.concat`.

@@CRITERION@@ таблица «набор → MAE»; выводы о том, какая группа помогла.
""",
            19: """
---
## Задание 8. Комбинированный набор (~8 мин)

1. Соберите набор из базового + групп, которые улучшили MAE в задании 7.
2. Если ни одна группа не улучшила MAE, возьмите базовый набор и одну лучшую по MAE группу — как эксперимент для сравнения.
3. Посчитайте `mae_final` на validation и сравните с `mae_base`.

@@CRITERION@@ `mae_final` посчитан; состав набора обоснован числами из задания 7. Улучшение не гарантируется на каждой validation-выборке, поэтому важнее честное сравнение и объяснение.
""",
        },
    )

    patch_notebook(
        "13_linear_regression/linear_regression_practice.ipynb",
        {
            0: """
# Занятие 26. Практика: линейная регрессия и цена квартиры

Это второе занятие по теме линейной регрессии: практическая работа в формате учебного leaderboard. Команды по 1–2 человека строят модель, сохраняют `submission.csv`, а преподаватель считает результат на скрытой test-выборке.

В блокноте уже есть рабочий starter baseline. Сначала запустите его и убедитесь, что получается `submission.csv`, затем улучшайте признаки, модель и validation-качество.

**Задача:** предсказать `price_mln` — цену квартиры в миллионах рублей.

**Главная метрика занятия:** MAE, то есть средняя абсолютная ошибка в млн рублей. Меньше — лучше.

Файл с ответами для test специально не дан в этом блокноте. Для честной проверки не открывайте `competition/data/private_target.csv`, если он лежит рядом в учительской версии материалов.
""",
            1: """
## Правила и ориентир на 90 минут

1. 0–10 мин — понять данные, метрику и baseline.
2. 10–30 мин — запустить starter baseline и получить первый `submission.csv`.
3. 30–70 мин — улучшать признаки и модель на validation.
4. 70–85 мин — выбрать финальную версию и сохранить submission.
5. 85–90 мин — обсудить, какие признаки помогли и почему.

Можно использовать только `train.csv` для обучения и выбора идей. `test.csv` нужен только для финального прогноза.
""",
            4: """
## Что есть в данных

Одна строка — одна квартира из объявления.

Цель `price_mln` есть только в `train.csv`. В `test.csv` её нет: её нужно предсказать.

Подумайте перед моделированием:

- какие признаки числовые по смыслу;
- какие признаки категориальные;
- какие столбцы записаны числами, но по смыслу являются флагами, кодами или порядковыми категориями, а не обычными величинами для расстояния;
- что делать с пропусками в `metro_min`.
""",
            13: """
## Идеи для улучшения

Попробуйте 2–4 идеи и сравнивайте их только по validation:

- добавить признаки `area_per_room`, `kitchen_share`, `is_first_floor`, `is_last_floor`;
- заменить `views_30d` на `log_views = log1p(views_30d)`;
- сравнить `LinearRegression`, `Ridge`, `Lasso`;
- проверить удаление отдельного признака: убрать его, переобучить модель и сравнить MAE на той же validation-выборке;
- проверить, не стала ли модель лучше на MAE, но хуже на RMSE.

Важно: если вы создаёте признак по train, такой же признак нужно создать и в test.
""",
        },
    )

    patch_notebook(
        "14_logistic_regression/logistic_regression_practice.ipynb",
        {
            0: """
# Занятие 28. Практика: логистическая регрессия и завершение курса

Это второе занятие по теме логистической регрессии: практическая работа в формате учебного leaderboard. Команды по 1–2 человека строят модель, сохраняют `submission.csv`, а преподаватель считает результат на скрытой test-выборке.

В блокноте уже есть рабочий starter baseline. Сначала запустите его и убедитесь, что получается `submission.csv`, затем улучшайте признаки, регуляризацию и порог.

**Задача:** предсказать, завершит ли ученик курс. Целевой столбец в train — `will_finish`.

**Главная метрика занятия:** F1-score. Больше — лучше.

F1 полезна здесь потому, что нам важны и найденные будущие “завершившие курс”, и количество ложных срабатываний.
""",
            1: """
## Правила и ориентир на 90 минут

1. 0–10 мин — понять данные, целевой класс и метрику.
2. 10–30 мин — запустить starter baseline логистической регрессии.
3. 30–60 мин — улучшить признаки и регуляризацию.
4. 60–80 мин — подобрать порог вероятности для F1.
5. 80–90 мин — сохранить submission и обсудить ошибки.

Используйте `test.csv` только для финального прогноза. Ответы к test не должны участвовать в выборе модели.
""",
            12: """
## Идеи для улучшения

Попробуйте несколько вариантов и сравнивайте их на validation:

- `LogisticRegression(C=0.3)`, `C=1`, `C=3`;
- `class_weight="balanced"`;
- новые признаки:
  - `activity_total` — общий учебный ритм: часы практики + часть посещений клуба;
  - `score_per_missed_class` — последняя оценка с поправкой на число пропусков;
- проверить удаление слабого или дублирующего признака: убрать его, переобучить модель и сравнить F1 на той же validation-выборке;
- другой порог вероятности.

Не подбирайте порог по test: там нет честных ответов.
""",
        },
    )

    for path in [
        "15_overfitting_validation/overfitting_validation_practice.ipynb",
    ]:
        if (ROOT / path).exists():
            patch_notebook(
                path,
                {
                    9: """
---
## Задание 3. Learning curve (~15 мин)

По мотивам п. 9. Для `best_degree` постройте **learning curve** (`learning_curve`, 5-fold CV, `neg_mean_squared_error`).

Важно: в sklearn метрика `neg_mean_squared_error` возвращается со знаком минус, потому что sklearn считает «чем больше score, тем лучше». Чтобы получить обычный MSE, умножьте score на `-1`.

@@CRITERION@@ график train vs CV MSE от размера train.
""",
                    11: """
---
## Задание 4. K-fold CV (~12 мин)

По мотивам п. 10. `cross_validate` с 5-fold для модели с `best_degree`. Выведите среднюю validation MSE и std.

В выводе `cross_validate` колонка `test_score` означает score на fold-validation, а не финальный test из train/validation/test.

@@CRITERION@@ напечатаны `test_score`, переведённый в положительный MSE, среднее и стандартное отклонение.
""",
                    15: """
---
## Задание 6. Утечка в preprocessing (~10 мин)

По мотивам п. 12. Объясните в markdown, почему нельзя делать `StandardScaler.fit_transform` или `SimpleImputer.fit_transform` на **всей** таблице до split.

Подсказка: такие преобразования запоминают статистики данных — среднее, стандартное отклонение или медиану. Если считать их по всей таблице, validation/test частично влияют на preprocessing.

@@CRITERION@@ объяснено, что статистики preprocessing нужно `fit` только на train, а validation/test обрабатывать через `transform`.
""",
                    17: """
---
## Задание 7. Выбранная модель (~12 мин)

Обучите выбранную модель: `best_degree`, `fit` poly на **train**, оцените MSE на **validation** ещё раз.

Это ещё не финальный test, а проверка выбранной модели на validation.

@@CRITERION@@ validation MSE близка к минимуму из задания 1.
""",
                },
            )

    for path in [
        "16_decision_tree/decision_tree_practice.ipynb",
    ]:
        if (ROOT / path).exists():
            patch_notebook(
                path,
                {
                    3: """
---
## Задание 0. Split и импорты (~8 мин)

Подключите `train_test_split`, `DecisionTreeClassifier`, `accuracy_score`, `plot_tree`, `confusion_matrix`.

Задайте `RANDOM_STATE=42` и сделайте разбиение на train/validation: `test_size=0.35`, `stratify=y`. Здесь параметр называется `test_size`, потому что так устроена функция sklearn, но получившуюся часть мы используем как **validation**, а не как финальный test.

@@CRITERION@@ получены `X_train`, `X_val`, `y_train`, `y_val`.
""",
                    5: """
---
## Задание 1. Глубина и переобучение (~15 мин)

По п. 11. Обучите деревья с `max_depth` in `[1, 3, 6, None]`. Соберите таблицу train/validation accuracy.

@@CRITERION@@ для очень глубокого дерева (`None`) train accuracy обычно близка к 1.0, а validation accuracy заметно ниже train accuracy. Если на вашем split разрыв небольшой, всё равно сравните числа и сделайте вывод.
""",
                    9: """
---
## Задание 3. Визуализация дерева (~12 мин)

По п. 1. `plot_tree` для модели с `best_depth` (figsize ≥ 10). Для признаков используйте понятные имена, например `feature_names=['x1', 'x2']`.

@@CRITERION@@ график отображается, на нём видны условия разбиения.
""",
                    13: """
---
## Задание 5. ccp_alpha (~12 мин)

По п. 11: `ccp_alpha` управляет post-pruning — обрезкой дерева после обучения.

Для полного дерева (`max_depth=None`) выведите число листьев при `ccp_alpha` in `[0, 0.005, 0.02, 0.08]`.

@@CRITERION@@ при росте `ccp_alpha` листьев обычно становится меньше.
""",
                    17: """
---
## Задание 7. Feature importance (~10 мин)

По п. 13. Выведите `feature_importances_` для признаков `x1` и `x2`.

Кратко в markdown объясните: если importance у признака большая, это значит, что дерево часто и полезно использовало признак для разбиений. Это не доказывает причинность.

@@CRITERION@@ показаны два числа importance и есть короткое объяснение.
""",
                },
            )

    for path in [
        "17_bagging_random_forest/bagging_random_forest_practice.ipynb",
    ]:
        if (ROOT / path).exists():
            patch_notebook(
                path,
                {
                    5: """
---
## Задание 1. Три модели (~15 мин)

Обучите три модели:

1. одно решающее дерево;
2. `BaggingClassifier` с 150 базовыми моделями;
3. `RandomForestClassifier` с 150 деревьями.

Для каждой модели посчитайте train и validation accuracy.

@@CRITERION@@ есть таблица сравнения. Bagging и Random Forest часто оказываются лучше одного дерева на validation, но это не гарантировано на каждом split — если улучшения нет, объясните по числам, что получилось.
""",
                    11: """
---
## Задание 4. max_depth (~12 мин)

Проверьте `max_depth` in `[None, 5, 10, 20]` при `n_estimators=150`. Выберите лучший вариант по validation accuracy.

Если несколько вариантов дают одинаковое качество, выберите более простую модель: меньшую глубину.

@@CRITERION@@ `best_depth` выбран и есть таблица сравнения.
""",
                    13: """
---
## Задание 5. Bootstrap вручную (~12 мин)

Сделайте 100 повторов bootstrap длины `len(X_train)`.

Bootstrap означает: случайно выбираем индексы объектов **с возвращением**, поэтому один объект может попасть несколько раз, а другой — ни разу.

В каждом повторе посчитайте долю уникальных индексов, затем найдите среднюю долю уникальных.

@@CRITERION@@ средняя доля уникальных примерно около 0.63 при большом n.
""",
                    15: """
---
## Задание 6. Permutation importance (~15 мин)

Посчитайте permutation importance на validation для лучшего `RandomForestClassifier` из предыдущих заданий.

Выведите top-5 признаков: таблицей или горизонтальным bar-chart (`barh`).

@@CRITERION@@ показаны top-5 признаков и понятно, по какой модели они посчитаны.
""",
                    17: """
---
## Задание 7. Train vs val gap (~8 мин)

В markdown объясните: почему высокий train accuracy у случайного леса сам по себе ещё не доказывает переобучение?

Подсказка: смотреть нужно не только на train, но и на validation/OOB. Если validation или OOB тоже хорошие, высокий train accuracy менее тревожен.

@@CRITERION@@ 2–3 предложения с опорой на train, validation и/или OOB.
""",
                },
            )

    patch_notebook(
        "18_gradient_boosting/gradient_boosting_practice.ipynb",
        {
            0: """
# Занятие 36. Практика: бустинг в реальных библиотеках

**Тема:** предсказание прогрессирования диабета по медицинским признакам (реальный датасет `load_diabetes`).  
**Модельный фокус:** градиентный бустинг для регрессии.  
**Библиотеки:** `sklearn`, `xgboost`, `lightgbm`, `catboost` доступны в учебной среде и сравниваются обязательно.
""",
            2: """
## Что нужно сделать

1. Подготовить данные и baseline.
2. Разделить данные на train / validation / test.
3. Обучить и сравнить `GradientBoostingRegressor` и `HistGradientBoostingRegressor` из `sklearn`.
4. Обязательно сравнить внешние библиотеки (`xgboost`, `lightgbm`, `catboost`).
5. Выбрать модель по validation, а test использовать один раз — для финальной проверки выбранного решения.
6. Сформулировать практические выводы: когда какую библиотеку выбирать, какие у неё плюсы и минусы.

### Оценивание (30 баллов)

| № | Тема | Баллы |
|---|------|------:|
| 0 | Загрузка и split | 3 |
| 1 | Baseline и метрики | 3 |
| 2 | Sklearn GradientBoostingRegressor | 4 |
| 3 | Подбор гиперпараметров для GBDT | 5 |
| 4 | HistGradientBoostingRegressor и early stopping | 4 |
| 5 | Внешние библиотеки бустинга | 5 |
| 6 | Единый рейтинг моделей | 2 |
| 7 | Преимущества и недостатки библиотек | 2 |
| 8 | Финальный инженерный вывод | 2 |
| | **Итого** | **30** |
""",
            3: """
---
## Среда и библиотеки

В учебной среде `xgboost`, `lightgbm` и `catboost` уже установлены. Если вы запускаете блокнот дома и импорт ниже падает, установите их отдельно перед занятием.
""",
            4: """
# В учебной среде установка не нужна.
# Если запускаете дома и библиотек нет, выполните в терминале:
# pip install xgboost lightgbm catboost
""",
            5: """
import importlib
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25  # 25% от train_val ≈ 20% от всех данных


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_regression(model_name, y_true, y_pred):
    return {
        "model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
""",
            6: """
---
## Задание 0. Загрузка и split — **3 балла**

**Сделайте:**
- загрузите датасет `load_diabetes(as_frame=True)`;
- соберите `X` и `y`;
- сначала отделите финальный `test`;
- из оставшейся части выделите `validation`;
- выведите размеры train/validation/test и первые 5 строк `X_train`.

### Подробные критерии (для проверки LLM)
- **1 балл:** датасет загружен через `load_diabetes(as_frame=True)`, переменные `X` и `y` созданы корректно.
- **1 балл:** получены `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`; test не используется для выбора модели.
- **1 балл:** выведены размеры всех выборок и показаны первые 5 строк `X_train`.

### Снижение баллов
- Использован другой датасет или `load_diabetes` без `as_frame=True` → минус **1.0**.
- Нет отдельной validation или test → минус **1.0**.
- Не выведены размеры выборок или отсутствует `X_train.head()` → минус **0.5** за каждый пропуск.
""",
            7: """
diabetes = load_diabetes(as_frame=True)
X = diabetes.data.copy()
y = diabetes.target.copy()

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=VAL_SIZE, random_state=RANDOM_STATE
)

print("X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_val:", y_val.shape, "y_test:", y_test.shape)
X_train.head()
""",
            8: """
---
## Задание 1. Baseline и метрики — **3 балла**

**Сделайте:**
- обучите `DummyRegressor(strategy='mean')` на train;
- посчитайте MAE, RMSE, R2 на validation;
- сохраните результат в `baseline_metrics`.

### Подробные критерии (для проверки LLM)
- **1 балл:** baseline-модель обучена корректно.
- **1 балл:** метрики MAE/RMSE/R2 посчитаны на validation.
- **1 балл:** результат сохранён в `baseline_metrics` (словарь/таблица).

### Снижение баллов
- `DummyRegressor(strategy='mean')` не обучен или заменён другой baseline-моделью → минус **1.0**.
- Метрики посчитаны не на validation (`X_val`, `y_val`) → минус **1.0**.
- Результат не сохранён в `baseline_metrics` или отсутствует одна из метрик (MAE/RMSE/R2) → минус **0.5** за каждый пропуск.
""",
            9: """
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)

y_pred_dummy = dummy.predict(X_val)
baseline_metrics = evaluate_regression("DummyRegressor(mean)", y_val, y_pred_dummy)

baseline_metrics
""",
            10: """
---
## Задание 2. Sklearn GradientBoostingRegressor — **4 балла**

**Сделайте:**
- обучите базовую модель `GradientBoostingRegressor(random_state=RANDOM_STATE)`;
- посчитайте метрики на validation;
- сравните с baseline (лучше/хуже и насколько).

### Подробные критерии (для проверки LLM)
- **1 балл:** модель `GradientBoostingRegressor` обучена.
- **1 балл:** корректно посчитаны MAE/RMSE/R2 на validation.
- **1 балл:** создано сравнение с baseline в единой таблице.
- **1 балл:** есть вывод, где baseline лучше/хуже по ключевым метрикам.

### Снижение баллов
- Не обучен `GradientBoostingRegressor(random_state=RANDOM_STATE)` → минус **1.0**.
- Сравнение с baseline отсутствует или выполнено не по MAE/RMSE → минус **1.0**.
- Нет явного вывода «лучше/хуже baseline» по числам → минус **0.5**.
""",
            11: """
gbr = GradientBoostingRegressor(random_state=RANDOM_STATE)
gbr.fit(X_train, y_train)

y_pred_gbr = gbr.predict(X_val)
gbr_metrics = evaluate_regression("GradientBoostingRegressor", y_val, y_pred_gbr)

compare_2 = pd.DataFrame([baseline_metrics, gbr_metrics]).set_index("model")
compare_2["delta_to_baseline_RMSE"] = (
    compare_2["RMSE"] - compare_2.loc["DummyRegressor(mean)", "RMSE"]
)
compare_2["delta_to_baseline_MAE"] = (
    compare_2["MAE"] - compare_2.loc["DummyRegressor(mean)", "MAE"]
)

print("Отрицательная delta => лучше baseline (меньше ошибка)")
compare_2
""",
            12: """
---
## Задание 3. Подбор гиперпараметров для GBDT — **5 баллов**

GBDT = Gradient Boosting Decision Trees, то есть градиентный бустинг над решающими деревьями.

**Сделайте:**
- настройте `GridSearchCV` для `GradientBoostingRegressor`;
- переберите минимум: `n_estimators`, `learning_rate`, `max_depth`;
- найдите лучшую модель по отрицательному MAE (`scoring='neg_mean_absolute_error'`);
- оцените лучшую модель на validation.

### Подробные критерии (для проверки LLM)
- **1 балл:** настроен `GridSearchCV` с осмысленной сеткой.
- **1 балл:** в сетке есть все 3 параметра (`n_estimators`, `learning_rate`, `max_depth`).
- **1 балл:** выбран корректный `scoring='neg_mean_absolute_error'`.
- **1 балл:** выведены `best_params_` и best CV score.
- **1 балл:** рассчитаны validation-метрики лучшей модели.

### Снижение баллов
- Отсутствует `GridSearchCV` или используется ручной подбор без CV → минус **1.0**.
- В `param_grid` нет одного из обязательных параметров (`n_estimators`, `learning_rate`, `max_depth`) → минус **0.5** за каждый.
- Нет `best_params_`/best CV score или нет validation-оценки лучшей модели → минус **0.5** за каждый пропуск.
""",
            13: """
param_grid = {
    "n_estimators": [100, 200, 400],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4],
}

base_gbr = GradientBoostingRegressor(random_state=RANDOM_STATE)
grid = GridSearchCV(
    estimator=base_gbr,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1,
)

grid.fit(X_train, y_train)
best_gbr = grid.best_estimator_
y_pred_best_gbr = best_gbr.predict(X_val)
best_gbr_metrics = evaluate_regression(
    "GradientBoostingRegressor (tuned)", y_val, y_pred_best_gbr
)

print("Best params:", grid.best_params_)
print("Best CV MAE:", -grid.best_score_)
best_gbr_metrics
""",
            14: """
---
## Задание 4. HistGradientBoostingRegressor и early stopping — **4 балла**

**Сделайте:**
- обучите `HistGradientBoostingRegressor`;
- включите early stopping (`early_stopping=True`);
- сравните метрики и время обучения с лучшей моделью из задачи 3.

### Подробные критерии (для проверки LLM)
- **1 балл:** обучен `HistGradientBoostingRegressor` с early stopping.
- **1 балл:** посчитаны метрики на validation.
- **1 балл:** замерено и показано время обучения.
- **1 балл:** есть прямое сравнение с лучшим GBDT из задачи 3.

### Снижение баллов
- Не включен `early_stopping=True` в `HistGradientBoostingRegressor` → минус **1.0**.
- Нет замера времени обучения хотя бы для одной модели сравнения → минус **0.5**.
- Нет прямого сравнения с лучшей моделью из задачи 3 по метрикам → минус **1.0**.
""",
            15: """
start = time.perf_counter()
hgbr = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=6,
    max_iter=500,
    early_stopping=True,
    random_state=RANDOM_STATE,
)
hgbr.fit(X_train, y_train)
hgbr_train_time = time.perf_counter() - start

y_pred_hgbr = hgbr.predict(X_val)
hgbr_metrics = evaluate_regression("HistGradientBoostingRegressor", y_val, y_pred_hgbr)
hgbr_metrics["fit_time_sec"] = hgbr_train_time

start = time.perf_counter()
_ = best_gbr.fit(X_train, y_train)
best_gbr_train_time = time.perf_counter() - start

compare_hist = pd.DataFrame(
    [
        {**best_gbr_metrics, "fit_time_sec": best_gbr_train_time},
        hgbr_metrics,
    ]
).set_index("model")

compare_hist
""",
            16: """
---
## Задание 5. Внешние библиотеки бустинга — **5 баллов**

**Сделайте:**
1. Импортируйте `XGBRegressor`, `LGBMRegressor`, `CatBoostRegressor`.
2. Обучите по одной базовой модели каждой библиотеки: `xgboost`, `lightgbm`, `catboost`.
3. Посчитайте метрики на validation и соберите общую таблицу `external_results`.

Подсказка по моделям:
- `xgboost.XGBRegressor(...)`
- `lightgbm.LGBMRegressor(...)`
- `catboost.CatBoostRegressor(verbose=0, ...)`

### Подробные критерии (для проверки LLM)
- **1 балл:** корректно импортированы все 3 внешние модели.
- **2 балла:** обучены все 3 модели (`XGBoost`, `LightGBM`, `CatBoost`) без ошибок.
- **1 балл:** рассчитаны метрики на validation для всех 3 моделей.
- **1 балл:** сформирована итоговая таблица `external_results` с тремя строками внешних моделей.

### Снижение баллов
- Не импортирована одна из обязательных библиотек (`xgboost`, `lightgbm`, `catboost`) → минус **0.5** за каждую.
- Для одной из моделей отсутствует `fit` или `predict` на validation → минус **0.5** за каждую.
- В `external_results` отсутствует строка модели или одна из метрик (MAE/RMSE/R2) → минус **0.5** за каждый пропуск.
""",
            17: """
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

external_rows = []

xgb = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE,
    objective="reg:squarederror",
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_val)
external_rows.append(evaluate_regression("XGBoost", y_val, xgb_pred))

lgbm = LGBMRegressor(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    random_state=RANDOM_STATE,
)
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_val)
external_rows.append(evaluate_regression("LightGBM", y_val, lgbm_pred))

cbr = CatBoostRegressor(
    iterations=400,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    verbose=0,
    random_seed=RANDOM_STATE,
)
cbr.fit(X_train, y_train)
cbr_pred = cbr.predict(X_val)
external_rows.append(evaluate_regression("CatBoost", y_val, cbr_pred))

external_results = pd.DataFrame(external_rows)
external_results
""",
            18: """
---
## Задание 6. Единый рейтинг моделей — **2 балла**

**Сделайте:**
- объедините результаты baseline, sklearn-моделей и внешних библиотек в один DataFrame `all_results`;
- отсортируйте по validation RMSE (по возрастанию);
- постройте bar-chart по RMSE для всех моделей.

### Подробные критерии (для проверки LLM)
- **1 балл:** корректно собран единый рейтинг `all_results`.
- **1 балл:** есть сортировка по RMSE и читаемый график сравнения.

### Снижение баллов
- В `all_results` не включены baseline и sklearn-модели из предыдущих задач → минус **1.0**.
- Нет сортировки по RMSE по возрастанию → минус **0.5**.
- График отсутствует или построен не по RMSE → минус **0.5**.
""",
            20: """
---
## Задание 7. Преимущества и недостатки библиотек — **2 балла**

Заполните таблицу (markdown **или** `DataFrame`) минимум для 4 библиотек:
- `sklearn GradientBoosting`
- `sklearn HistGradientBoosting`
- `XGBoost`
- `LightGBM`
- `CatBoost`

Для каждой укажите:
1. что в библиотеке удобно;
2. что может мешать;
3. когда её разумно брать для рабочего сервиса, а когда лучше не брать.

### Подробные критерии (для проверки LLM)
- **1 балл:** минимум 4 библиотеки с корректными плюсами/минусами.
- **1 балл:** есть условия выбора для рабочего сервиса, формат ответа — markdown-таблица или `DataFrame`.

### Снижение баллов
- Описаны менее 4 библиотек → минус **1.0**.
- Для библиотеки отсутствует один из блоков: преимущества / ограничения / когда использовать → минус **0.5** за каждый пропуск.
- Формулировки слишком общие и не связаны с качеством, скоростью, устойчивостью или поддержкой → минус **0.5**.
""",
            22: """
---
## Задание 8. Финальный инженерный вывод — **2 балла**

Напишите 8–12 предложений:
- какая модель лучшая по validation;
- насколько она лучше baseline по validation RMSE/MAE;
- какую модель вы бы выбрали для финальной проверки на test;
- какую библиотеку взяли бы при ограничении по времени обучения;
- какую — при максимальном фокусе на качестве;
- какие риски нужно проверить перед использованием модели в рабочем сервисе.

После выбора модели по validation выполните **одну финальную оценку на test**.

### Подробные критерии (для проверки LLM)
- **1 балл:** вывод опирается на фактические метрики из ноутбука.
- **1 балл:** явно рассмотрены компромиссы: качество, скорость, сложность поддержки и риски рабочего сервиса.

### Снижение баллов
- В выводе нет фактических метрик (RMSE/MAE/R2) из `all_results` → минус **1.0**.
- Нет сравнения с baseline по числам улучшения → минус **0.5**.
- Нет итогового выбора на основе данных → минус **0.5**.
""",
            23: """
best_row = all_results.loc[all_results['RMSE'].idxmin()]
baseline_row = all_results[all_results['model'] == 'DummyRegressor(mean)'].iloc[0]
worst_row = all_results.loc[all_results['RMSE'].idxmax()]

rmse_gain = baseline_row['RMSE'] - best_row['RMSE']
mae_gain = baseline_row['MAE'] - best_row['MAE']

model_candidates = {
    "DummyRegressor(mean)": DummyRegressor(strategy="mean"),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "GradientBoostingRegressor (tuned)": GradientBoostingRegressor(
        random_state=RANDOM_STATE,
        **grid.best_params_,
    ),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
    ),
    "CatBoost": CatBoostRegressor(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=0,
        random_seed=RANDOM_STATE,
    ),
}

best_name = best_row['model']
final_model = model_candidates[best_name]
final_model.fit(X_train_val, y_train_val)
test_pred = final_model.predict(X_test)
final_test_metrics = evaluate_regression(f"{best_name} (final test)", y_test, test_pred)

summary = pd.DataFrame(
    {
        'metric': [
            'Best validation model',
            'Validation RMSE',
            'Validation MAE',
            'Validation R2',
            'Baseline validation RMSE',
            'Baseline validation MAE',
            'Validation RMSE gain vs baseline',
            'Validation MAE gain vs baseline',
            'Worst validation model',
            'Final test RMSE',
            'Final test MAE',
            'Final test R2',
        ],
        'value': [
            best_row['model'],
            round(best_row['RMSE'], 2),
            round(best_row['MAE'], 2),
            round(best_row['R2'], 3),
            round(baseline_row['RMSE'], 2),
            round(baseline_row['MAE'], 2),
            round(rmse_gain, 2),
            round(mae_gain, 2),
            worst_row['model'],
            round(final_test_metrics['RMSE'], 2),
            round(final_test_metrics['MAE'], 2),
            round(final_test_metrics['R2'], 3),
        ],
    }
)

print('Выводы по validation:')
print(f"- Лучшая модель: {best_row['model']} (RMSE={best_row['RMSE']:.2f}, MAE={best_row['MAE']:.2f}, R2={best_row['R2']:.3f})")
print(f"- Baseline: RMSE={baseline_row['RMSE']:.2f}, MAE={baseline_row['MAE']:.2f}, R2={baseline_row['R2']:.3f}")
print(f"- Улучшение относительно baseline: RMSE на {rmse_gain:.2f}, MAE на {mae_gain:.2f}")
print(f"- Худшая модель в сравнении: {worst_row['model']} (RMSE={worst_row['RMSE']:.2f})")
print('\\nФинальная проверка на test выбранной модели:')
print(final_test_metrics)

summary
""",
        },
    )


if __name__ == "__main__":
    main()
