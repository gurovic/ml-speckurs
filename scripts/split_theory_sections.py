#!/usr/bin/env python3
"""Split mixed theory sections and renumber; update п./пп. references in lesson folders."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# old_section -> new section(s); int means single new number, tuple means range
REF_MAPS: dict[str, dict[int, int | tuple[int, int]]] = {
    "13_workflow": {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 12,
        13: (13, 14),
        14: 15,
        15: 16,
        16: (17, 18),
        17: 19,
        18: 20,
    },
    "14_feature_engineering": {
        **{i: i for i in range(1, 13)},
        13: (13, 14),
        14: 15,
        15: 16,
        16: 17,
        17: 18,
    },
    "15_linear_regression": {
        **{i: i for i in range(1, 7)},
        7: (7, 8),
        8: 9,
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 14,
        14: 15,
        15: 16,
    },
    "16_logistic_regression": {
        **{i: i for i in range(1, 11)},
        11: (11, 12),
        12: 13,
        13: (14, 15),
        14: 16,
        15: 17,
        16: (18, 19),
        17: 20,
        18: 21,
        19: 22,
        20: (23, 24),
        21: 25,
    },
    "18_decision_tree": {
        **{i: i for i in range(1, 12)},
        12: 12,
        13: (13, 14),
        14: 15,
        15: 16,
        16: 17,
        17: 18,
    },
    "19_bagging_random_forest": {
        **{i: i for i in range(1, 11)},
        11: (11, 12),
        12: 13,
        13: 14,
        14: 15,
        15: 16,
        16: 17,
    },
    "20_gradient_boosting": {
        **{i: i for i in range(1, 8)},
        8: (8, 9),
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: (14, 15),
        14: (16, 17),
        15: (18, 19),
        16: 20,
    },
}


def format_ref(value: int | tuple[int, int]) -> str:
    if isinstance(value, tuple):
        a, b = value
        if a == b:
            return str(a)
        return f"{a}–{b}"
    return str(value)


def build_replace_patterns(ref_map: dict[int, int | tuple[int, int]]):
    """Patterns sorted by number desc to avoid partial replacements."""
    singles = []
    for old in sorted(ref_map, reverse=True):
        new = ref_map[old]
        if isinstance(new, tuple):
            singles.append((old, format_ref(new)))
        else:
            singles.append((old, str(new)))
    return singles


def update_refs_in_text(text: str, ref_map: dict[int, int | tuple[int, int]]) -> str:
    singles = build_replace_patterns(ref_map)

    def repl_range(m: re.Match) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        na = ref_map.get(a, a)
        nb = ref_map.get(b, b)
        start = na[0] if isinstance(na, tuple) else na
        end = nb[1] if isinstance(nb, tuple) else nb
        if start == end:
            return f"п. {start}"
        return f"пп. {start}–{end}"

    text = re.sub(r"пп\.\s*(\d+)\s*[–-]\s*(\d+)", repl_range, text)

    def repl_pp_list(m: re.Match) -> str:
        nums = [int(x) for x in re.findall(r"\d+", m.group(0))]
        mapped = []
        for n in nums:
            v = ref_map.get(n, n)
            if isinstance(v, tuple):
                mapped.extend(v)
            else:
                mapped.append(v)
        mapped = sorted(set(mapped))
        if len(mapped) == 1:
            return f"п. {mapped[0]}"
        if mapped == list(range(mapped[0], mapped[-1] + 1)):
            return f"пп. {mapped[0]}–{mapped[-1]}"
        return "пп. " + ", ".join(str(x) for x in mapped)

    text = re.sub(r"пп\.\s*[\d,\s]+(?= теории| теор| из |$|[.)])", repl_pp_list, text)

    for old, new_str in singles:
        text = re.sub(rf"\bп\.\s*{old}\b", f"п. {new_str}", text)
    return text


def split_markdown_cell(source: str, split_after_heading: str, new_heading: str) -> list[str]:
    """Split one markdown cell into two at paragraph boundary after first heading block."""
    if not source.startswith(split_after_heading.split("\n")[0][:30]):
        raise ValueError(f"Cell does not start with expected heading: {split_after_heading[:40]!r}")
    parts = source.split("\n\n", 1)
    if len(parts) < 2:
        raise ValueError("Cannot split: no blank line paragraph boundary")
    first, rest = parts[0], parts[1]
    # find second logical block in rest for split point
    return [first + "\n\n" + rest.split("\n\n", 1)[0] + "\n", new_heading + "\n\n" + rest.split("\n\n", 1)[1]]


def set_heading(source: str, num: int, title: str) -> str:
    return re.sub(r"^##\s*\d+\.\s*[^\n]+", f"## {num}. {title}", source, count=1)


def renumber_headings_in_notebook(cells: list, start_from: int = 1) -> list:
    n = start_from - 1
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        if re.match(r"^##\s*\d+\.", src):
            n += 1
            cell["source"] = [
                re.sub(r"^##\s*\d+\.\s*", f"## {n}. ", src, count=1)
            ]
    return cells


def load_nb(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_nb(path: Path, nb: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_cell_text(cell: dict, text: str) -> None:
    cell["source"] = [text] if text.endswith("\n") else [text + "\n"] if "\n" in text else [text]


def insert_markdown_after(cells: list, idx: int, text: str) -> list:
    new_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text if text.endswith("\n") else text + "\n"],
    }
    return cells[: idx + 1] + [new_cell] + cells[idx + 1 :]


def apply_workflow_splits(nb: dict) -> dict:
    cells = nb["cells"]
    for i, c in enumerate(cells):
        if c["cell_type"] != "markdown":
            continue
        t = cell_text(c)
        if t.startswith("## 13. Предобработка и обучение kNN"):
            set_cell_text(
                c,
                "## 13. StandardScaler: fit на train, transform везде\n\n"
                "**Предобработка** — привести данные к виду, удобному для модели. Для kNN важно **масштабирование**.\n\n"
                "**`StandardScaler`** для каждого признака на train считает среднее и стандартное отклонение, "
                "затем преобразует значения: «отнять среднее, поделить на σ». После этого столбцы сравнимы по масштабу.\n\n"
                "Два метода scikit-learn:\n\n"
                "- **`fit`** — «выучить» параметры преобразования **только на train**;\n"
                "- **`transform`** — применить **те же** параметры к train, validation и test.\n\n"
                "На validation и test **`fit` вызывать нельзя** — иначе модель увидит статистику test при настройке (утечка).\n",
            )
            cells = insert_markdown_after(
                cells,
                i,
                "## 14. Обучение kNN на масштабированных данных\n\n"
                "**Порядок одного эксперимента:**\n\n"
                "1. инжиниринг признаков;\n"
                "2. `scaler.fit(X_train)` → `scaler.transform` для train, val, test;\n"
                "3. `knn.fit(X_train_scaled, y_train)`;\n"
                "4. метрика на validation;\n"
                "5. **один раз** test (п. 19).\n",
            )
            break

    for i, c in enumerate(cells):
        if c["cell_type"] != "markdown":
            continue
        t = cell_text(c)
        if t.startswith("## 16. Анализ ошибок"):
            set_cell_text(
                c,
                "## 17. Confusion matrix на validation\n\n"
                "Средняя метрика (accuracy, recall) **скрывает детали**. На validation строят **матрицу ошибок** "
                "(`confusion_matrix`) — та же таблица TP/FN/FP/TN из п. 8, но с **числами** вместо букв:\n\n"
                "- **строки** — истинный класс (как было на самом деле);\n"
                "- **столбцы** — что **предсказала** модель.\n\n"
                "Код ниже выводит матрицу с подписями. Для recall по классу 0 смотрим **первую строку** (ячейки TP и FN).\n",
            )
            cells = insert_markdown_after(
                cells,
                i,
                "## 18. Разбор ошибок по группам\n\n"
                "Полезно разбить validation на **группы** (крупные / мелкие опухоли) и посмотреть "
                "**долю ошибок в каждой группе отдельно**. Эти доли **не обязаны** в сумме давать 1 — "
                "это доли **внутри** своей группы.\n",
            )
            break

    nb["cells"] = renumber_headings_in_notebook(cells)
    return nb


def apply_feature_engineering_splits(nb: dict) -> dict:
    cells = nb["cells"]
    for i, c in enumerate(cells):
        if c["cell_type"] != "markdown":
            continue
        t = cell_text(c)
        if "## 13. Агрегаты и утечка" in t:
            set_cell_text(
                c,
                "## 13. Агрегаты по прошлым событиям\n\n"
                "Для клиента полезны признаки «число прошлых покупок» и «средняя сумма прошлых заказов». "
                "Ключевое слово — **прошлых**.\n\n"
                "Если в агрегат попали события после момента прогноза, возникла утечка.\n",
            )
            cells = insert_markdown_after(
                cells,
                i,
                "## 14. Утечка через агрегаты и target encoding *\n\n"
                "**Target encoding (*)** заменяет категорию средним ответом. Наивный расчёт сообщает учебному объекту "
                "часть его собственного ответа. Безопасный вариант требует специальных out-of-fold вычислений внутри train, "
                "поэтому пока достаточно запомнить предупреждение: самостоятельно считать среднее цели по всей таблице нельзя.\n\n"
                "**Текстовые признаки — обзор.** Текст можно представить частотами слов или более сложными числовыми векторами. "
                "Словарь также обучают только на train; подробная работа с текстом относится к отдельной теме.\n",
            )
            break
    nb["cells"] = renumber_headings_in_notebook(cells)
    return nb


def apply_linear_regression_splits(nb: dict) -> dict:
    cells = nb["cells"]
    for i, c in enumerate(cells):
        if c["cell_type"] != "markdown":
            continue
        t = cell_text(c)
        if t.startswith("## 7. Проверяем модель на новых данных"):
            set_cell_text(
                c,
                "## 7. Train / validation / test для регрессии\n\n"
                "Оценивать модель только на тех же примерах, по которым она училась, нечестно: "
                "это как давать на контрольной заранее известные задачи. Данные заранее делят на:\n\n"
                "- **train** — модель подбирает коэффициенты;\n"
                "- **validation** — выбираются признаки, регуляризация и другие настройки;\n"
                "- **test** — одна итоговая проверка уже выбранного решения.\n\n"
                "Небольшой кодовый пример ниже сравнивает train и test для одной заранее зафиксированной модели. "
                "В полном проекте варианты модели выбирают только по validation или кросс-валидации.\n",
            )
            cells = insert_markdown_after(
                cells,
                i,
                "## 8. Когда линейная модель переобучается\n\n"
                "Линейная регрессия тоже может переобучиться:\n\n"
                "1. Если признаков много, а объектов мало, модель может запомнить обучающие примеры вместе с шумом.\n"
                "2. Сложные признаки ($x^2$, $x^3$, $x_1x_2$ и другие) позволяют построить очень гибкую зависимость, "
                "которая «обовьёт» случайные отклонения.\n"
                "3. Если решения о признаках и настройках принимались после просмотра test, модель косвенно подстроилась и под него.\n\n"
                "Мультиколлинеарность похожих признаков усиливает нестабильность весов — подробнее в п. 15.\n",
            )
            break
    nb["cells"] = renumber_headings_in_notebook(cells)
    return nb


def apply_logistic_regression_splits(nb: dict) -> dict:
    cells = nb["cells"]
    for i, c in enumerate(cells):
        t = cell_text(c) if c["cell_type"] == "markdown" else ""
        if t.startswith("## 11. Baseline и дисбаланс"):
            set_cell_text(
                c,
                "## 11. Baseline для классификации\n\n"
                "Перед сложной моделью нужен простой соперник — **baseline**. "
                "Сравнивать модель нужно с baseline на **одних и тех же** данных.\n",
            )
            cells = insert_markdown_after(
                cells,
                i,
                "## 12. Дисбаланс классов и выбор метрики\n\n"
                "Если 95% писем не являются спамом, модель «всегда не спам» получит accuracy 95%, "
                "хотя не найдёт ни одного спам-письма.\n\n"
                "Такой перекос называется **дисбалансом классов**. Поэтому нужно:\n\n"
                "1. смотреть доли классов;\n"
                "2. не ограничиваться accuracy;\n"
                "3. выбирать метрику по цене FP и FN.\n",
            )
            break

    for i, c in enumerate(cells):
        t = cell_text(c) if c["cell_type"] == "markdown" else ""
        if t.startswith("## 13. Финальная проверка"):
            set_cell_text(
                c,
                "## 14. Первый честный test\n\n"
                "После выбора модели, порога и метрики на validation **один раз** открывают test. "
                "Повторные правки по test делают его оценку оптимистичной (см. занятие 33, п. 6).\n",
            )
            cells = insert_markdown_after(
                cells,
                i,
                "## 15. ROC- и PR-кривые *\n\n"
                "ROC и PR показывают качество **ранжирования** объектов по вероятности **без выбора одного порога**. "
                "Порог подбирают отдельно на validation (п. 13).\n",
            )
            break

    for i, c in enumerate(cells):
        t = cell_text(c) if c["cell_type"] == "markdown" else ""
        if t.startswith("## 16. Масштабирование и регуляризация"):
            set_cell_text(
                c,
                "## 18. Зачем масштабировать перед L1/L2\n\n"
                "Признаки в разных единицах измерения по-разному попадают под штраф регуляризации. "
                "Перед Ridge/Lasso/ElasticNet числовые признаки обычно масштабируют (`StandardScaler`, `fit` только на train).\n",
            )
            cells = insert_markdown_after(
                cells,
                i,
                "## 19. Параметр C и сила регуляризации\n\n"
                "В `LogisticRegression` параметр **`C`** — обратная сила регуляризации: большой `C` — слабый штраф, "
                "малый `C` — сильный. **`l1_ratio`** задаёт долю L1 в ElasticNet. Подбирают по validation.\n",
            )
            break

    for i, c in enumerate(cells):
        t = cell_text(c) if c["cell_type"] == "markdown" else ""
        if t.startswith("## 20. Ограничения вероятностей"):
            set_cell_text(
                c,
                "## 23. Ограничения интерпретации и экстраполяции\n\n"
                "Коэффициенты и вероятности логрег описывают связь **внутри диапазона обучающих данных**. "
                "Далеко за пределами train прогнозы могут быть ненадёжными.\n",
            )
            cells = insert_markdown_after(
                cells,
                i,
                "## 24. Калибровка вероятностей *\n\n"
                "Предсказанные вероятности не гарантированно совпадают с частотами событий на новых данных. "
                "**Калибровка** (например, isotonic или Platt) выравнивает вероятности по validation. "
                "После внедрения качество и калибровку мониторят на свежих данных.\n",
            )
            break

    nb["cells"] = renumber_headings_in_notebook(cells)
    return nb


def apply_decision_tree_splits(nb: dict) -> dict:
    cells = nb["cells"]
    for i, c in enumerate(cells):
        t = cell_text(c) if c["cell_type"] == "markdown" else ""
        if t.startswith("## 12. Масштаб, категории и пропуски"):
            set_cell_text(
                c,
                "## 12. Нужно ли масштабирование дереву\n\n"
                "Дерево сравнивает порядок значений с порогом, поэтому масштабирование обычно не требуется. "
                "Строго монотонное преобразование меняет пороги, но сохраняет порядок.\n",
            )
            break

    for i, c in enumerate(cells):
        t = cell_text(c) if c["cell_type"] == "markdown" else ""
        if t.startswith("## 13. Числовые пропуски и категории"):
            set_cell_text(
                c,
                "## 13. Пропуски в DecisionTree\n\n"
                "Современные версии sklearn умеют направлять числовые `NaN` при некоторых настройках обычного дерева; "
                "поддержка зависит от версии, критерия и режима. Перед проектом проверяют документацию своей версии "
                "и явно фиксируют способ обработки пропусков.\n\n"
                "Даже если версия sklearn принимает NaN, полезно понимать, как пропуски будут направляться "
                "и сохранится ли это поведение после внедрения.\n",
            )
            cells = insert_markdown_after(
                cells,
                i,
                "## 14. Кодирование категорий для дерева\n\n"
                "`DecisionTreeClassifier` sklearn не поддерживает категории напрямую: их нужно закодировать "
                "без создания ложного порядка. Поддержка NaN не превращает категорию в число — "
                "категориальные признаки всё равно требуют осмысленного кодирования или алгоритма с нативной поддержкой категорий.\n",
            )
            break

    for i, c in enumerate(cells):
        t = cell_text(c) if c["cell_type"] == "markdown" else ""
        if t.startswith("## 14. Важность признаков"):
            set_cell_text(
                c,
                "## 15. Impurity importance в дереве\n\n"
                "`feature_importances_` суммирует уменьшения impurity, сделанные признаком. "
                "Это не доказательство причинности и не гарантирует, что признак полезен на новых данных.\n\n"
                "Важность может завышаться у признаков с множеством возможных разбиений и делиться между коррелирующими признаками. "
                "Надёжнее дополнять её permutation importance (п. 16).\n",
            )
            break

    nb["cells"] = renumber_headings_in_notebook(cells)
    return nb


def apply_bagging_splits(nb: dict) -> dict:
    cells = nb["cells"]
    for i, c in enumerate(cells):
        t = cell_text(c) if c["cell_type"] == "markdown" else ""
        if t.startswith("## 11. Дисбаланс классов"):
            set_cell_text(
                c,
                "## 11. Метрики при дисбалансе\n\n"
                "При редком классе accuracy вводит в заблуждение. Сравнивают precision, recall, F1 и PR-кривую на validation.\n",
            )
            cells = insert_markdown_after(
                cells,
                i,
                "## 12. class_weight в Random Forest\n\n"
                "`class_weight` меняет веса классов при обучении деревьев. "
                "Это другой инструмент, чем смена порога вероятности (п. 13).\n",
            )
            break

    for i, c in enumerate(cells):
        t = cell_text(c) if c["cell_type"] == "markdown" else ""
        if t.startswith("## 13. Важность признаков"):
            set_cell_text(
                c,
                "## 14. Impurity importance в лесу\n\n"
                "Лес усредняет impurity importance по деревьям, но сохраняет её ограничения: "
                "коррелирующие признаки могут «делить» важность.\n",
            )
            break

    nb["cells"] = renumber_headings_in_notebook(cells)
    return nb


def apply_boosting_splits(nb: dict) -> dict:
    cells = nb["cells"]
    for i, c in enumerate(cells):
        t = cell_text(c) if c["cell_type"] == "markdown" else ""
        if t.startswith("## 8. Post-hoc"):
            set_cell_text(
                c,
                "## 8. Early stopping и post-hoc выбор числа деревьев\n\n"
                "Число деревьев можно подобрать post-hoc по validation-кривой или остановить обучение early stopping, "
                "если validation-метрика перестала улучшаться.\n",
            )
            cells = insert_markdown_after(
                cells,
                i,
                "## 9. Когда внутренняя validation неверна\n\n"
                "Early stopping и post-hoc выбор требуют честной validation: при временных рядах и группах объектов "
                "случайный split может завышать качество. Используют разбиение по времени или группам (занятие 33, п. 12).\n",
            )
            break

    for i, c in enumerate(cells):
        t = cell_text(c) if c["cell_type"] == "markdown" else ""
        if t.startswith("## 13. Предобработка и реализации"):
            set_cell_text(
                c,
                "## 14. Нужна ли предобработка для бустинга\n\n"
                "Деревьям бустинга обычно не нужно масштабирование. "
                "sklearn GradientBoosting ожидает числовые признаки и обработанные пропуски; "
                "HistGradientBoosting поддерживает пропуски.\n",
            )
            cells = insert_markdown_after(
                cells,
                i,
                "## 15. Обзор реализаций Gradient Boosting *\n\n"
                "XGBoost и LightGBM предлагают эффективные реализации; CatBoost особенно удобен для категориальных признаков. "
                "Названия параметров и возможности различаются; настройки не переносят между библиотеками без проверки документации.\n",
            )
            break

    for i, c in enumerate(cells):
        t = cell_text(c) if c["cell_type"] == "markdown" else ""
        if t.startswith("## 14. Ограничения и эксплуатация"):
            set_cell_text(
                c,
                "## 16. Ограничения бустинга\n\n"
                "Бустинг чувствителен к гиперпараметрам и изменениям распределения, плохо экстраполирует, "
                "хуже распараллеливается, чем лес. При squared error он чувствителен к выбросам цели; "
                "существуют устойчивые функции потерь.\n",
            )
            cells = insert_markdown_after(
                cells,
                i,
                "## 17. Эксплуатация и мониторинг после внедрения\n\n"
                "Вероятности не гарантированно откалиброваны. После внедрения следят за данными, качеством, "
                "временем и размером зафиксированной модели.\n",
            )
            break

    for i, c in enumerate(cells):
        t = cell_text(c) if c["cell_type"] == "markdown" else ""
        if t.startswith("## 15. Анализ ошибок и importance"):
            set_cell_text(
                c,
                "## 18. Анализ ошибок бустинга\n\n"
                "Изучают группы с крупными ошибками. Новый признак должен возникать из гипотезы об этих ошибках "
                "и проверяться отдельным validation-экспериментом.\n",
            )
            cells = insert_markdown_after(
                cells,
                i,
                "## 19. Importance: impurity vs permutation\n\n"
                "Impurity importance имеет ограничения деревьев; для разработки её дополняют permutation importance на validation. "
                "Importance объясняет использование признака моделью, но не причинность.\n",
            )
            break

    nb["cells"] = renumber_headings_in_notebook(cells)
    return nb


SPLITTERS = {
    "13_workflow/workflow_theory.ipynb": apply_workflow_splits,
    "14_feature_engineering/feature_engineering_theory.ipynb": apply_feature_engineering_splits,
    "15_linear_regression/linear_regression_theory.ipynb": apply_linear_regression_splits,
    "16_logistic_regression/logistic_regression_theory.ipynb": apply_logistic_regression_splits,
    "18_decision_tree/decision_tree_theory.ipynb": apply_decision_tree_splits,
    "19_bagging_random_forest/bagging_random_forest_theory.ipynb": apply_bagging_splits,
    "20_gradient_boosting/gradient_boosting_theory.ipynb": apply_boosting_splits,
}


def update_folder_refs(folder: str) -> None:
    ref_map = REF_MAPS[folder]
    folder_path = ROOT / folder
    for path in folder_path.rglob("*"):
        if path.suffix not in {".ipynb", ".md"}:
            continue
        if path.name.endswith(".ipynb") and "theory" not in path.name and path != folder_path / f"{folder.split('_', 1)[1]}_theory.ipynb":
            pass  # update all ipynb and md in folder
        try:
            if path.suffix == ".ipynb":
                nb = load_nb(path)
                changed = False
                for cell in nb.get("cells", []):
                    src = cell.get("source", [])
                    if not src:
                        continue
                    new_parts = []
                    for part in src:
                        new_part = update_refs_in_text(part, ref_map)
                        if new_part != part:
                            changed = True
                        new_parts.append(new_part)
                    cell["source"] = new_parts
                if changed:
                    save_nb(path, nb)
                    print(f"  refs: {path.relative_to(ROOT)}")
            elif path.suffix == ".md":
                text = path.read_text(encoding="utf-8")
                new_text = update_refs_in_text(text, ref_map)
                if new_text != text:
                    path.write_text(new_text, encoding="utf-8")
                    print(f"  refs: {path.relative_to(ROOT)}")
        except Exception as e:
            print(f"  skip {path}: {e}", file=sys.stderr)


def fix_workflow_internal_refs(nb: dict) -> dict:
    """Manual fixes after split for workflow-specific wording."""
    ref_map = REF_MAPS["13_workflow"]
    for cell in nb["cells"]:
        if cell.get("source"):
            cell["source"] = [update_refs_in_text("".join(cell["source"]), ref_map)]
            if len(cell["source"]) == 1 and not cell["source"][0].endswith("\n") and "\n" in cell["source"][0]:
                pass
            # keep as single string in source list
    # Fix §2 cycle list explicitly
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "Краткая памятка по циклу из п. 2" in src:
            set_cell_text(
                cell,
                "Краткая памятка по циклу из п. 2:\n\n"
                "1. Тип задачи и постановка (пп. 1, 5–8).\n"
                "2. Разбиение train / validation / test без утечек (пп. 4, 9–10).\n"
                "3. Baseline (п. 11).\n"
                "4. Признаки, предобработка, модель — параметры преобразований только на train (пп. 12–14).\n"
                "5. Сравнение идей и подбор `k` на validation, журнал экспериментов (пп. 15–16).\n"
                "6. Анализ ошибок (пп. 17–18).\n"
                "7. Один раз test и мониторинг после внедрения (п. 19).\n",
            )
        if src.startswith("## 2. Сквозной цикл"):
            set_cell_text(
                cell,
                re.sub(
                    r"3\. \*\*Предобработка\*\*[^\n]+\n",
                    "3. **Предобработка** — заполнить пропуски, закодировать категории, привести числа к сравнимому масштабу; "
                    "параметры таких преобразований считают **только на train** (пп. 13–14).\n",
                    src,
                ),
            )
            src2 = cell_text(cell)
            set_cell_text(
                cell,
                src2.replace("7. **Анализ ошибок** — понять, где модель ошибается, и сформулировать следующую идею (п. 16).\n",
                             "7. **Анализ ошибок** — понять, где модель ошибается, и сформулировать следующую идею (пп. 17–18).\n")
                .replace("один раз проверяют **test** (п. 17) и готовят мониторинг после внедрения (п. 18).",
                         "один раз проверяют **test** (п. 19) и готовят мониторинг после внедрения (п. 20)."),
            )
        if "## 3. Метод k" in src:
            set_cell_text(
                cell,
                src.replace("проверим в п. 15)", "проверим в п. 13)")
                .replace("Как подбирать `k` — п. 14.", "Как подбирать `k` — п. 15."),
            )
    return nb


def main() -> None:
    for rel, splitter in SPLITTERS.items():
        path = ROOT / rel
        print(f"Split: {rel}")
        nb = load_nb(path)
        nb = splitter(nb)
        if "13_workflow" in rel:
            nb = fix_workflow_internal_refs(nb)
        save_nb(path, nb)

    for folder in REF_MAPS:
        print(f"Update refs in {folder}/")
        update_folder_refs(folder)

    print("Done.")


if __name__ == "__main__":
    main()
