# -*- coding: utf-8 -*-
"""Align existing Тест_к_теории.ipynb to theory-tests.md standard."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(r"c:\Users\Гуровиц Владимир\ml-speckurs")

INTROS = {
    21: """# Занятие 21. Мини-тест: ML workflow и kNN

Короткая проверка теории. Типы заданий: сопоставление, выбор ответа, открытый ответ, вычисление / код, по картинке.

В ячейках с кодом заполните пропуски (`...`), чтобы прошли `assert`.
""",
    23: """# Занятие 23. Мини-тест: конструирование признаков

Короткая проверка теории. Типы заданий: сопоставление, выбор ответа, открытый ответ, вычисление / код, по картинке.

В ячейках с кодом заполните пропуски (`...`), чтобы прошли `assert`.
""",
    25: """# Занятие 25. Мини-тест: линейная регрессия

Короткая проверка теории. Типы заданий: сопоставление, выбор ответа, открытый ответ, вычисление / код, по картинке.

В ячейках с кодом заполните пропуски (`...`), чтобы прошли `assert`.
""",
    27: """# Занятие 27. Мини-тест: логистическая регрессия

Короткая проверка теории. Типы заданий: сопоставление, выбор ответа, открытый ответ, вычисление / код, по картинке.

В ячейках с кодом заполните пропуски (`...`), чтобы прошли `assert`.
""",
    29: """# Занятие 29. Мини-тест: переобучение и валидация

Короткая проверка теории. Типы заданий: сопоставление, выбор ответа, открытый ответ, вычисление / код, по картинке.

В ячейках с кодом заполните пропуски (`...`), чтобы прошли `assert`.
""",
    31: """# Занятие 31. Мини-тест: решающее дерево

Короткая проверка теории. Типы заданий: сопоставление, выбор ответа, открытый ответ, вычисление / код, по картинке.

В ячейках с кодом заполните пропуски (`...`), чтобы прошли `assert`.
""",
    33: """# Занятие 33. Мини-тест: bagging и случайный лес

Короткая проверка теории. Типы заданий: сопоставление, выбор ответа, открытый ответ, вычисление / код, по картинке.

В ячейках с кодом заполните пропуски (`...`), чтобы прошли `assert`.
""",
    35: """# Занятие 35. Мини-тест: градиентный бустинг

Короткая проверка теории. Типы заданий: сопоставление, выбор ответа, открытый ответ, вычисление / код, по картинке.

В ячейках с кодом заполните пропуски (`...`), чтобы прошли `assert`.
""",
}

# Expected answers for compute markdown cells (by lesson + title start)
COMPUTE_ANSWERS = {
    (21, "Голосование"): "**Правильный ответ:** класс `A`, 2 голоса (проверяется assert).",
    (23, "Доля возвратов"): "**Правильный ответ:** доли `0.05` и `0.2`; рискованнее `B` (assert).",
    (23, "Масштаб и расстояние"): "**Правильный ответ:** доминирует `income`, отношение вкладов `10000` (assert).",
    (25, "Формула прогноза"): "**Правильный ответ:** `ŷ = 7`; при увеличении `x2` на 1 прогноз уменьшается на `1` (assert).",
    (25, "MSE vs MAE"): "**Правильный ответ:** MAE `4`, MSE `34`; сильнее реагирует MSE (assert).",
    (27, "Расчёт сигмоиды"): "**Правильный ответ:** `σ(0)=0.5`; при `z=-2` и пороге `0.5` класс `0` (assert).",
    (27, "Recall по числам"): "**Правильный ответ:** recall `0.75` (assert).",
    (29, "Underfit"): "**Правильный ответ:** M1 — underfit, M2 — overfit, M3 — ok (assert).",
    (29, "Среднее по K-fold"): "**Правильный ответ:** среднее `0.80` (assert).",
    (31, "Выбор split"): "**Правильный ответ:** лучше split L, взвешенный Gini `0` против `0.5` (assert).",
    (31, "Вероятность в листе"): "**Правильный ответ:** вероятность класса 1 равна `0.75`, класс `1` (assert).",
    (33, "Majority vote"): "**Правильный ответ:** класс `1` с `3` голосами (assert).",
    (33, "Усреднение"): "**Правильный ответ:** MSE одиночных `8/3`, MSE среднего `0` (assert).",
    (35, "Шаг на остатках"): "**Правильный ответ:** `F1=[11, 9.5, 10.5]`, новые остатки `[1, -0.5, 0.5]` (assert).",
    (35, "Learning rate уменьшает"): "**Правильный ответ:** шаг меньше в `10` раз (assert).",
    (35, "Learning rate и величина"): "**Правильный ответ:** шаг меньше в `10` раз (assert).",
}

TITLE_FIXES = {
    "## 4. Усреднение снижает разброс": "## 4. Усреднение прогнозов ансамбля",
    "## 7. Learning rate уменьшает шаг": "## 7. Learning rate и величина шага",
}


def to_source(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip("\n") + "\n"
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] != "" else [])


def get_src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    cell["source"] = to_source(text)


def lesson_num(path: Path) -> int:
    m = re.search(r"Урок_(\d+)_", path.parent.name)
    return int(m.group(1)) if m else 0


def normalize_type_line(line: str) -> str:
    if not line.startswith("**Тип:**"):
        return line
    raw = line.split(":**", 1)[1].strip()
    # Map composites to standard vocabulary (primary = по картинке when present)
    mapping = {
        "вопрос по картинке + выбор ответа": "по картинке",
        "вопрос по картинке + открытый ответ": "по картинке",
        "вопрос по картинке + сопоставление": "по картинке",
        "вопрос по картинке + вычисление / код": "по картинке",
    }
    return f"**Тип:** {mapping.get(raw, raw)}"


def ensure_format_answer(src: str) -> str:
    if "**Тип:** открытый ответ" not in src and "**Тип:** по картинке" not in src:
        # only add format for pure open answer
        pass
    typ = ""
    for ln in src.splitlines():
        if ln.startswith("**Тип:**"):
            typ = ln.split(":**", 1)[1].strip()
            break
    if typ == "открытый ответ" and "**Формат ответа:**" not in src:
        # insert before Правильный ответ
        if "**Правильный ответ:**" in src:
            src = src.replace(
                "**Правильный ответ:**",
                "**Формат ответа:** 1–3 предложения.\n\n**Правильный ответ:**",
                1,
            )
    return src


def ensure_compute_answer(src: str, lesson: int) -> str:
    if "**Тип:** вычисление / код" not in src and "**Тип:** по картинке" not in src:
        return src
    # for compute-only or image+code without answer key
    if "**Правильный ответ:**" in src:
        return src
    title = src.strip().splitlines()[0]
    for (L, key), ans in COMPUTE_ANSWERS.items():
        if L == lesson and key in title:
            if not src.endswith("\n"):
                src += "\n"
            src = src.rstrip() + "\n\n" + ans + "\n"
            return src
    # generic for compute
    if "**Тип:** вычисление / код" in src:
        src = src.rstrip() + "\n\n**Правильный ответ:** см. проходящие `assert` в code-ячейке.\n"
    return src


def make_31_matching() -> str:
    return """## 1. Части решающего дерева

**Тип:** сопоставление

**Задание.** Сопоставьте часть дерева с её ролью.

**Варианты для сопоставления:**

A. Корень  
B. Внутренний узел  
C. Лист

1. узел, где модель выдаёт ответ (класс или число);  
2. вершина, с которой начинается путь объекта;  
3. узел, где задаётся вопрос вида «признак ≤ порог?».

**Правильный ответ:** A-2, B-3, C-1.
"""


def fix_notebook(path: Path) -> None:
    lesson = lesson_num(path)
    nb = json.loads(path.read_text(encoding="utf-8"))

    # metadata
    meta = nb.setdefault("metadata", {})
    meta["name"] = "Тест_к_теории"
    meta.setdefault(
        "kernelspec",
        {"display_name": "Python 3", "language": "python", "name": "python3"},
    )
    meta.setdefault(
        "language_info",
        {"name": "python", "pygments_lexer": "ipython3"},
    )

    # intro
    if lesson in INTROS and nb["cells"]:
        set_src(nb["cells"][0], INTROS[lesson])

    # lesson 31: replace task 1 (open about path) with matching; keep greedy as open
    if lesson == 31:
        for i, cell in enumerate(nb["cells"]):
            src = get_src(cell)
            if src.strip().startswith("## 1. Как дерево задаёт правило"):
                set_src(cell, make_31_matching())
                break

    for cell in nb["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        src = get_src(cell)
        if not src.strip().startswith("## "):
            continue

        # title fixes
        first = src.strip().splitlines()[0]
        for old, new in TITLE_FIXES.items():
            if first == old:
                src = src.replace(old, new, 1)
                break

        # normalize type
        lines = src.splitlines(keepends=True)
        new_lines = []
        for ln in lines:
            if ln.startswith("**Тип:**"):
                # keep newline style
                nl = "\n" if ln.endswith("\n") else ""
                core = ln.rstrip("\n")
                new_lines.append(normalize_type_line(core) + nl)
            else:
                new_lines.append(ln)
        src = "".join(new_lines)

        src = ensure_format_answer(src)
        src = ensure_compute_answer(src, lesson)
        set_src(cell, src)

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"fixed {path.relative_to(ROOT)}")


def audit() -> None:
    print("\n=== AUDIT ===")
    for f in sorted(ROOT.glob("Урок_*_Теория/Тест_к_теории.ipynb")):
        nb = json.loads(f.read_text(encoding="utf-8"))
        types = {}
        missing_ans = []
        imgs = 0
        code = 0
        titles = []
        for c in nb["cells"]:
            src = get_src(c)
            if c["cell_type"] == "code":
                code += 1
            if "data:image/png;base64" in src:
                imgs += 1
            if src.strip().startswith("## "):
                titles.append(src.strip().splitlines()[0])
                typ = next(
                    (ln.split(":**", 1)[1].strip() for ln in src.splitlines() if ln.startswith("**Тип:**")),
                    "?",
                )
                types[typ] = types.get(typ, 0) + 1
                if "**Правильный ответ:**" not in src:
                    missing_ans.append(titles[-1])
        name = nb.get("metadata", {}).get("name")
        has_match = types.get("сопоставление", 0) >= 1
        has_compute = types.get("вычисление / код", 0) >= 1 or code >= 1
        has_img = types.get("по картинке", 0) >= 1 or imgs >= 1
        has_open = types.get("открытый ответ", 0) >= 1
        has_mcq = types.get("выбор ответа", 0) >= 1
        ntypes = sum(1 for k, v in types.items() if v > 0)
        ok = (
            name == "Тест_к_теории"
            and has_match
            and has_compute
            and has_img
            and has_open
            and has_mcq
            and ntypes >= 3
            and not missing_ans
            and 8 <= len(titles) <= 15
        )
        print(f.parent.name)
        print(f"  name={name!r} tasks={len(titles)} types={types}")
        print(f"  code={code} img_cells={imgs} missing_ans={missing_ans}")
        print(f"  checklist_ok={ok}")


def main() -> None:
    for f in sorted(ROOT.glob("Урок_*_Теория/Тест_к_теории.ipynb")):
        fix_notebook(f)
    audit()


if __name__ == "__main__":
    main()
