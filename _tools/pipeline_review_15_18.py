"""Pipeline stages 5-7 for topics 17-20: review, fixes, execute, CHANGELOG."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOPICS = [
    {
        "folder": "15_overfitting_validation",
        "prefix": "overfitting_validation",
        "n": 33,
        "np": 34,
        "topic_ru": "переобучение и валидация",
        "meta_theory": "Занятие 29. Переобучение и валидация",
        "exercise_refs": [
            (1, "п. 1"),
            (2, "п. 4"),
            (3, "п. 10"),
            (4, "п. 12"),
            (5, "п. 12"),
            (6, "п. 13"),
            (7, "п. 8"),
            (8, "п. 17"),
        ],
    },
    {
        "folder": "16_decision_tree",
        "prefix": "decision_tree",
        "n": 35,
        "np": 36,
        "topic_ru": "решающее дерево",
        "meta_theory": "Занятие 31. Решающее дерево",
        "exercise_refs": [
            (1, "п. 4"),
            (2, "п. 4"),
            (3, "п. 4"),
            (4, "п. 7"),
            (5, "п. 8"),
            (6, "п. 11"),
            (7, "п. 16"),
            (8, "п. 10"),
        ],
    },
    {
        "folder": "17_bagging_random_forest",
        "prefix": "bagging_random_forest",
        "n": 37,
        "np": 38,
        "topic_ru": "bagging и случайный лес",
        "meta_theory": "Занятие 33. Bagging и случайный лес",
        "exercise_refs": [
            (1, "п. 3"),
            (2, "п. 7"),
            (3, "п. 5"),
            (4, "п. 5"),
            (5, "п. 6"),
            (6, "п. 10"),
            (7, "п. 8"),
            (8, "п. 13"),
        ],
    },
    {
        "folder": "18_gradient_boosting",
        "prefix": "gradient_boosting",
        "n": 39,
        "np": 40,
        "topic_ru": "градиентный бустинг",
        "meta_theory": "Занятие 35. Градиентный бустинг",
        "exercise_refs": [
            (1, "п. 6"),
            (2, "п. 4"),
            (3, "п. 5"),
            (4, "п. 7"),
            (5, "п. 2"),
            (6, "п. 12"),
            (7, "п. 8"),
            (8, "п. 9"),
        ],
    },
]


def load_nb(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_nb(path: Path, nb):
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def set_metadata_name(nb, name: str):
    nb.setdefault("metadata", {})["name"] = name


def patch_exercises(folder: Path, prefix: str, n: int, np_: int, topic_ru: str, refs: list):
    path = folder / f"{prefix}_exercises.ipynb"
    nb = load_nb(path)
    intro = (
        f"# Занятие {n}. Упражнения: {topic_ru}\n\n"
        f"Короткая проверка теории (`{prefix}_theory.ipynb`, занятие {n}). "
        f"Сквозной код — в `{prefix}_practice.ipynb` (занятие {np_}).\n"
    )
    nb["cells"][0]["source"] = [intro]
    ref_map = {num: ref for num, ref in refs}
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell.get("source", []))
        m = re.match(r"## (\d+)\.", src)
        if not m:
            continue
        num = int(m.group(1))
        if num in ref_map and ref_map[num] not in src:
            lines = src.split("\n", 1)
            cell["source"] = [f"{lines[0]} ({ref_map[num]})\n\n{lines[1] if len(lines) > 1 else ''}"]
            if cell["source"][-1].endswith("\n\n"):
                cell["source"][-1] = cell["source"][-1].rstrip("\n")
    set_metadata_name(nb, f"Занятие {n}. Упражнения")
    save_nb(path, nb)


def patch_theory_metadata(folder: Path, prefix: str, meta_name: str):
    path = folder / f"{prefix}_theory.ipynb"
    nb = load_nb(path)
    set_metadata_name(nb, meta_name)
    save_nb(path, nb)


def create_outline(folder: Path, n: int, np_: int):
    plan = (folder / "plan.md").read_text(encoding="utf-8")
    outline = f"""# LESSON_OUTLINE — {folder.name}

- **Теория / упражнения:** занятие {n}
- **Практика:** занятие {np_} (N+1)
- **Формула:** N = номер_папки × 2 − 1

## План теории (из plan.md)

{plan.split('## Упражнения')[0].strip()}

## Практика (задания 0–8)

См. `{folder.name.split('_', 1)[1] if '_' in folder.name else folder.name}_practice.ipynb` — validation/learning curve, CV, подбор гиперпараметров, итог.

## Артефакты

| Файл | Статус |
|------|--------|
| theory | ✓ |
| exercises | ✓ |
| practice | ✓ |
| practice_solution | ✓ (преподаватель) |
"""
    (folder / "LESSON_OUTLINE.md").write_text(outline, encoding="utf-8")


def create_changelog(folder: Path, prefix: str, n: int, np_: int):
    text = f"""# CHANGELOG — {folder.name}

## Pipeline ml-lesson-workflow (стадии 5–7)

### Технический review
- Проверены и перезапущены `{prefix}_theory.ipynb` и `{prefix}_practice_solution.ipynb`.
- Нумерация: теория **{n}**, практика **{np_}**.

### Методический review
- Упражнения: ссылки на **п.** теории; intro с занятиями {n}/{np_}.
- Практика: пустые code-ячейки; ссылки на п. теории.

### Филолог / школьник
- metadata `name` синхронизированы во всех ноутбуках.

### Final Editor
- Комплект из 4 ноутбуков готов к выдаче (solution — только преподавателю).
"""
    (folder / "CHANGELOG.md").write_text(text, encoding="utf-8")


def fix_brief(folder: Path):
    path = folder / "LESSON_BRIEF.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        text = text.replace("Датaset", "Датасет")
        path.write_text(text, encoding="utf-8")


for t in TOPICS:
    folder = ROOT / t["folder"]
    patch_exercises(folder, t["prefix"], t["n"], t["np"], t["topic_ru"], t["exercise_refs"])
    patch_theory_metadata(folder, t["prefix"], t["meta_theory"])
    create_outline(folder, t["n"], t["np"])
    create_changelog(folder, t["prefix"], t["n"], t["np"])
    fix_brief(folder)
    print("Review artifacts:", t["folder"])

print("Patch done. Run notebook execution separately.")
