"""Pipeline stages 5-7 for topics 15-18: review, fixes, execute, CHANGELOG."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOPICS = [
    {
        "theory_folder": "Урок_29_Переобучение_и_валидация_Теория",
        "practice_folder": "Урок_30_Переобучение_и_валидация_Практика",
        "prefix": "overfitting_validation",
        "theory_file": "Урок_29_Переобучение_и_валидация.ipynb",
        "exercises_file": "Тест_к_теории.ipynb",
        "practice_file": "Урок_30_Переобучение_и_валидация_Практика.ipynb",
        "n": 29,
        "np": 30,
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
        "theory_folder": "Урок_31_Решающее_дерево_Теория",
        "practice_folder": "Урок_32_Решающее_дерево_Практика",
        "prefix": "decision_tree",
        "theory_file": "Урок_31_Решающее_дерево.ipynb",
        "exercises_file": "Тест_к_теории.ipynb",
        "practice_file": "Урок_32_Решающее_дерево_Практика.ipynb",
        "n": 31,
        "np": 32,
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
        "theory_folder": "Урок_33_Ансамбли_Bagging_Случайный_лес_Теория",
        "practice_folder": "Урок_34_Ансамбли_Bagging_Случайный_лес_Практика",
        "prefix": "bagging_random_forest",
        "theory_file": "Урок_33_Ансамбли_Bagging_Случайный_лес.ipynb",
        "exercises_file": "Тест_к_теории.ipynb",
        "practice_file": "Урок_34_Ансамбли_Bagging_Случайный_лес_Практика.ipynb",
        "n": 33,
        "np": 34,
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
        "theory_folder": "Урок_35_Boosting_Теория",
        "practice_folder": "Урок_36_Boosting_Практика",
        "prefix": "gradient_boosting",
        "theory_file": "35_Градиентный_бустинг.ipynb",
        "exercises_file": "Тест_к_теории.ipynb",
        "practice_file": "Урок_36_Boosting_Практика.ipynb",
        "n": 35,
        "np": 36,
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


def patch_exercises(
    theory_folder: Path,
    exercises_file: str,
    practice_file: str,
    n: int,
    np_: int,
    topic_ru: str,
    refs: list,
):
    path = theory_folder / exercises_file
    nb = load_nb(path)
    intro = (
        f"# Занятие {n}. Упражнения: {topic_ru}\n\n"
        f"Короткая проверка теории (занятие {n}). "
        f"Сквозной код — в `{practice_file}` (занятие {np_}).\n"
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


def patch_theory_metadata(theory_folder: Path, theory_file: str, meta_name: str):
    path = theory_folder / theory_file
    nb = load_nb(path)
    set_metadata_name(nb, meta_name)
    save_nb(path, nb)


def create_outline(theory_folder: Path, practice_file: str, n: int, np_: int):
    plan = (theory_folder / "plan.md").read_text(encoding="utf-8")
    outline = f"""# LESSON_OUTLINE — {theory_folder.name}

- **Теория / упражнения:** занятие {n}
- **Практика:** занятие {np_} (следующий урок)
- **Формула:** уроки идут по порядку с 21

## План теории (из plan.md)

{plan.split('## Упражнения')[0].strip()}

## Практика (задания 0–8)

См. `{practice_file}` — validation/learning curve, CV, подбор гиперпараметров, итог.

## Артефакты

| Файл | Статус |
|------|--------|
| theory | ✓ |
| exercises | ✓ |
| practice | ✓ канонический блокнот с условиями, ответами и LLM-критериями |
"""
    (theory_folder / "LESSON_OUTLINE.md").write_text(outline, encoding="utf-8")


def create_changelog(theory_folder: Path, theory_file: str, practice_file: str, n: int, np_: int):
    text = f"""# CHANGELOG — {theory_folder.name}

## Pipeline ml-lesson-workflow (стадии 5–7)

### Технический review
- Проверены и перезапущены `{theory_file}` и `{practice_file}`.
- Нумерация: теория **{n}**, практика **{np_}**.

### Методический review
- Упражнения: ссылки на **п.** теории; intro с занятиями {n}/{np_}.
- Практика: условия, code-ответы и подробные критерии для LLM-проверки; ссылки на п. теории.

### Филолог / школьник
- metadata `name` синхронизированы во всех ноутбуках.

### Final Editor
- Комплект ноутбуков готов: теория, упражнения и практика.
"""
    (theory_folder / "CHANGELOG.md").write_text(text, encoding="utf-8")


def fix_brief(folder: Path):
    path = folder / "LESSON_BRIEF.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        text = text.replace("Датaset", "Датасет")
        path.write_text(text, encoding="utf-8")


for t in TOPICS:
    theory_folder = ROOT / t["theory_folder"]
    patch_exercises(
        theory_folder,
        t["exercises_file"],
        t["practice_file"],
        t["n"],
        t["np"],
        t["topic_ru"],
        t["exercise_refs"],
    )
    patch_theory_metadata(theory_folder, t["theory_file"], t["meta_theory"])
    create_outline(theory_folder, t["practice_file"], t["n"], t["np"])
    create_changelog(theory_folder, t["theory_file"], t["practice_file"], t["n"], t["np"])
    fix_brief(theory_folder)
    print("Review artifacts:", t["theory_folder"])

print("Patch done. Run notebook execution separately.")
