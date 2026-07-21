from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

UROK = "\u0423\u0440\u043e\u043a_"
THEORY = "\u0422\u0435\u043e\u0440\u0438\u044f"
TEST = "\u0422\u0435\u0441\u0442"
EXERCISES_HEADING = "## \u0423\u043f\u0440\u0430\u0436\u043d\u0435\u043d\u0438\u044f"
EM_DASH = "\u2014"

TYPE_LABEL = "\u0422\u0438\u043f"
ANSWER_LABEL = "\u041f\u0440\u0430\u0432\u0438\u043b\u044c\u043d\u044b\u0439 \u043e\u0442\u0432\u0435\u0442"
CHOICE_OPTIONS_LABEL = "\u0412\u0430\u0440\u0438\u0430\u043d\u0442\u044b \u043e\u0442\u0432\u0435\u0442\u0430"
MATCH_OPTIONS_LABEL = "\u0412\u0430\u0440\u0438\u0430\u043d\u0442\u044b \u0434\u043b\u044f \u0441\u043e\u043f\u043e\u0441\u0442\u0430\u0432\u043b\u0435\u043d\u0438\u044f"
FORMAT_LABEL = "\u0424\u043e\u0440\u043c\u0430\u0442 \u043e\u0442\u0432\u0435\u0442\u0430"

OPEN = "\u043e\u0442\u043a\u0440\u044b\u0442\u044b\u0439 \u043e\u0442\u0432\u0435\u0442"
CHOICE = "\u0432\u044b\u0431\u043e\u0440 \u043e\u0442\u0432\u0435\u0442\u0430"
MATCHING = "\u0441\u043e\u043f\u043e\u0441\u0442\u0430\u0432\u043b\u0435\u043d\u0438\u0435"
ALLOWED_TYPES = {OPEN, CHOICE, MATCHING}


def theory_exercise_notebooks() -> list[Path]:
    files: list[Path] = []
    for lesson_dir in ROOT.iterdir():
        if not lesson_dir.is_dir():
            continue
        if not lesson_dir.name.startswith(UROK) or not lesson_dir.name.endswith(THEORY):
            continue
        for notebook in lesson_dir.glob("*.ipynb"):
            if notebook.name.startswith(TEST):
                files.append(notebook)
    return sorted(files, key=lambda path: str(path))


def markdown_tasks(path: Path) -> list[tuple[int, str, str | None, str]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[tuple[int, str, str | None, str]] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", ""))
        heading = re.match(r"^##\s+(\d+)\.\s+(.+)", text)
        if not heading:
            continue
        type_match = re.search(r"\*\*" + TYPE_LABEL + r":\*\*\s*([^\n]+)", text)
        task_type = type_match.group(1).strip().rstrip(".") if type_match else None
        tasks.append((int(heading.group(1)), heading.group(2).strip(), task_type, text))
    return tasks


def validate_task(path: Path, index: int, text: str) -> list[str]:
    errors: list[str] = []
    type_match = re.search(r"\*\*" + TYPE_LABEL + r":\*\*\s*([^\n]+)", text)
    if not type_match:
        return [f"{path}: task {index}: missing type"]

    task_type = type_match.group(1).strip().rstrip(".")
    if task_type not in ALLOWED_TYPES:
        errors.append(f"{path}: task {index}: unsupported type {task_type!r}")

    if ANSWER_LABEL not in text:
        errors.append(f"{path}: task {index}: missing correct answer")

    if task_type == CHOICE and CHOICE_OPTIONS_LABEL not in text:
        errors.append(f"{path}: task {index}: missing answer options")
    elif task_type == MATCHING and MATCH_OPTIONS_LABEL not in text:
        errors.append(f"{path}: task {index}: missing matching variants")
    elif task_type == OPEN and FORMAT_LABEL not in text:
        errors.append(f"{path}: task {index}: missing answer format")

    return errors


def plan_exercise_items(path: Path) -> list[tuple[int, str, str]]:
    text = path.read_text(encoding="utf-8")
    start = text.find(EXERCISES_HEADING)
    if start < 0:
        return []

    section = text[start:]
    items: list[tuple[int, str, str]] = []
    pattern = re.compile(r"^(\d+)\.\s+(.+?)\s+" + EM_DASH + r"\s+(.+)$")
    for line in section.splitlines():
        match = pattern.match(line.strip())
        if match:
            items.append(
                (
                    int(match.group(1)),
                    match.group(2).strip(),
                    match.group(3).strip().rstrip("."),
                )
            )
    return items


def validate_plan_matches_notebook(path: Path, tasks: list[tuple[int, str, str | None, str]]) -> list[str]:
    plan = path.parent / "plan.md"
    if not plan.exists():
        return [f"{path}: missing plan.md"]

    notebook_items = [(number, title, task_type) for number, title, task_type, _ in tasks]
    plan_items = plan_exercise_items(plan)
    if notebook_items == plan_items:
        return []

    return [
        f"{plan}: exercises section does not match {path.name}",
        f"  expected: {notebook_items}",
        f"  actual:   {plan_items}",
    ]


def main() -> None:
    files = theory_exercise_notebooks()
    errors: list[str] = []

    for path in files:
        tasks = markdown_tasks(path)
        for index, (_, _, _, task_text) in enumerate(tasks, start=1):
            errors.extend(validate_task(path, index, task_text))
        errors.extend(validate_plan_matches_notebook(path, tasks))

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)

    print(f"LMS exercise format and plan mapping are valid: {len(files)} notebooks checked.")


if __name__ == "__main__":
    main()
