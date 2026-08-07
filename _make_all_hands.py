# -*- coding: utf-8 -*-
"""Build *_hand_solution.ipynb for every practice: student-facing text + empty answer cells."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (source_notebook, hand_notebook)
PAIRS = [
    (
        ROOT / "Урок_22_ML_workflow_и_kNN_Практика" / "knn_imba.ipynb",
        ROOT / "Урок_22_ML_workflow_и_kNN_Практика" / "knn_imba_hand_solution.ipynb",
    ),
    (
        ROOT / "Урок_24_Feature_Engineering_Практика" / "Урок_24_Feature_Engineering_Практика.ipynb",
        ROOT / "Урок_24_Feature_Engineering_Практика" / "Урок_24_Feature_Engineering_Практика_hand_solution.ipynb",
    ),
    (
        ROOT / "Урок_26_Линейная_регрессия_Практика" / "Урок_26_Линейная_регрессия_Практика.ipynb",
        ROOT / "Урок_26_Линейная_регрессия_Практика" / "Урок_26_Линейная_регрессия_Практика_hand_solution.ipynb",
    ),
    (
        ROOT / "Урок_28_Логистическая_регрессия_Практика" / "Урок_28_Логистическая_регрессия_Практика.ipynb",
        ROOT / "Урок_28_Логистическая_регрессия_Практика" / "Урок_28_Логистическая_регрессия_Практика_hand_solution.ipynb",
    ),
    (
        ROOT / "Урок_30_Переобучение_и_валидация_Практика" / "Урок_30_Переобучение_и_валидация_Практика.ipynb",
        ROOT / "Урок_30_Переобучение_и_валидация_Практика" / "Урок_30_Переобучение_и_валидация_Практика_hand_solution.ipynb",
    ),
    (
        ROOT / "Урок_32_Решающее_дерево_Практика" / "Урок_32_Решающее_дерево_Практика.ipynb",
        ROOT / "Урок_32_Решающее_дерево_Практика" / "Урок_32_Решающее_дерево_Практика_hand_solution.ipynb",
    ),
    (
        ROOT / "Урок_34_Ансамбли_Bagging_Случайный_лес_Практика" / "Урок_34_Ансамбли_Bagging_Случайный_лес_Практика.ipynb",
        ROOT / "Урок_34_Ансамбли_Bagging_Случайный_лес_Практика" / "Урок_34_Ансамбли_Bagging_Случайный_лес_Практика_hand_solution.ipynb",
    ),
    (
        ROOT / "Урок_36_Boosting_Практика" / "Урок_36_Boosting_Практика.ipynb",
        ROOT / "Урок_36_Boosting_Практика" / "Урок_36_Boosting_Практика_hand_solution.ipynb",
    ),
]

ASSIGNMENT_RE = re.compile(
    r"^##\s+(Задание\b|.+—\s*\*\*\d+\s*балл)",
    re.MULTILINE,
)
# also competition-style scored sections without "Задание"
SCORED_SECTION_RE = re.compile(
    r"^##\s+.+\*\*\d+\s*балл",
    re.MULTILINE,
)

AUTHOR_MD_RE = re.compile(
    r"^\s*(\*\*(Ответ|Вывод|Итоговые выводы|Диагноз|Эталонные тезисы|Вывод по протоколу)"
    r"[^*]*\*\*|###\s*Postmortem\s*\(автор|"
    r"\*Почему |\*Вывод )",
    re.IGNORECASE,
)

STUDENT_INTRO = (
    "Вы **пишете код и текст сами** в пустых ячейках после каждого задания. "
    "Блоки **«Легенда»** и **«Дано»** (и сломанный код в «Дано», если он есть) не меняйте.\n"
)


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, text: str) -> None:
    if not text.endswith("\n") and text:
        text = text + "\n"
    cell["source"] = [line + "\n" for line in text.rstrip("\n").split("\n")] if text else []


def empty_code() -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [],
    }


def empty_md_placeholder() -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["*(Ваш ответ)*\n"],
    }


def is_assignment_md(text: str) -> bool:
    """True for student-facing task prompts (possibly after a --- separator)."""
    if re.search(r"(?m)^##\s+Задание\b", text):
        return True
    # competition-style: ## Something — **N балл**
    for line in text.splitlines():
        s = line.strip()
        if s == "---" or not s:
            continue
        if s.startswith("## ") and re.search(r"\*\*\d+\s*балл", s):
            return True
        # first non-empty meaningful heading line decides
        if s.startswith("## "):
            return False
    return False


def is_author_answer_md(text: str) -> bool:
    t = text.lstrip()
    if AUTHOR_MD_RE.match(t):
        return True
    if t.startswith("**Ответ:**") or t.startswith("**Вывод:**"):
        return True
    if "авторский" in t.lower() and (
        t.startswith("**") or t.startswith("###") or t.startswith("*")
    ):
        return True
    if t.startswith("# --- Решение") or "Решение задания" in t[:80]:
        return True
    return False


def is_preamble_keep_code(text: str, idx: int, first_assign_idx: int) -> bool:
    """Code before first assignment stays (Дано / broken prod / data load)."""
    return idx < first_assign_idx


def fix_intro(text: str) -> str:
    text = re.sub(
        r"Вы[^\n]{0,300}(авторск[^\n]*|решен[^\n]*преподавател[^\n]*)\n",
        STUDENT_INTRO,
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    if "пишете код сами" not in text and "пишете код и текст сами" not in text:
        # inject after first heading line
        lines = text.split("\n")
        if lines:
            lines.insert(1, "")
            lines.insert(2, STUDENT_INTRO.rstrip("\n"))
            text = "\n".join(lines)
    # remove teacher-only hints
    text = text.replace(
        "ниже уже лежат авторские решения для преподавателя",
        "ниже — пустые ячейки для вашей работы",
    )
    text = re.sub(r"\(ниже уже лежат авторские решения[^\)]*\)", "", text)
    return text


def find_first_assignment_idx(cells: list[dict]) -> int:
    for i, c in enumerate(cells):
        if c.get("cell_type") == "markdown" and is_assignment_md(cell_text(c)):
            return i
    # fallback: first ## after Дано
    for i, c in enumerate(cells):
        t = cell_text(c)
        if c.get("cell_type") == "markdown" and re.search(r"^##\s+Задание", t, re.M):
            return i
    return len(cells)


def build_hand(src: Path, dst: Path) -> dict:
    nb = json.loads(src.read_text(encoding="utf-8"))
    cells = nb["cells"]
    first_asg = find_first_assignment_idx(cells)
    out: list[dict] = []
    stats = {"kept_preamble": 0, "kept_tasks": 0, "emptied_code": 0, "emptied_md": 0}

    for i, cell in enumerate(cells):
        c = deepcopy(cell)
        text = cell_text(c)
        ctype = c.get("cell_type")

        if i == 0 and ctype == "markdown":
            set_source(c, fix_intro(text))
            # clear outputs N/A
            out.append(c)
            stats["kept_preamble"] += 1
            continue

        if i < first_asg:
            # preamble: keep markdown and code as-is (Дано), strip outputs
            if ctype == "code":
                c["outputs"] = []
                c["execution_count"] = None
            out.append(c)
            stats["kept_preamble"] += 1
            continue

        # --- assignment region ---
        if ctype == "markdown":
            if is_assignment_md(text):
                out.append(c)
                stats["kept_tasks"] += 1
                continue
            if is_author_answer_md(text):
                out.append(empty_md_placeholder())
                stats["emptied_md"] += 1
                continue
            # other markdown in assignment area (rare) — keep if looks like instructions
            if text.lstrip().startswith("###") and "критери" in text.lower():
                out.append(c)
                continue
            # default: treat as author filler → placeholder
            if text.strip():
                out.append(empty_md_placeholder())
                stats["emptied_md"] += 1
            continue

        if ctype == "code":
            # empty student workspace
            out.append(empty_code())
            stats["emptied_code"] += 1
            continue

        out.append(c)

    # Ensure every assignment markdown is followed by at least one empty code or md cell
    ensured = []
    for i, c in enumerate(out):
        ensured.append(c)
        if c.get("cell_type") == "markdown" and is_assignment_md(cell_text(c)):
            nxt = out[i + 1] if i + 1 < len(out) else None
            if nxt is None:
                ensured.append(empty_code())
            elif nxt.get("cell_type") == "markdown" and is_assignment_md(cell_text(nxt)):
                ensured.append(empty_code())
    out = ensured

    nb["cells"] = out
    nb.setdefault("metadata", {})["name"] = dst.stem
    # drop large widgets/state if any
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def main() -> None:
    print("Building hand notebooks…")
    for src, dst in PAIRS:
        if not src.exists():
            print("MISSING", src)
            continue
        stats = build_hand(src, dst)
        size = dst.stat().st_size
        print(f"OK {dst.relative_to(ROOT)} ({size} bytes) {stats}")
    print("Done.")


if __name__ == "__main__":
    main()
