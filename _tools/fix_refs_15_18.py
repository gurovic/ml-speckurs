"""Fix cross-ref п. numbers and student-reader tweaks for topics 17-20."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPLACEMENTS = {
    "15_overfitting_validation/overfitting_validation_practice.ipynb": [
        ("п. 7–9 теории", "п. 8–10 теории"),
        ("(как в теории, п. 4)", "(как в теории, п. 6)"),
        ("По мотивам п. 7.", "По мотивам п. 8."),
        ("где переобучение? (п. 2)", "где переобучение? (п. 1)"),
        ("По мотивам п. 8. Для `best_degree`", "По мотивам п. 9. Для `best_degree`"),
        ("По мотивам п. 9. `cross_validate`", "По мотивам п. 10. `cross_validate`"),
        ("гиперпараметр**? (п. 6)", "гиперпараметр**? (п. 4)"),
    ],
    "15_overfitting_validation/overfitting_validation_practice_solution.ipynb": [
        ("п. 7–9 теории", "п. 8–10 теории"),
        ("(как в теории, п. 4)", "(как в теории, п. 6)"),
        ("По мотивам п. 7.", "По мотивам п. 8."),
        ("где переобучение? (п. 2)", "где переобучение? (п. 1)"),
        ("По мотивам п. 8. Для `best_degree`", "По мотивам п. 9. Для `best_degree`"),
        ("По мотивам п. 9. `cross_validate`", "По мотивам п. 10. `cross_validate`"),
        ("гиперпараметр**? (п. 6)", "гиперпараметр**? (п. 4)"),
    ],
    "16_decision_tree/decision_tree_practice.ipynb": [
        ("По п. 16. `plot_tree`", "По п. 1. `plot_tree`"),
        ("По п. 15. Выведите `feature_importances_`", "По п. 14. Выведите `feature_importances_`"),
    ],
    "16_decision_tree/decision_tree_practice_solution.ipynb": [
        ("По п. 16. `plot_tree`", "По п. 1. `plot_tree`"),
        ("По п. 15. Выведите `feature_importances_`", "По п. 14. Выведите `feature_importances_`"),
    ],
    "17_bagging_random_forest/bagging_random_forest_practice.ipynb": [
        ("По п. 3. Сгенерируйте bootstrap", "По п. 2. Сгенерируйте bootstrap"),
    ],
    "17_bagging_random_forest/bagging_random_forest_practice_solution.ipynb": [
        ("По п. 3. Сгенерируйте bootstrap", "По п. 2. Сгенерируйте bootstrap"),
    ],
    "18_gradient_boosting/gradient_boosting_practice.ipynb": [
        ("По п. 17. `RandomForestRegressor", "По п. 12. `RandomForestRegressor"),
        ("от bagging (п. 2).", "от bagging (п. 11)."),
    ],
    "18_gradient_boosting/gradient_boosting_practice_solution.ipynb": [
        ("По п. 17. `RandomForestRegressor", "По п. 12. `RandomForestRegressor"),
        ("от bagging (п. 2).", "от bagging (п. 11)."),
    ],
}

EXERCISE_HEADER_FIXES = {
    "15_overfitting_validation/overfitting_validation_exercises.ipynb": {
        "## 1. Диагноз (п. 2)": "## 1. Диагноз (п. 1)",
        "## 2. Параметры (п. 6)": "## 2. Параметры (п. 4)",
        "## 3. K-fold (п. 9)": "## 3. K-fold (п. 10)",
        "## 4. Группы (п. 11)": "## 4. Группы (п. 12)",
        "## 6. Fold preprocessing (п. 14)": "## 6. Fold preprocessing (п. 13)",
        "## 7. Validation curve (п. 7)": "## 7. Validation curve (п. 8)",
        "## 8. Финальный fit (п. 16)": "## 8. Финальный fit (п. 17)",
    },
    "16_decision_tree/decision_tree_exercises.ipynb": {
        "## 5. Лист регрессии (п. 8)": "## 5. Лист регрессии (п. 6)",
    },
}

SECTION6_OLD = (
    "## 6. Train, validation и test\n\n"
    "Train обучает параметры. Validation выбирает гиперпараметры и идеи. Test открывается после выбора.\n\n"
    "Если мы посмотрели test, изменили степень полинома и снова посмотрели test, то фактически начали обучаться на test. "
    "Его оценка становится оптимистичной.\n"
)
SECTION6_NEW = (
    "## 6. Train, validation и test\n\n"
    "Train обучает параметры. Validation выбирает гиперпараметры и идеи. Test открывается после выбора.\n\n"
    "Аналогия: **train** — учебник, на котором учимся; **validation** — пробник, по которому подбираем стратегию; "
    "**test** — контрольная, которую открываем один раз в конце.\n\n"
    "Если мы посмотрели test, изменили степень полинома и снова посмотрели test, то фактически начали обучаться на test. "
    "Его оценка становится оптимистичной.\n"
)


def apply_replacements(path: Path, pairs: list):
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        text = "".join(cell.get("source", []))
        new = text
        for old, repl in pairs:
            new = new.replace(old, repl)
        if new != text:
            cell["source"] = [new]
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("patched", path.relative_to(ROOT))


def fix_exercise_headers(path: Path, mapping: dict):
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell.get("source", []))
        new = text
        for old, repl in mapping.items():
            new = new.replace(old, repl)
        if new != text:
            cell["source"] = [new]
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("exercises", path.relative_to(ROOT))


def fix_section6_analogy():
    path = ROOT / "15_overfitting_validation/overfitting_validation_theory.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell.get("source", []))
        if SECTION6_OLD in text and "Аналогия:" not in text:
            cell["source"] = [text.replace(SECTION6_OLD, SECTION6_NEW, 1)]
            path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print("student fix", path.relative_to(ROOT))
            return


def main():
    for rel, pairs in REPLACEMENTS.items():
        apply_replacements(ROOT / rel, pairs)
    for rel, mapping in EXERCISE_HEADER_FIXES.items():
        fix_exercise_headers(ROOT / rel, mapping)
    fix_section6_analogy()
    print("Done.")


if __name__ == "__main__":
    main()
