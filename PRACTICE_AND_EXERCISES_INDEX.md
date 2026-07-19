# Практики и мини-задачки по разделу

Индекс показывает, какие материалы уже есть к каждому теоретическому занятию.

Мини-задачки на понимание оформлены как LMS-ready notebook: только markdown-ячейки, без кода, `assert` и скрытых ответов.

| Теория | Мини-задачки на понимание | Следующая практика | Комментарий |
|---:|---|---|---|
| 21. ML workflow и kNN | `11_workflow/workflow_exercises.ipynb` | 22. `11_workflow/workflow_practice.ipynb` | Полный цикл: постановка, split, baseline, kNN, ошибки |
| 23. Конструирование признаков | `12_feature_engineering/feature_engineering_exercises.ipynb` | 24. `12_feature_engineering/feature_engineering_practice.ipynb` | Признаки для регрессии цены квартиры |
| 25. Линейная регрессия | `13_linear_regression/linear_regression_exercises.ipynb` | 26. `13_linear_regression/linear_regression_practice.ipynb` | Практика с leaderboard-форматом и скрытым test |
| 27. Логистическая регрессия | `14_logistic_regression/logistic_regression_exercises.ipynb` | 28. `14_logistic_regression/logistic_regression_practice.ipynb` | Практика с вероятностями, F1 и подбором порога |
| 29. Переобучение и валидация | `15_overfitting_validation/overfitting_validation_exercises.ipynb` | 30. `15_overfitting_validation/overfitting_validation_practice.ipynb` | Validation curve, learning curve, k-fold CV |
| 31. Решающее дерево | `16_decision_tree/decision_tree_exercises.ipynb` | 32. `16_decision_tree/decision_tree_practice.ipynb` | Глубина, листья, pruning, confusion matrix |
| 33. Bagging и случайный лес | `17_bagging_random_forest/bagging_random_forest_exercises.ipynb` | 34. `17_bagging_random_forest/bagging_random_forest_practice.ipynb` | Bagging, Random Forest, OOB, permutation importance |
| 35. Градиентный бустинг | `18_gradient_boosting/gradient_boosting_exercises.ipynb` | 36. `18_gradient_boosting/gradient_boosting_practice.ipynb` | Sklearn-бустинг и внешние библиотеки бустинга |

## Дополнительные материалы для практик в формате leaderboard

Для занятий 26 и 28 есть отдельные папки с данными, baseline, авторским решением и учительской проверкой внутри соответствующих тем:

- `13_linear_regression/competition/`
- `14_logistic_regression/competition/`

Итоговое соревнование по модулю вынесено из папок уроков в отдельную корневую папку:

- `final_ensemble_competition/`

Внутри каждой папки:

- `data/train.csv`, `data/test.csv`, `data/sample_submission.csv`;
- `data/private_target.csv` — не выдавать ученикам до проверки;
- `simple_baseline.ipynb` — можно выдать ученикам по решению учителя;
- `author_solution.ipynb` — авторское решение;
- `solution_notes.md` — разбор идей;
- `score_submissions.ipynb` — учительская проверка файлов команд.
