# Практики и мини-задачки по разделу

Индекс показывает, какие материалы уже есть к каждому теоретическому занятию.

Мини-задачки на понимание оформлены как LMS-ready notebook: только markdown-ячейки, без кода, `assert` и скрытых ответов.

| Теория | Мини-задачки на понимание | Следующая практика | Комментарий |
|---:|---|---|---|
| 25. ML workflow и kNN | `13_workflow/workflow_exercises.ipynb` | 26. `13_workflow/workflow_practice.ipynb` | Полный цикл: постановка, split, baseline, kNN, ошибки |
| 27. Конструирование признаков | `14_feature_engineering/feature_engineering_exercises.ipynb` | 28. `14_feature_engineering/feature_engineering_practice.ipynb` | Признаки для регрессии цены квартиры |
| 29. Линейная регрессия | `15_linear_regression/linear_regression_exercises.ipynb` | 30. `15_linear_regression/linear_regression_practice.ipynb` | Практика с leaderboard-форматом и скрытым test |
| 31. Логистическая регрессия | `16_logistic_regression/logistic_regression_exercises.ipynb` | 32. `16_logistic_regression/logistic_regression_practice.ipynb` | Практика с вероятностями, F1 и подбором порога |
| 33. Переобучение и валидация | `17_overfitting_validation/overfitting_validation_exercises.ipynb` | 34. `17_overfitting_validation/overfitting_validation_practice.ipynb` | Validation curve, learning curve, k-fold CV |
| 35. Решающее дерево | `18_decision_tree/decision_tree_exercises.ipynb` | 36. `18_decision_tree/decision_tree_practice.ipynb` | Глубина, листья, pruning, confusion matrix |
| 37. Bagging и случайный лес | `19_bagging_random_forest/bagging_random_forest_exercises.ipynb` | 38. `19_bagging_random_forest/bagging_random_forest_practice.ipynb` | Bagging, Random Forest, OOB, permutation importance |
| 39. Градиентный бустинг | `20_gradient_boosting/gradient_boosting_exercises.ipynb` | 40. `20_gradient_boosting/gradient_boosting_practice.ipynb` | Sklearn-бустинг и внешние библиотеки бустинга |

## Дополнительные материалы для практик в формате leaderboard

Для занятий 30 и 32 есть отдельные папки с данными, baseline, авторским решением и учительской проверкой внутри соответствующих тем:

- `15_linear_regression/competition/`
- `16_logistic_regression/competition/`

Итоговое соревнование по модулю вынесено из папок уроков в отдельную корневую папку:

- `final_ensemble_competition/`

Внутри каждой папки:

- `data/train.csv`, `data/test.csv`, `data/sample_submission.csv`;
- `data/private_target.csv` — не выдавать ученикам до проверки;
- `simple_baseline.ipynb` — можно выдать ученикам по решению учителя;
- `author_solution.ipynb` — авторское решение;
- `solution_notes.md` — разбор идей;
- `score_submissions.ipynb` — учительская проверка файлов команд.
