# Итоговое соревнование по модулю: классификация ансамблями

Участникам раздать:

- `ensemble_classification_competition.ipynb`
- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

Не раздавать участникам до конца соревнования:

- `data/private_target.csv`
- `score_submissions.ipynb`

Главная метрика: ROC-AUC по столбцу `churn_probability`, больше — лучше.

Для проверки соберите CSV команд в папку `submissions` и запустите `score_submissions.ipynb`.

## Дополнительные материалы

Для преподавателя:

- `solution_notes.md` — разбор решения и список идей.
- `author_solution.ipynb` — авторское решение с RandomForest, ExtraTrees, sklearn-бустингом, XGBoost, LightGBM и CatBoost.
- `simple_baseline.ipynb` — простой baseline на RandomForest, который можно выдать ученикам по желанию.

Для итогового соревнования baseline удобно выдавать слабым группам, а сильным оставить только условие и данные.

## Результаты на private

Эти числа посчитаны по скрытому `data/private_target.csv`. До проверки их лучше не показывать ученикам.

Главная метрика: **ROC_AUC** (больше — лучше).

| Решение | Private score | Дополнительные метрики |
|---|---:|---|
| `simple_baseline.ipynb` | ROC-AUC = 0.6800 | AP = 0.4373, F1@0.5 = 0.1157 |
| `author_solution.ipynb` | ROC-AUC = 0.6994 | AP = 0.4571, F1@0.5 = 0.1157 |
