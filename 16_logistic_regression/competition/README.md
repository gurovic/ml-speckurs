# Занятие 32. Практика: логистическая регрессия

Участникам раздать:

- `../logistic_regression_practice.ipynb`
- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

Не раздавать участникам до конца занятия:

- `data/private_target.csv`
- `score_submissions.ipynb`

Главная метрика: F1 по столбцу `will_finish`, больше — лучше.

Для проверки соберите CSV команд в папку `submissions` и запустите `score_submissions.ipynb`.

## Дополнительные материалы

Для преподавателя:

- `solution_notes.md` — разбор решения и список идей.
- `author_solution.ipynb` — авторское решение.
- `simple_baseline.ipynb` — простой baseline, который можно выдать ученикам по желанию.

Если важно проверить, умеют ли ученики сами строить pipeline и подбирать порог, baseline не выдавайте. Если важнее сравнение идей, baseline можно дать.

## Результаты на private

Эти числа посчитаны по скрытому `data/private_target.csv`. До проверки их лучше не показывать ученикам.

Главная метрика: **F1** (больше — лучше).

| Решение | Private score | Дополнительные метрики |
|---|---:|---|
| `simple_baseline.ipynb` | F1 = 0.7752 | precision = 0.7479, recall = 0.8044, ROC-AUC = 0.7389 |
| `author_solution.ipynb` | F1 = 0.7924 | precision = 0.6933, recall = 0.9244, ROC-AUC = 0.7338 |
