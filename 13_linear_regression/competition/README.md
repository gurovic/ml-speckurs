# Занятие 26. Практика: линейная регрессия

Участникам раздать:

- `../linear_regression_practice.ipynb`
- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

Не раздавать участникам до конца занятия:

- `data/private_target.csv`
- `score_submissions.ipynb`

Главная метрика: MAE по `price_mln`, меньше — лучше.

Для проверки соберите CSV команд в папку `submissions` и запустите `score_submissions.ipynb`.

## Дополнительные материалы

Для преподавателя:

- `solution_notes.md` — разбор решения и список идей.
- `author_solution.ipynb` — авторское решение.
- `simple_baseline.ipynb` — простой baseline, который можно выдать ученикам по желанию.

Если цель — проверить полный цикл работы, baseline лучше не выдавать. Если цель — сосредоточиться на улучшении признаков и модели, baseline можно дать как старт.

## Результаты на private

Эти числа посчитаны по скрытому `data/private_target.csv`. До проверки их лучше не показывать ученикам.

Главная метрика: **MAE** (меньше — лучше).

| Решение | Private score | Дополнительные метрики |
|---|---:|---|
| `simple_baseline.ipynb` | MAE = 0.6366 | RMSE = 0.7941, R² = 0.9477 |
| `author_solution.ipynb` | MAE = 0.6147 | RMSE = 0.7719, R² = 0.9506 |
