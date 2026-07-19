# CHANGELOG — 18_gradient_boosting

## Рецензия и правки (занятие 35)

### Теория (`gradient_boosting_theory.ipynb`)
- **§3:** пояснение, что для MSE остаток — направление сдвига прогноза.
- **§5:** связь микро-примера (пп. 1–4) со сквозным `make_regression`; введение train/validation через `train_test_split`.
- **§7 / §8:** разделены post-hoc кривая и early stopping; определение early stopping перенесено в §8; упоминание test.
- **Early stopping:** `learning_rate=0.1` — демо останавливается на 258 деревьях из 350.
- **metadata:** единое имя «Занятие 35. Градиентный бустинг».

### Упражнения (`gradient_boosting_exercises.ipynb`)
- Убраны `(п. N)` из заголовков задач (ссылка на теорию — только в шапке).

### Student Reader
- Расширен `STUDENT_REVIEW.md` (6 типичных вопросов по теории и практике).
