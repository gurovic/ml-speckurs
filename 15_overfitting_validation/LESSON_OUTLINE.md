# LESSON_OUTLINE — 15_overfitting_validation

- **Теория / упражнения:** занятие 29
- **Практика:** занятие 30 (N+1)
- **Формула:** N = номер_папки × 2 − 1

## Сквозной пример

- **Модель:** `LinearRegression` + `PolynomialFeatures`
- **Данные:** $y \approx x^2$ + шум, 180 точек
- **Метрика:** MSE; split 70/30 (`random_state=42`)

## План теории (16 разделов)

0. Шапка + code: сквозной датасет и split 70/30 (scatter train/val)
1. Ошибка на train и на новых данных (привязка к сквозному примеру)
2. Три состояния модели (таблица + реальные MSE степеней 1/2/15 + code: три кривые)
3. Откуда берётся переобучение (утечка — ссылка на занятие 21, п. 10)
4. * Смещение и разброс (+ code: 5 обучений на подвыборках, степени 1 и 15)
5. Параметры и гиперпараметры (таблица на сквозном примере)
6. Train, validation и test (напоминание, ссылка на занятие 21, п. 4; split уже сделан в шапке)
7. Перебор гиперпараметров и переобучение на validation
8. Validation curve (+ code)
9. Learning curve (+ code)
10. K-fold cross-validation (+ code)
11. Стратификация, группы и время (+ code)
12. Предобработка внутри fold
13. Утечки в CV и grid search
14. * Вложенная валидация
15. Воспроизводимость: seeds, фиксация случайности, версии артефактов
16. Финальный протокол

**Code-ячейки:** шапка (датасет + split), п. 2 (три состояния), п. 4 (bias/variance), п. 8, 9, 10, 11.

## Упражнения (10 задач)

1. Validation error — п. 1
2. Диагноз переобучения — п. 2
3. Параметры vs гиперпараметры — п. 5
4. Test один раз — п. 6
5. K-fold: 5 fit — п. 10
6. GroupKFold — п. 11
7. TimeSeriesSplit — п. 11
8. Scaler на fold-train — п. 12
9. Validation curve — п. 8
10. Финальный fit — п. 16

## Практика (задания 0–8)

См. `overfitting_validation_practice.ipynb` — split, validation/learning curve, CV, диагноз, утечка, итог.

## Артефакты

| Файл | Статус |
|------|--------|
| theory | ✓ пересборка |
| exercises | ✓ 10 задач |
| practice | ✓ |
| practice_solution | ✓ (преподаватель) |

## Скрипты

- `scripts/rebuild_overfitting_validation_theory.py`
- `scripts/rebuild_overfitting_validation_exercises.py`
