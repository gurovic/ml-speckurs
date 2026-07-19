# LESSON_BRIEF — bagging_random_forest

## Мета
- **Папка:** `17_bagging_random_forest/` (**N = 17 × 2 − 1 = 33**, практика **34**)
- **Теория:** `bagging_random_forest_theory.ipynb`
- **Упражнения:** `bagging_random_forest_exercises.ipynb`
- **Практика:** `bagging_random_forest_practice.ipynb` (~90 мин)
- **Решение:** `bagging_random_forest_practice_solution.ipynb` (только преподаватель)

## Тема и цель
- **Тема:** ансамбли — математическая идея, виды методов, bagging и случайный лес.
- **После занятия студент умеет:**
  - объяснить, почему усреднение / голосование слабых моделей может обогнать одну сильную;
  - отличить bagging, boosting и stacking;
  - построить bagging и random forest, интерпретировать OOB и гиперпараметры;
  - сравнить impurity и permutation importance.

## Структура теории (запрос автора)
1. **Математическая основа** (пп. 1–5): bias–variance, усреднение, голосование, разнообразие.
2. **Виды ансамблей** (п. 6) — обзор; bagging/boosting/stacking.
3. **Основная тема** (пп. 7–27): нестабильность дерева → bootstrap → bagging → RF → гиперпараметры → importance → эксплуатация.

## Сквозной пример
- **Главная модель:** `RandomForestClassifier` (bagging — промежуточный шаг сравнения).
- **Baseline для сравнения:** одно `DecisionTreeClassifier` (не Dummy).
- **Датасет:** `make_classification` (1000 объектов, 20 признаков, 7 информативных).
- **Split:** 70/30, stratify, `random_state=42`.
- **Метрика:** accuracy на validation (дисбаланс — отдельный п. 21).

## Ограничения
- [x] Не редактировать `plan.md`
- [x] Практика: пустые code-ячейки, студент пишет весь код
- [x] Упражнения: `...` + assert, 10 задач
- [x] Один сквозной датасет во всех ноутбуках
- Ссылки **п.** / **пп.** — минимум в тексте (см. skill: «Минимум перекрёстных ссылок»); в упражнениях и практике — ссылка на теорию в шапке, не в каждом задании.

## Предпосылки
- Занятие 29: bias–variance, train/validation.
- Занятие 31: решающее дерево, переобучение, `max_depth`, importance.

## Чек-лист покрытия
- [x] Bias–variance и формула снижения разброса при усреднении
- [x] Голосование / majority vote (интуиция Кондорсе)
- [x] Условие разнообразия моделей
- [x] Таблица видов ансамблей
- [x] Bootstrap, bagging, OOB
- [x] Random Forest, `max_features`, гиперпараметры
- [x] Дисбаланс, class_weight, порог
- [x] Impurity vs permutation importance
- [x] Extra Trees (факультативно)

## Запреты
- Не смешивать датасеты (только `make_classification` из brief).
- Не начинать с bootstrap до математики и обзора ансамблей.
- Не выдавать `practice_solution` студентам.
