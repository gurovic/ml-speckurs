import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.strip() + "\n"
    }


def notebook(cells, name):
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


theory = [
md(r"""
# Занятие 1. Линейная регрессия: как провести линию через данные

На прошлом занятии кластеризация помогала **находить группы без готовых ответов**. Теперь задача другая: по известным примерам будем **предсказывать число** — цену, температуру, время или результат.

К концу занятия вы сможете:

- отличать регрессию от классификации и кластеризации;
- объяснить, как линейная модель строит прогноз;
- сравнивать модели по MAE, MSE и $R^2$;
- понимать переобучение, регуляризацию и влияние масштаба;
- замечать выбросы, связанные признаки и нелинейные зависимости.
"""),
md(r"""
## 1. Задача регрессии

**Регрессия** — предсказание числового ответа.

| Данные об объекте | Что предсказываем |
|---|---|
| площадь квартиры, район, этаж | цену в рублях |
| часы подготовки | балл за контрольную |
| температура и влажность | расход электричества |

Сравним знакомые задачи:

- **кластеризация:** готовых ответов нет, ищем похожие группы;
- **классификация:** ответ — категория, например «спам / не спам»;
- **регрессия:** ответ — число, например 73 балла.

Будем изучать связь между временем подготовки $x$ и результатом $y$.
"""),
code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hours = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
scores = np.array([42, 48, 55, 61, 68, 74, 82], dtype=float)
data = pd.DataFrame({'часы подготовки': hours, 'балл': scores})
data
"""),
code("""
plt.figure(figsize=(7, 4.5))
plt.scatter(hours, scores, s=90, color='steelblue')
plt.xlabel('Часы подготовки')
plt.ylabel('Балл')
plt.title('Каждый ученик — точка на графике')
plt.grid(alpha=0.3)
plt.show()
"""),
md(r"""
## 2. Линейная регрессия с одним признаком

Простейшая модель проводит прямую:

$$\hat y = wx + b$$

- $x$ — известный признак;
- $\hat y$ (читается «игрек с крышкой») — прогноз;
- $w$ — **наклон**: насколько меняется прогноз при увеличении $x$ на 1;
- $b$ — **смещение**: прогноз при $x=0$.

Если $w=6{,}5$, каждый дополнительный час связан примерно с ростом прогноза на 6,5 балла. Это связь в данных, а не доказательство причины: сама по себе формула не доказывает, что именно часы вызвали рост.
"""),
code("""
from sklearn.linear_model import LinearRegression

X = hours.reshape(-1, 1)  # sklearn ожидает таблицу признаков
model = LinearRegression()
model.fit(X, scores)

w = model.coef_[0]
b = model.intercept_
print(f'Формула: балл ≈ {w:.2f} · часы + {b:.2f}')
print(f'Прогноз для 8 часов: {model.predict([[8]])[0]:.1f}')
"""),
code("""
grid = np.linspace(0, 8, 100)
plt.figure(figsize=(7, 4.5))
plt.scatter(hours, scores, s=90, label='наблюдения')
plt.plot(grid, model.predict(grid.reshape(-1, 1)), color='crimson', label='линия модели')
plt.xlabel('Часы подготовки'); plt.ylabel('Балл')
plt.title('Линейная регрессия')
plt.grid(alpha=0.3); plt.legend(); plt.show()
"""),
md(r"""
## 3. Ошибка модели и MSE

Для каждой точки есть **остаток** (ошибка):

$$e_i = y_i - \hat y_i$$

У одной точки ошибка положительная, у другой отрицательная. Простое среднее может дать ноль, хотя прогнозы неточны. Поэтому в **MSE** ошибки возводят в квадрат:

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2$$

Геометрически $(y_i-\hat y_i)^2$ — площадь квадрата со стороной, равной длине ошибки. Большая ошибка получает особенно большой штраф.
"""),
code("""
pred = model.predict(X)
residuals = scores - pred

plt.figure(figsize=(7, 4.5))
plt.scatter(hours, scores, s=90, label='факт')
plt.plot(hours, pred, color='crimson', label='прогноз')
for x, y, y_hat in zip(hours, scores, pred):
    plt.plot([x, x], [y_hat, y], color='gray', linestyle='--')
plt.title('Пунктирные отрезки — ошибки модели')
plt.xlabel('Часы'); plt.ylabel('Балл'); plt.grid(alpha=0.3); plt.legend(); plt.show()
"""),
md(r"""
## 4. Как находится лучшая прямая

Алгоритм подбирает $w$ и $b$, чтобы MSE была минимальна. Представьте холмистую поверхность: высота — ошибка, координаты — значения $w$ и $b$. Нужно попасть в самую низкую точку.

Есть два основных пути.

1. **Точное решение.** С помощью производных можно записать условия минимума и решить систему уравнений. Для одного признака получается
   $$w=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2},\qquad b=\bar y-w\bar x.$$
2. **Градиентный спуск.** Начать с произвольных коэффициентов и много раз делать маленький шаг в сторону уменьшения ошибки.

Точное решение быстро для небольших таблиц. Градиентный спуск удобнее для огромных данных и сложных моделей, но требует подобрать размер шага и число итераций.
"""),
code("""
# Проверим формулу точного решения
x_mean, y_mean = hours.mean(), scores.mean()
w_manual = np.sum((hours - x_mean) * (scores - y_mean)) / np.sum((hours - x_mean)**2)
b_manual = y_mean - w_manual * x_mean
print(f'Вручную по формуле: w={w_manual:.2f}, b={b_manual:.2f}')
assert np.isclose(w_manual, w) and np.isclose(b_manual, b)
"""),
code("""
# Упрощённый градиентный спуск: ищем w и b маленькими шагами
w_gd, b_gd = 0.0, 0.0
learning_rate = 0.01
history = []
for step in range(3000):
    p = w_gd * hours + b_gd
    error = p - scores
    w_gd -= learning_rate * 2 * np.mean(error * hours)
    b_gd -= learning_rate * 2 * np.mean(error)
    history.append(np.mean(error**2))

print(f'Градиентный спуск: w={w_gd:.2f}, b={b_gd:.2f}')
plt.figure(figsize=(7, 3.5)); plt.plot(history)
plt.xlabel('Шаг'); plt.ylabel('MSE'); plt.title('Ошибка уменьшается шаг за шагом')
plt.grid(alpha=0.3); plt.show()
"""),
md(r"""
## 5. Метрики качества: MAE, MSE и RMSE

| Метрика | Формула | Особенность |
|---|---|---|
| MAE | $\frac1n\sum|y_i-\hat y_i|$ | понятна в единицах ответа, спокойнее к редким большим ошибкам |
| MSE | $\frac1n\sum(y_i-\hat y_i)^2$ | сильно штрафует большие промахи, измеряется в квадратных единицах |
| RMSE | $\sqrt{MSE}$ | тоже чувствительна к большим промахам, но снова в единицах ответа |

«Лучшая» метрика зависит от задачи. Если ошибка на 20 баллов намного опаснее двух ошибок по 10, полезна MSE/RMSE. Если важна типичная ошибка в понятных единицах — MAE.
"""),
code("""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(scores, pred)
mse = mean_squared_error(scores, pred)
rmse = np.sqrt(mse)
print(f'MAE = {mae:.2f} балла')
print(f'MSE = {mse:.2f} балла²')
print(f'RMSE = {rmse:.2f} балла')
"""),
md(r"""
## 6. Константный baseline и $R^2$

Перед сложной моделью нужен простой соперник — **baseline**. Для MSE разумный baseline всегда предсказывает среднее значение $\bar y$.

Коэффициент $R^2$ сравнивает нашу модель с таким прогнозом:

$$R^2 = 1-\frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}$$

- $R^2=1$ — идеальные прогнозы;
- $R^2=0$ — не лучше постоянного среднего;
- $R^2<0$ — даже хуже baseline.

$R^2$ не означает «процент правильных ответов» и сам по себе не доказывает полезность модели.
"""),
code("""
baseline = np.full_like(scores, scores.mean())
print(f'MSE baseline: {mean_squared_error(scores, baseline):.2f}')
print(f'MSE модели:   {mean_squared_error(scores, pred):.2f}')
print(f'R² модели:    {r2_score(scores, pred):.3f}')
"""),
md(r"""
## 7. Проверяем модель на новых данных

Оценивать модель на тех же примерах, по которым она училась, нечестно: это как проверять контрольную по заранее известным вопросам. Данные делят на:

- **обучающую выборку** — модель подбирает коэффициенты;
- **тестовую выборку** — итоговая проверка на невиданных примерах.

Линейная регрессия тоже может переобучиться, особенно если признаков много, данных мало или мы добавили сложные степени $x^2,x^3,\ldots$. Тогда учебная ошибка мала, а тестовая велика.
"""),
code("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, scores, test_size=2, random_state=42)
check_model = LinearRegression().fit(X_train, y_train)
print(f'MAE на обучении: {mean_absolute_error(y_train, check_model.predict(X_train)):.2f}')
print(f'MAE на тесте:    {mean_absolute_error(y_test, check_model.predict(X_test)):.2f}')
"""),
md(r"""
## 8. Регуляризация: ограничитель сложности

Когда коэффициенты становятся огромными, модель слишком резко реагирует на небольшие изменения признаков. **Регуляризация** добавляет к ошибке штраф за большие коэффициенты.

- **Ridge (L2):** штраф $\lambda\sum w_j^2$. Обычно уменьшает все веса, но редко делает их ровно нулевыми.
- **Lasso (L1):** штраф $\lambda\sum|w_j|$. Может обнулить часть весов — получается отбор признаков.
- **ElasticNet:** сочетает L1 и L2.

$\lambda$ (в sklearn — `alpha`) управляет силой штрафа. Слишком малый штраф почти ничего не меняет; слишком большой приводит к недообучению.
"""),
code("""
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Два почти одинаковых признака: часы и минуты, переведённые обратно в часы
X_related = np.column_stack([hours, hours + np.array([0.1, -0.1, 0, 0.1, -0.1, 0, 0.1])])
for name, estimator in [
    ('Без штрафа', LinearRegression()),
    ('L2 / Ridge', Ridge(alpha=10)),
    ('L1 / Lasso', Lasso(alpha=1)),
    ('ElasticNet', ElasticNet(alpha=1, l1_ratio=0.5)),
]:
    estimator.fit(X_related, scores)
    print(f'{name:12}: веса {np.round(estimator.coef_, 2)}')
"""),
md(r"""
## 9. Почему важно масштабирование

Признак «площадь» может быть от 20 до 200, а «цена» — в миллионах. Для градиентного спуска это создаёт вытянутый «овраг»: к минимуму идти труднее. При регуляризации признак с крупными числами может получить маленький коэффициент только из-за единиц измерения и меньший штраф.

**StandardScaler** преобразует каждый признак примерно к среднему 0 и стандартному отклонению 1. Масштабирование особенно важно перед градиентными методами, Ridge, Lasso и ElasticNet. Обычной линейной регрессии с точным решением оно для прогноза не обязательно, но помогает сравнивать веса.
"""),
code("""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

scaled_ridge = make_pipeline(StandardScaler(), Ridge(alpha=1))
scaled_ridge.fit(X_related, scores)
print('Pipeline сначала масштабирует признаки, затем обучает Ridge.')
"""),
md(r"""
## 10. Множественная линейная регрессия

Чаще признаков несколько:

$$\hat y=w_1x_1+w_2x_2+\ldots+w_px_p+b.$$

Например, цену квартиры можно оценивать по площади, расстоянию до центра и этажу. Модель всё ещё **линейна по коэффициентам**: каждый признак умножается на свой вес, результаты складываются.

Если остальные признаки не меняются, $w_j$ показывает, как изменится прогноз при росте $x_j$ на 1. Но это осторожная интерпретация, а не доказательство причинности.
"""),
code("""
apartments = pd.DataFrame({
    'площадь': [30, 40, 50, 60, 70, 80],
    'до_центра_км': [2, 8, 5, 10, 4, 12],
    'цена_млн': [7.1, 6.9, 9.8, 9.5, 13.5, 12.2]
})
multi = LinearRegression().fit(apartments[['площадь', 'до_центра_км']], apartments['цена_млн'])
print(dict(zip(['площадь', 'до_центра_км'], np.round(multi.coef_, 3))))
print(f'Смещение: {multi.intercept_:.3f}')
"""),
md(r"""
## 11. Выбросы и робастные методы

**Выброс** — необычно далёкое наблюдение. Из-за квадратов в MSE одна такая точка способна заметно потянуть прямую к себе.

Сначала выброс нужно проверить: это ошибка измерения или редкий, но настоящий случай? Автоматически удалять его нельзя. Если необычные случаи реальны, можно сравнить MAE и MSE или использовать робастные модели, например `HuberRegressor` и `RANSACRegressor`.
"""),
code("""
from sklearn.linear_model import HuberRegressor

x_out = np.append(hours, 7.5)
y_out = np.append(scores, 25)  # необычная точка
ordinary = LinearRegression().fit(x_out.reshape(-1, 1), y_out)
robust = HuberRegressor().fit(x_out.reshape(-1, 1), y_out)

grid = np.linspace(0, 8, 100).reshape(-1, 1)
plt.figure(figsize=(7, 4.5))
plt.scatter(x_out, y_out, s=80, label='данные')
plt.plot(grid, ordinary.predict(grid), label='обычная регрессия')
plt.plot(grid, robust.predict(grid), label='Huber — устойчивее')
plt.xlabel('Часы'); plt.ylabel('Балл'); plt.title('Выброс тянет прямую')
plt.grid(alpha=0.3); plt.legend(); plt.show()
"""),
md(r"""
## 12. Мультиколлинеарность

**Мультиколлинеарность** возникает, когда признаки сильно связаны друг с другом: например, длина в метрах и та же длина в сантиметрах.

Как заметить:

- посмотреть корреляции числовых признаков;
- построить диаграммы рассеяния;
- проверить, сильно ли меняются веса при небольшом изменении данных.

Почему это плохо: прогноз может оставаться хорошим, но модель не понимает, как «поделить заслугу» между похожими признаками. Коэффициенты становятся нестабильными, иногда огромными и с неожиданными знаками. Помогают удаление дубликатов смысла, объединение признаков и Ridge-регуляризация.
"""),
code("""
corr = pd.DataFrame(X_related, columns=['часы', 'почти те же часы']).corr()
corr.style.format('{:.3f}')
"""),
md(r"""
## 13. Когда линейная модель не работает

Прямая плохо описывает:

- изгибы и волны;
- резкие пороги;
- взаимодействия признаков («эффект одного зависит от другого»);
- прогнозы далеко за пределами изученного диапазона (**экстраполяцию**).

Это видно по графику остатков: если вместо случайного облака появляется дуга или узор, модель пропустила закономерность.
"""),
code("""
x_curve = np.linspace(-3, 3, 30)
y_curve = x_curve**2 + np.random.default_rng(3).normal(0, 0.5, len(x_curve))
line = LinearRegression().fit(x_curve.reshape(-1, 1), y_curve)
curve_pred = line.predict(x_curve.reshape(-1, 1))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(x_curve, y_curve); axes[0].plot(x_curve, curve_pred, color='crimson')
axes[0].set_title('Прямая пропускает изгиб'); axes[0].grid(alpha=0.3)
axes[1].scatter(curve_pred, y_curve - curve_pred)
axes[1].axhline(0, color='crimson'); axes[1].set_title('В остатках виден узор')
axes[1].set_xlabel('Прогноз'); axes[1].set_ylabel('Остаток'); axes[1].grid(alpha=0.3)
plt.show()
"""),
md(r"""
## 14. Линейная регрессия умеет строить кривые

Слово «линейная» относится не обязательно к форме графика, а к тому, **как входят коэффициенты**. Добавим новый признак $x^2$:

$$\hat y=w_1x+w_2x^2+b.$$

Это парабола по $x$, но линейная комбинация признаков $x$ и $x^2$. Аналогично можно добавить $x^3$, синусы или произведения признаков. Главное — не увлечься: много степеней легко приводит к переобучению.
"""),
code("""
from sklearn.preprocessing import PolynomialFeatures

poly_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
poly_model.fit(x_curve.reshape(-1, 1), y_curve)

dense = np.linspace(-3, 3, 200).reshape(-1, 1)
plt.figure(figsize=(7, 4.5))
plt.scatter(x_curve, y_curve, label='данные')
plt.plot(dense, poly_model.predict(dense), color='darkgreen', label='линейная модель с $x^2$')
plt.title('Линейная комбинация новых признаков даёт кривую')
plt.grid(alpha=0.3); plt.legend(); plt.show()
"""),
md(r"""
## 15. Алгоритм работы над задачей регрессии

1. Понять, какое число предсказываем и зачем.
2. Исследовать данные: пропуски, распределения, выбросы, связи.
3. Отделить тестовые данные **до** подбора модели.
4. Построить константный baseline.
5. Обучить простую линейную регрессию.
6. Выбрать метрику по смыслу задачи и сравнить с baseline.
7. Посмотреть остатки и ошибки на отдельных объектах.
8. При необходимости добавить признаки, масштабирование или регуляризацию.
9. Проверить качество на тесте и честно описать ограничения.

> **Главная мысль.** Линейная регрессия — не просто кнопка `fit`. Это понятная модель «взвесить признаки и сложить», качество которой нужно сравнивать, проверять на новых данных и интерпретировать с осторожностью.
""")
]


practice = [
md(r"""
# Занятие 1. Практика по линейной регрессии

Выполните задания после теоретического занятия. В ячейках замените `...` своим ответом и запустите код. Если всё верно, проверка `assert` завершится без ошибки.

Можно пользоваться формулами и кодом из теории. Решения намеренно не приведены.
"""),
code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
"""),
md(r"""
## 1. Прямая по двум точкам

Прямая проходит через точки $(1, 3)$ и $(3, 7)$. Найдите коэффициенты в формуле $\hat y=wx+b$.

Подсказка: $w=\frac{y_2-y_1}{x_2-x_1}$.
"""),
code("""
w = ...
b = ...

assert np.isclose(w, 2)
assert np.isclose(b, 1)
print('Верно: получилась формула ŷ = 2x + 1')
"""),
md(r"""
## 2. Лучшая прямая по трём точкам

Даны точки $(1,2)$, $(2,3)$, $(3,5)$. Обучите `LinearRegression` и получите прогноз при $x=4$.
"""),
code("""
X = np.array([[1], [2], [3]])
y = np.array([2, 3, 5])

model = ...
prediction = ...

assert isinstance(model, LinearRegression)
assert np.isclose(prediction, 6.3333333333)
print(f'Верно: прогноз равен {prediction:.2f}')
"""),
md(r"""
## 3. Сравнение MAE и MSE

Фактические ответы: `[10, 10, 10, 10]`, прогнозы: `[9, 11, 10, 18]`. Посчитайте MAE и MSE. Объясните в комментарии, почему MSE сильнее реагирует на последний прогноз.
"""),
code("""
y_true = np.array([10, 10, 10, 10])
y_pred = np.array([9, 11, 10, 18])

mae = ...
mse = ...
# Ваше объяснение: ...

assert np.isclose(mae, 2.5)
assert np.isclose(mse, 16.5)
print('Верно')
"""),
md(r"""
## 4. Baseline и $R^2$

Для `y_true` и `y_pred` ниже:

1. создайте baseline, который всегда предсказывает среднее `y_true`;
2. вычислите $R^2$ модели;
3. запишите вывод: лучше ли модель baseline?
"""),
code("""
y_true = np.array([2, 4, 6, 8, 10])
y_pred = np.array([3, 4, 5, 8, 10])

baseline = ...
r2 = ...
better_than_baseline = ...  # True или False

assert np.allclose(baseline, [6, 6, 6, 6, 6])
assert np.isclose(r2, 0.94)
assert better_than_baseline is True
print('Верно: модель заметно лучше константного прогноза')
"""),
md(r"""
## 5. Без регуляризации, L1 и L2

Обучите три модели на готовых данных:

- `LinearRegression()`;
- `Ridge(alpha=10)`;
- `Lasso(alpha=1)`.

Сохраните их коэффициенты. Найдите модель, у которой хотя бы один коэффициент стал равен нулю.
"""),
code("""
rng = np.random.default_rng(42)
X = rng.normal(size=(60, 4))
y = 5 * X[:, 0] + 0.15 * X[:, 1] + rng.normal(0, 0.2, 60)

plain = ...
ridge = ...
lasso = ...

plain_coef = ...
ridge_coef = ...
lasso_coef = ...
model_with_zero_weights = ...  # строка 'plain', 'ridge' или 'lasso'

assert len(plain_coef) == len(ridge_coef) == len(lasso_coef) == 4
assert model_with_zero_weights == 'lasso'
assert np.any(np.isclose(lasso_coef, 0))
print('Верно: L1-регуляризация умеет обнулять веса')
"""),
md(r"""
## 6. Какие признаки коллинеарны?

В таблице есть четыре признака. Вычислите матрицу корреляций и укажите пару с корреляцией по модулю больше 0.99. Ответ запишите как множество имён, например `{'рост', 'масса'}`.
"""),
code("""
df = pd.DataFrame({
    'метры': [1.2, 1.5, 1.7, 2.0, 2.4, 2.8],
    'сантиметры': [120, 150, 170, 200, 240, 280],
    'температура': [18, 21, 17, 25, 19, 23],
    'номер': [5, 1, 6, 2, 4, 3],
})

corr = ...
collinear_pair = ...

display(corr)
assert collinear_pair == {'метры', 'сантиметры'}
print('Верно: это один и тот же смысл в разных единицах')
"""),
md(r"""
## 7. Что говорят веса?

Модель цены квартиры использует признаки в таком порядке: `['площадь_м2', 'до_центра_км', 'этаж']`. Её веса равны `[0.18, -0.25, 0.03]`.

Считая остальные признаки неизменными, ответьте:

1. На сколько миллионов рублей меняется прогноз при увеличении площади на 1 м²?
2. Какой признак уменьшает прогноз при росте?
"""),
code("""
feature_names = ['площадь_м2', 'до_центра_км', 'этаж']
weights = np.array([0.18, -0.25, 0.03])

price_change_per_m2 = ...
negative_feature = ...

assert np.isclose(price_change_per_m2, 0.18)
assert negative_feature == 'до_центра_км'
print('Верно')
"""),
md(r"""
## 8. Линейная модель для нелинейной зависимости

Данные приблизительно подчиняются формуле $y=x^3$. Обычная прямая здесь не подходит. Создайте таблицу признаков из столбцов $x$, $x^2$ и $x^3$, обучите `LinearRegression` и сделайте прогноз для $x=2$.
"""),
code("""
x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
y = x**3

X_poly = ...
poly_model = ...
new_object = ...  # признаки x, x², x³ для x=2; форма (1, 3)
prediction = ...

assert X_poly.shape == (7, 3)
assert new_object.shape == (1, 3)
assert np.isclose(prediction, 8)
print('Верно: линейная комбинация признаков x, x² и x³ описала кубическую зависимость')
"""),
md(r"""
## 9. Мини-исследование без автоматической проверки

Возьмите небольшой набор данных из знакомой области: спорт, погода, учёба или игры. Выберите числовую цель и выполните полный цикл:

1. исследуйте таблицу и постройте графики;
2. разделите данные на обучение и тест;
3. создайте константный baseline;
4. обучите линейную регрессию;
5. сравните MAE и $R^2$;
6. изучите остатки и сформулируйте 2–3 ограничения модели.

Добавляйте ниже свои markdown- и code-ячейки.
""")
]


root = Path(__file__).parent
(root / 'lesson_1_Урок_25_Линейная_регрессия.ipynb').write_text(
    json.dumps(notebook(theory, 'Занятие 1. Теория линейной регрессии'), ensure_ascii=False, indent=1),
    encoding='utf-8'
)
(root / 'lesson_1_Тест_к_теории.ipynb').write_text(
    json.dumps(notebook(practice, 'Занятие 1. Практика по линейной регрессии'), ensure_ascii=False, indent=1),
    encoding='utf-8'
)
print('Notebooks created:', len(theory), 'theory cells,', len(practice), 'practice cells')
