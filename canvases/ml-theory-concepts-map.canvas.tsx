import { useCanvasState, useHostTheme } from "cursor/canvas";
import {
  H1,
  H2,
  Stack,
  Table,
  Text,
} from "cursor/canvas";

type ConceptItem = { label: string };

type Lesson = { num: number; short: string; concepts: ConceptItem[] };

/** Крупные сквозные концепты — по одному на строку таблицы */
const BIG_CONCEPTS: {
  key: string;
  title: string;
}[] = [
  { key: "workflow", title: "Постановка задачи и ML-workflow" },
  { key: "metrics", title: "Метрики качества и baseline" },
  { key: "train-split", title: "Train / validation / test" },
  {
    key: "cv",
    title: "Кросс-валидация, curves и схемы разбиения",
  },
  { key: "leakage", title: "Утечка данных" },
  {
    key: "features",
    title: "Конструирование и отбор признаков",
  },
  { key: "missing", title: "Пропуски" },
  { key: "categorical", title: "Категориальные признаки" },
  { key: "outliers", title: "Выбросы" },
  { key: "multicollinearity", title: "Мультиколлинеарность" },
  { key: "scaling", title: "Масштабирование признаков" },
  { key: "imbalance", title: "Дисбаланс классов" },
  {
    key: "models",
    title: "Модели (kNN, регрессии, деревья, ансамбли)",
  },
  {
    key: "regularization",
    title: "Регуляризация",
  },
  { key: "hyperparams", title: "Подбор гиперпараметров" },
  {
    key: "reproducibility",
    title: "Воспроизводимость: seeds и версии",
  },
  {
    key: "overfitting",
    title: "Переобучение, недообучение, bias–variance",
  },
  {
    key: "importance",
    title: "Важность признаков и интерпретация модели",
  },
];

type GridRow = {
  key: string;
  title: string;
  byLesson: Map<number, ConceptItem[]>;
};

function assignBigConcept(lessonNum: number, label: string): string {
  if (/утечк|агрегаты по прошл|target encoding/i.test(label)) return "leakage";
  if (
    /Permutation importance|Impurity importance|Feature importance|интерпретац.*коэфф|importance vs|Importance: impurity/i.test(
      label,
    )
  ) {
    return "importance";
  }
  if (
    /^[Чч]ек-лист|сквозной цикл|типы ML-задач|постановк|критерий успеха|единица наблюд|Confusion matrix|Разбор ошибок по группам|алгоритм работы|порядок работ|Эксплуатация и мониторинг|Анализ ошибок бустинга/i.test(
      label,
    )
  ) {
    return "workflow";
  }
  if (
    /Grid search|гиперпараметр|Параметры модели vs|Подбор k по validation|настраивать в лес|Гиперпараметры random|Learning rate|Early stopping|post-hoc выбор числа/i.test(
      label,
    )
  ) {
    return "hyperparams";
  }
  if (
    /регуляризац|Ridge|Lasso|ElasticNet|l1_ratio|ccp_alpha|post-pruning|Параметр C и сила/i.test(
      label,
    )
  ) {
    return "regularization";
  }
  if (/^Ошибка на train/i.test(label)) return "train-split";
  if (/Утечки в CV/i.test(label)) return "leakage";
  if (
    /воспроизвод|random_state|seed|верси.*артефакт/i.test(label)
  ) {
    return "reproducibility";
  }
  if (
    /^[Пп]ереобуч|при добавлении дерев|Три состояния|Причины переобуч|bias–variance|не описывает данные|недостаточно|нестабильн|Ограничения и эксплуата/i.test(
      label,
    )
  ) {
    return "overfitting";
  }
  if (
    /K-fold|Validation curve|Learning curve|Стратиф|разбиен.*групп|fold|Nested cross|Одно разбиение|Переобучение на validation|протокол сравн|Предобработка внутри|независимост.*объект|Разбиение по групп|Вложенная cross|внутренняя validation неверна/i.test(
      label,
    )
  ) {
    return "cv";
  }
  if (
    /Train\s*\//i.test(label) ||
    /Финальный test/i.test(label) ||
    /Финальный протокол/i.test(label) ||
    /Первый честный test/i.test(label)
  ) {
    return "train-split";
  }
  if (
    /[Bb]aseline|метрик|MSE|MAE|R²|confusion|Precision|recall|ROC|Log loss|калибров|DummyClassifier|Матрица ошибок|Out-of-bag|Разброс ансамбл|FN и FP|функци.*потерь|OOB-прогноз/i.test(
      label,
    )
  ) {
    return "metrics";
  }
  if (/Bagging vs|лес vs|лес, а когда/i.test(label)) return "models";
  if (/[Mm]асштаб|StandardScaler/i.test(label)) return "scaling";
  if (/выброс|ограничение выброс|робастн/i.test(label)) return "outliers";
  if (
    /категор|one-hot|ordinal encoding|редкие и неизвестные/i.test(label)
  ) {
    return "categorical";
  }
  if (/пропуск|MNAR|Пропуски в DecisionTree/i.test(label)) return "missing";
  if (/Кодирование категорий для дерева/i.test(label)) return "categorical";
  if (/Предобработка данных для бустинга|Нужна ли предобработка для бустинга|Обзор реализаций Gradient Boosting/i.test(label)) {
    return "features";
  }
  if (/мультиколлин/i.test(label)) return "multicollinearity";
  if (
    /Дисбаланс|class_weight|Взвешивание классов|Порог вероятности|смена порога|Метрики при дисбалансе|class_weight в Random/i.test(
      label,
    )
  ) {
    return "imbalance";
  }

  switch (lessonNum) {
    case 25:
      if (/kNN/i.test(label)) return "models";
      if (/Инжиниринг признаков/i.test(label)) return "features";
      return "workflow";
    case 27:
      return "features";
    case 29:
      if (/полином/i.test(label)) return "features";
      return "models";
    case 31:
      return "models";
    case 33:
      return "cv";
    case 35:
      return "models";
    case 37:
      return "models";
    case 39:
      return "models";
    default:
      return "workflow";
  }
}

function buildGridRows(): GridRow[] {
  const map = new Map<string, GridRow>();
  for (const bc of BIG_CONCEPTS) {
    map.set(bc.key, { ...bc, byLesson: new Map() });
  }
  for (const lesson of LESSONS) {
    for (const item of lesson.concepts) {
      const bigKey = assignBigConcept(lesson.num, item.label);
      const row = map.get(bigKey);
      if (!row) continue;
      if (!row.byLesson.has(lesson.num)) row.byLesson.set(lesson.num, []);
      row.byLesson.get(lesson.num)!.push(item);
    }
  }
  return BIG_CONCEPTS.filter((bc) => {
    const row = map.get(bc.key)!;
    return Array.from(row.byLesson.values()).some((items) => items.length > 0);
  }).map((bc) => map.get(bc.key)!);
}

function buildVisibleRows(placements: Placements) {
  const defaultKeys = new Set(buildGridRows().map((r) => r.key));
  return BIG_CONCEPTS.filter(
    (bc) =>
      defaultKeys.has(bc.key) ||
      Object.keys(placements).some((k) => k.startsWith(`${bc.key}:`)),
  );
}

type Placements = Record<string, string[]>;

function cellKey(rowKey: string, lessonNum: number) {
  return `${rowKey}:${lessonNum}`;
}

function placementsFromGrid(grid: GridRow[]): Placements {
  const placements: Placements = {};
  for (const row of grid) {
    for (const lesson of LESSONS) {
      const items = row.byLesson.get(lesson.num) ?? [];
      if (items.length > 0) {
        placements[cellKey(row.key, lesson.num)] = items.map((i) => i.label);
      }
    }
  }
  return placements;
}

/** Одноразовый перенос плашек в новые строки (не трогает остальную раскладку) */
const LABEL_SPLIT: Record<string, string[]> = {
  "StandardScaler и обучение kNN на train": [
    "StandardScaler: fit на train, transform везде",
    "Обучение kNN на масштабированных данных",
  ],
  "Анализ ошибок и confusion matrix": [
    "Confusion matrix на validation",
    "Разбор ошибок по группам",
  ],
  "Агрегаты по прошлому и риск утечки": [
    "Агрегаты по прошлым событиям",
    "Утечка через агрегаты и target encoding",
  ],
  "Train / test и риск переобучения регрессии": [
    "Train / validation / test для регрессии",
    "Когда линейная модель переобучается",
  ],
  "Baseline и дисбаланс классов": [
    "Baseline для классификации",
    "Дисбаланс классов и выбор метрики",
  ],
  "Калибровка и ограничения вероятностей": [
    "Ограничения интерпретации и экстраполяции",
    "Калибровка вероятностей",
  ],
  "Пропуски и категории в решающем дереве": [
    "Пропуски в DecisionTree",
    "Кодирование категорий для дерева",
  ],
  "Ограничения и эксплуатация бустинга": [
    "Ограничения бустинга",
    "Эксплуатация и мониторинг после внедрения",
  ],
  "Анализ ошибок и importance в бустинге": [
    "Анализ ошибок бустинга",
    "Importance: impurity vs permutation",
  ],
  "Числовые преобразования (отношения, log1p)": [
    "Отношения и разности",
    "Логарифмирование скошенных величин",
  ],
  "Качество разбиения: Gini и MSE": [
    "Gini impurity — мера смешения классов",
    "Information gain и качество разбиения",
    "MSE как критерий разбиения в регрессии",
  ],
  "Плюсы и ограничения решающего дерева": [
    "Почему одно дерево редко в проде",
    "Когда одно дерево всё-таки уместно",
  ],
  "Математическая идея ансамбля": [
    "Одна сильная модель или много слабых?",
    "Смещение, разброс и шум (bias–variance)",
    "Почему усреднение уменьшает разброс",
    "Голосование классификаторов",
    "Разнообразие — главное условие ансамбля",
    "Bagging, boosting и stacking — обзор",
  ],
  "Bagging: bootstrap и усреднение": [
    "Нестабильность одиночного дерева",
    "Сквозной датасет и разбиение train/validation",
    "Bootstrap-выборка",
    "Доля уникальных объектов в bootstrap",
    "Алгоритм bagging",
    "Усреднение, голосование и сравнение дерева с bagging",
    "Связь ошибок и выигрыш ансамбля",
    "Out-of-bag оценка",
    "Как собирается OOB-прогноз",
  ],
  "Random Forest: разнообразие деревьев": [
    "Случайный лес",
    "Случайные признаки и разнообразие деревьев",
    "Классификация и регрессия в лесу",
    "Основные гиперпараметры",
    "Что настраивать первым",
    "Дисбаланс: метрики, class_weight и порог",
  ],
  "Оценка и эксплуатация ансамблей": [
    "Impurity importance в лесу",
    "Permutation importance",
    "OOB, validation и test",
    "Скорость, память и параллельное обучение",
    "Когда лес полезен и где ограничен",
    "Bagging, RF и Extra Trees",
  ],
};

const LABEL_RENAME: Record<string, string> = {
  "Устройство решающего дерева (узлы и листья)":
    "Правила «если — то» и устройство дерева",
  "Выбор вопроса в узле (порог признака)": "Выбор признака и порога в узле",
  "Рекурсивное построение дерева": "Рекурсивное построение и остановка",
  "Прогноз в листе (класс или среднее)":
    "Прогноз в листе: класс и вероятность",
  "Вероятность класса и размер листа":
    "Прогноз в листе: класс и вероятность",
  "Ступенчатая граница классификации":
    "Ступенчатая и осевая граница решений",
  "Почему граница дерева осевая":
    "Ступенчатая и осевая граница решений",
  "Переобучение глубокого дерева":
    "Переобучение, гиперпараметры и обрезка",
  "Ограничение глубины и post-pruning (ccp_alpha)":
    "Переобучение, гиперпараметры и обрезка",
  "Нужно ли масштабировать признаки для дерева":
    "Масштаб, пропуски и категории в sklearn",
  "Пропуски в DecisionTree": "Масштаб, пропуски и категории в sklearn",
  "Кодирование категорий для дерева":
    "Масштаб, пропуски и категории в sklearn",
  "Impurity importance в дереве": "Impurity и permutation importance",
  "Permutation importance — проверка важности":
    "Impurity и permutation importance",
  "Нестабильность дерева при новых данных":
    "Почему одно дерево редко в проде",
  "Feature importance в дереве": "Impurity и permutation importance",
  "Feature importance в случайном лесе": "Impurity importance в лесу",
  "Сколько уникальных объектов в bootstrap":
    "Доля уникальных объектов в bootstrap",
  "Out-of-bag оценка без отдельной validation": "Out-of-bag оценка",
  "Как собирается OOB-прогноз по деревьям": "Как собирается OOB-прогноз",
  "Random Forest — bagging + случайные признаки": "Случайный лес",
  "Лес для классификации и регрессии":
    "Классификация и регрессия в лесу",
  "Что настраивать в лесу в первую очередь": "Что настраивать первым",
  "Метрики при дисбалансе": "Дисбаланс: метрики, class_weight и порог",
  "class_weight в Random Forest":
    "Дисбаланс: метрики, class_weight и порог",
  "Порог вероятности для бинарной классификации":
    "Дисбаланс: метрики, class_weight и порог",
  "Impurity importance в лесу": "Impurity importance в лесу",
  "Permutation importance на validation": "Permutation importance",
  "Random Forest vs Extra Trees": "Bagging, RF и Extra Trees",
  "Усреднение, голосование и сравнение трёх моделей":
    "Усреднение, голосование и сравнение дерева с bagging",
  "Регуляризация C и l1_ratio в логрег": "Параметр C и сила регуляризации",
  "Предобработка данных для бустинга": "Нужна ли предобработка для бустинга",
  "Стратификация по классам в CV": "Стратификация, группы и время в CV",
  "Разбиение по группам, времени и задаче": "Стратификация, группы и время в CV",
  "Типичные утечки внутри cross-validation": "Утечки в CV и grid search",
  "Grid search по сетке гиперпараметров": "Утечки в CV и grid search",
  "Одно разбиение vs повторная проверка":
    "Переобучение на validation при многократном переборе",
  "Критерии хорошего признака": "Что делает признак хорошим (момент прогноза)",
  "Нелинейность через новые признаки (x², x₁x₂)": "Нелинейность и взаимодействия",
  "Когда нужно взаимодействие признаков": "Как увидеть пользу взаимодействия",
  "Даты, сезонность и цикличность (sin/cos)": "Даты и цикличность (sin/cos)",
  "Время с прошлого события (recency)": "Время с события (дней с публикации)",
  "Редкие и неизвестные категории в test": "Редкие и неизвестные категории",
  "Пропуски — заполнение и индикатор": "Пропуски и индикатор",
  "Почему появляются пропуски (MNAR и др.)": "Почему пропуски появились",
  "Агрегаты по прошлым событиям": "Агрегаты по прошлому и утечка; target encoding",
  "Утечка через агрегаты и target encoding":
    "Агрегаты по прошлому и утечка; target encoding",
  "Отбор и проверка новых признаков": "Проверка признака на validation",
  "Стоимость признака в продакшене": "Стоимость признака и чек-лист",
  "Чек-лист конструирования признаков": "Стоимость признака и чек-лист",
};

function migrateConceptLabels(placements: Placements): Placements {
  const next: Placements = {};
  for (const [ck, labels] of Object.entries(placements)) {
    const out: string[] = [];
    for (const label of labels) {
      if (LABEL_SPLIT[label]) out.push(...LABEL_SPLIT[label]);
      else if (LABEL_RENAME[label]) out.push(LABEL_RENAME[label]);
      else out.push(label);
    }
    if (out.length > 0) next[ck] = [...new Set(out)];
  }
  return next;
}

const ROW_SPLIT_TARGET: Record<string, string> = {
  "StandardScaler: fit на train, transform везде": "scaling",
  "Обучение kNN на масштабированных данных": "models",
  "Масштабирование StandardScaler (fit на train)": "scaling",
  "Зачем масштабировать перед L1/L2": "scaling",
  "Масштабирование перед Ridge / Lasso": "scaling",
  "Нужно ли масштабировать признаки для дерева": "scaling",
  "Масштаб, пропуски и категории в sklearn": "scaling",
  "Подбор k по validation (журнал экспериментов)": "hyperparams",
  "Параметры модели vs гиперпараметры": "hyperparams",
  "Grid search по сетке гиперпараметров": "hyperparams",
  "Гиперпараметры random forest (n_estimators, max_depth…)": "hyperparams",
  "Что настраивать в лесу в первую очередь": "hyperparams",
  "Learning rate и число деревьев в бустинге": "hyperparams",
  "Early stopping и post-hoc выбор числа деревьев": "hyperparams",
  "Baseline для классификации": "metrics",
  "Дисбаланс классов и выбор метрики": "imbalance",
  "Порог вероятности и итоговый класс": "imbalance",
  "Взвешивание классов (class_weight)": "imbalance",
  "Как смена порога меняет FN и FP": "imbalance",
  "Метрики при дисбалансе": "imbalance",
  "class_weight в Random Forest": "imbalance",
  "Порог вероятности для бинарной классификации": "imbalance",
  "Интервалы и ограничение выбросов": "outliers",
  "Выбросы и робастная регрессия": "outliers",
  "Категориальные признаки и one-hot": "categorical",
  "Порядковые категории (ordinal encoding)": "categorical",
  "Редкие и неизвестные категории в test": "categorical",
  "Пропуски — заполнение и индикатор": "missing",
  "Почему появляются пропуски (MNAR и др.)": "missing",
  "Пропуски в DecisionTree": "missing",
  "Мультиколлинеарность похожих признаков": "multicollinearity",
  "Нужна ли предобработка для бустинга": "features",
  "Обзор реализаций Gradient Boosting": "models",
  "Параметр C и сила регуляризации": "regularization",
};

function migrateRowSplitPlacements(placements: Placements): Placements {
  const next: Placements = {};
  for (const [ck, labels] of Object.entries(placements)) {
    if (ck.startsWith("preprocessing:")) continue;
    next[ck] = [...labels];
  }

  const toMove: { label: string; lessonNum: number; targetRow: string }[] = [];
  for (const [ck, labels] of Object.entries(placements)) {
    const lessonNum = Number(ck.split(":")[1]);
    for (const label of labels) {
      const targetRow = ROW_SPLIT_TARGET[label];
      if (targetRow && !ck.startsWith(`${targetRow}:`)) {
        toMove.push({ label, lessonNum, targetRow });
      }
    }
  }

  for (const { label, lessonNum, targetRow } of toMove) {
    for (const ck of Object.keys(next)) {
      next[ck] = next[ck].filter((l) => l !== label);
      if (next[ck].length === 0) delete next[ck];
    }
    const targetCk = cellKey(targetRow, lessonNum);
    next[targetCk] = [...(next[targetCk] ?? []), label];
  }

  return next;
}

function moveChip(
  placements: Placements,
  from: { rowKey: string; lessonNum: number; label: string },
  to: { rowKey: string; lessonNum: number; beforeLabel?: string },
): Placements {
  if (from.lessonNum !== to.lessonNum) return placements;

  const fromCk = cellKey(from.rowKey, from.lessonNum);
  const toCk = cellKey(to.rowKey, to.lessonNum);
  const fromList = [...(placements[fromCk] ?? [])];
  const fromIdx = fromList.indexOf(from.label);
  if (fromIdx === -1) return placements;

  fromList.splice(fromIdx, 1);
  const next: Placements = { ...placements };
  if (fromList.length === 0) delete next[fromCk];
  else next[fromCk] = fromList;

  const toList = fromCk === toCk ? fromList : [...(next[toCk] ?? [])];
  if (to.beforeLabel !== undefined) {
    const insertAt = toList.indexOf(to.beforeLabel);
    toList.splice(insertAt >= 0 ? insertAt : toList.length, 0, from.label);
  } else {
    toList.push(from.label);
  }
  next[toCk] = toList;
  return next;
}

type DragPayload = { rowKey: string; lessonNum: number; label: string };

function parseDragPayload(raw: string): DragPayload | null {
  try {
    const data = JSON.parse(raw) as DragPayload;
    if (
      typeof data.rowKey === "string" &&
      typeof data.lessonNum === "number" &&
      typeof data.label === "string"
    ) {
      return data;
    }
  } catch {
    /* ignore */
  }
  return null;
}

const LESSONS: Lesson[] = [
  {
    num: 25,
    short: "kNN / workflow",
    concepts: [
      { label: "Типы ML-задач (классификация, регрессия, кластеризация)" },
      { label: "Сквозной цикл ML-проекта от данных до внедрения" },
      { label: "kNN — прогноз по k ближайшим соседям" },
      { label: "Train / validation / test в workflow" },
      { label: "От бизнес-вопроса к постановке ML-задачи" },
      { label: "Критерий успеха проекта и метрика" },
      { label: "Единица наблюдения и момент прогноза" },
      { label: "Выбор метрики из цены FN и FP" },
      { label: "Разбиение с учётом независимости объектов" },
      { label: "Утечка данных — признаки из будущего" },
      { label: "Baseline DummyClassifier vs kNN" },
      { label: "Инжиниринг признаков (обзор в workflow)" },
      { label: "StandardScaler: fit на train, transform везде" },
      { label: "Обучение kNN на масштабированных данных" },
      { label: "Подбор k по validation (журнал экспериментов)" },
      { label: "Одинаковый протокол сравнения моделей" },
      { label: "Confusion matrix на validation" },
      { label: "Разбор ошибок по группам" },
      { label: "Финальный test и внедрение модели" },
      { label: "Чек-лист ML-эксперимента" },
    ],
  },
  {
    num: 27,
    short: "признаки (FE)",
    concepts: [
      { label: "Что делает признак хорошим (момент прогноза)" },
      { label: "Отношения и разности" },
      { label: "Логарифмирование скошенных величин" },
      { label: "Интервалы и ограничение выбросов" },
      { label: "Нелинейность и взаимодействия" },
      { label: "Как увидеть пользу взаимодействия" },
      { label: "Даты и цикличность (sin/cos)" },
      { label: "Время с события (дней с публикации)" },
      { label: "Категориальные признаки и one-hot" },
      { label: "Порядковые категории (ordinal encoding)" },
      { label: "Редкие и неизвестные категории" },
      { label: "Пропуски и индикатор" },
      { label: "Почему пропуски появились" },
      { label: "Агрегаты по прошлому и утечка; target encoding" },
      { label: "Масштабирование StandardScaler (fit на train)" },
      { label: "Проверка признака на validation" },
      { label: "Стоимость признака и чек-лист" },
    ],
  },
  {
    num: 29,
    short: "лин. регрессия",
    concepts: [
      { label: "Обучение с учителем и без учителя" },
      { label: "Задача регрессии — предсказание числа" },
      { label: "Линейная регрессия с одним признаком (ŷ = wx + b)" },
      { label: "Остатки и функция потерь MSE" },
      { label: "Подбор коэффициентов (прямое решение / градиентный спуск)" },
      { label: "Метрики MAE, MSE и RMSE" },
      { label: "Константный baseline и коэффициент R²" },
      { label: "Train / validation / test для регрессии" },
      { label: "Когда линейная модель переобучается" },
      { label: "Множественная линейная регрессия" },
      { label: "Когда прямая не описывает данные" },
      { label: "Кривые через полиномиальные признаки" },
      { label: "Регуляризация Ridge, Lasso, ElasticNet" },
      { label: "Масштабирование перед Ridge / Lasso" },
      { label: "Выбросы и робастная регрессия" },
      { label: "Мультиколлинеарность похожих признаков" },
      { label: "Алгоритм работы над задачей регрессии" },
    ],
  },
  {
    num: 31,
    short: "логрег",
    concepts: [
      { label: "Задача бинарной классификации" },
      { label: "Классы, вероятности и согласованность" },
      { label: "Сигмоида: из признаков в вероятность класса 1" },
      { label: "Порог вероятности и итоговый класс" },
      { label: "LogisticRegression в sklearn" },
      { label: "Log loss и градиентная оптимизация" },
      { label: "Учебный пример с двумя признаками" },
      { label: "Train / validation / test для логрег" },
      { label: "Матрица ошибок (TP, FN, FP, TN)" },
      { label: "Precision, recall, F1 и выбор метрики" },
      { label: "Baseline для классификации" },
      { label: "Дисбаланс классов и выбор метрики" },
      { label: "Как смена порога меняет FN и FP" },
      { label: "Первый честный test" },
      { label: "ROC-кривая и PR-кривая" },
      { label: "Несколько признаков и линейная граница" },
      { label: "Интерпретация коэффициентов логрег" },
      { label: "Зачем масштабировать перед L1/L2" },
      { label: "Параметр C и сила регуляризации" },
      { label: "Взвешивание классов (class_weight)" },
      { label: "Многоклассовая классификация (OvR / OvO)" },
      { label: "Когда линейной границы недостаточно" },
      { label: "Ограничения интерпретации и экстраполяции" },
      { label: "Калибровка вероятностей" },
      { label: "Порядок работы с задачей классификации" },
    ],
  },
  {
    num: 33,
    short: "валидация",
    concepts: [
      { label: "Ошибка на train и на новых данных" },
      { label: "Три состояния: недо-, пере- и хорошее обобщение" },
      { label: "Причины переобучения (шум, гибкость, мало данных)" },
      { label: "Смещение и разброс (bias–variance)" },
      { label: "Параметры модели vs гиперпараметры" },
      { label: "Train / validation / test — роли выборок" },
      { label: "Переобучение на validation при многократном переборе" },
      { label: "Validation curve — подбор сложности модели" },
      { label: "Learning curve — нужно ли больше данных" },
      { label: "K-fold cross-validation" },
      { label: "Стратификация, группы и время в CV" },
      { label: "Предобработка внутри каждого fold" },
      { label: "Утечки в CV и grid search" },
      { label: "Вложенная cross-validation" },
      { label: "Воспроизводимость: seeds и версии артефактов" },
      { label: "Финальный протокол: fit на train+val, оценка на test" },
    ],
  },
  {
    num: 35,
    short: "дерево",
    concepts: [
      { label: "Правила «если — то» и устройство дерева" },
      { label: "Выбор признака и порога в узле" },
      { label: "Gini impurity — мера смешения классов" },
      { label: "Information gain и качество разбиения" },
      { label: "Энтропия как критерий разбиения" },
      { label: "Рекурсивное построение и остановка" },
      { label: "MSE как критерий разбиения в регрессии" },
      { label: "Прогноз в листе: класс и вероятность" },
      { label: "Путь одного объекта через дерево" },
      { label: "Ступенчатая и осевая граница решений" },
      { label: "Переобучение, гиперпараметры и обрезка" },
      { label: "Масштаб, пропуски и категории в sklearn" },
      { label: "Impurity и permutation importance" },
      { label: "Почему одно дерево редко в проде" },
      { label: "Когда одно дерево всё-таки уместно" },
    ],
  },
  {
    num: 37,
    short: "bagging / RF",
    concepts: [
      { label: "Одна сильная модель или много слабых?" },
      { label: "Смещение, разброс и шум (bias–variance)" },
      { label: "Почему усреднение уменьшает разброс" },
      { label: "Голосование классификаторов" },
      { label: "Разнообразие — главное условие ансамбля" },
      { label: "Bagging, boosting и stacking — обзор" },
      { label: "Нестабильность одиночного дерева" },
      { label: "Сквозной датасет и разбиение train/validation" },
      { label: "Bootstrap-выборка" },
      { label: "Доля уникальных объектов в bootstrap" },
      { label: "Алгоритм bagging" },
      { label: "Усреднение, голосование и сравнение дерева с bagging" },
      { label: "Связь ошибок и выигрыш ансамбля" },
      { label: "Out-of-bag оценка" },
      { label: "Как собирается OOB-прогноз" },
      { label: "Случайный лес" },
      { label: "Случайные признаки и разнообразие деревьев" },
      { label: "Классификация и регрессия в лесу" },
      { label: "Основные гиперпараметры" },
      { label: "Что настраивать первым" },
      { label: "Дисбаланс: метрики, class_weight и порог" },
      { label: "Impurity importance в лесу" },
      { label: "Permutation importance" },
      { label: "OOB, validation и test" },
      { label: "Скорость, память и параллельное обучение" },
      { label: "Когда лес полезен и где ограничен" },
      { label: "Bagging, RF и Extra Trees" },
    ],
  },
  {
    num: 39,
    short: "бустинг",
    concepts: [
      { label: "Boosting — последовательное исправление ошибок" },
      { label: "Несколько шагов бустинга на числовом примере" },
      { label: "Почему бустинг называется «градиентным»" },
      { label: "Learning rate и число деревьев в бустинге" },
      { label: "Слабые (мелкие) деревья в бустинге" },
      { label: "Функции потерь в градиентном бустинге" },
      { label: "Переобучение при добавлении деревьев" },
      { label: "Early stopping и post-hoc выбор числа деревьев" },
      { label: "Когда внутренняя validation неверна" },
      { label: "Градиентный бустинг для классификации" },
      { label: "Вероятности классов в бустинге" },
      { label: "Bagging vs boosting — параллель vs последовательно" },
      { label: "Когда выбрать random forest, когда boosting" },
      { label: "Нужна ли предобработка для бустинга" },
      { label: "Обзор реализаций Gradient Boosting" },
      { label: "Ограничения бустинга" },
      { label: "Эксплуатация и мониторинг после внедрения" },
      { label: "Анализ ошибок бустинга" },
      { label: "Importance: impurity vs permutation" },
      { label: "Чек-лист настройки градиентного бустинга" },
    ],
  },
];

const INITIAL_PLACEMENTS = placementsFromGrid(buildGridRows());

function collectPlacedLabels(
  placements: Placements,
  visibleRowKeys: string[],
): Set<string> {
  const placed = new Set<string>();
  for (const rowKey of visibleRowKeys) {
    for (const lesson of LESSONS) {
      for (const label of placements[cellKey(rowKey, lesson.num)] ?? []) {
        placed.add(label);
      }
    }
  }
  return placed;
}

function missingSectionsByLesson(
  placements: Placements,
  visibleRowKeys: string[],
) {
  const placed = collectPlacedLabels(placements, visibleRowKeys);
  return LESSONS.map((lesson) => ({
    lesson,
    labels: lesson.concepts
      .filter((item) => !placed.has(item.label))
      .map((item) => item.label),
  })).filter((group) => group.labels.length > 0);
}

type DragEl = {
  style: {
    opacity: string;
    outline: string;
    outlineOffset: string;
  };
};

type DragEvt = {
  preventDefault: () => void;
  stopPropagation?: () => void;
  dataTransfer: {
    setData: (type: string, value: string) => void;
    getData: (type: string) => string;
    effectAllowed: string;
    dropEffect: string;
  };
  currentTarget: DragEl;
};

function ConceptChip({
  label,
  bg,
  color,
  rowKey,
  lessonNum,
  dragging,
}: {
  label: string;
  bg: string;
  color: string;
  rowKey: string;
  lessonNum: number;
  dragging?: boolean;
}) {
  return (
    <span
      draggable
      onDragStart={(e: DragEvt) => {
        const payload: DragPayload = { rowKey, lessonNum, label };
        e.dataTransfer.setData("application/json", JSON.stringify(payload));
        e.dataTransfer.effectAllowed = "move";
        e.currentTarget.style.opacity = "0.45";
      }}
      onDragEnd={(e: DragEvt) => {
        e.currentTarget.style.opacity = "1";
      }}
      style={{
        display: "inline-block",
        padding: "4px 8px",
        borderRadius: 4,
        background: bg,
        color,
        fontSize: 11,
        fontWeight: 600,
        lineHeight: 1.35,
        whiteSpace: "normal",
        wordBreak: "break-word",
        border: "1px solid rgba(128,128,128,0.35)",
        cursor: "grab",
        opacity: dragging ? 0.45 : 1,
        userSelect: "none",
      }}
    >
      {label}
    </span>
  );
}

function ConceptCell({
  rowKey,
  lessonNum,
  labels,
  chipStyle,
  dropOutline,
  onDropChip,
}: {
  rowKey: string;
  lessonNum: number;
  labels: string[];
  chipStyle: { bg: string; color: string };
  dropOutline: string;
  onDropChip: (
    from: DragPayload,
    to: { rowKey: string; lessonNum: number; beforeLabel?: string },
  ) => void;
}) {
  const handleDragOver = (e: DragEvt) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  };

  const handleDragEnter = (e: DragEvt) => {
    e.currentTarget.style.outline = dropOutline;
    e.currentTarget.style.outlineOffset = "2px";
  };

  const handleDragLeave = (e: DragEvt) => {
    e.currentTarget.style.outline = "";
    e.currentTarget.style.outlineOffset = "";
  };

  const handleDrop = (e: DragEvt, beforeLabel?: string) => {
    e.preventDefault();
    e.currentTarget.style.outline = "";
    e.currentTarget.style.outlineOffset = "";
    const from = parseDragPayload(
      e.dataTransfer.getData("application/json"),
    );
    if (!from) return;
    onDropChip(from, { rowKey, lessonNum, beforeLabel });
  };

  if (labels.length === 0) {
    return (
      <span
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={(e: DragEvt) => handleDrop(e)}
        style={{
          display: "block",
          minHeight: 28,
          minWidth: 40,
          borderRadius: 4,
        }}
      />
    );
  }

  return (
    <span
      onDragOver={handleDragOver}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDrop={(e: DragEvt) => handleDrop(e)}
      style={{ display: "block", minHeight: 28, borderRadius: 4 }}
    >
      <Stack gap={4}>
        {labels.map((label) => (
          <span
            key={label}
            onDragOver={handleDragOver}
            onDrop={(e: DragEvt) => {
              e.stopPropagation?.();
              handleDrop(e, label);
            }}
          >
            <ConceptChip
              label={label}
              rowKey={rowKey}
              lessonNum={lessonNum}
              {...chipStyle}
            />
          </span>
        ))}
      </Stack>
    </span>
  );
}

function buildUnifiedTableRows(
  visibleRows: { key: string; title: string }[],
  placements: Placements,
  chipStyle: { bg: string; color: string },
  dropOutline: string,
  onDropChip: (
    from: DragPayload,
    to: { rowKey: string; lessonNum: number; beforeLabel?: string },
  ) => void,
) {
  const rows = [] as Parameters<typeof Table>[0]["rows"];

  for (const row of visibleRows) {
    rows.push([
      <span key={`t-${row.key}`}>
        <Text weight="semibold" size="small">
          {row.title}
        </Text>
      </span>,
      ...LESSONS.map((lesson) => (
        <span key={`${row.key}-${lesson.num}`}>
          <ConceptCell
            rowKey={row.key}
            lessonNum={lesson.num}
            labels={placements[cellKey(row.key, lesson.num)] ?? []}
            chipStyle={chipStyle}
            dropOutline={dropOutline}
            onDropChip={onDropChip}
          />
        </span>
      )),
    ]);
  }

  return rows;
}

export default function MlTheoryConceptsMap() {
  const { text, fill, accent } = useHostTheme();
  const [labelSplitDone, setLabelSplitDone] = useCanvasState("label-split-v5", false);
  const [rowSplitDone, setRowSplitDone] = useCanvasState("row-split-v4", false);
  const [placements, setPlacements] = useCanvasState(
    "placements-v2",
    INITIAL_PLACEMENTS,
  );

  if (!labelSplitDone) {
    setPlacements(migrateConceptLabels(placements));
    setLabelSplitDone(true);
  }

  if (!rowSplitDone) {
    setPlacements(migrateRowSplitPlacements(placements));
    setRowSplitDone(true);
  }

  const gridRows = buildVisibleRows(placements);
  const visibleRowKeys = gridRows.map((row) => row.key);
  const missingByLesson = missingSectionsByLesson(placements, visibleRowKeys);
  const missingCount = missingByLesson.reduce(
    (sum, group) => sum + group.labels.length,
    0,
  );
  const totalConcepts = LESSONS.reduce(
    (sum, lesson) => sum + lesson.concepts.length,
    0,
  );
  const chipStyle = { bg: fill.secondary, color: text.primary };
  const dropOutline = `2px dashed ${accent.primary}`;

  const handleDropChip = (
    from: DragPayload,
    to: { rowKey: string; lessonNum: number; beforeLabel?: string },
  ) => {
    setPlacements((prev) => moveChip(prev, from, to));
  };

  const tableHeaders = [
    <Text weight="semibold" size="small">
      Концепт
    </Text>,
    ...LESSONS.map((l) => (
      <span key={`h-${l.num}`}>
        <Stack gap={2} style={{ minWidth: 110 }}>
          <Text weight="semibold" size="small">
            {l.num}
          </Text>
          <Text tone="secondary" size="small">
            {l.short}
          </Text>
        </Stack>
      </span>
    )),
  ];

  return (
    <Stack gap={20} style={{ padding: "8px 4px 24px", maxWidth: 1600 }}>
      <Stack gap={6}>
        <H1>Карта концептов теории ML-курса</H1>
        <Text tone="secondary" size="small">
          Строки — крупные концепты (регуляризация, kNN, валидация…).
          Столбцы — занятия 25–39. Перетащите плашку внутри столбца: смена
          порядка в ячейке или перенос в другую строку-концепт того же занятия.
          Раскладка сохраняется между сессиями.
        </Text>
      </Stack>

      <Table
        headers={tableHeaders}
        rows={buildUnifiedTableRows(
          gridRows,
          placements,
          chipStyle,
          dropOutline,
          handleDropChip,
        )}
        striped
        stickyHeader
        style={{ fontSize: 12 }}
      />

      {missingCount > 0 ? (
        <Stack gap={10}>
          <Stack gap={4}>
            <H2>Разделы теории вне таблицы</H2>
            <Text tone="secondary" size="small">
              {missingCount} разделов из *_theory.ipynb не размещены ни в одной
              ячейке текущей таблицы.
            </Text>
          </Stack>
          {missingByLesson.map(({ lesson, labels }) => (
            <span key={`missing-${lesson.num}`}>
              <Stack gap={4}>
                <Text weight="semibold" size="small">
                  З{lesson.num} · {lesson.short}
                </Text>
                <Stack gap={3}>
                  {labels.map((label) => (
                    <span key={label}>
                      <Text size="small">{label}</Text>
                    </span>
                  ))}
                </Stack>
              </Stack>
            </span>
          ))}
        </Stack>
      ) : (
        <Text tone="secondary" size="small">
          Все разделы теории ({totalConcepts}) размещены в таблице.
        </Text>
      )}

      <Text tone="tertiary" size="small">
        Источник: *_theory.ipynb, папки 13–20 курса ml-speckurs · названия по
        заголовкам разделов ## N.
      </Text>
    </Stack>
  );
}
