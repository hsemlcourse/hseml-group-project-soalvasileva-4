# ML Project — Предсказание удовлетворенности авиапассажира

**Студент:** Васильева Софья Александровна

**Группа:** БИВ233


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

В проекте необходимо предсказать, будет ли пассажир удовлетворён перелётом, на основе характеристик поездки, класса обслуживания, параметров задержек и оценок качества сервиса.

Целевой признак: `satisfaction`

В используемой версии датасета классы целевой переменной:

- `satisfied` — пассажир удовлетворён;
- `dissatisfied` — пассажир не удовлетворён.

Положительный класс: `satisfied`.

**Задача:** Классификация

**Датасет:** Airline Customer Satisfaction Prediction, Kaggle

**Целевая метрика:** ROC-AUC


## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_EDA.ipynb            # EDA
│   └── 02_Baseline.ipynb       # Baseline-модель? эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   ├── tables                  # Таблицы для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-soalvasileva-4
cd hseml-group-project-soalvasileva-4

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Положить CSV-файл датасета Airline Customer Satisfaction в папку:
data/raw/


# 5. Запустить ноутбуки в следующем порядке:
notebooks/EDA_01.ipynb
notebooks/Baseline_02.ipynb

# 6. Проверить код, запустив в терминале:
ruff check src
```

## Данные
- `data/raw/` — исходный CSV-файл датасета. Файл не коммитится в репозиторий, его нужно положить в папку вручную.
- `data/processed/` — подготовленные train / validation / test выборки, которые создаются после запуска `EDA_01.ipynb`.
- `report/tables/` — таблицы с результатами анализа данных и экспериментов.
- `report/images/` — графики, построенные в ходе EDA и моделирования.


## Результаты
Основная метрика: ROC-AUC.

| Модель | ROC-AUC | F1-score | Accuracy | Примечание |
|--------|--------:|---------:|---------:|------------|
| DummyClassifier | 0.500000 | — | — | Нижняя граница качества |
| LogisticRegression | ~0.902 | — | — | Простая baseline-модель |
| RandomForestClassifier, max_depth=14 | 0.989989 | 0.949442 | 0.944924 | Лучшая модель CP1 на validation |

Полная таблица экспериментов сохраняется в:

```bash
report/tables/cp1_model_results.csv
```


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
