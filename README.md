# ML Project — Предсказание удовлетворенности авиапассажира

**Студент:** Васильева Софья Александровна

**Группа:** БИВ233


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#запуск)
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

**Задача:** Бинарная классификация

**Датасет:** Airline Customer Satisfaction Prediction, Kaggle

**Источник данных:**  
https://www.kaggle.com/datasets/raminhuseyn/airline-customer-satisfaction

**Целевая метрика:** ROC-AUC


## Структура репозитория

```text
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходный файл датасета
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_EDA.ipynb            # EDA
│   └── 02_Baseline.ipynb       # Baseline-модель, эксперименты и ablation study
│   └── 03_Experiments.ipynb    # Тюнинг моделей, test evaluation, feature importance
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
├── Dockerfile
├── docker-compose.yml
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

# 3. Активировать окружение
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate    # Windows

# 4. Установить зависимости
pip install -r requirements.txt
```

Положить CSV-файл датасета Airline Customer Satisfaction в папку:

```text
data/raw/
```

Запустить ноутбуки в следующем порядке:

```text
notebooks/01_EDA.ipynb
notebooks/02_Baseline.ipynb
notebooks/03_Experiments.ipynb
```

Проверить код, запустив в терминале:

```bash
ruff check src
```

Запуск через Docker:

```bash
docker compose up --build
```

Для запуска через Docker должны быть подготовлены файлы в `data/processed/`. Они создаются после выполнения ноутбука `notebooks/01_EDA.ipynb`.

## Данные
- `data/raw/` — исходный CSV-файл датасета. Файл не коммитится в репозиторий, его нужно положить в папку вручную.
- `data/processed/` — подготовленные train / validation / test выборки, которые создаются после запуска `01_EDA.ipynb`.
- `report/tables/` — таблицы с результатами анализа данных и экспериментов.
- `report/images/` — графики, построенные в ходе EDA и моделирования.


## Результаты
Основная метрика: ROC-AUC.

### CP1

| Модель | ROC-AUC | F1-score | Accuracy | Примечание |
|--------|--------:|---------:|---------:|------------|
| DummyClassifier | 0.500000 | — | — | Нижняя граница качества |
| LogisticRegression | ~0.902 | — | — | Простая baseline-модель |
| RandomForestClassifier, max_depth=14 | 0.989989 | 0.949442 | 0.944924 | Лучшая модель CP1 на validation |

Полная таблица CP1-экспериментов сохраняется в:

```text
report/tables/cp1_model_results.csv
```

### CP2

В CP2 был проведён подбор гиперпараметров через `RandomizedSearchCV` для RandomForest и GradientBoosting.

Лучшая модель по validation ROC-AUC: `gradient_boosting_tuned`.

| Модель | Dataset | ROC-AUC | F1-score | Accuracy | Precision | Recall |
|--------|---------|--------:|---------:|---------:|----------:|-------:|
| gradient_boosting_tuned | validation | 0.991339 | 0.954270 | 0.950416 | 0.963483 | 0.945231 |
| gradient_boosting_tuned | test | 0.991728 | 0.954134 | 0.950108 | 0.960205 | 0.948138 |

Также был добавлен анализ важности признаков. Наиболее значимые признаки:

- `inflight_entertainment`;
- `seat_comfort`;
- `ease_of_online_booking`;
- `class_Business`.

Результаты CP2 сохраняются в:

```text
report/tables/cp2_tuned_validation_results.csv
report/tables/cp2_final_test_results.csv
report/tables/cp2_feature_importance.csv
```

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
