# ML Project — Предсказание удовлетворенности авиапассажира

**Студент:** Васильева Софья Александровна

**Группа:** БИВ233


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Деплой](#деплой)
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
├── api
│   ├── __init__.py             # FastAPI package
│   └── app.py                  # FastAPI endpoints
├── app
│   └── streamlit_app.py        # Streamlit-интерфейс
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходный файл датасета
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_EDA.ipynb            # EDA
│   ├── 02_Baseline.ipynb       # Baseline-модель, эксперименты и ablation study
│   └── 03_Experiments.ipynb    # Тюнинг моделей, test evaluation, feature importance
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   ├── tables                  # Таблицы для отчёта
│   ├── report.md               # Отчёт в Markdown
│   └── report.pdf              # Отчёт в PDF
├── src
│   ├── preprocessing.py        # Предобработка данных
│   ├── modeling.py             # Обучение и оценка моделей
│   └── train_final_model.py    # Обучение финальной модели
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
.venv\Scripts\Activate.ps1    # Windows PowerShell

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
ruff check src api app
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

### CP3

Для деплоя финальная модель была переобучена на train + validation и проверена на test.

Финальная модель: `GradientBoostingClassifier`.

| Метрика | Значение |
|--------|---------:|
| ROC-AUC | 0.991833 |
| F1-score | 0.955114 |
| Accuracy | 0.951186 |
| Precision | 0.961422 |
| Recall | 0.948889 |

Финальные результаты сохраняются в:

```text
report/tables/cp3_final_model_test_results.csv
```

## Деплой

Для CP3 реализованы:

- FastAPI API для отправки запросов к модели;
- Streamlit-интерфейс для ручного ввода признаков и получения предсказания;
- Docker Compose для локального запуска сервисов.

Перед запуском API или Streamlit нужно обучить и сохранить финальную модель:

```bash
python -m src.train_final_model
```

### FastAPI

Запуск API:

```bash
uvicorn api.app:app --reload
```

Документация API будет доступна по адресу:

```text
http://127.0.0.1:8000/docs
```

Реализованы endpoints:

| Endpoint | Метод | Назначение |
|----------|-------|------------|
| `/health` | GET | Проверка состояния сервиса |
| `/predict` | POST | Получение предсказания удовлетворённости |

### Streamlit

Запуск интерфейса:

```bash
streamlit run app/streamlit_app.py
```

Интерфейс будет доступен по адресу:

```text
http://localhost:8501
```

### Docker Compose

Запуск через Docker:

```bash
docker compose up --build
```

Скриншоты работы деплоя находятся в `report/images/`:

- `api_docs.png`;
- `api_predict.png`;
- `streamlit_form.png`;
- `streamlit_prediction.png`.

## Отчёт

Финальный отчёт находится в папке `report/`:

- [`report/report.md`](report/report.md)
- [`report/report.pdf`](report/report.pdf)

Видео демонстрации работы деплоя:  
https://disk.yandex.ru/i/R6DeGPolCgNDLw