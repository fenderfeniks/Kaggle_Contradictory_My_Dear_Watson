# Contradictory, My Dear Watson — Production NLI Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-8A2BE2)](https://github.com/huggingface/peft)
[![Lightning](https://img.shields.io/badge/Lightning-2.2%2B-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Airflow](https://img.shields.io/badge/Apache%20Airflow-2.8%2B-017CEE?logo=apacheairflow&logoColor=white)](https://airflow.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/Ruff-linter-D7FF64?logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff)
[![Kaggle](https://img.shields.io/badge/Kaggle-Contradictory%20My%20Dear%20Watson-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/contradictory-my-dear-watson)

> **Соревнование:** [Contradictory, My Dear Watson](https://www.kaggle.com/competitions/contradictory-my-dear-watson)
> **Задача:** Natural Language Inference (NLI) — трёхклассовая классификация на 15 языках
> **Базовая модель:** `cross-encoder/nli-deberta-v3-base` + дообучение через LoRA (PEFT)
> **Финальный результат:** Test Acc **0.773** | Test F1 **0.773**

Production-ready фреймворк для NLP-классификации, построенный на основе учебного Kaggle-датасета. Проект выходит за рамки соревновательной задачи — вокруг эксперимента выстроен полноценный MLOps-стек: сервер инференса на FastAPI, трекинг экспериментов в MLflow с Model Registry, оркестрация через Airflow с автоматическим промоутом в продакшн, мониторинг Prometheus/Grafana, Telegram-бот и деплой в Docker/Kubernetes.

---

## Структура проекта

```
.
├── configs/                  # Иерархия конфигов Hydra (модель, данные, обучение, API)
│   ├── environment/          # local.yaml / prod.yaml / colab.yaml
│   ├── model/
│   │   ├── architecture/     # nli_deberta.yaml, xlm_roberta_base.yaml и др.
│   │   └── finetuning/       # lora.yaml, head_only.yaml, full.yaml
│   └── data/source/          # kaggle_dataset.yaml, hf_dataset.yaml, local_csv.yaml
├── dags/                     # DAG-и Apache Airflow
│   ├── retrain_model_dag.py  # Еженедельное переобучение (KubernetesPodOperator)
│   ├── promote_to_prod.py    # Ручной промоут Staging → Production
│   ├── quality_control.py    # Мониторинг качества модели
│   ├── batch_analytics.py    # Пакетная аналитика
│   └── system_maintenance.py # Обслуживание инфраструктуры
├── deploy/
│   ├── airflow/              # Переменные Airflow
│   └── k8s/                  # Kubernetes манифесты (Deployment, Service, Ingress, PVC, RBAC)
├── notebooks/                # EDA, отбор моделей, настройка LoRA, оценка метрик
├── src/
│   ├── api/
│   │   ├── rest/             # FastAPI: эндпоинты, rate limiting, middleware, метрики
│   │   └── tg_bot/           # Aiogram 3: хендлеры, клавиатуры, webhook/polling
│   ├── core/
│   │   ├── data/             # DataModule, DynamicTextCollator, cleaners, fetcher
│   │   └── models/           # HFModelBuilder, HFTokenizerBuilder
│   ├── jobs/                 # promote.py, batch_analytics.py, maintenance.py
│   ├── sdk/                  # NLPPipeline — единый интерфейс инференса
│   ├── training/             # NLPModel (LightningModule), Optuna tuner
│   ├── utils/                # config_schema.py, hydra_utils.py, torch_utils.py
│   ├── train.py              # Точка входа обучения
│   ├── eval.py               # Точка входа оценки
│   ├── infer.py              # Точка входа инференса
│   └── submit.py             # Генерация submission для Kaggle
├── tests/
│   ├── api/                  # Тесты эндпоинтов и схем
│   ├── core/                 # Тесты коллатора и очистки текста
│   ├── dags/                 # Тесты структуры и контрактов DAG-ов
│   ├── jobs/                 # Тесты fetcher и eval exit code
│   ├── sdk/                  # Тесты NLPPipeline
│   └── training/             # Тесты NLPModel (loss, оптимизатор, PR-кривая)
├── .github/workflows/        # CI/CD
├── docker-compose.yml        # Полный локальный стек
├── Dockerfile                # Многоцелевая сборка (api / training)
├── Makefile                  # Task runner
├── pyproject.toml            # Зависимости (uv), конфиг Ruff
└── airflow-values.yaml       # Helm values для Airflow на Kubernetes
```

---

## Результаты

Финальное дообучение `cross-encoder/nli-deberta-v3-base` с LoRA на полном датасете (Google Colab, GPU):

| Метрика | Значение |
|---------|----------|
| Test Accuracy | **0.773** |
| Test F1 (macro) | **0.773** |
| Test Loss | 0.506 |

Для сравнения — baseline (head-only, 3 эпохи, 800 примеров): Test F1 **0.757**. Переход к LoRA на полном датасете дал стабильный прирост при сохранении компактности адаптера (0.24% обучаемых параметров от общего числа весов модели).

---

## Исследование

### Датасет и задача

Источник: [Kaggle — Contradictory, My Dear Watson](https://www.kaggle.com/competitions/contradictory-my-dear-watson)

Датасет содержит пары «посылка–гипотеза» на **15 языках** (включая китайские иероглифы, арабскую вязь и кириллицу), с тремя целевыми метками: **Entailment**, **Neutral**, **Contradiction**. Классы распределены идеально равномерно — оверсэмплинг и взвешивание классов не потребовались.

### EDA и предобработка

**Анализ длины токенов по всей тренировочной выборке:**

| Метрика | Токены |
|---------|--------|
| Медиана | 36 |
| 95-й перцентиль | 85 |
| 99-й перцентиль | 123 |
| Максимум | 171 |

Выбрано `max_length = 128` как продакшен-стандарт. Значение 64 токена небезопасно для NLI: поскольку посылка и гипотеза конкатенируются, медианная пара уже приближается к 60–70 токенам — жёсткое ограничение в 64 токена обрезало бы значительную часть гипотез.

**Многоязычная очистка текста:** наивный анализатор отметил ~33% токенов как «непечатаемые» — ложноположительное срабатывание на китайских, арабских и кириллических символах. Реализован `MultilingualTextCleaner`, удаляющий только системные управляющие символы (`\x00–\x08`, `\x0e–\x1f`) с сохранением всех языковых особенностей Unicode. Очистка вынесена в компонуемый `TextCleaningPipeline` — список клинеров конфигурируется через Hydra без изменения кода.

**Ключевое архитектурное решение — `DynamicTextCollator`:** подача посылки и гипотезы единой конкатенированной строкой давала ~33% accuracy (случайное угадывание). Реализован кастомный коллатор, передающий оба текста как `text` и `text_pair` напрямую в токенизатор HuggingFace — тот корректно вставляет `[SEP]` и генерирует `token_type_ids` для cross-attention. Коллатор поддерживает как парные тексты (NLI), так и одиночные (обычная классификация) через единый интерфейс.

### Выбор модели — Smoke Test (Этап 1)

Четыре предобученные NLI-модели протестированы в режиме быстрого прогона (head-only fine-tuning, 3 эпохи, 800 обучающих примеров на эпоху):

| Модель | Train F1 | Val F1 | Test Acc | Test F1 | Test Loss | Время |
|--------|----------|--------|----------|---------|-----------|-------|
| `cross-encoder/nli-deberta-v3-base` | 0.733 | 0.700 | 0.756 | **0.757** | 0.650 | 6.6 мин |
| `sileod/deberta-v3-base-tasksource-nli` | 0.769 | **0.768** | 0.674 | 0.676 | 0.669 | 6.4 мин |
| `symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli` | 0.454 | 0.436 | 0.470 | 0.462 | 1.063 | 21.0 мин |
| `tasksource/ModernBERT-large-nli` | 0.410 | 0.451 | 0.393 | 0.377 | 1.234 | 2.8 мин |

Архитектуры на базе DeBERTa v3 доминируют — их Disentangled Self-Attention особенно хорошо подходит для перекрёстного внимания между посылкой и гипотезой. XLM-RoBERTa и ModernBERT не смогли адаптироваться в режиме head-only (результаты близки к случайному угадыванию) и требуют полной разморозки сети.

### Выбор модели — Расширенный тест (Этап 2)

Два финалиста на DeBERTa протестированы на увеличенном объёме данных (4 эпохи, 2000 примеров на эпоху):

| Метрика | `sileod` (Модель A) | `cross-encoder` (Модель B) |
|---------|---------------------|---------------------------|
| Train F1 | 0.768 | **0.787** |
| Val F1 | 0.727 | **0.762** |
| Val Loss | 0.590 | **0.564** |
| Test F1 | **0.760** | 0.739 |
| Время обучения | 23.9 мин | **19.4 мин** |

**Победитель: `cross-encoder/nli-deberta-v3-base`**

`cross-encoder` оказался ~19% быстрее, показал лучшую динамику train/val и более низкий val loss. Небольшое преимущество `sileod` на тестовой выборке (400 примеров) признано статистически незначимым из-за малого размера выборки.

### Настройка LoRA / PEFT

**Шаг 1 — Коэффициент масштабирования (соотношение α/r):**

Анализ норм градиентов при фиксированном `r=8` для соотношений α/r: 1.0, 2.0, 4.0, 8.0. Выбрано соотношение **2.0** (α=16, r=8) — достаточная амплитуда градиентов для быстрой сходимости без риска NaN.

**Шаг 2 — Целевые модули (Overfitting Probe):**

100-шаговый стресс-тест (LR=1e-3) на одном батче для оценки выразительности против числа параметров:

| Целевые модули | Обучаемых параметров | Start Loss | End Loss | Train Acc |
|----------------|---------------------|-----------|---------|-----------| 
| query_proj, value_proj | 0.16% | 6.1764 | 0.0005 | 1.0 |
| query_proj, key_proj, value_proj | **0.24%** | 5.9207 | **0.0004** | 1.0 |
| query_proj, value_proj, dense | 0.64% | 5.9017 | 0.0002 | 1.0 |
| query_proj, key_proj, value_proj, dense | 0.72% | 5.6413 | 0.0001 | 1.0 |

**Выбрано: `query_proj, key_proj, value_proj`** — все конфигурации сошлись к accuracy 1.0; добавление `dense` утраивает число параметров ради пренебрежимо малого прироста (0.0004 → 0.0001). Включение `key_proj` позволяет механизму Disentangled Attention полноценно перестраивать фокус внимания между посылкой и гипотезой при спартанском объёме весов адаптера.

**Итоговый конфиг LoRA:** `r=8`, `α=16`, `dropout=0.1`, целевые модули: `query_proj, key_proj, value_proj`

### Направления для дальнейшей оптимизации

- **Снижение ранга LoRA** — уменьшить `r` с 8 до 4 или 2 (текущий ~1M обучаемых параметров избыточен для 12K примеров); протестировать `lora_dropout=0.2`.
- **Аугментация для мультиязычности** — NLI-логика на редких языках недопредставлена; варианты: аугментация машинным переводом или переход на бэкбон, предобученный на корпусах масштаба XNLI.
- **Метрики модели в Prometheus** — добавить распределение предсказанных классов и confidence distribution для детектирования data drift в продакшне.

---

## Архитектура пайплайна

### Обучение и регистрация модели

`src/train.py` оркестрирует полный цикл: токенизатор → модель → DataModule → LightningModule → Trainer. После завершения обучения лучший чекпоинт загружается обратно в память (`torch.load` + `load_state_dict`), LoRA-адаптеры впекаются в базовую модель (`merge_and_unload`), и результат логируется в MLflow Model Registry нативным `mlflow.transformers.log_model` — с явными pip-зависимостями из секции `inference-core` в `pyproject.toml`. Новая версия автоматически получает алиас `Staging`.

### Промоут в продакшн

`src/jobs/promote.py` реализует gate-проверку перед промоутом: сравнивает `val_f1` тег текущей `Staging`-версии с `Production`. Промоут происходит только если новая модель превосходит действующую — случайный откат метрики из-за нестабильного рана исключён. DAG `promote_to_prod` запускается строго вручную и после промоута перезапускает API-деплоймент через `kubectl rollout restart`.

### DataModule и кэширование

`NLPDataModule` вычисляет MD5-хэш конфигурации обработки данных (источник, клинеры, колонки, seed, max_length) и сохраняет очищенный датасет на диск. Повторные запуски с той же конфигурацией пропускают шаг очистки — данные грузятся из кэша. При изменении любого параметра хэш меняется и пересчёт происходит автоматически.

### API и инференс

`NLPPipeline` (`src/sdk/inference.py`) — единый SDK для инференса, используемый и FastAPI-сервером, и Telegram-ботом. Поддерживает загрузку весов из MLflow Model Registry, HuggingFace Hub или локального чекпоинта. Инференс в FastAPI запускается через `asyncio.to_thread` чтобы не блокировать event loop.

---

## Быстрый старт

### Требования

- Docker + Docker Compose
- NVIDIA GPU (опционально; есть CPU-фолбэк через `accelerator: auto`)
- Пакетный менеджер `uv`

### 1. Клонирование и настройка

```bash
git clone https://github.com/fenderfeniks/Kaggle_Contradictory_My_Dear_Watson.git
cd Kaggle_Contradictory_My_Dear_Watson

cp .env.example .env
# Заполнить HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY в .env
```

### 2. Установка зависимостей локально

```bash
make install
# Аналог: uv venv && uv pip install -e ".[dev,training,api]"
```

### 3. Запуск обучения

```bash
make train
# Запускает: docker compose run --rm trainer python -m src.train
```

Для обучения в Google Colab использовать `notebooks/train_colab.ipynb` с переопределением окружения:

```bash
python -m src.train environment=colab paths.log_dir=/content/drive/MyDrive/watson_logs
```

### 4. Запуск сервера инференса

```bash
make api
# Запускает: docker compose up -d --build api
# API доступен на http://localhost:8000
# Документация: http://localhost:8000/docs
```

### 5. Полный стек с мониторингом

```bash
docker compose up -d
# Сервисы:
#   API         → http://localhost:8000
#   Airflow     → http://localhost:8080  (admin/admin)
#   Prometheus  → http://localhost:9090
#   Grafana     → http://localhost:3000  (admin/admin)
#   Demo        → http://localhost:8501
```

### 6. Промоут модели в продакшн

```bash
# После завершения обучения модель автоматически попадает в Staging.
# Для промоута в Production запустить вручную:
python -m src.jobs.promote
# Или через Airflow DAG: promote_to_prod
```

### 7. Остановка

```bash
make down
```

---

## Тесты

```bash
pytest
```

Тест-сьют покрывает: логику loss с весами классов, шаги обучения/валидации/теста `NLPModel`, поиск оптимального порога по PR-кривой, `DynamicTextCollator` (паддинг, усечение, парные тексты, отсутствие лейблов), `TextCleaningPipeline`, схемы API, эндпоинты через `httpx`, структуру и контракты DAG-ов Airflow, `NLPPipeline` SDK.

---

## Стек технологий

| Слой | Инструменты |
|------|-------------|
| **Модель** | `cross-encoder/nli-deberta-v3-base`, HuggingFace Transformers |
| **Обучение** | PyTorch Lightning, PEFT (LoRA), MLflow + Model Registry, Optuna |
| **Конфигурация** | Hydra (иерархические конфиги, переопределение через CLI) |
| **Данные** | HuggingFace Datasets, кэширование по MD5-хэшу конфига |
| **API** | FastAPI + Uvicorn, SlowAPI (rate limiting), Redis, Prometheus |
| **Бот** | Aiogram 3 (Telegram), webhook + polling режимы |
| **Мониторинг** | Prometheus + Grafana, `prometheus-fastapi-instrumentator` |
| **Оркестрация** | Apache Airflow + KubernetesPodOperator, Slack-нотификации |
| **Деплой** | Docker, Docker Compose, Kubernetes (Helm, PVC, RBAC) |
| **Пакетный менеджер** | `uv` + `pyproject.toml` (группы зависимостей по слоям) |
| **Качество кода** | Ruff (lint + format), pre-commit, mypy |
| **Тесты** | pytest, pytest-asyncio, httpx |
| **Демо** | Streamlit |

---

## Переменные окружения

Скопировать `.env.example` в `.env` и заполнить:

| Переменная | Описание |
|------------|----------|
| `HF_TOKEN` | Токен HuggingFace (нужен для загрузки gated-моделей) |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Kaggle API credentials для скачивания датасета |
| `TG_BOT_TOKEN` | Токен Telegram-бота (опционально) |
| `MLFLOW_TRACKING_URI` | URL MLflow-сервера (по умолчанию: `sqlite:///logs/mlflow.db`) |
| `API_PORT` | Порт FastAPI-сервера (по умолчанию: `8000`) |
| `LOG_LEVEL` | Уровень логирования: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `API_KEY` | Ключ для защиты эндпоинтов API (опционально) |

---

## Примечания

- Датасет используется в исследовательских и учебных целях. Формальный сабмит в соревнование не производился.
- Целевой бюджет VRAM: **8 ГБ**. Все решения по размеру батча и рангу LoRA приняты с учётом этого ограничения. Финальное обучение проводилось на Google Colab (T4, 15 ГБ).
- Сервис `trainer` в `docker-compose.yml` запускается под профилем `training` и не стартует по умолчанию — предназначен для запуска по требованию.
- MLflow по умолчанию пишет в локальный SQLite (`logs/mlflow.db`). Для продакшн-деплоя заменить на удалённый tracking server через `MLFLOW_TRACKING_URI`.
ENDOFFILE