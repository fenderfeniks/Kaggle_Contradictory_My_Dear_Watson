# Contradictory, My Dear Watson — Production NLI Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-8A2BE2)](https://github.com/huggingface/peft)
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

Production-ready фреймворк для NLP-классификации, построенный на основе учебного Kaggle-датасета. Проект выходит за рамки обычной соревновательной задачи — вокруг эксперимента выстроен полноценный MLOps-стек: сервер инференса на FastAPI, трекинг экспериментов в MLflow, оркестрация через Airflow, мониторинг Prometheus/Grafana и деплой в Docker/Kubernetes.

---

## Структура проекта

```
.
├── configs/                  # Иерархия конфигов Hydra (модель, данные, обучение, API)
├── dags/                     # DAG-и Apache Airflow для оркестрации пайплайна
├── demo/                     # Streamlit-демо + отдельный Dockerfile
├── deploy/                   # Переменные Airflow, K8s-манифесты
├── notebooks/                # Ноутбуки для EDA и исследований
├── src/                      # Основной код (обучение, инференс, API, данные)
├── tests/                    # Тест-сьют на pytest
├── .github/workflows/        # CI/CD пайплайны
├── docker-compose.yml        # Полный локальный стек (API, Trainer, Airflow, Prometheus, Grafana)
├── Dockerfile                # Многоцелевая сборка (api / training)
├── Makefile                  # Task runner для разработчика
├── pyproject.toml            # Зависимости (uv), конфиг Ruff
└── airflow-values.yaml       # Helm values для Airflow на Kubernetes
```

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

**Многоязычная очистка текста:** наивный анализатор отметил ~33% токенов как «непечатаемые» — ложноположительное срабатывание на китайских, арабских и кириллических символах. Пайплайн был настроен на удаление только системных управляющих символов (`\x00–\x08`, `\x0e–\x1f`) с сохранением всех языковых особенностей Unicode.

**Ключевое архитектурное решение:** подача посылки и гипотезы единой конкатенированной строкой давала ~33% accuracy (случайное угадывание). Был реализован кастомный `DynamicTextCollator`, передающий оба текста как `text` и `text_pair` напрямую в токенизатор HuggingFace — тот корректно вставляет `[SEP]` и генерирует `token_type_ids` для cross-attention.

---

### Выбор модели — Smoke Test (Этап 1)

Четыре предобученные NLI-модели протестированы в режиме быстрого прогона (head-only fine-tuning, 3 эпохи, 800 обучающих примеров):

| Модель | Train F1 | Val F1 | Test Acc | Test F1 | Test Loss | Время |
|--------|----------|--------|----------|---------|-----------|-------|
| `cross-encoder/nli-deberta-v3-base` | 0.733 | 0.700 | 0.756 | **0.757** | 0.650 | 6.6 мин |
| `sileod/deberta-v3-base-tasksource-nli` | 0.769 | **0.768** | 0.674 | 0.676 | 0.669 | 6.4 мин |
| `symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli` | 0.454 | 0.436 | 0.470 | 0.462 | 1.063 | 21.0 мин |
| `tasksource/ModernBERT-large-nli` | 0.410 | 0.451 | 0.393 | 0.377 | 1.234 | 2.8 мин |

Архитектуры на базе DeBERTa v3 доминируют — их Disentangled Self-Attention особенно хорошо подходит для задачи перекрёстного внимания между посылкой и гипотезой. XLM-RoBERTa и ModernBERT не смогли адаптироваться в режиме head-only (результаты близки к случайному угадыванию) и требуют полной разморозки сети.

### Выбор модели — Расширенный тест (Этап 2)

Два финалиста на DeBERTa протестированы на увеличенном объёме данных (4 эпохи, 2000 примеров):

| Метрика | `sileod` (Модель A) | `cross-encoder` (Модель B) |
|---------|---------------------|---------------------------|
| Train F1 | 0.768 | **0.787** |
| Val F1 | 0.727 | **0.762** |
| Val Loss | 0.590 | **0.564** |
| Test F1 | **0.760** | 0.739 |
| Время обучения | 23.9 мин | **19.4 мин** |

**Победитель: `cross-encoder/nli-deberta-v3-base`**

`cross-encoder` оказался ~19% быстрее, показал лучшую динамику train/val и более низкий val loss, свидетельствующий об уверенности модели в предсказаниях. Небольшое преимущество `sileod` на тестовой выборке (400 примеров) признано статистически незначимым из-за малого размера выборки.

---

### Настройка LoRA / PEFT

**Шаг 1 — Коэффициент масштабирования (соотношение α/r):**

Анализ норм градиентов при фиксированном `r=8` для соотношений α/r: 1.0, 2.0, 4.0, 8.0. Выбрано соотношение **2.0** (α=16, r=8) — достаточная амплитуда градиентов для быстрой сходимости без риска NaN.

**Шаг 2 — Целевые модули (Overfitting Probe):**

100-шаговый стресс-тест (LR=1e-3) на одном батче для оценки выразительности против числа параметров:

| Целевые модули | Обучаемых параметров | Start Loss | End Loss | Train Acc |
|----------------|---------------------|-----------|---------|-----------|
| query_proj, value_proj | 0.16% | 6.1764 | 0.0005 | 1.0 |
| query_proj, key_proj, value_proj | 0.24% | 5.9207 | 0.0004 | 1.0 |
| query_proj, value_proj, dense | 0.64% | 5.9017 | 0.0002 | 1.0 |
| query_proj, key_proj, value_proj, dense | 0.72% | 5.6413 | 0.0001 | 1.0 |

**Выбрано: `query_proj, key_proj, value_proj`** — все конфигурации сошлись к accuracy 1.0; добавление `dense` утраивает число параметров ради пренебрежимо малого прироста (0.0004 → 0.0001). Включение `key_proj` позволяет механизму Disentangled Attention полноценно перестраивать фокус внимания между посылкой и гипотезой при спартанском объёме весов адаптера.

**Итоговый конфиг LoRA:** `r=8`, `α=16`, `dropout=0.1`, целевые модули: `query_proj, key_proj, value_proj`

---

### Направления для дальнейшей оптимизации (TODO)

Первый прогон (5 эпох) зафиксировал Test F1 ~0.76, однако val_loss начинает расти после ~1000-го шага, тогда как train_f1 продолжает тянуться к 0.88 — явное переобучение на ограниченном датасете (12 120 обучающих примеров).

Запланированные улучшения:

- **Полный датасет + Early Stopping** — переход от отладочных срезов к полной выборке с ранней остановкой по минимуму `val_loss`.
- **Регуляризация размером батча** — снижение с 16 до 8; меньший батч добавляет стохастический шум в градиенты как естественную регуляризацию, оставаясь в рамках 8 ГБ VRAM.
- **Снижение ранга LoRA** — уменьшить `r` с 8 до 4 или 2 (текущий ~1M обучаемых параметров избыточен для 12K примеров); протестировать `lora_dropout=0.2`.
- **Аугментация для мультиязычности** — логика NLI на редких языках недопредставлена; варианты: аугментация машинным переводом или переход на бэкбон, предобученный на корпусах масштаба XNLI.

---

## Быстрый старт

### Требования

- Docker + Docker Compose
- NVIDIA GPU (опционально, рекомендуется; есть CPU-фолбэк)
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

### 4. Запуск сервера инференса

```bash
make api
# Запускает: docker compose up -d --build api
# API доступен на http://localhost:8000
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

### 6. Остановка

```bash
make down
```

---

## Стек технологий

| Слой | Инструменты |
|------|-------------|
| **Модель** | `cross-encoder/nli-deberta-v3-base`, HuggingFace Transformers |
| **Обучение** | PyTorch Lightning, PEFT (LoRA), MLflow, Optuna |
| **Конфигурация** | Hydra |
| **API** | FastAPI + Uvicorn, SlowAPI (rate limiting), Redis |
| **Бот** | Aiogram 3 (Telegram) |
| **Мониторинг** | Prometheus + Grafana |
| **Оркестрация** | Apache Airflow + Kubernetes (Helm) |
| **Контейнеризация** | Docker, Docker Compose |
| **Пакетный менеджер** | `uv` + `pyproject.toml` |
| **Линтер** | Ruff |
| **Тесты** | pytest, pytest-asyncio |
| **Демо** | Streamlit |

---

## Переменные окружения

Скопировать `.env.example` в `.env` и заполнить:

| Переменная | Описание |
|------------|----------|
| `HF_TOKEN` | Токен HuggingFace (нужен для загрузки gated-моделей) |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Kaggle API credentials для скачивания датасета |
| `TG_BOT_TOKEN` | Токен Telegram-бота (опционально) |
| `MLFLOW_TRACKING_URI` | URL MLflow-сервера (по умолчанию: `http://localhost:5000`) |
| `API_PORT` | Порт FastAPI-сервера (по умолчанию: `8000`) |
| `LOG_LEVEL` | Уровень логирования: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Запуск тестов

```bash
pytest
```

Конфигурация в `pyproject.toml` (`testpaths = ["tests"]`, `asyncio_mode = "auto"`).

---

## Примечания

- Датасет используется в исследовательских и учебных целях. Сабмит в соревнование не производился.
- Целевой бюджет VRAM: **8 ГБ** (потребительские GPU класса RTX). Все решения по размеру батча и рангу LoRA приняты с учётом этого ограничения.
- Сервис `trainer` в `docker-compose.yml` запускается под профилем `training` и не стартует по умолчанию — он предназначен для запуска обучения по требованию.
