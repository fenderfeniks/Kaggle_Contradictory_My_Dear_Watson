# src/submit.py
import logging
import os

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm.auto import tqdm


load_dotenv()
import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.core.data.builder import NLPDataModule  # noqa: E402
from src.utils.hydra_utils import setup_config  # noqa: E402


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def generate_submission(cfg: DictConfig) -> None:
    setup_config(cfg)

    # 1. Загрузка токенизатора и модели
    logger.info("Загрузка токенизатора...")
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()

    model_builder = hydra.utils.instantiate(cfg.model.builder, tokenizer=tokenizer)
    model = model_builder.build()

    if getattr(model_builder, "loaded_from_mlflow", False):
        logger.info(
            f"Модель загружена из MLflow Production "
            f"('{cfg.model.builder.mlflow_model_name}@{cfg.model.builder.mlflow_model_alias}')"
        )
    else:
        logger.info(f"Модель загружена с базового пути: {cfg.model.builder.model_name_or_path}")

    # Подгрузка весов (LoRA/State Dict)
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        logger.info(f"Подгрузка кастомных весов из: {ckpt_path}")
        if os.path.isdir(ckpt_path) and os.path.exists(
            os.path.join(ckpt_path, "adapter_config.json")
        ):
            logger.info("Обнаружен PEFT/LoRA адаптер. Оборачиваем модель...")
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, ckpt_path)
        else:
            logger.warning("adapter_config.json не найден. Попытка загрузки как state_dict...")
            try:
                weight_path = (
                    os.path.join(ckpt_path, "pytorch_model.bin")
                    if os.path.isdir(ckpt_path)
                    else ckpt_path
                )
                state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                logger.info("Веса state_dict успешно загружены.")
            except Exception as e:
                logger.error(f"Не удалось загрузить чекпоинт. Ошибка: {e}")
                raise e

    # Авто-определение устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Модель находится на устройстве: {device}")

    # 2. Инициализация данных
    logger.info("Подготовка данных...")
    # Инициализируем DataModule (передаем конфиг данных и токенизатор)
    datamodule = NLPDataModule(data_cfg=cfg.data, tokenizer=tokenizer)
    datamodule.prepare_data()
    datamodule.setup(stage="test")

    test_loader = datamodule.test_dataloader()

    # Извлекаем оригинальные ID из тестового датасета
    test_ids = datamodule.test_dataset["id"]

    if len(test_ids) == 0:
        logger.error("Тестовый датасет пуст! Проверь сплиты данных.")
        return

    logger.info(f"Начинаем инференс для {len(test_ids)} примеров...")

    # 3. Генерация предсказаний
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating Kaggle Submission"):
            # Переносим только тензоры (input_ids, attention_mask) на GPU
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            outputs = model(**batch_gpu)

            # В соревновании NLI классы просто выбираются по максимальному логиту
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())

    # 4. Формирование Submission
    # Требования соревнования "Contradictory, My Dear Watson": колонки `id` и `prediction`
    submission_df = pd.DataFrame({"id": test_ids, "prediction": all_preds})

    output_path = "submission.csv"
    submission_df.to_csv(output_path, index=False)

    print("\n" + "=" * 50)
    logger.info(f"Файл {output_path} успешно сохранен!")
    logger.info(f"Количество строк: {len(submission_df)}")
    print("Пример сабмишена:")
    print(submission_df.head())
    print("=" * 50 + "\n")


if __name__ == "__main__":
    generate_submission()
