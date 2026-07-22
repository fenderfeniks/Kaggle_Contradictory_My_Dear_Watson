import logging
import os
import zipfile
from datasets import load_dataset, load_from_disk
from kaggle.api.kaggle_api_extended import KaggleApi

logger = logging.getLogger(__name__)


class RawDataFetcher:
    """
    Универсальный класс для получения сырых данных.
    Проверяет локальное наличие, при необходимости скачивает с Kaggle или HuggingFace.
    """

    def __init__(
        self,
        source_type: str,
        raw_dir: str,
        dataset_name: str = None,
        file_name: str = None,
        # ИЗМЕНЕНО: token убран — Kaggle читает KAGGLE_USERNAME / KAGGLE_KEY из env сам.
        # Параметр оставлен только для HuggingFace (HF_TOKEN).
        token: str = None,
        **kwargs,
    ):
        self.source_type = source_type
        self.raw_dir = raw_dir
        self.dataset_name = dataset_name
        self.file_name = file_name
        self.token = token  # используется только в _load_hf
        self.kwargs = kwargs

    def load(self):
        """Единая точка входа для получения DatasetDict."""
        os.makedirs(self.raw_dir, exist_ok=True)

        if self.source_type == "local_csv":
            return self._load_local()
        elif self.source_type == "kaggle":
            return self._load_kaggle()
        elif self.source_type == "hf":
            return self._load_hf()
        else:
            raise ValueError(f"Неизвестный тип источника данных: {self.source_type}")

    def _load_local(self):
        file_path = os.path.join(self.raw_dir, self.file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Критическая ошибка: Локальный файл {file_path} не найден! "
                "Данные должны лежать в этой папке."
            )
        logger.info(f"Загрузка локального файла: {file_path}")
        return load_dataset("csv", data_files=file_path, **self.kwargs)

    def _load_kaggle(self):
        file_path = os.path.join(self.raw_dir, self.file_name)

        if os.path.exists(file_path):
            logger.info(f"Kaggle датасет найден локально: {file_path}. Скачивание пропущено.")
            return load_dataset("csv", data_files=file_path, **self.kwargs)

        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")

        # Если переменных нет в env, проверяем наличие файла ~/.kaggle/kaggle.json
        kaggle_json_exists = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))

        if not (username and key) and not kaggle_json_exists:
            raise EnvironmentError(
                "Не найдены ключи Kaggle API! "
                "Задайте KAGGLE_USERNAME и KAGGLE_KEY в окружении "
                "или положите файл kaggle.json в ~/.kaggle/"
            )

        # Явно передаем конфиг, если ключи заданы через переменные окружения
        if username and key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key

        logger.info(f"Скачиваем соревнование {self.dataset_name or 'contradictory-my-dear-watson'} с Kaggle...")

        api = KaggleApi()
        api.authenticate()
        
        # Скачиваем архив соревнования
        comp_name = self.dataset_name or 'contradictory-my-dear-watson'
        api.competition_download_files(comp_name, path=self.raw_dir)

        # Распаковываем
        zip_path = os.path.join(self.raw_dir, f'{comp_name}.zip')
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir)
            os.remove(zip_path)

        logger.info("Скачивание и распаковка с Kaggle завершены.")

        return load_dataset("csv", data_files=file_path, **self.kwargs)

    def _load_hf(self):
        hf_local_path = os.path.join(self.raw_dir, self.dataset_name.replace("/", "_"))

        if not os.path.exists(hf_local_path):
            logger.info(
                f"Данные не найдены локально. Скачиваем {self.dataset_name} из HuggingFace..."
            )
            # token здесь — HF_TOKEN для приватных репо, для публичных = None
            dataset = load_dataset(self.dataset_name, token=self.token, **self.kwargs)
            dataset.save_to_disk(hf_local_path)
            logger.info(f"HF датасет сохранен в {hf_local_path}")
            return dataset

        logger.info(f"HF датасет найден локально: {hf_local_path}")
        return load_from_disk(hf_local_path)