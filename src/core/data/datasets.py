from datasets import Dataset as HFDataset
from typing import Callable, Optional

class NLPDatasetAdapter:
    """
    Адаптер для параллельной и батчевой обработки HuggingFace datasets.
    """
    def __init__(
        self, 
        hf_dataset: HFDataset, 
        text_column: str = "text",
        text_pair_column: Optional[str] = None, # Добавлено
        cleaning_pipeline: Optional[Callable[[str], str]] = None,
        num_proc: int = 4,
        batch_size: int = 1000
    ):
        self.dataset = hf_dataset
        self.text_column = text_column
        self.text_pair_column = text_pair_column # Инициализируем
        self.cleaning_pipeline = cleaning_pipeline
        self.num_proc = num_proc
        self.batch_size = batch_size

    def prepare_dataset(self) -> HFDataset:
        if not self.cleaning_pipeline:
            return self.dataset

        def _apply_cleaning_batched(examples: dict) -> dict:
            # Очистка первой колонки (premise)
            examples[self.text_column] = [
                self.cleaning_pipeline(text) for text in examples[self.text_column]
            ]
            # Очистка второй колонки (hypothesis), если она задана
            if self.text_pair_column and self.text_pair_column in examples:
                examples[self.text_pair_column] = [
                    self.cleaning_pipeline(text) for text in examples[self.text_pair_column]
                ]
            return examples

        return self.dataset.map(
            _apply_cleaning_batched, 
            batched=True,
            batch_size=self.batch_size, 
            num_proc=self.num_proc, 
            desc="Cleaning text (Batched)"
        )