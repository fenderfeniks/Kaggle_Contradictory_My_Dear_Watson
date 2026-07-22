import torch
from typing import Any
from transformers import PreTrainedTokenizerBase

class DynamicTextCollator:
    """
    Коллатор для сборки батчей, токенизации и динамического паддинга.
    Поддерживает как одиночные тексты, так и пары текстов (NLI).
    """
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int = 512,
        text_column: str = "text",
        text_pair_column: str = None,  # Добавлено для hypothesis
        target_column: str = "label",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.text_pair_column = text_pair_column # Инициализируем
        self.target_column = target_column

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [feature[self.text_column] for feature in features]
        
        # Проверяем, есть ли вторая колонка с текстом
        if self.text_pair_column and self.text_pair_column in features[0]:
            text_pairs = [feature[self.text_pair_column] for feature in features]
            batch = self.tokenizer(
                texts,
                text_pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        else:
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

        # Безопасная проверка: добавляем таргеты только если они есть в батче
        if self.target_column in features[0]:
            targets = [feature[self.target_column] for feature in features]
            batch["labels"] = torch.tensor(targets, dtype=torch.long)
            
        return batch