from abc import ABC, abstractmethod
import re

class BaseCleaner(ABC):
    """
    Базовый класс для всех обработчиков текста.
    Задает единый интерфейс очистки.
    """
    
    @abstractmethod
    def clean(self, text: str) -> str:
        """
        Основной метод очистки текста.
        
        Args:
            text (str): Исходный сырой текст.
            
        Returns:
            str: Очищенный текст.
        """
        pass

class RegexCleaner(BaseCleaner):
    """
    Класс для очистки текста на основе регулярных выражений.
    Удобен для удаления ссылок, HTML-тегов или спецсимволов.
    """
    
    def __init__(self, pattern: str, replacement: str = ""):
        """
        Инициализация регулярного выражения.
        
        Args:
            pattern (str): Регулярное выражение для поиска.
            replacement (str): Строка, на которую заменяем найденные совпадения.
        """
        self.pattern = re.compile(pattern)
        self.replacement = replacement

    def clean(self, text: str) -> str:
        """
        Применяет регулярное выражение к тексту.
        """
        return self.pattern.sub(self.replacement, text)

class MultilingualTextCleaner(BaseCleaner):
    """
    Безопасная очистка для многоязычных данных:
    Удаляет HTML-теги, спец-маркеры, управляющие символы и лишние пробелы, 
    но бережно сохраняет все буквы и знаки препинания разных языков.
    """
    def clean(self, text: str) -> str:
        if not text:
            return ""
        
        # 1. Удаляем HTML-теги (например, <br>, <i>)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 2. Удаляем специфичные спец-маркеры моделей
        text = re.sub(r'<\|.*?\|>', '', text)
        
        # 3. БЕЗОПАСНОЕ УДАЛЕНИЕ НЕПЕЧАТАЕМЫХ СИМВОЛОВ (Control Characters)
        # \x00-\x08, \x0b-\x0c, \x0e-\x1f, \x7f-\x9f — это технические коды, которых не должно быть в тексте.
        # Мы оставляем \n, \r и \t (они нормализуются на следующем шаге).
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # 4. Схлопываем множественные пробелы, табы и переносы строк в один пробел
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
# ===================

class TextCleaningPipeline:
    """
    Пайплайн, объединяющий несколько шагов очистки в один вызов.
    """
    
    def __init__(self, cleaners: list[BaseCleaner]):
        """
        Инициализирует пайплайн списком клинеров.
        
        Args:
            cleaners (list[BaseCleaner]): Список инстансов классов-наследников BaseCleaner.
        """
        self.cleaners = cleaners

    def __call__(self, text: str) -> str:
        """
        Прогоняет текст через все клинеры по очереди.
        """
        for cleaner in self.cleaners:
            text = cleaner.clean(text)
        return text