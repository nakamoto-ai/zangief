import random
import re
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset
from .base_dataset import BaseDataset
from loguru import logger
from zangief.validator.reward import Reward
from typing import Any, Dict, List, Union


class CC100(BaseDataset):
    LANGUAGE_ALIAS = {"zh": "zh-Hans", "zht": "zh-Hant"}

    def __init__(self):
        super().__init__()
        self.all_languages = self.initialize_all_languages()
        self.selected_languages = self.select_random_languages(10)
        self.languages_by_buffer_size = self.assign_buffer_sizes()
        self.datasets = self.load_datasets()

    def initialize_all_languages(self) -> List[str]:
        return [
            "ar", "bn", "cs", "de", "el", "en", "es", "fa", "fr",
            "he", "hi", "hu", "it", "ja", "jv", "ko", "my", "nl",
            "pa", "pl", "pt", "ro", "ru", "sv", "ta", "te", "th",
            "tr", "uk", "ur", "vi", "zh"
        ]

    def select_random_languages(self, count: int) -> List[str]:
        return random.sample(self.all_languages, count)

    def assign_buffer_sizes(self) -> Dict[str, int]:
        return {lang: 50_000 for lang in self.selected_languages}

    def load_datasets(self) -> Dict[str, List[str]]:
        datasets = {}
        for language, buffer_size in self.languages_by_buffer_size.items():
            dataset_language = self.LANGUAGE_ALIAS.get(language, language)
            logger.info(f"Loading dataset for {language}")
            datasets[language] = self.prepare_dataset(dataset_language, buffer_size)
            logger.info(f"Loaded {language} ({len(datasets[language])} records)")
        return datasets

    def prepare_dataset(self, language: str, buffer_size: int) -> List[str]:
        streaming_dataset = load_dataset("cc100", language, split="train", streaming=True)
        dataset = streaming_dataset.shuffle(seed=1137, buffer_size=buffer_size).filter(self.filter_dataset)
        return self.buffer_dataset(dataset, language)

    @staticmethod
    def filter_dataset(example: Dict[str]) -> bool:
        text = example["text"].strip()
        length_filter = len(text) > 50
        url_filter = CC100.contains_url(text)
        return length_filter and url_filter

    @staticmethod
    def contains_url(text: str) -> bool:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return not bool(url_pattern.search(text))

    def buffer_dataset(self, dataset: Dict[str, Any], language: str) -> List[Any]:
        buffer_size = self.languages_by_buffer_size[language]
        buffer = []
        try:
            for item in dataset:
                if len(buffer) < buffer_size:
                    buffer.append(item)
                else:
                    break
        except StopIteration:
            pass
        return buffer

    def get_random_record(self, language: str = "es") -> str:
        row = random.choice(self.datasets[language])
        return row["text"]
