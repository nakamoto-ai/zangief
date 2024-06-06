import random
from datasets import load_dataset
from .base_dataset import BaseDataset
from loguru import logger


class CC100(BaseDataset):

    def __init__(self):
        super().__init__()
        self.languages_by_buffer_size = {
            "ar": 100_000,
            "de": 100_000,
            "en": 100_000,
            "es": 100_000,
            "fa": 100_000,
            "fr": 100_000,
            "hi": 100_000,
            "he": 100_000,
            "pt": 100_000,
            "ru": 100_000,
            "ur": 100_000,
            "vi": 100_000,
            "zh": 100_000,
        }
        language_alias = {"zh": "zh-Hans"}
        self.datasets = {}
        for language in self.languages_by_buffer_size:
            buffer_size = self.languages_by_buffer_size[language]
            dataset_language = language
            if language in language_alias:
                dataset_language = language_alias[language]
            streaming_dataset = load_dataset(
                "cc100", dataset_language, split="train", streaming=True
            )
            dataset = streaming_dataset.shuffle(
                seed=42, buffer_size=buffer_size
            ).filter(self.filter_dataset)
            logger.info(f"Loading dataset for {language}")
            # streaming_dataset = load_dataset("cc100", dataset_language, split='train')
            # dataset = streaming_dataset.shuffle(seed=42).filter(self.filter_dataset)
            buffered_dataset = self.buffer_dataset(dataset, language)
            self.datasets[language] = buffered_dataset
            logger.info(f"Loaded {language} ({len(buffered_dataset)} records)")

    @staticmethod
    def filter_dataset(example):
        text = example["text"].strip()
        return len(text) > 50

    def buffer_dataset(self, dataset, language):
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

    def get_random_record(self, language="es") -> str:
        row = random.choice(self.datasets[language])
        return row["text"]
