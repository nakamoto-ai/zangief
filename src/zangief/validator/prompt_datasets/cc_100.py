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

    def __init__(self):
        super().__init__()
        self.all_languages = [
            "en",
            "zht"
        ]
        self.selected_languages = random.sample(self.all_languages, 2)
        self.languages_by_buffer_size = {
            lb: 50_000 for
            lb in self.selected_languages
        }
        language_alias = {"zh": "zh-Hans", "zht": "zh-Hant"}
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
                seed=1137, buffer_size=buffer_size
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
        length_filter = len(text) > 50
        url_filter = CC100.contains_url(text)
        return length_filter and url_filter

    @staticmethod
    def contains_url(text: str) -> bool:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return not bool(url_pattern.search(text))

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