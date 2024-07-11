import random
import re
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset
from .base_dataset import BaseDataset
from loguru import logger
from reward_script import Reward
from googletrans import Translator
from typing import Any, Dict, List, Union


class CC100(BaseDataset):

    def __init__(self):
        super().__init__()
        self.languages_by_buffer_size = {
            "ar": 120_000,
            "de": 120_000,
            "en": 120_000,
            "es": 120_000,
            "fa": 120_000,
            "fr": 120_000,
            "hi": 120_000,
            "he": 120_000,
            "it": 120_000,
            "nl": 120_000,
            "pl": 120_000,
            "pt": 120_000,
            "ru": 120_000,
            "ur": 120_000,
            "vi": 120_000,
            "zh": 120_000,
        }
        language_alias = {"zh": "zh-Hans"}
        self.datasets = {}
        self.reward_model = Reward(device="cpu")
        self.composite_score_threshold = 0.5
        self.translator = Translator()

        self.english_dataset = load_dataset("cc100", "en", split="train", streaming=True).shuffle(
            seed=1137, buffer_size=120_000
        ).filter(self.filter_dataset)
        self.english_buffered = self.buffer_dataset(self.english_dataset, "en")

        for language in self.languages_by_buffer_size:
            if language == "en":
                continue
            buffer_size = self.languages_by_buffer_size[language]
            dataset_language = language
            if language in language_alias:
                dataset_language = language_alias[language]
            streaming_dataset = load_dataset(
                "cc100", dataset_language, split="train", streaming=True
            )
            dataset = streaming_dataset.shuffle(
                seed=1137, buffer_size=buffer_size
            ).filter(self.filter_dataset_with_source)
            logger.info(f"Loading dataset for {language}")
            buffered_dataset = self.buffer_dataset(dataset, language)
            self.datasets[language] = buffered_dataset
            logger.info(f"Loaded {language} ({len(buffered_dataset)} records)")

    @staticmethod
    def contains_url(text: str) -> bool:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return bool(url_pattern.search(text))

    @staticmethod
    def is_truncated(source_text: str, target_text: str, threshold: float = 0.7) -> bool:
        return len(target_text) < len(source_text) * threshold

    @staticmethod
    def filter_dataset(example: Dict[str, str]) -> bool:
        text = example["text"].strip()
        if len(text) <= 50:
            return False
        if CC100.contains_url(text):
            return False
        return True

    def translate_to_english(self, text: str, src_language: str) -> str:
        translation = self.translator.translate(text, src=src_language, dest='en')
        return translation.text

    def filter_dataset_with_source(self, example: Dict[str, str]) -> bool:
        text = example["text"].strip()
        if len(text) <= 50:
            return False
        if CC100.contains_url(text):
            return False

        translated_text = self.translate_to_english(text, example["language"])

        for source_example in self.english_buffered:
            source_text = source_example["text"].strip()
            if self.is_truncated(source_text, translated_text):
                return False

        source_texts = [source_example["text"].strip() for source_example in self.english_buffered]
        target_texts = [translated_text] * len(source_texts)
        bert_scores = self.reward_model.get_bert_score(source_texts, target_texts)
        comet_scores = self.reward_model.get_comet_score(source_texts, target_texts)
        for bert_score, comet_score in zip(bert_scores, comet_scores):
            composite_score = self.reward_model.get_composite_score(bert_score, comet_score)
            if composite_score < self.composite_score_threshold:
                return False

        return True

    def buffer_dataset(
        self,
        dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
        language: str
    ) -> List[Dict[str, Any]]:
        buffer_size = self.languages_by_buffer_size[language]
        buffer = []
        try:
            for item in dataset:
                item["language"] = language
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
