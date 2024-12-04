from distutils.command.clean import clean

from comet import download_model, load_from_checkpoint
from comet.models.base import CometModel
from bert_score import BERTScorer
from typing import List, Dict, Any, Tuple
import langid


def get_comet_model() -> CometModel:
    comet_model_path = download_model("Unbabel/wmt20-comet-qe-da")
    comet_model = load_from_checkpoint(comet_model_path)
    return comet_model


def get_bert_model() -> BERTScorer:
    bert_model = BERTScorer(
        model_type="bert-base-multilingual-cased"
    )
    return bert_model


class Reward:

    def __init__(self, comet_model: CometModel, bert_model: BERTScorer, device: str = "cpu"):
        self.comet_model = comet_model
        self.comet_model.eval()
        self.bert_model = bert_model
        self.bert_model.device = device

    def get_bert_score(self, sources: List[str], targets: List[str]) -> List[float]:
        _, _, f1 = self.bert_model.score(sources, targets)
        return f1.tolist()

    def prep_comet_data(self, sources: List[str], targets: List[str]) -> List[Dict[str, str]]:
        data = [
            {"src": source, "mt": target} for source, target in zip(sources, targets)
        ]
        return data

    def get_comet_score(self, sources: List[str], targets: List[str]) -> List[float]:
        comet_data = self.prep_comet_data(sources, targets)
        comet_scores = self.comet_model.predict(comet_data)["scores"]
        normalized_scores = [(score + 1) / 2 for score in comet_scores]
        return normalized_scores

    def get_composite_score(self, bert_score: float, comet_score: float) -> float:
        raw_score = 0.5 * bert_score + 0.5 * comet_score
        clipped_score = min(max(raw_score, 0), 1)
        if clipped_score > 1:
            composite_score = 1
        elif clipped_score < 0:
            composite_score = 0
        else:
            composite_score = clipped_score
        return composite_score

    def is_valid_response(self, target_language: str, value: Any) -> bool:
        if value is None or not isinstance(value, str):
            return False
        elif not self.is_correct_langauge(target_language, value):
            return False
        return True

    def is_correct_langauge(self, target_language: str, target: str) -> bool:
        classified_language, confidence = langid.classify(target)
        if target_language != classified_language:
            return False
        else:
            return True

    def get_targets_and_indexes(self, targets: List[str], target_language: str) -> Tuple[List[str], List[int]]:
        cleaned_targets = []
        empty_indexes = []
        for index, value in enumerate(targets):
            if self.is_valid_response(target_language, value):
                cleaned_targets.append(value)
            else:
                empty_indexes.append(index)
        return cleaned_targets, empty_indexes

    def get_sources_and_scores(self, source: str, cleaned_targets: List[str]) -> Tuple[List[str], List[float], List[float]]:
        sources = [source] * len(cleaned_targets)
        bert_scores = self.get_bert_score(sources, cleaned_targets)
        comet_scores = self.get_comet_score(sources, cleaned_targets)
        return sources, bert_scores, comet_scores

    def get_full_score_objects(self, cleaned_targets: List[str], bert_scores: List[float],
                               comet_scores: List[float], composite_scores: list) -> List[Dict[str, str]]:
        fulls = []
        for target, bert_score, comet_score in zip(
            cleaned_targets, bert_scores, comet_scores
        ):
            composite_score = self.get_composite_score(bert_score, comet_score)
            composite_scores.append(composite_score)
            full = {
                'bert': str(bert_score),
                'comet': str(comet_score),
                'composite': str(composite_score)
            }
            fulls.append(full)
        return fulls

    def get_final_full_scores(self, empty_indexes: List[int], composite_scores: List[float],
                              targets: List[str], fulls: List[Dict[str, str]])\
            -> Tuple[List[int], Dict[int, Dict[str, str]]]:
        final_scores = [
            0 if i in empty_indexes else composite_scores.pop(0)
            for i in range(len(targets))
        ]

        full_scores = {
            i: fulls[j]
            for j, i in enumerate([i for i in range(len(targets)) if i not in empty_indexes])
        }
        return final_scores, full_scores

    def get_scores(self, source: str, target_language: str, targets: List[str], logger)\
            -> Tuple[List[int], Dict[int, Dict[str, str]]]:
        logger.info(f"Source: {source}\nTargets: {targets}\nTarget Language: {target_language}")

        cleaned_targets, empty_indexes = self.get_targets_and_indexes(targets, target_language)

        logger.info(f"Cleaned Targets: {cleaned_targets}\nEmpty Indexes: {empty_indexes}")

        composite_scores = []

        fulls = []
        if len(cleaned_targets) > 0:
            sources, bert_scores, comet_scores = self.get_sources_and_scores(source, cleaned_targets)
            logger.info(f"Sources: {sources}\nBert Scores: {bert_scores}\nComet Scores: {comet_scores}")
            fulls = self.get_full_score_objects(cleaned_targets, bert_scores, comet_scores, composite_scores)
            logger.info(f"Fulls: {fulls}\nComposite Scores: {composite_scores}")

        final_scores, full_scores = self.get_final_full_scores(empty_indexes, composite_scores, targets, fulls)

        logger.info(f"Final Scores: {final_scores}\nFull Scores: {full_scores}")

        return final_scores, full_scores
