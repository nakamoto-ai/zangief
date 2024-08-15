from comet import download_model, load_from_checkpoint
from bert_score import BERTScorer
import langid


class Reward:

    def __init__(self, device="cpu"):
        comet_model_path = download_model("Unbabel/wmt20-comet-qe-da")
        self.comet_model = load_from_checkpoint(comet_model_path)
        self.comet_model.eval()
        self.bert_model = BERTScorer(
            model_type="bert-base-multilingual-cased", device=device
        )

    def get_bert_score(self, sources, targets):
        _, _, f1 = self.bert_model.score(sources, targets)
        return f1.tolist()

    def prep_comet_data(self, sources, targets):
        data = [
            {"src": source, "mt": target} for source, target in zip(sources, targets)
        ]
        return data

    def get_comet_score(self, sources, targets):
        comet_data = self.prep_comet_data(sources, targets)
        comet_scores = self.comet_model.predict(comet_data)["scores"]
        normalized_scores = [(score + 1) / 2 for score in comet_scores]
        return normalized_scores

    def get_composite_score(self, bert_score, comet_score):
        raw_score = 0.5 * bert_score + 0.5 * comet_score
        clipped_score = min(max(raw_score, 0), 1)
        return clipped_score

    def is_valid_response(self, target_language, value):
        if value is None or not isinstance(value, str):
            return False
        elif not self.is_correct_langauge(target_language, value):
            return False
        return True

    def is_correct_langauge(self, target_language, target):
        classified_language, confidence = langid.classify(target)
        if target_language != classified_language:
            return False
        else:
            return True

    def get_scores(self, source, target_language, targets):
        cleaned_targets = []
        empty_indexes = []

        for index, value in enumerate(targets):
            if self.is_valid_response(target_language, value):
                cleaned_targets.append(value)
            else:
                empty_indexes.append(index)

        composite_scores = []
        fulls = []
        if len(cleaned_targets) > 0:
            sources = [source] * len(cleaned_targets)
            bert_scores = self.get_bert_score(sources, cleaned_targets)
            comet_scores = self.get_comet_score(sources, cleaned_targets)
            for target, bert_score, comet_score in zip(
                cleaned_targets, bert_scores, comet_scores
            ):
                composite_score = self.get_composite_score(bert_score, comet_score)
                if composite_score > 1:
                    composite_score = 1
                elif composite_score < 0:
                    composite_score = 0
                composite_scores.append(composite_score)
                full = {
                    'prompt': bert_score,
                    'source_language': comet_score,
                    'target_language': composite_score,
                    'scores': True
                }
                fulls.append(full)

        final_scores = []
        full_scores = []
        for i in range(0, len(targets)):
            if i in empty_indexes:
                final_scores.insert(i, 0)
                empty_full_score = {
                    'prompt': 0,
                    'source_language': 0,
                    'target_language': 0,
                    'scores': True
                }
                full_scores.insert(i, empty_full_score)
            else:
                score = composite_scores.pop(0)
                full_score = full_scores.pop(0)
                final_scores.insert(i, score)
                full_scores.insert(i, full_score)

        return final_scores, full_scores
