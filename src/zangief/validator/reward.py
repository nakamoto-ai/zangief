
from collections import Counter
from comet import download_model, load_from_checkpoint
from bert_score import BERTScorer
import langid
import math
from nltk.util import ngrams
from typing import List
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import Levenshtein
from typing import List, Dict, Tuple, Any


class Reward:

    def __init__(self, device: str = "cpu"):
        comet_model_path = download_model("Unbabel/wmt20-comet-qe-da")
        self.comet_model = load_from_checkpoint(comet_model_path)
        self.comet_model.eval()
        self.bert_model = BERTScorer(
            model_type="bert-base-multilingual-cased", device=device
        )
        self.sem_fluency_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.sem_fluency_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.sentiment_pipeline = pipeline('sentiment-analysis')

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

    def get_ngram_precision(self, source: str, target: str, n: int) -> float:
        target_ngrams = list(ngrams(target.split(), n))
        source_ngrams = list(ngrams(source.split(), n))

        target_counter = Counter(target_ngrams)
        source_counter = Counter(source_ngrams)

        match_count = sum(min(target_counter[ng], source_counter[ng]) for ng in target_counter)
        total_target_ngrams = len(target_ngrams)

        if total_target_ngrams == 0:
            return 0.0

        return match_count / total_target_ngrams

    def get_ngram_score(self, sources: List[str], targets: List[str]) -> List[float]:
        scores = []
        for target, source in zip(targets, sources):
            word_count = len(target.split())

            if word_count == 1:
                score = 1.0  # Max score for 1 word
            elif word_count == 2:
                score = self.get_ngram_precision(source, target, 2)  # Use bigrams for 2 words
            else:
                bigram_score = self.get_ngram_precision(source, target, 2)  # Bigram score
                trigram_score = self.get_ngram_precision(source, target, 3)  # Trigram score
                score = (bigram_score + trigram_score) / 2  # Average of bigram and trigram scores

            scores.append(score)

        return scores

    # def get_semantic_adequacy_score(self, sources: List[str], targets: List[str]) -> List[float]:
    #     model = self.sem_adequacy_model
    #
    #     # Compute embeddings for targets and sources
    #     target_embeddings = model.encode(targets, convert_to_tensor=True)
    #     source_embeddings = model.encode(sources, convert_to_tensor=True)
    #
    #     # Compute cosine similarities
    #     scores = []
    #     for target_emb, source_emb in zip(target_embeddings, source_embeddings):
    #         similarity = util.pytorch_cos_sim(target_emb, source_emb).item()
    #         normalized_similarity = (similarity + 1) / 2  # Normalize to range [0, 1]
    #         scores.append(normalized_similarity)
    #
    #     return scores

    def calculate_perplexity(self, model, tokenizer, text: str) -> float:
        encodings = tokenizer(text, return_tensors='pt')
        max_length = model.config.n_positions
        stride = 512

        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                log_likelihood = outputs.loss * trg_len

            lls.append(log_likelihood)

        perplexity = torch.exp(torch.stack(lls).sum() / end_loc)
        return perplexity.item()

    def get_semantic_fluency_score(self, sources: List[str], targets: List[str]) -> List[float]:
        model = self.sem_fluency_model
        tokenizer = self.sem_fluency_tokenizer

        scores = []
        for target, source in zip(targets, sources):
            source_perplexity = self.calculate_perplexity(model, tokenizer, source)
            target_perplexity = self.calculate_perplexity(model, tokenizer, target)

            # Normalize target perplexity by source perplexity to account for complexity
            normalized_fluency_score = target_perplexity / source_perplexity

            # Transform to ensure the score is within [0, 1]
            final_score = 1 / (1 + normalized_fluency_score)
            scores.append(final_score)

        return scores

    def get_sentiment_score(self, sources: List[str], targets: List[str]) -> List[float]:
        sentiment_pipeline = self.sentiment_pipeline

        scores = []
        for target, source in zip(targets, sources):
            target_sentiment = sentiment_pipeline(target)[0]
            source_sentiment = sentiment_pipeline(source)[0]

            # Compare sentiment labels and scores
            if target_sentiment['label'] == source_sentiment['label']:
                score = 1 - abs(target_sentiment['score'] - source_sentiment['score'])
            else:
                score = 0.0  # If sentiments are different, assign a score of 0

            # Ensure the score is within [0, 1] range
            score = max(0.0, min(score, 1.0))
            scores.append(score)

        return scores

    def calculate_levenshtein_distance(self, target: str, source: str) -> int:
        return Levenshtein.distance(target, source)

    def get_levenshtein_score(self, sources: List[str], targets: List[str]) -> List[float]:
        scores = []
        for target, source in zip(targets, sources):
            distance = self.calculate_levenshtein_distance(target, source)
            max_len = max(len(target), len(source))
            if max_len == 0:
                score = 1.0  # If both strings are empty, they are perfectly similar
            else:
                score = 1 - (distance / max_len)  # Normalize the distance to [0, 1]
            scores.append(score)
        return scores

    def get_literal_score(self, bert_score: float, comet_score: float, levenshtein_score: float) -> float:
        raw_score = bert_score / 3 + comet_score / 3 + levenshtein_score / 3
        clipped_score = min(max(raw_score, 0), 1)
        return clipped_score

    def get_contextual_score(self, ngram_score: float, semantic_fluency_score: float, sentiment_score: float):
        raw_score = ngram_score / 3 + semantic_fluency_score / 3 + sentiment_score / 3
        clipped_score = min(max(raw_score, 0), 1)
        return clipped_score

    def get_composite_score(self, literal_score: float, contextual_score: float, speed_score: float) -> float:
        raw_score = contextual_score * 0.5 + literal_score * 0.3 + speed_score * 0.2
        clipped_score = min(max(raw_score, 0), 1)
        return clipped_score

    def is_valid_response(self, target_language: str, value: str | None | Any) -> bool:
        if value is None or not isinstance(value, str):
            return False
        elif not self.is_correct_language(target_language, value):
            return False
        return True

    def is_correct_language(self, target_language: str, target: str) -> bool:
        classified_language, confidence = langid.classify(target)
        if target_language != classified_language:
            return False
        else:
            return True

    def get_speed_score(self, elapsed_times: List[float], threshold: int = 3, max_score: float = 1.0,
                        decay_factor: float = 0.1) -> List[float]:
        speed_scores = []
        for elapsed in elapsed_times:
            speed_score = max_score
            if elapsed > threshold:
                speed_score = max_score * math.exp(-decay_factor * (elapsed - threshold))
            speed_scores.append(speed_score)
        return speed_scores

    def get_scores(self, source: str, target_language: str, targets: List[Tuple[str, float]]) -> List[float]:
        cleaned_targets = []
        cleaned_times = []
        empty_indexes = []

        for index, (value, elapsed) in enumerate(targets):
            if self.is_valid_response(target_language, value):
                cleaned_targets.append(value)
                cleaned_times.append(elapsed)
            else:
                empty_indexes.append(index)

        composite_scores = []
        if len(cleaned_targets) > 0:
            sources = [source] * len(cleaned_targets)
            bert_scores = self.get_bert_score(sources, cleaned_targets)
            print(f"BERT Scores: {bert_scores}")
            comet_scores = self.get_comet_score(sources, cleaned_targets)
            print(f"COMET Scores: {comet_scores}")
            ngram_scores = self.get_ngram_score(sources, cleaned_targets)
            print(f"N-Gram Scores: {ngram_scores}")
            levenshtein_scores = self.get_levenshtein_score(sources, cleaned_targets)
            print(f"Levenshtein Scores: {levenshtein_scores}")
            semantic_fluency_scores = self.get_semantic_fluency_score(sources, cleaned_targets)
            print(f"Semantic Fluency Scores: {semantic_fluency_scores}")
            sentiment_scores = self.get_sentiment_score(sources, cleaned_targets)
            print(f"Sentiment Scores: {sentiment_scores}")
            speed_scores = self.get_speed_score(cleaned_times)
            for (target, bert_score, comet_score, ngram_score, levenshtein_score,
                 semantic_fluency_score, sentiment_score, speed_score) in zip(
                cleaned_targets, bert_scores, comet_scores, ngram_scores, levenshtein_scores,
                semantic_fluency_scores, sentiment_scores, speed_scores
            ):
                literal_score = self.get_literal_score(bert_score, comet_score, levenshtein_score)
                contextual_score = self.get_contextual_score(ngram_score, semantic_fluency_score, sentiment_score)
                composite_score = self.get_composite_score(literal_score, contextual_score, speed_score)
                if composite_score > 1:
                    composite_score = 1
                elif composite_score < 0:
                    composite_score = 0
                composite_scores.append(composite_score)

        final_scores = []
        for i in range(0, len(targets)):
            if i in empty_indexes:
                final_scores.insert(i, 0)
            else:
                score = composite_scores.pop(0)
                final_scores.insert(i, score)

        return final_scores
