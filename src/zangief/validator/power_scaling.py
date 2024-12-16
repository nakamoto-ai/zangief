
from typing import Dict


def conditional_power_scaling(score_dict: Dict[int, float]) -> Dict[int, float]:
    scaling_factor = 0.2
    mean_score = sum(score_dict.values()) / len(score_dict)

    transformed_scores = []
    for uid, score in score_dict.items():
        if score > mean_score:
            transformed_score = (score / mean_score) ** (1 - scaling_factor)
        else:
            transformed_score = (score / mean_score) ** (1 + scaling_factor)

        transformed_scores.append(transformed_score)

    for uid, t in zip(score_dict.keys(), transformed_scores):
        score_dict[uid] = (t / max(transformed_scores))

    return score_dict


def conditional_cubic_scaling(score_dict: Dict[int, float], abs_bound: float = 2) -> Dict[int, float]:
    scores = list(score_dict.values())
    min_score, max_score = min(scores), max(scores)

    lower_bound = 0 - abs_bound
    upper_bound = abs_bound

    scaled_scores = []
    for score in scores:
        scaled_score = lower_bound + (upper_bound - lower_bound) * (score - min_score) / (max_score - min_score)
        scaled_scores.append(scaled_score)

    cubic_transformed_scores = [x ** 3 for x in scaled_scores]

    min_transformed = min(cubic_transformed_scores)
    max_transformed = max(cubic_transformed_scores)
    normalized_scores = [(score - min_transformed) / (max_transformed - min_transformed) for score in
                         cubic_transformed_scores]

    for uid, new_score in zip(score_dict.keys(), normalized_scores):
        score_dict[uid] = new_score

    return score_dict
