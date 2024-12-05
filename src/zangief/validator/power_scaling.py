
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
