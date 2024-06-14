
import math
from typing import Dict


def sigmoid(x: float):
    return 1 / (1 + math.exp(-x))

import math
from typing import Dict

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def sigmoid_rewards(score_dict: Dict[int, float]) -> Dict[int, float]:
    """
    Adjusts the distribution of scores, such that the best miners are rewarded significantly more than the rest.
    This ensures that it's profitable to run a high-end model, in comparison to cheap models.

    Args:
        score_dict (dict[int, float]): A dictionary mapping miner UIDs to their scores.

    Returns:
        A dictionary mapping miner UIDs to their adjusted scores.
    """
    # Calculate the mean score
    mean_score = sum(score_dict.values()) / len(score_dict)

    # mean_score = mean(scores)
    transformed_scores = []
    for uid, score in score_dict.items():
        if score > mean_score:
            # Apply a power < 1 to scores above the mean to raise them, retaining separation
            transformed_score = (score / mean_score) ** 0.9
        else:
            # Apply a power > 1 to scores below the mean to lower them, retaining separation
            transformed_score = (score / mean_score) ** 1.1

        # Ensure the transformed score is scaled back to between 0 and 1
        transformed_scores.append(transformed_score)

    for uid, t in zip(score_dict.keys(), transformed_scores):
        score_dict[uid] = (t / max(transformed_scores))

    return score_dict


# def sigmoid_rewards(score_dict: dict[int, float]) -> dict[int, float]:
#     mean_score = sum(score_dict.values()) / len(score_dict)

#     threshold_percentage = 0.2
#     threshold = mean_score * (1 + threshold_percentage)

#     steepness = 5.0

#     high_reward = 1.0
#     low_reward = 0.01

#     adjusted_scores: dict[int, float] = {}
#     for model_id, score in score_dict.items():
#         normalized_score = (score - threshold) * steepness
#         reward_ratio = sigmoid(normalized_score)
#         adjusted_score = low_reward + (high_reward - low_reward) * reward_ratio
#         adjusted_scores[model_id] = adjusted_score

#     return adjusted_scores