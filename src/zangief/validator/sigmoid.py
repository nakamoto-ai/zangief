
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
    Adjusts the distribution of scores, such that the worst miners are punished significantly more than the rest.
    This ensures that it's unprofitable to run a low-end model, in comparison to high-end models.

    Args:
        score_dict (Dict[int, float]): A dictionary mapping miner UIDs to their scores.

    Returns:
        A dictionary mapping miner UIDs to their adjusted scores.
    """
    # Calculate the mean score
    mean_score = sum(score_dict.values()) / len(score_dict)

    # Calculate standard deviation
    std_dev = (sum((x - mean_score) ** 2 for x in score_dict.values()) / len(score_dict)) ** 0.5

    # Set the steepness for the sigmoid function
    steepness = 10.0 / std_dev if std_dev > 0 else 10.0  # Avoid division by zero

    # Calculate the adjusted scores using the sigmoid function
    adjusted_scores: Dict[int, float] = {}
    for model_id, score in score_dict.items():
        normalized_score = (score - mean_score) * steepness
        adjusted_score = sigmoid(normalized_score)
        adjusted_scores[model_id] = adjusted_score

    return adjusted_scores


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