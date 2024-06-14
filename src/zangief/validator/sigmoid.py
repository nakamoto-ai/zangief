
import math
from typing import Dict


def sigmoid(x: float):
    return 1 / (1 + math.exp(-x))

def sigmoid_rewards(score_dict: dict[int, float]) -> dict[int, float]:
    """
    Adjusts the distribution of scores, such that the worst miners are punished significantly more than the rest.
    This ensures that it's unprofitable to run a low-end model, in comparison to high-end models.

    Args:
        score_dict (dict[int, float]): A dictionary mapping miner UIDs to their scores.

    Returns:
        A dictionary mapping miner UIDs to their adjusted scores.
    """
    # Calculate the mean score
    mean_score = sum(score_dict.values()) / len(score_dict)

    # Set the threshold as a percentage above the mean score
    threshold_percentage = 0.2  
    threshold = mean_score * (1 - threshold_percentage)

    steepness = 7.5  # steepness for sharper punishment

    # Set the high and low rewards
    high_reward = 1.0
    low_reward = 0.01  

    # Calculate the adjusted scores using the sigmoid function
    adjusted_scores: Dict[int, float] = {}
    for model_id, score in score_dict.items():
        if score < threshold:
            normalized_score = (threshold - score) * steepness
            penalty_ratio = sigmoid(normalized_score)
            adjusted_score = low_reward + (high_reward - low_reward) * penalty_ratio
        else:
            adjusted_score = high_reward
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