import math


def sigmoid(x: float):
    return 1 / (1 + math.exp(-x))


def sigmoid_rewards(score_dict: dict[int, float]) -> dict[int, float]:
    mean_score = sum(score_dict.values()) / len(score_dict)

    threshold_percentage = 0.2
    threshold = mean_score * (1 + threshold_percentage)

    steepness = 8.0

    high_reward = 1.0
    low_reward = 0.01

    adjusted_scores: dict[int, float] = {}
    for model_id, score in score_dict.items():
        normalized_score = (score - threshold) * steepness
        reward_ratio = sigmoid(normalized_score)
        adjusted_score = low_reward + (high_reward - low_reward) * reward_ratio
        adjusted_scores[model_id] = adjusted_score

    return adjusted_scores