import pytest
from dataclasses import dataclass
from typing import Dict
from zangief.validator.power_scaling import conditional_power_scaling


@dataclass
class ConditionalPowerScalingInputData:
    score_dict: Dict[int, float]
    expected_output: Dict[int, float]


@pytest.mark.parametrize("input_data", [
    ConditionalPowerScalingInputData(
        score_dict={1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2},
        expected_output={1: 1.0, 2: 0.79, 3: 0.53, 4: 0.228},
    ),
    ConditionalPowerScalingInputData(
        score_dict={1: 1.0, 2: 0.5, 3: 0.3},
        expected_output={1: 1.0, 2: 0.53, 3: 0.289},
    ),
    ConditionalPowerScalingInputData(
        score_dict={1: 0.4, 2: 0.6},
        expected_output={1: 0.66, 2: 1.0},
    ),
])
def test_conditional_power_scaling(input_data: ConditionalPowerScalingInputData):
    result = conditional_power_scaling(input_data.score_dict.copy())

    assert set(result.keys()) == set(input_data.expected_output.keys())
    for key in result.keys():
        assert pytest.approx(result[key], rel=1e-2) == input_data.expected_output[key]
