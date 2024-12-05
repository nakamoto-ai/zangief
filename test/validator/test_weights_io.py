import pytest
from dataclasses import dataclass
from typing import Dict, Any
from unittest.mock import MagicMock, mock_open, patch
import os
import json
from zangief.validator.weights_io import ensure_weights_file, write_weight_file, read_weight_file

module = "zangief.validator.weights_io"


@dataclass
class EnsureWeightsFileInputData:
    zangief_dir_name: str
    weights_file_name: str


@pytest.mark.parametrize("input_data", [
    EnsureWeightsFileInputData(zangief_dir_name="zangief_dir", weights_file_name="zangief_dir/weights.json"),
])
@patch("os.makedirs")
@patch("os.path.exists")
@patch("builtins.open", new_callable=mock_open)
def test_ensure_weights_file(mock_open, mock_path_exists, mock_makedirs, input_data: EnsureWeightsFileInputData):
    mock_path_exists.side_effect = lambda path: path != input_data.zangief_dir_name and path != input_data.weights_file_name

    ensure_weights_file(input_data.zangief_dir_name, input_data.weights_file_name)

    mock_makedirs.assert_called_once_with(input_data.zangief_dir_name)
    mock_open.assert_called_once_with(input_data.weights_file_name, 'w')
    mock_open().write.assert_called_once_with('{}')


@dataclass
class WriteWeightFileInputData:
    weights_file: str
    modules_info: Dict[int, Dict[str, Any]]


@pytest.mark.parametrize("input_data", [
    WriteWeightFileInputData(
        weights_file="weights.json",
        modules_info={
            1: {"address": "address1", "score": 0.5},
            2: {"address": "address2", "score": 0.7},
        }
    )
])
@patch("builtins.open", new_callable=mock_open)
def test_write_weight_file(mock_open, input_data: WriteWeightFileInputData):
    write_weight_file(input_data.weights_file, input_data.modules_info)

    mock_open.assert_called_once_with(input_data.weights_file, 'w')

    file_handle = mock_open()
    written_content = ''.join(call.args[0] for call in file_handle.write.mock_calls)
    expected_content = json.dumps(input_data.modules_info, indent=4)

    assert written_content == expected_content



@dataclass
class ReadWeightFileInputData:
    weights_file: str
    file_exists: bool
    file_content: str
    expected_output: Dict[int, Dict[str, Any]]


@pytest.mark.parametrize("input_data", [
    ReadWeightFileInputData(
        weights_file="weights.json",
        file_exists=True,
        file_content=json.dumps({
            "1": {"address": "address1", "score": 0.5},
            "2": {"address": "address2", "score": 0.7},
        }),
        expected_output={
            "1": {"address": "address1", "score": 0.5},
            "2": {"address": "address2", "score": 0.7},
        }
    ),
    ReadWeightFileInputData(
        weights_file="weights.json",
        file_exists=False,
        file_content="",
        expected_output={}
    ),
])
@patch("os.path.exists")
@patch("builtins.open", new_callable=mock_open)
def test_read_weight_file(mock_open, mock_path_exists, input_data: ReadWeightFileInputData):
    mock_path_exists.return_value = input_data.file_exists
    mock_open.return_value.read.return_value = input_data.file_content

    result = read_weight_file(input_data.weights_file)

    if input_data.file_exists:
        mock_open.assert_called_once_with(input_data.weights_file, 'r')
    else:
        mock_open.assert_not_called()

    assert result == input_data.expected_output

