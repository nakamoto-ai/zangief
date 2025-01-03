import pytest
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from unittest.mock import MagicMock, patch

from zangief.validator.prompt_datasets.cc_100 import CC100

module = "zangief.validator.prompt_datasets.cc_100"


@pytest.fixture
def cc100() -> CC100:
    with patch(f"{module}.load_dataset"):
        cc = CC100()
        cc.all_languages = ["en", "es", "fr", "de", "zh"]
        cc.selected_languages = ["en", "zh"]
        cc.languages_by_buffer_size = {"en": 50_000, "zh-Hans": 50_000}
        cc.datasets = {"en": [{"text": "Sample text"}], "zh-Hans": [{"text": "中文样本"}]}
    return cc


class TestCC100:
    @dataclass
    class SelectRandomLanguagesInputData:
        count: int
        all_languages: List[str]
        expected_count: int

    @pytest.mark.parametrize("input_data", [
        SelectRandomLanguagesInputData(count=3, all_languages=["en", "es", "fr"], expected_count=3),
        SelectRandomLanguagesInputData(count=2, all_languages=["en", "zh"], expected_count=2),
    ])
    def test_select_random_languages(self, cc100: CC100, input_data: SelectRandomLanguagesInputData) -> None:
        cc100.all_languages = input_data.all_languages
        selected_languages = cc100.select_random_languages(input_data.count)
        assert len(selected_languages) == input_data.expected_count
        assert set(selected_languages).issubset(set(input_data.all_languages))

    @dataclass
    class AssignBufferSizesInputData:
        selected_languages: List[str]
        expected_buffer_sizes: Dict[str, int]

    @pytest.mark.parametrize("input_data", [
        AssignBufferSizesInputData(
            selected_languages=["en", "zh"], expected_buffer_sizes={"en": 50000, "zh-Hans": 50000}),
        AssignBufferSizesInputData(
            selected_languages=["es", "fr"], expected_buffer_sizes={"es": 50000, "fr": 50000}),
    ])
    def test_assign_buffer_sizes(self, cc100: CC100, input_data: AssignBufferSizesInputData) -> None:
        cc100.selected_languages = input_data.selected_languages
        buffer_sizes = cc100.assign_buffer_sizes()
        assert buffer_sizes == input_data.expected_buffer_sizes

    @dataclass
    class ContainsUrlInputData:
        text: str
        expected_result: bool

    @pytest.mark.parametrize("input_data", [
        ContainsUrlInputData(text="This is a normal text without URL.", expected_result=False),
        ContainsUrlInputData(text="Check this out: https://example.com", expected_result=True),
        ContainsUrlInputData(text="Visit www.example.com for details.", expected_result=True),
    ])
    def test_contains_url(self, input_data: ContainsUrlInputData) -> None:
        assert CC100.contains_url(input_data.text) == input_data.expected_result

    @dataclass
    class FilterDatasetInputData:
        example: Dict[str, str]
        expected_result: bool

    @pytest.mark.parametrize("input_data", [
        FilterDatasetInputData(
            example={"text": "This is a long text example that does not contain any URLs and meets all conditions."},
            expected_result=True),
        FilterDatasetInputData(
            example={"text": "Short text without URLs."},
            expected_result=False),
        FilterDatasetInputData(
            example={"text": "This is a long text example that unfortunately contains a URL: https://example.com."},
            expected_result=False),
    ])
    def test_filter_dataset(self, input_data: FilterDatasetInputData) -> None:
        assert CC100.filter_dataset(input_data.example) == input_data.expected_result

    @dataclass
    class BufferDatasetInputData:
        dataset: List[Dict[str, str]]
        buffer_size: int
        expected_length: int

    @pytest.mark.parametrize("input_data", [
        BufferDatasetInputData(dataset=[{"text": f"Sample {i}"} for i in range(100)], buffer_size=10, expected_length=10),
        BufferDatasetInputData(dataset=[{"text": f"Sample {i}"} for i in range(5)], buffer_size=10, expected_length=5),
    ])
    def test_buffer_dataset(self, cc100: CC100, input_data: BufferDatasetInputData) -> None:
        cc100.languages_by_buffer_size = {"en": input_data.buffer_size}
        buffered_dataset = cc100.buffer_dataset(input_data.dataset, "en")
        assert len(buffered_dataset) == input_data.expected_length

    @dataclass
    class GetRandomRecordInputData:
        language: str
        datasets: Dict[str, List[Dict[str, str]]]
        expected_language_found: bool

    @pytest.mark.parametrize("input_data", [
        GetRandomRecordInputData(language="en", datasets={"en": [{"text": "Sample text"}]}, expected_language_found=True),
        GetRandomRecordInputData(language="es", datasets={}, expected_language_found=False),
    ])
    def test_get_random_record(self, cc100: CC100, input_data: GetRandomRecordInputData) -> None:
        cc100.datasets = input_data.datasets
        if input_data.expected_language_found:
            assert cc100.get_random_record(input_data.language) == "Sample text"
        else:
            with pytest.raises(KeyError):
                cc100.get_random_record(input_data.language)

    @patch(f"{module}.load_dataset")
    def test_load_datasets(self, mock_load_dataset: MagicMock) -> None:
        mock_load_dataset.return_value = MagicMock(
            shuffle=lambda seed, buffer_size: MagicMock(
                filter=lambda func: [{"text": "Sample text"}]
            )
        )
        cc100 = CC100()
        cc100.languages_by_buffer_size = {"en": 50_000}
        datasets = cc100.load_datasets()
        assert "en" in datasets
        assert datasets["en"] == [{"text": "Sample text"}]
