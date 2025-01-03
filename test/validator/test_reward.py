import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch
from typing import List, Dict, Tuple, Any
from bert_score import BERTScorer
from comet.models.base import CometModel
from zangief.validator.reward import Reward, get_comet_model, get_bert_model

module = "zangief.validator.reward"


@patch(f"{module}.download_model")
@patch(f"{module}.load_from_checkpoint")
def test_get_comet_model(mock_load_from_checkpoint, mock_download_model):
    mock_download_model.return_value = "model_path"
    mock_load_from_checkpoint.return_value = MagicMock(spec=CometModel)

    model = get_comet_model()

    mock_download_model.assert_called_once_with("Unbabel/wmt20-comet-qe-da")
    mock_load_from_checkpoint.assert_called_once_with("model_path")
    assert isinstance(model, CometModel)


@patch(f"{module}.BERTScorer")
def test_get_bert_model(mock_bert_scorer):
    mock_bert_scorer.return_value = MagicMock(spec=BERTScorer)

    model = get_bert_model()

    mock_bert_scorer.assert_called_once_with(model_type="bert-base-multilingual-cased")
    assert isinstance(model, BERTScorer)


class TestReward:
    @dataclass
    class BertScoreInputData:
        sources: List[str]
        targets: List[str]
        expected_scores: List[float]

    @pytest.mark.parametrize("input_data", [
        BertScoreInputData(
            sources=["source1", "source2"],
            targets=["target1", "target2"],
            expected_scores=[0.8, 0.7],
        ),
    ])
    def test_get_bert_score(self, input_data: BertScoreInputData):
        mock_f1 = MagicMock()
        mock_f1.tolist.return_value = input_data.expected_scores

        bert_model = MagicMock()
        bert_model.score.return_value = (MagicMock(), MagicMock(), mock_f1)
        reward = Reward(MagicMock(spec=CometModel), bert_model)

        result = reward.get_bert_score(input_data.sources, input_data.targets)

        assert result == input_data.expected_scores

    @dataclass
    class PrepCometDataInputData:
        sources: List[str]
        targets: List[str]
        expected_data: List[Dict[str, str]]

    @pytest.mark.parametrize("input_data", [
        PrepCometDataInputData(
            sources=["source1", "source2"],
            targets=["target1", "target2"],
            expected_data=[
                {"src": "source1", "mt": "target1"},
                {"src": "source2", "mt": "target2"},
            ],
        ),
    ])
    def test_prep_comet_data(self, input_data: PrepCometDataInputData):
        reward = Reward(MagicMock(spec=CometModel), MagicMock(spec=BERTScorer))

        result = reward.prep_comet_data(input_data.sources, input_data.targets)

        assert result == input_data.expected_data

    @dataclass
    class CometScoreInputData:
        sources: List[str]
        targets: List[str]
        comet_scores: List[float]
        expected_scores: List[float]

    @pytest.mark.parametrize("input_data", [
        CometScoreInputData(
            sources=["source1", "source2"],
            targets=["target1", "target2"],
            comet_scores=[0.5, 0.7],
            expected_scores=[0.75, 0.85],
        ),
    ])
    def test_get_comet_score(self, input_data: CometScoreInputData):

        comet_model = MagicMock()
        comet_model.predict.return_value = {"scores": input_data.comet_scores}
        reward = Reward(comet_model, MagicMock(spec=BERTScorer))

        result = reward.get_comet_score(input_data.sources, input_data.targets)

        assert result == input_data.expected_scores

    @dataclass
    class CompositeScoreInputData:
        bert_score: float
        comet_score: float
        expected_score: float

    @pytest.mark.parametrize("input_data", [
        CompositeScoreInputData(bert_score=0.8, comet_score=0.6, expected_score=0.7),
        CompositeScoreInputData(bert_score=1.2, comet_score=1.2, expected_score=1.0),
        CompositeScoreInputData(bert_score=-0.5, comet_score=-0.3, expected_score=0.0),
    ])
    def test_get_composite_score(self, input_data: CompositeScoreInputData):
        reward = Reward(MagicMock(spec=CometModel), MagicMock(spec=BERTScorer))

        result = reward.get_composite_score(input_data.bert_score, input_data.comet_score)

        assert result == input_data.expected_score
