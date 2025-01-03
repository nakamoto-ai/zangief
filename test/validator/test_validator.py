
import asyncio
import pytest
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from unittest.mock import MagicMock, patch, Mock
import argparse
from substrateinterface import Keypair
from communex.client import CommuneClient
from communex.module.client import ModuleClient

from zangief.validator.prompt_datasets.cc_100 import CC100
from zangief.validator.client import ModuleClientFactory
from zangief.validator.reward import Reward
from zangief.validator.validator import (
    TranslateValidator,
    extract_address,
    get_miner_ip_port,
    get_ip_port,
    get_netuid,
    normalize_scores,
    all_miners_queried,
    load_validator_config,
    get_validator_args,
    get_key,
    return_net_type,
    create_validator,
    run_validator,
)

module = "zangief.validator.validator"


@pytest.fixture
def mock_validator_environment(mocker):
    mocker.patch(f"{module}.TranslateValidator", return_value="mock_validator")
    mocker.patch(f"{module}.CC100", return_value=mocker.Mock())
    mocker.patch(f"{module}.ValidatorConfig")
    mocker.patch("sys.argv", ["test_script"])


@dataclass
class ExtractAddressInputData:
    string: str
    expected_match: Optional[str]


@pytest.mark.parametrize("input_data", [
    ExtractAddressInputData(string="192.168.1.1:8080", expected_match="192.168.1.1:8080"),
    ExtractAddressInputData(string="No IP here", expected_match=None),
    ExtractAddressInputData(string="Server at 10.0.0.1:9000 is live", expected_match="10.0.0.1:9000")
])
def test_extract_address(input_data: ExtractAddressInputData) -> None:
    match = extract_address(input_data.string)
    assert (match.group(0) if match else None) == input_data.expected_match


@dataclass
class GetMinerIpPortInputData:
    modules: Dict[str, Any]
    expected_miners: List[Dict[str, Any]]


@pytest.mark.parametrize("input_data", [
    GetMinerIpPortInputData(
        modules={"module1": {"incentive": 0, "dividends": 0}, "module2": {"incentive": 5, "dividends": 2}},
        expected_miners=[{"incentive": 0, "dividends": 0}, {"incentive": 5, "dividends": 2}]
    ),
    GetMinerIpPortInputData(
        modules={"module1": {"incentive": 3, "dividends": 5}},
        expected_miners=[]
    )
])
def test_get_miner_ip_port(input_data: GetMinerIpPortInputData, mocker) -> None:
    mocker.patch(f"{module}.get_map_modules", return_value=input_data.modules)
    client = mocker.Mock()
    result = get_miner_ip_port(client, netuid=23)
    assert result == input_data.expected_miners


@dataclass
class GetIpPortInputData:
    modules_addresses: Dict[int, str]
    expected_ip_port: Dict[str, List[str]]


@pytest.mark.parametrize("input_data", [
    GetIpPortInputData(
        modules_addresses={1: "192.168.1.1:8080", 2: "10.0.0.1:9000"},
        expected_ip_port={1: ["192.168.1.1", "8080"], 2: ["10.0.0.1", "9000"]}
    ),
    GetIpPortInputData(
        modules_addresses={1: "Invalid address"},
        expected_ip_port={1: ["0.0.0.0", "00"]}
    )
])
def test_get_ip_port(input_data: GetIpPortInputData) -> None:
    result = get_ip_port(input_data.modules_addresses)
    assert result == input_data.expected_ip_port


@dataclass
class GetNetuidInputData:
    is_testnet: bool
    expected_netuid: int


@pytest.mark.parametrize("input_data", [
    GetNetuidInputData(is_testnet=True, expected_netuid=23),
    GetNetuidInputData(is_testnet=False, expected_netuid=13)
])
def test_get_netuid(input_data: GetNetuidInputData) -> None:
    result = get_netuid(input_data.is_testnet)
    assert result == input_data.expected_netuid


@dataclass
class NormalizeScoresInputData:
    scores: List[float]
    expected_normalized: List[float]


@pytest.mark.parametrize("input_data", [
    NormalizeScoresInputData(scores=[1.0, 2.0, 3.0], expected_normalized=[0.0, 0.5, 1.0]),
    NormalizeScoresInputData(scores=[5.0, 5.0, 5.0], expected_normalized=[1, 1, 1])
])
def test_normalize_scores(input_data: NormalizeScoresInputData) -> None:
    result = normalize_scores(input_data.scores)
    assert result == input_data.expected_normalized


@dataclass
class AllMinersQueriedInputData:
    remaining_miners: List[dict]
    expected_result: bool


@pytest.mark.parametrize("input_data", [
    AllMinersQueriedInputData(remaining_miners=[], expected_result=True),
    AllMinersQueriedInputData(remaining_miners=[{"id": 1}], expected_result=False)
])
def test_all_miners_queried(input_data: AllMinersQueriedInputData) -> None:
    result = all_miners_queried(input_data.remaining_miners)
    assert result == input_data.expected_result


@dataclass
class LoadValidatorConfigInputData:
    args: argparse.Namespace
    mock_config: Dict[str, Any]
    expected_config: Dict[str, Any]


@pytest.mark.parametrize("input_data", [
    LoadValidatorConfigInputData(
        args=argparse.Namespace(env="test.env", ignore_env_file=False),
        mock_config={
            'testnet': True,
            'key_name': "test_key",
            'netuid': 23,
            'call_timeout': 30,
            'interval': 10,
            'key_password': "test_pass"
        },
        expected_config={
            'testnet': True,
            'key_name': "test_key",
            'netuid': 23,
            'call_timeout': 30,
            'interval': 10,
            'key_password': "test_pass"
        }
    )
])
def test_load_validator_config(input_data: LoadValidatorConfigInputData, mocker):
    mock_validator_config = mocker.patch(f"{module}.ValidatorConfig")
    mock_validator_config.return_value.get_testnet.return_value = input_data.mock_config['testnet']
    mock_validator_config.return_value.get_key_name.return_value = input_data.mock_config['key_name']
    mock_validator_config.return_value.get_netuid.return_value = input_data.mock_config['netuid']
    mock_validator_config.return_value.get_validator_call_timeout.return_value = input_data.mock_config['call_timeout']
    mock_validator_config.return_value.get_validator_interval.return_value = input_data.mock_config['interval']
    mock_validator_config.return_value.get_key_password.return_value = input_data.mock_config['key_password']

    config = load_validator_config(input_data.args)
    assert config == input_data.expected_config


@dataclass
class RunValidatorInputData:
    validator: str
    interval: int


@pytest.mark.parametrize("input_data", [
    RunValidatorInputData(validator="mock_validator", interval=10)
])
def test_run_validator(input_data: RunValidatorInputData, mock_validator_environment, mocker):
    mock_create_validator = mocker.patch(f"{module}.create_validator", return_value=(input_data.validator, input_data.interval))
    mock_validator_instance = mocker.Mock()
    mock_create_validator.return_value = (mock_validator_instance, input_data.interval)

    run_validator()

    mock_create_validator.assert_called_once()
    mock_validator_instance.validation_loop.assert_called_once_with(interval=input_data.interval)


@dataclass
class GetValidatorArgsInputData:
    args: List[str]
    expected_env: str
    expected_ignore_env_file: bool


@pytest.mark.parametrize("input_data", [
    GetValidatorArgsInputData(args=["--env", "custom.env"], expected_env="custom.env", expected_ignore_env_file=False),
    GetValidatorArgsInputData(args=["--ignore-env-file"], expected_env=".env", expected_ignore_env_file=True),
    GetValidatorArgsInputData(args=[], expected_env=".env", expected_ignore_env_file=False),
])
def test_get_validator_args(input_data: GetValidatorArgsInputData, mocker):
    mocker.patch("sys.argv", ["test_script"] + input_data.args)
    args = get_validator_args()
    assert args.env == input_data.expected_env
    assert args.ignore_env_file == input_data.expected_ignore_env_file


@dataclass
class GetKeyInputData:
    config: Dict[str, Any]
    expected_key_name: str
    expected_password: Optional[str]


@pytest.mark.parametrize("input_data", [
    GetKeyInputData(config={'key_name': 'test_key', 'key_password': 'test_pass'}, expected_key_name="test_key",
                    expected_password="test_pass"),
    GetKeyInputData(config={'key_name': 'test_key'}, expected_key_name="test_key", expected_password=None),
])
def test_get_key(input_data: GetKeyInputData, mocker):
    mock_load_key = mocker.patch(f"{module}.classic_load_key", return_value="mock_key")
    key = get_key(input_data.config)
    mock_load_key.assert_called_once_with(input_data.expected_key_name, password=input_data.expected_password)
    assert key == "mock_key"


@dataclass
class ReturnNetTypeInputData:
    input_config: Dict[str, Any]
    expected_output: str


@pytest.mark.parametrize("input_data", [
    ReturnNetTypeInputData(input_config={'testnet': True}, expected_output="TEST"),
    ReturnNetTypeInputData(input_config={'testnet': False}, expected_output="Main"),
    ReturnNetTypeInputData(input_config={}, expected_output="Main"),
])
def test_return_net_type(input_data: ReturnNetTypeInputData):
    result = return_net_type(input_data.input_config)
    assert result == input_data.expected_output


@dataclass
class CreateValidatorInputData:
    args: argparse.Namespace
    config: Dict[str, Any]
    key: str
    expected_validator: str
    expected_interval: int


@pytest.mark.parametrize("input_data", [
    CreateValidatorInputData(
        args=argparse.Namespace(env=".env", ignore_env_file=False),
        config={
            'testnet': True,
            'key_name': "test_key",
            'netuid': 23,
            'call_timeout': 30,
            'interval': 10,
            'key_password': "test_pass"
        },
        key="mock_key",
        expected_validator="mock_validator",
        expected_interval=10
    ),
])
@patch("datasets.load_dataset")
@patch(f"comet.load_from_checkpoint")
@patch(f"{module}.get_comet_model")
def test_create_validator(mock_get_comet_model, mock_load_from_checkpoint, mock_load_dataset,
                          input_data: CreateValidatorInputData, mocker):
    mock_get_comet_model.return_value = MagicMock()
    mock_load_from_checkpoint.return_value = MagicMock()

    mock_load_dataset.return_value = MagicMock(
        shuffle=MagicMock(
            return_value=MagicMock(
                filter=MagicMock(return_value=[{"text": "Sample text"}] * 10)
            )
        )
    )
    mock_get_args = mocker.patch(f"{module}.get_validator_args", return_value=input_data.args)
    mock_load_config = mocker.patch(f"{module}.load_validator_config", return_value=input_data.config)
    mock_get_key = mocker.patch(f"{module}.get_key", return_value=input_data.key)
    mock_validator = mocker.patch(f"{module}.TranslateValidator", return_value=input_data.expected_validator)
    mock_cc100 = mocker.patch(f"{module}.CC100", return_value=mocker.Mock())

    validator, interval = create_validator()

    mock_get_args.assert_called_once()
    mock_load_config.assert_called_once_with(input_data.args)
    mock_get_key.assert_called_once_with(input_data.config)
    mock_cc100.assert_called_once()
    mock_validator.assert_called_once_with(
        key=input_data.key,
        netuid=input_data.config['netuid'],
        client=mocker.ANY,
        module_client=mocker.ANY,
        reward=mocker.ANY,
        cc100=mocker.ANY,
        call_timeout=input_data.config['call_timeout'],
        use_testnet=input_data.config['testnet']
    )
    assert validator == input_data.expected_validator
    assert interval == input_data.expected_interval


class TestTranslateValidator:
    @pytest.fixture
    def mock_cc100(self):
        mock_cc100 = MagicMock()
        mock_cc100.selected_languages = ["en", "fr"]
        return mock_cc100

    def setup_validator(self, mock_cc100):
        mock_cc100.selected_languages = ["en", "fr", "es"]

        key = MagicMock(spec=Keypair)
        netuid = 1
        client = MagicMock(spec=CommuneClient)
        module_client = MagicMock(spec=ModuleClientFactory)
        reward = MagicMock(spec=Reward)

        validator = TranslateValidator(
            key=key,
            netuid=netuid,
            client=client,
            module_client=module_client,
            reward=reward,
            cc100=mock_cc100
        )
        return validator

    @dataclass
    class SplitIpPortInputData:
        ip_port: str
        expected_output: Tuple[str, str]

    @pytest.mark.parametrize("input_data", [
        SplitIpPortInputData(ip_port="192.168.1.1:8080", expected_output=("192.168.1.1", "8080")),
        SplitIpPortInputData(ip_port="invalid_ip", expected_output=(None, None)),
        SplitIpPortInputData(ip_port="", expected_output=(None, None)),
    ])
    def test_split_ip_port(self, mock_cc100, input_data: SplitIpPortInputData):
        validator = self.setup_validator(mock_cc100)

        result = validator.split_ip_port(input_data.ip_port)
        assert result == input_data.expected_output

    @dataclass
    class IpPortInvalidInputData:
        ip: str
        port: str
        expected_result: bool

    @pytest.mark.parametrize("input_data", [
        IpPortInvalidInputData(ip=None, port="8080", expected_result=True),
        IpPortInvalidInputData(ip="192.168.1.1", port=None, expected_result=True),
        IpPortInvalidInputData(ip="192.168.1.1", port="8080", expected_result=False),
    ])
    def test_ip_port_invalid(self, mock_cc100, input_data: IpPortInvalidInputData):
        validator = self.setup_validator(mock_cc100)

        result = validator.ip_port_invalid(input_data.ip, input_data.port)
        assert result == input_data.expected_result

    @dataclass
    class MinerCallInputData:
        endpoint: str
        data: Dict[Any, Any]
        expected_result: Any

    @pytest.mark.parametrize("input_data", [
        MinerCallInputData(endpoint="score", data={"score": 0.8}, expected_result={"answer": "mock_answer"}),
    ])
    @patch("asyncio.run")
    def test_miner_call(self, mock_async_run, mock_cc100, input_data: MinerCallInputData):
        mock_async_run.return_value = {"answer": input_data.expected_result}

        validator = self.setup_validator(mock_cc100)

        result = validator.miner_call(
            endpoint=input_data.endpoint,
            client=MagicMock(),
            miner_key="miner_key",
            data=input_data.data,
            timeout=10,
        )
        assert result == input_data.expected_result

    @dataclass
    class GetMinerPromptInputData:
        languages: List[str]
        datasets: Dict[str, Any]

    @pytest.mark.parametrize("input_data", [
        GetMinerPromptInputData(
            languages=["en", "fr"],
            datasets={"en": [MagicMock(get_random_record=lambda lang: "Hello, world!")],
                      "fr": [MagicMock(get_random_record=lambda lang: "Bonjour, le monde!")]}
        ),
    ])
    @patch("random.randint", return_value=0)
    def test_get_miner_prompt(self, mock_randint, mock_cc100, input_data: GetMinerPromptInputData):
        validator = self.setup_validator(mock_cc100)
        validator.languages = input_data.languages
        validator.datasets = input_data.datasets

        source_text, source_language, target_language = validator.get_miner_prompt()

        assert source_text in ["Hello, world!", "Bonjour, le monde!"]
        assert source_language in ["en", "fr"]
        assert target_language in ["en", "fr"]
        assert source_language != target_language

    @dataclass
    class VerifyValidatorKeyInputData:
        netuid: int
        validator_registered: bool
        expected_result: bool

    @pytest.mark.parametrize("input_data", [
        VerifyValidatorKeyInputData(netuid=1, validator_registered=True, expected_result=True),
        VerifyValidatorKeyInputData(netuid=1, validator_registered=False, expected_result=False),
    ])
    def test_verify_validator_key(self, mock_cc100, input_data: VerifyValidatorKeyInputData):
        if input_data.validator_registered:
            mock_qmk_return_value = {0: "key1", 1: "key2"}
        else:
            mock_qmk_return_value = {}

        validator = self.setup_validator(mock_cc100)
        validator.key.ss58_address = "key2"
        validator.client.query_map_key.return_value = mock_qmk_return_value

        result = validator.verify_validator_key(input_data.netuid)

        assert result == input_data.expected_result

        if input_data.validator_registered:
            assert validator.uid == 1
        else:
            assert validator.uid is None

    @dataclass
    class LoadLanguagesInputData:
        selected_languages: List[str]
        expected_languages: List[str]

    @pytest.mark.parametrize("input_data", [
        LoadLanguagesInputData(selected_languages=["en", "fr", "es"], expected_languages=["en", "fr", "es"]),
    ])
    def test_load_languages(self, input_data: LoadLanguagesInputData):
        mock_cc100 = MagicMock(spec=CC100)
        mock_cc100.selected_languages = input_data.selected_languages

        validator = self.setup_validator(mock_cc100)

        validator.load_languages(mock_cc100)

        assert validator.languages == input_data.expected_languages
        assert validator.datasets == {lang: [mock_cc100] for lang in input_data.expected_languages}

    @dataclass
    class GetMinerPredictionInputData:
        prompt: Tuple[str, str, str]
        miner_info: Dict[str, str]
        miner_response: str
        expected_result: str

    @pytest.mark.parametrize("input_data", [
        GetMinerPredictionInputData(
            prompt=("Translate this", "en", "fr"),
            miner_info={"address": "127.0.0.1:8080", "key": "miner_key"},
            miner_response="Translation result",
            expected_result="Translation result",
        ),
    ])
    @patch.object(TranslateValidator, "miner_call")
    def test_get_miner_prediction(self, mock_miner_call, input_data: GetMinerPredictionInputData):
        mock_miner_call.return_value = input_data.miner_response

        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)

        result = validator._get_miner_prediction(input_data.prompt, input_data.miner_info)

        assert result == input_data.expected_result
        mock_miner_call.assert_called_once()

    @dataclass
    class ReturnMinerScoresInputData:
        score: Dict[str, float]
        miner_info: Dict[str, str]
        miner_response: bool
        expected_result: bool

    @pytest.mark.parametrize("input_data", [
        ReturnMinerScoresInputData(
            score={"accuracy": 0.95},
            miner_info={"address": "127.0.0.1:8080", "key": "miner_key"},
            miner_response=True,
            expected_result=True,
        ),
    ])
    @patch.object(TranslateValidator, "miner_call")
    def test_return_miner_scores(self, mock_miner_call, input_data: ReturnMinerScoresInputData):
        mock_miner_call.return_value = input_data.miner_response

        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)

        result = validator._return_miner_scores(input_data.score, input_data.miner_info)

        assert result == input_data.expected_result
        mock_miner_call.assert_called_once()

    @dataclass
    class GetMinersToQueryInputData:
        miners: List[Dict[str, Any]]
        current_weights: Dict[str, Dict[str, Any]]
        expected_remaining_miners: List[Dict[str, Any]]
        expected_miners_to_query: List[Dict[str, Any]]

    @pytest.mark.parametrize("input_data", [
        GetMinersToQueryInputData(
            miners=[
                {"uid": 1, "key": "key1"},
                {"uid": 2, "key": "key2"},
                {"uid": 3, "key": "key3"},
            ],
            current_weights={"1": {"ss58": "key1"}},
            expected_remaining_miners=[
                {"uid": 2, "key": "key2"},
                {"uid": 3, "key": "key3"},
            ],
            expected_miners_to_query=[
                {"uid": 2, "key": "key2"},
                {"uid": 3, "key": "key3"},
            ],
        ),
    ])
    @patch(f"{module}.read_weight_file")
    @patch(f"{module}.write_weight_file")
    def test_get_miners_to_query(self, mock_write_weight_file, mock_read_weight_file,
                                 input_data: GetMinersToQueryInputData):
        mock_read_weight_file.return_value = input_data.current_weights

        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)

        remaining_miners, miners_to_query = validator.get_miners_to_query(input_data.miners)

        assert remaining_miners == input_data.expected_remaining_miners
        assert miners_to_query == input_data.expected_miners_to_query

    @dataclass
    class PromptMinersInputData:
        miners_to_query: List[Dict[str, Any]]
        miner_responses: List[str]

    @pytest.mark.parametrize("input_data", [
        PromptMinersInputData(
            miners_to_query=[
                {"uid": 1, "key": "key1"},
                {"uid": 2, "key": "key2"},
            ],
            miner_responses=["Response1", "Response2"],
        ),
    ])
    @patch.object(TranslateValidator, "_get_miner_prediction")
    def test_prompt_miners(self, mock_get_miner_prediction, input_data: PromptMinersInputData):
        mock_get_miner_prediction.side_effect = input_data.miner_responses

        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)

        miner_answers = validator.prompt_miners(mock_get_miner_prediction, input_data.miners_to_query)

        assert miner_answers == input_data.miner_responses
        assert mock_get_miner_prediction.call_count == len(input_data.miners_to_query)

    @dataclass
    class ReturnMinerScoresInputData:
        full_scores: Dict[int, str]
        miners_to_query: List[Dict[str, Any]]

    @pytest.mark.parametrize("input_data", [
        ReturnMinerScoresInputData(
            full_scores={0: "score1", 1: "score2"},
            miners_to_query=[{"uid": 0, "key": "key1"}, {"uid": 1, "key": "key2"}],
        ),
    ])
    @patch.object(TranslateValidator, "_return_miner_scores")
    def test_return_miner_scores(self, mock_return_miner_scores, input_data: ReturnMinerScoresInputData):
        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)

        validator.return_miner_scores(input_data.full_scores, input_data.miners_to_query)

        assert mock_return_miner_scores.call_count == len(input_data.miners_to_query)

    @dataclass
    class GetScoreDictInputData:
        miners_to_query: List[Dict[str, Any]]
        scores: List[float]
        expected_result: Dict[int, float]

    @pytest.mark.parametrize("input_data", [
        GetScoreDictInputData(
            miners_to_query=[{"uid": 1}, {"uid": 2}, {"uid": 3}],
            scores=[0.9, 0.8, 0.7],
            expected_result={1: 0.9, 2: 0.8, 3: 0.7},
        ),
    ])
    def test_get_score_dict(self, input_data: GetScoreDictInputData):
        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)

        result = validator.get_score_dict(input_data.miners_to_query, input_data.scores)

        assert result == input_data.expected_result

    @dataclass
    class GetDataToWriteInputData:
        miners_to_query: List[Dict[str, Any]]
        score_dict: Dict[int, float]
        expected_result: Dict[int, Dict[str, Any]]

    @pytest.mark.parametrize("input_data", [
        GetDataToWriteInputData(
            miners_to_query=[{"uid": 1, "key": "key1"}, {"uid": 2, "key": "key2"}],
            score_dict={1: 0.9, 2: 0.8},
            expected_result={
                1: {"ss58": "key1", "score": 0.9},
                2: {"ss58": "key2", "score": 0.8},
            },
        ),
    ])
    def test_get_data_to_write(self, input_data: GetDataToWriteInputData):
        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)

        result = validator.get_data_to_write(input_data.miners_to_query, input_data.score_dict)

        assert result == input_data.expected_result

    @dataclass
    class GetCurrentWeightsInputData:
        data_to_write: Dict[int, Dict[str, Any]]
        current_weights: Dict[int, Dict[str, Any]]
        expected_result: Dict[int, Dict[str, Any]]

    @pytest.mark.parametrize("input_data", [
        GetCurrentWeightsInputData(
            data_to_write={1: {"ss58": "key1", "score": 0.9}},
            current_weights={2: {"ss58": "key2", "score": 0.8}},
            expected_result={
                2: {"ss58": "key2", "score": 0.8},
                1: {"ss58": "key1", "score": 0.9},
            },
        ),
    ])
    @patch(f"{module}.read_weight_file")
    def test_get_current_weights(self, mock_read_weight_file, input_data: GetCurrentWeightsInputData):
        mock_read_weight_file.return_value = input_data.current_weights

        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)

        result = validator.get_current_weights(input_data.data_to_write)

        assert result == input_data.expected_result

    @dataclass
    class WriteCurrentWeightsInputData:
        miners_to_query: List[Dict[str, Any]]
        score_dict: Dict[int, float]
        expected_data_to_write: Dict[int, Dict[str, Any]]

    @pytest.mark.parametrize("input_data", [
        WriteCurrentWeightsInputData(
            miners_to_query=[{"uid": 1, "key": "key1"}, {"uid": 2, "key": "key2"}],
            score_dict={1: 0.9, 2: 0.8},
            expected_data_to_write={
                1: {"ss58": "key1", "score": 0.9},
                2: {"ss58": "key2", "score": 0.8},
            },
        ),
    ])
    @patch(f"{module}.write_weight_file")
    @patch(f"{module}.read_weight_file", return_value={})
    def test_write_current_weights(self, mock_read_weight_file, mock_write_weight_file,
                                   input_data: WriteCurrentWeightsInputData):
        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)

        validator.write_current_weights(input_data.miners_to_query, input_data.score_dict)

        assert mock_write_weight_file.call_count == 2
        mock_write_weight_file.assert_any_call(validator.weights_file, input_data.expected_data_to_write)

    @dataclass
    class GetFullScoreDictInputData:
        current_weights: Dict[int, Dict[str, Any]]
        expected_result: Dict[int, float]

    @pytest.mark.parametrize("input_data", [
        GetFullScoreDictInputData(
            current_weights={1: {"ss58": "key1", "score": 0.9}, 2: {"ss58": "key2", "score": 0.8}},
            expected_result={1: 0.9, 2: 0.8},
        ),
    ])
    @patch(f"{module}.read_weight_file")
    def test_get_full_score_dict(self, mock_read_weight_file, input_data: GetFullScoreDictInputData):
        mock_read_weight_file.return_value = input_data.current_weights

        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)

        result = validator.get_full_score_dict()

        assert result == input_data.expected_result

    @patch(f"{module}.write_weight_file")
    def test_reset_validator(self, mock_write_weight_file):
        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)

        validator.reset_validator()

        assert mock_write_weight_file.call_count == 2
        mock_write_weight_file.assert_any_call(validator.weights_file, {})

    @dataclass
    class GetMinerAnswersInputData:
        prompt: str
        miners_to_query: List[Dict[str, Any]]
        miner_responses: List[str]
        expected_answers: List[str]

    @pytest.mark.parametrize("input_data", [
        GetMinerAnswersInputData(
            prompt="Translate this",
            miners_to_query=[{"uid": 1, "key": "key1"}, {"uid": 2, "key": "key2"}],
            miner_responses=["Response1", "Response2"],
            expected_answers=["Response1", "Response2"],
        ),
    ])
    @patch.object(TranslateValidator, "_get_miner_prediction")
    def test_get_miner_answers(self, mock_get_miner_prediction, input_data: GetMinerAnswersInputData):
        mock_get_miner_prediction.side_effect = input_data.miner_responses

        mock_cc100 = MagicMock()
        validator = self.setup_validator(mock_cc100)

        result = validator.get_miner_answers(input_data.prompt, input_data.miners_to_query)

        assert result == input_data.expected_answers

    @dataclass
    class GetUnweightedScoresInputData:
        full_score_dict: Dict[int, float]
        normalized_scores: Dict[int, float]
        expected_unweighted_sum: float

    @pytest.mark.parametrize("input_data", [
        GetUnweightedScoresInputData(
            full_score_dict={1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2, 5: 0.25},
            normalized_scores={1: 1.0, 2: 0.5185, 3: 0.4815, 4: 0.0, 5: 0.2106},
            expected_unweighted_sum=2.2106,
        ),
    ])
    @patch(f"{module}.normalize_scores")
    @patch(f"{module}.conditional_cubic_scaling")
    def test_get_unweighted_scores(self, mock_cubic_scaling, mock_normalize_scores,
                                   input_data: GetUnweightedScoresInputData):
        mock_normalize_scores.return_value = input_data.normalized_scores.values()
        mock_cubic_scaling.return_value = input_data.normalized_scores

        mock_cc100 = MagicMock()
        validator = self.setup_validator(mock_cc100)

        unweighted_sum, result = validator.get_unweighted_scores(input_data.full_score_dict)

        assert unweighted_sum == input_data.expected_unweighted_sum
        assert result == input_data.normalized_scores

    @dataclass
    class NormalizeWeightedScoresInputData:
        weighted_scores: Dict[int, float]
        expected_normalized_scores: Dict[int, float]

    @pytest.mark.parametrize("input_data", [
        NormalizeWeightedScoresInputData(
            weighted_scores={1: 0.9, 2: 0.7},
            expected_normalized_scores={1: 1.0, 2: 0.8},
        ),
    ])
    @patch(f"{module}.normalize_scores")
    def test_normalize_weighted_scores(self, mock_normalize_scores, input_data: NormalizeWeightedScoresInputData):
        mock_normalize_scores.return_value = input_data.expected_normalized_scores.values()

        mock_cc100 = MagicMock()
        validator = self.setup_validator(mock_cc100)

        result = validator.normalize_weighted_scores(input_data.weighted_scores)

        assert result == input_data.expected_normalized_scores

    @dataclass
    class GetWeightedScoresInputData:
        unweighted_scores: float
        power_scaled_scores: Dict[int, float]
        expected_weighted_scores: Dict[int, float]

    @pytest.mark.parametrize("input_data", [
        GetWeightedScoresInputData(
            unweighted_scores=1.6,
            power_scaled_scores={1: 0.9, 2: 0.7},
            expected_weighted_scores={1: 562.5, 2: 437.5},
        ),
    ])
    @patch.object(TranslateValidator, "normalize_weighted_scores")
    def test_get_weighted_scores(self, mock_normalize_weighted_scores, input_data: GetWeightedScoresInputData):
        mock_normalize_weighted_scores.return_value = input_data.expected_weighted_scores

        mock_cc100 = MagicMock()
        validator = self.setup_validator(mock_cc100)

        result = validator.get_weighted_scores(input_data.unweighted_scores, input_data.power_scaled_scores)

        assert result == input_data.expected_weighted_scores

    @dataclass
    class RemoveValidatorUIDInputData:
        weighted_scores: Dict[int, float]
        uid: int
        expected_scores: Dict[int, float]

    @pytest.mark.parametrize("input_data", [
        RemoveValidatorUIDInputData(
            weighted_scores={"1": 1000, "2": 800},
            uid=1,
            expected_scores={"2": 800},
        ),
    ])
    def test_remove_validator_uid(self, input_data: RemoveValidatorUIDInputData):
        mock_cc100 = MagicMock()
        validator = self.setup_validator(mock_cc100)
        validator.uid = input_data.uid

        validator.remove_validator_uid(input_data.weighted_scores)

        assert input_data.weighted_scores == input_data.expected_scores

    @dataclass
    class GetFinalUIDsWeightsInputData:
        weighted_scores: Dict[int, float]
        expected_uids: List[int]
        expected_weights: List[int]

    @pytest.mark.parametrize("input_data", [
        GetFinalUIDsWeightsInputData(
            weighted_scores={"1": 1.0, "2": 0.8},
            expected_uids=[1, 2],
            expected_weights=[1000, 0],
        ),
    ])
    def test_get_final_uids_weights(self, input_data: GetFinalUIDsWeightsInputData):
        mock_cc100 = MagicMock()
        validator = self.setup_validator(mock_cc100)

        uids, weights = validator.get_final_uids_weights(input_data.weighted_scores)

        assert uids == input_data.expected_uids
        assert weights == input_data.expected_weights

    @dataclass
    class ValidateStepInputData:
        netuid: int
        miners: List[Dict[str, Any]]

    @pytest.mark.parametrize("input_data", [
        ValidateStepInputData(
            netuid=1,
            miners=[{"uid": 1, "key": "key1"}],
        ),
    ])
    @patch.object(TranslateValidator, "verify_validator_key", return_value=True)
    @patch.object(TranslateValidator, "get_miners_to_query", return_value=([], [{"uid": 1, "key": "key1"}]))
    @patch.object(TranslateValidator, "get_miner_prompt", return_value=("Translate this", "en", "fr"))
    @patch.object(TranslateValidator, "get_miner_answers", return_value=["Translation result"])
    @patch.object(TranslateValidator, "return_miner_scores")
    @patch(f"{module}.read_weight_file", return_value={})
    @patch(f"{module}.write_weight_file")
    def test_validate_step(self, mock_write_weight_file, mock_read_weight_file, mock_return_miner_scores,
                           mock_get_miner_answers, mock_get_miner_prompt,
                           mock_get_miners_to_query, mock_verify_validator_key, input_data):
        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)
        validator.reward.get_scores = MagicMock(return_value=([100], {"1": 100}))
        validator.get_final_uids_weights = MagicMock(return_value=([1], [100]))

        asyncio.run(validator.validate_step(input_data.netuid))

        mock_verify_validator_key.assert_called_once_with(input_data.netuid)
        mock_get_miners_to_query.assert_called_once()
        mock_get_miner_prompt.assert_called_once()
        mock_get_miner_answers.assert_called_once()
        validator.reward.get_scores.assert_called_once()
        mock_return_miner_scores.assert_called_once()
        mock_write_weight_file.assert_called()

    @dataclass
    class SetWeightsInputData:
        full_score_dict: Dict[int, float]
        unweighted_scores: float
        weighted_scores: Dict[int, float]
        final_uids: List[int]
        final_weights: List[int]

    @pytest.mark.parametrize("input_data", [
        SetWeightsInputData(
            full_score_dict={1: 1000, 2: 800},
            unweighted_scores=1800,
            weighted_scores={1: 1000, 2: 800},
            final_uids=[1, 2],
            final_weights=[1000, 800],
        ),
    ])
    @patch.object(TranslateValidator, "get_full_score_dict")
    @patch.object(TranslateValidator, "get_unweighted_scores")
    @patch.object(TranslateValidator, "get_weighted_scores")
    @patch.object(TranslateValidator, "remove_validator_uid")
    @patch.object(TranslateValidator, "get_final_uids_weights")
    @patch.object(TranslateValidator, "commune_client_vote")
    def test_set_weights(self, mock_commune_client_vote, mock_get_final_uids_weights,
                         mock_remove_validator_uid, mock_get_weighted_scores,
                         mock_get_unweighted_scores, mock_get_full_score_dict, input_data):
        mock_get_full_score_dict.return_value = input_data.full_score_dict
        mock_get_unweighted_scores.return_value = (input_data.unweighted_scores, input_data.weighted_scores)
        mock_get_weighted_scores.return_value = input_data.weighted_scores
        mock_get_final_uids_weights.return_value = (input_data.final_uids, input_data.final_weights)

        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)

        validator.set_weights()

        mock_get_full_score_dict.assert_called_once()
        mock_get_unweighted_scores.assert_called_once()
        mock_get_weighted_scores.assert_called_once()
        mock_remove_validator_uid.assert_called_once()
        mock_get_final_uids_weights.assert_called_once()
        mock_commune_client_vote.assert_called_once_with(input_data.final_uids, input_data.final_weights)

    @dataclass
    class AttemptVoteInputData:
        uids: List[int]
        weights: List[int]

    @pytest.mark.parametrize("input_data", [
        AttemptVoteInputData(
            uids=[1, 2],
            weights=[1000, 800],
        ),
    ])
    @patch.object(CommuneClient, "vote")
    def test_attempt_vote(self, mock_vote, input_data: AttemptVoteInputData):
        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)

        mock_client = MagicMock(spec=CommuneClient)
        validator.client = mock_client
        validator.key = MagicMock(spec=Keypair)
        validator.netuid = 1

        validator._attempt_vote(input_data.uids, input_data.weights)

        mock_client.vote.assert_called_once_with(
            key=validator.key,
            uids=input_data.uids,
            weights=input_data.weights,
            netuid=validator.netuid
        )

    @dataclass
    class CommuneClientVoteInputData:
        uids: List[int]
        weights: List[int]
        max_retries: int
        retry_delay: float

    @pytest.mark.parametrize("input_data", [
        CommuneClientVoteInputData(
            uids=[1, 2],
            weights=[1000, 800],
            max_retries=3,
            retry_delay=0.1,
        ),
    ])
    @patch.object(TranslateValidator, "_attempt_vote", side_effect=Exception("Test retry"))
    @patch("time.sleep", side_effect=None)
    def test_commune_client_vote(self, mock_sleep, mock_attempt_vote, input_data: CommuneClientVoteInputData):
        mock_cc100 = MagicMock(spec=CC100)
        validator = self.setup_validator(mock_cc100)

        with pytest.raises(Exception):
            validator.commune_client_vote(input_data.uids, input_data.weights, max_retries=input_data.max_retries,
                                          retry_delay=input_data.retry_delay)

        mock_attempt_vote.assert_called()
        mock_sleep.assert_called()
