import pytest
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from unittest.mock import MagicMock, patch
import argparse

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
        return TranslateValidator(
            key=MagicMock(),
            netuid=1,
            client=MagicMock(),
            module_client=MagicMock(),
            reward=MagicMock(spec=Reward),
            cc100=mock_cc100,
        )

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

