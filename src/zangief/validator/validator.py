import os
import asyncio
import concurrent.futures
import re
import time
from functools import partial
import numpy as np
import random
import argparse
from typing import cast, Any, Dict

from communex.client import CommuneClient
from communex.module.client import ModuleClient
from communex._common import get_node_url
from communex.compat.key import classic_load_key
from communex.module.module import Module
from communex.types import Ss58Address
from communex.misc import get_map_modules

from substrateinterface import Keypair

from loguru import logger

from weights_io import ensure_weights_file, write_weight_file, read_weight_file
from power_scaling import conditional_power_scaling
from reward import Reward
from prompt_datasets.cc_100 import CC100

from zangief.config.validator import ValidatorConfig

logger.add("logs/log_{time:YYYY-MM-DD}.log", rotation="1 day", level="INFO")

IP_REGEX = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+")


def extract_address(string: str):
    """
    Extracts an address from a string.
    """
    return re.search(IP_REGEX, string)


def get_miner_ip_port(client: CommuneClient, netuid: int, balances=False):
    modules = cast(dict[str, Any], get_map_modules(
        client, netuid=netuid, include_balances=balances))

    # Convert the values to a human readable format
    modules_to_list = [value for _, value in modules.items()]

    miners: list[Any] = []

    for module in modules_to_list:
        if module["incentive"] == module["dividends"] == 0:
            miners.append(module)
        elif module["incentive"] > module["dividends"]:
            miners.append(module)

    return miners


def get_ip_port(modules_adresses: dict[int, str]):
    """
    Get the IP and port information from module addresses.

    Args:
        modules_addresses: A dictionary mapping module IDs to their addresses.

    Returns:
        A dictionary mapping module IDs to their IP and port information.
    """

    filtered_addr = {id: extract_address(addr) for id, addr in modules_adresses.items()}
    ip_port = {
        id: x.group(0).split(":") if x is not None else ["0.0.0.0", "00"] for id, x in filtered_addr.items()
    }

    return ip_port


def get_netuid(is_testnet):
    if is_testnet:
        return 23
    else:
        return 13


def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        # If all scores are the same, give all ones
        return [1] * len(scores)

    # Normalize scores from 0 to 1
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]

    return normalized_scores


class TranslateValidator(Module):
    """
    A class for validating text generated by modules in a subnet.

    Attributes:
        client: The CommuneClient instance used to interact with the subnet.
        key: The keypair used for authentication.
        netuid: The unique identifier of the subnet.
        call_timeout: The timeout value for module calls in seconds (default: 60).

    Methods:
        get_modules: Retrieve all module addresses from the subnet.
        _get_miner_prediction: Prompt a miner module to generate an answer to the given question.
        _score_miner: Score the generated answer against the validator's own answer.
        get_miner_prompt: Generate a prompt for the miner modules.
        validate_step: Perform a validation step by generating questions, prompting modules, and scoring answers.
        validation_loop: Run the validation loop continuously based on the provided settings.
    """

    def __init__(
        self,
        key: Keypair,
        netuid: int,
        client: CommuneClient,
        call_timeout: int = 30,
        use_testnet: bool = False,
    ) -> None:
        super().__init__()
        self.client = client
        self.key = key
        self.netuid = netuid
        self.call_timeout = call_timeout
        self.use_testnet = use_testnet
        self.uid = None
        home_dir = os.path.expanduser("~")
        commune_dir = os.path.join(home_dir, ".commune")
        self.zangief_dir = os.path.join(commune_dir, "zangief")
        self.weights_file = os.path.join(self.zangief_dir, "weights.json")
        ensure_weights_file(zangief_dir_name=self.zangief_dir, weights_file_name=self.weights_file)
        write_weight_file(self.weights_file, {})

        self.reward = Reward()
        self.languages = []
        self.datasets = {}
        self.load_languages()

    def load_languages(self):
        cc_100 = CC100()
        self.languages = cc_100.selected_languages
        self.datasets = {
            l: [cc_100] for
            l in self.languages
        }

    def get_addresses(self, client: CommuneClient, netuid: int) -> dict[int, str]:
        """
        Retrieve all module addresses from the subnet.

        Args:
            client: The CommuneClient instance used to query the subnet.
            netuid: The unique identifier of the subnet.

        Returns:
            A dictionary mapping module IDs to their addresses.
        """
        module_addresses = client.query_map_address(netuid)
        return module_addresses

    def split_ip_port(self, ip_port):
        # Check if the input is empty or None
        if not ip_port:
            return None, None

        # Split the input string by the colon
        parts = ip_port.split(":")

        # Check if the split resulted in exactly two parts
        if len(parts) == 2:
            ip, port = parts
            return ip, port
        else:
            return None, None

    def _get_miner_prediction(
        self,
        prompt: str,
        miner_info: tuple[list[str], Ss58Address],
    ) -> str | None:
        """
        Prompt a miner module to generate an answer to the given question.

        Args:
            question: The question to ask the miner module.
            miner_info: A tuple containing the miner's connection information and key.

        Returns:
            The generated answer from the miner module, or None if the miner fails to generate an answer.
        """
        question, source_language, target_language = prompt
        connection = miner_info['address']
        miner_key = miner_info['key']
        module_ip, module_port = self.split_ip_port(connection)

        if module_ip == "None" or module_port == "None" or module_ip is None or module_port is None:
            return ""

        client = ModuleClient(module_ip, int(module_port), self.key)

        try:
            miner_answer = asyncio.run(
                client.call(
                    "generate",
                    miner_key,
                    {"prompt": question, "source_language": source_language, "target_language": target_language},
                    timeout=self.call_timeout,
                )
            )
            miner_answer = miner_answer["answer"]
            return miner_answer
        except Exception as e:
            logger.error(f"Error getting miner response: {e}")
            return ""

    def _return_miner_scores(
        self,
        score: Dict[str, float],
        miner_info: tuple[list[str], Ss58Address],
    ):
        connection = miner_info['address']
        miner_key = miner_info['key']
        module_ip, module_port = self.split_ip_port(connection)

        if module_ip == "None" or module_port == "None" or module_ip is None or module_port is None:
            return False

        client = ModuleClient(module_ip, int(module_port), self.key)

        try:
            send_miner_score = asyncio.run(
                client.call(
                    "score",
                    miner_key,
                    score,
                    timeout=10
                )
            )
            return send_miner_score['answer']
        except Exception as e:
            return False

    def get_miners_to_query(self, miners: list[dict[str, Any]]):
        current_weights = read_weight_file(self.weights_file)
        miners_to_query = []
        excluded_uids = set()
        counter = 0
        weights_changed = False

        logger.info(f"Initial SCORED_MINERS: {current_weights}")

        for miner in miners:
            uid = str(miner['uid'])
            miner_key = miner['key']

            if uid in current_weights:
                if miner_key != current_weights[uid]['ss58']:
                    # Miner has been deregistered and must be re-scored
                    del current_weights[uid]
                    weights_changed = True

                # If the miner key matches and UID is in scored_miners, exclude it because it has already been scored
                if uid in current_weights and miner_key == current_weights[uid]['ss58']:
                    excluded_uids.add(uid)
                    continue

            miners_to_query.append(miner)
            counter += 1

            if counter == 8:
                break

        remaining_miners = [miner for miner in miners if str(miner['uid']) not in excluded_uids]

        if weights_changed:
            write_weight_file(self.weights_file, current_weights)

        logger.info(f"Updated SCORED_MINERS: {current_weights}")
        logger.info(f"MINERS_TO_QUERY: {miners_to_query}")

        return remaining_miners, miners_to_query

    def get_miner_prompt(self) -> tuple:
        """
        Generate a prompt for the miner modules.

        Returns:
            The generated prompt for the miner modules.
        """
        source_language = np.random.choice(self.languages).item()
        target_languages = [language for language in self.languages if language != source_language]
        target_language = np.random.choice(target_languages).item()

        source_datasets = self.datasets[source_language]
        random_dataset_index = random.randint(0, len(source_datasets) - 1)
        source_dataset = source_datasets[random_dataset_index]

        source_text = source_dataset.get_random_record(source_language)
        return source_text, source_language, target_language

    async def validate_step(
        self, netuid: int
    ) -> None:
        """
        Perform a validation step.

        Generates questions based on the provided settings, prompts modules to generate answers,
        and scores the generated answers against the validator's own answers.

        Args:
            netuid: The network UID of the subnet.
        """

        miners = get_miner_ip_port(self.client, self.netuid)

        modules_keys = self.client.query_map_key(netuid)
        val_ss58 = self.key.ss58_address
        if val_ss58 not in modules_keys.values():
            logger.error(f"Validator key {val_ss58} is not registered in subnet")
            return None

        for uid, ss58 in modules_keys.items():
            if ss58.__str__() == val_ss58:
                self.uid = uid

        remaining_miners, miners_to_query = self.get_miners_to_query(miners)

        miner_prompt, source_language, target_language = self.get_miner_prompt()

        logger.debug("Source")
        logger.debug(source_language)
        logger.debug("Target")
        logger.debug(target_language)
        logger.debug("Prompt")
        logger.debug(miner_prompt)

        prompt = (miner_prompt, source_language, target_language)
        logger.debug("Creating miner prediction partial...")
        get_miner_prediction = partial(self._get_miner_prediction, prompt)

        logger.debug("Prompting miners...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            it = executor.map(get_miner_prediction, miners_to_query)
            miner_answers = [*it]

        scores, full_scores = self.reward.get_scores(miner_prompt, target_language, miner_answers)

        for i, full_score in enumerate(full_scores):
            send_miner_score = partial(self._return_miner_scores, full_score)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                rs = executor.map(send_miner_score, [miners_to_query[i]])
                successes = [*rs]

        logger.debug("Miner prompt")
        logger.debug(miner_prompt)
        logger.debug("Miner answers")
        logger.debug(miner_answers)
        logger.debug("Raw scores")
        logger.debug(scores)

        score_dict: dict[int, float] = {}
        for uid, score in zip([m['uid'] for m in miners_to_query], scores):
            score_dict[uid] = score

        data_to_write = {}
        logger.info(f"SCORE DICT: {score_dict}")
        for item in miners_to_query:
            ss58 = item['key']
            uid = int(item['uid'])
            score = score_dict[uid]
            data_to_write[uid] = {"ss58": ss58, "score": score}

        current_weights = read_weight_file(self.weights_file)
        for key, data in data_to_write.items():
            current_weights[key] = data

        write_weight_file(self.weights_file, current_weights)
        ddd = read_weight_file(self.weights_file)
        logger.info(f"READ DATA: {ddd}")

        logger.info("Miner UIDs")
        logger.info([m['uid'] for m in miners_to_query])
        logger.info("Final scores")
        logger.info(scores)

        if len(remaining_miners) == 0:
            scores = read_weight_file(self.weights_file)

            s_dict: dict[int: float] = {}
            for uid, data in scores.items():
                s_dict[uid] = data['score']

            logger.info("SETTING WEIGHTS")
            self.set_weights(s_dict)
            write_weight_file(self.weights_file, {})
            self.load_languages()

    def validation_loop(self, interval: int = 20) -> None:
        while True:
            logger.info("Begin validator step ... ")
            asyncio.run(self.validate_step(self.netuid))
            logger.info(f"Sleeping for {interval} seconds ... ")
            time.sleep(interval)

    def set_weights(self, s_dict):
        """
        Set weights for miners based on their normalized and power scaled scores.
        """
        full_score_dict = s_dict
        weighted_scores: dict[int: float] = {}

        abnormal_scores = full_score_dict.values()
        normal_scores = normalize_scores(abnormal_scores)
        score_dict = {uid: score for uid, score in zip(full_score_dict.keys(), normal_scores)}
        power_scaled_scores = conditional_power_scaling(score_dict)
        scores = sum(power_scaled_scores.values())

        for uid, score in power_scaled_scores.items():
            weight = score * 1000 / scores
            weighted_scores[uid] = weight

        weighted_scores = {k: v for k, v in zip(
            weighted_scores.keys(), normalize_scores(weighted_scores.values())) if v != 0}

        if self.uid is not None and str(self.uid) in weighted_scores:
            del weighted_scores[str(self.uid)]
            logger.info(f"REMOVING UID !!!!!! {self.uid}")
        else:
            logger.info("NOT REMOVING ANY UID")

        uids = list(weighted_scores.keys())
        intuids = [eval(i) for i in uids]
        weights = list(weighted_scores.values())
        intweights = [int(weight * 1000) for weight in weights]

        logger.info("**********************************")
        logger.info(f"UIDS: {intuids}")
        logger.info(f"WEIGHTS TO SET: {intweights}")
        logger.info("**********************************")

        try:
            self.client.vote(key=self.key, uids=intuids, weights=intweights, netuid=self.netuid)
        except Exception as e:
            logger.error(f"WARNING: Failed to set weights with exception: {e}. Will retry.")
            sleepy_time = random.uniform(1, 2)
            time.sleep(sleepy_time)
            # retry with a different node
            self.client = CommuneClient(get_node_url(use_testnet=self.use_testnet))
            self.client.vote(key=self.key, uids=intuids, weights=intweights, netuid=self.netuid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="zangief validator")
    parser.add_argument("--env", type=str, default=".env", help="config file path")
    parser.add_argument('--ignore-env-file', action='store_true', help='If set, ignore .env file')
    args = parser.parse_args()

    logger.info("Loading validator config ... ")

    # Load config, and get the values.
    validator_config = ValidatorConfig(env_path=args.env, ignore_config_file=args.ignore_env_file)

    testnet = validator_config.get_testnet()
    keyname = validator_config.get_key_name()
    netuid = validator_config.get_netuid()
    call_timeout = validator_config.get_validator_call_timeout()
    interval = validator_config.get_validator_interval()
    key_password = validator_config.get_key_password()

    if key_password is not None:
        key = classic_load_key(keyname, password=key_password)
    else:
        key = classic_load_key(keyname)

    if testnet:
        logger.info("Connecting to TEST network ... ")
    else:
        logger.info("Connecting to Main network ... ")

    validator = TranslateValidator(
        key=key,
        netuid=netuid,
        client=CommuneClient(get_node_url(use_testnet=testnet)),
        call_timeout=call_timeout,
        use_testnet=testnet
    )

    logger.info("Running validator ... ")
    validator.validation_loop(interval=interval)
