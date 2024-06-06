import os
import asyncio
import concurrent.futures
import re
import time
from functools import partial
import numpy as np
import random
import argparse
import json
import math
from typing import Any, cast
from datetime import datetime

from communex.client import CommuneClient
from communex.module.client import ModuleClient
from communex._common import get_node_url
from communex.compat.key import classic_load_key
from communex.module.module import Module
from communex.types import Ss58Address
from communex.misc import get_map_modules
from substrateinterface import Keypair

from config import Config
from loguru import logger


from reward import Reward
from prompt_datasets.cc_100 import CC100


logger.add("logs/log_{time:YYYY-MM-DD}.log", rotation="1 day", level="INFO")

IP_REGEX = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+")

def set_weights(
    score_dict: dict[
        int, float
    ],  # implemented as a float score from 0 to 1, one being the best
    netuid: int,
    client: CommuneClient,
    key: Keypair,
) -> None:
    """
    Set weights for miners based on their scores.

    Args:
        score_dict: A dictionary mapping miner UIDs to their scores.
        netuid: The network UID.
        client: The CommuneX client.
        key: The keypair for signing transactions.
    """
    running_weights = get_running_weights()
    running_uids = running_weights.keys()
    new_uids = score_dict.keys()

    # Tracks registers/de-registers
    diff_registers = [n for n in new_uids if n not in running_uids] + [r for r in running_uids if r not in new_uids]
    for uid in diff_registers:
        running_weights[uid] = 0

    weighted_scores: dict[int, int] = {}
    # Filters out miners that weren't scored
    new_score_dict = {k: v for k, v in score_dict.items() if v != -1}

    abnormal_scores = new_score_dict.values()
    normal_scores = normalize_scores(abnormal_scores)

    scores = sum(normal_scores)
    new_score_uids = new_score_dict.keys()

    for uid, score in score_dict.items():
        if uid not in new_score_uids:
            weight = running_weights[uid]
        else:
            new_weight = int(score * 1000 / scores)
            old_weight = running_weights[uid].copy()
            weight = sigmoid_transition(old_weight, new_weight)
        weighted_scores[uid] = weight
        running_weights[uid] = weight

    # filter out 0 weights
    weighted_scores = {k: v for k, v in weighted_scores.items() if v != 0}

    weights = list(weighted_scores.values())
    uids = list(weighted_scores.keys())

    try:
        client.vote(key=key, uids=uids, weights=weights, netuid=netuid)
    except Exception as e:
        logger.error(f"WARNING: Failed to set weights with exception: {e}. Will retry.")
        sleepy_time = random.uniform(1, 2)
        time.sleep(sleepy_time)
        # retry with a different node
        client = CommuneClient(get_node_url())
        client.vote(key=key, uids=uids, weights=weights, netuid=netuid)
    finally:
        save_running_weights(running_weights)

def get_running_weights():
    with open('running_weights.json', 'r', encoding='utf-8') as rw_file:
        running_weights = json.load(rw_file)
    print(f"Running Weights Before Scoring: {running_weights}")
    return running_weights

def sigmoid_transition(current_weight, new_weight, num_iterations=6):
    """
    Perform a sigmoid transition from current weight to new weight.
    **Adjust 'num_iterations' according to needs**

    Parameters:
        current_weight (float): The current weight.
        new_weight (float): The desired new weight.
        num_iterations (int): Number of iterations for the transition.

    Returns:
        float: The updated weight after sigmoid transition.
    """
    sigmoid = lambda x: 1 / (1 + math.exp(-x))
    delta_weight = new_weight - current_weight
    alpha = 4 / num_iterations  # Adjusted alpha based on desired iterations
    updated_weight = current_weight
    for _ in range(num_iterations):
        transition = sigmoid(alpha * delta_weight)
        updated_weight += transition * delta_weight
    return updated_weight


def save_running_weights(running_weights):
    with open('running_weights.json', 'w', encoding='utf-8') as rw_file:
        json.dump(running_weights, rw_file)
    print(f"Running Weights After Scoring: {running_weights}")


def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        # If all scores are the same, give all ones
        return [1] * len(scores)

    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]

    return normalized_scores


def extract_address(string: str):
    """
    Extracts an address from a string.
    """
    return re.search(IP_REGEX, string)


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
        id: x.group(0).split(":") for id, x in filtered_addr.items() if x is not None
    }
    return ip_port


def get_netuid(is_testnet):
    if is_testnet:
        return 23
    else:
        return 1


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
        call_timeout: int = 20,
    ) -> None:
        super().__init__()
        self.client = client
        self.key = key
        self.netuid = netuid
        self.call_timeout = call_timeout

        self.reward = Reward()
        self.languages = [
            "ar",
            "de",
            "en",
            "es",
            "fa",
            "fr",
            "hi",
            "he",
            "pt",
            "ru",
            "ur",
            "vi",
            "zh"
        ]
        cc_100 = CC100()
        self.datasets = {
            "ar": [cc_100],
            "de": [cc_100],
            "en": [cc_100],
            "es": [cc_100],
            "fa": [cc_100],
            "fr": [cc_100],
            "hi": [cc_100],
            "he": [cc_100],
            "pt": [cc_100],
            "ru": [cc_100],
            "ur": [cc_100],
            "vi": [cc_100],
            "zh": [cc_100],
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
        connection, miner_key = miner_info
        module_ip, module_port = connection
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
        except Exception as e:
            logger.error(f"Error getting miner response: {e}")
            miner_answer = None
        return miner_answer

    def get_miners_to_query(self, miner_keys, num_miners_to_query=8):
        miner_queue = self.load_queue()
        for miner_key in miner_keys:
            if miner_key not in miner_queue:
                miner_queue.append(miner_key)
        miner_queue = [m for m in miner_queue if m in miner_keys]
        next_miners = miner_queue.copy()[:num_miners_to_query-1]
        new_miner_queue = miner_queue[num_miners_to_query:] + next_miners
        self.save_queue(new_miner_queue)
        return new_miner_queue

    def load_queue(self):
        with open('miner_queue.json', 'r', encoding='utf-8') as mq_file:
            miner_queue = json.load(mq_file)
        return miner_queue

    def save_queue(self, miner_queue):
        with open('miner_queue.json', 'w', encoding='utf-8') as mq_file:
            json.dump(miner_queue, mq_file)

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
        try:
            modules_addresses = self.get_addresses(self.client, netuid)
        except Exception as e:
            logger.error(f"Error syncing with the network: {e}")
            self.client = CommuneClient(get_node_url())
            modules_addresses = self.get_addresses(self.client, netuid)

        modules_keys = self.client.query_map_key(netuid)
        val_ss58 = self.key.ss58_address
        if val_ss58 not in modules_keys.values():
            logger.error(f"Validator key {val_ss58} is not registered in subnet")
            return None

        miners_to_query = self.get_miners_to_query(modules_keys.keys())

        modules_info: dict[int, tuple[list[str], Ss58Address]] = {}
        miner_uids = []
        modules_filtered_address = get_ip_port(modules_addresses)
        for module_id in miners_to_query:
            module_addr = modules_filtered_address.get(module_id, None)
            if not module_addr:
                continue
            modules_info[module_id] = (module_addr, modules_keys[module_id])
            miner_uids.append(module_id)

        score_dict: dict[int, float] = {}

        miner_prompt, source_language, target_language = self.get_miner_prompt()

        logger.debug("Source")
        logger.debug(source_language)
        logger.debug("Target")
        logger.debug(target_language)
        logger.debug("Prompt")
        logger.debug(miner_prompt)

        prompt = (miner_prompt, source_language, target_language)
        get_miner_prediction = partial(self._get_miner_prediction, prompt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            it = executor.map(get_miner_prediction, modules_info.values())
            miner_answers = [*it]

        scores = self.reward.get_scores(miner_prompt, target_language, miner_answers)

        logger.debug("Miner prompt")
        logger.debug(miner_prompt)
        logger.debug("Miner answers")
        logger.debug(miner_answers)
        logger.debug("Raw scores")
        logger.debug(scores)

        for uid, score in zip(modules_info.keys(), scores):
            if uid in miner_uids:
                score_dict[uid] = score
            else:
                score_dict[uid] = -1

        logger.info("Miner UIDs")
        logger.info(miner_uids)
        logger.info("Final scores")
        logger.info(scores)

        if not score_dict:
            logger.info("No miner returned a valid answer")
            return None

        set_weights(score_dict, self.netuid, self.client, self.key)

    def validation_loop(self, config: Config | None = None) -> None:
        while True:
            logger.info("Begin validator step ... ")
            asyncio.run(self.validate_step(self.netuid))

            interval = int(config.validator.get("interval"))
            logger.info(f"Sleeping for {interval} seconds ... ")
            time.sleep(interval)

    def reset_queue(self):
        with open('miner_queue.json', 'w', encoding='utf-8') as mq_file:
            json.dump([], mq_file)

    def reset_running_weights(self):
        emissions = self.get_current_emissions()
        weights = self.normalize_emissions(emissions)
        save_running_weights(weights)

    def normalize_emissions(self, emissions):
        values = emissions.values()
        min_val = min(values)
        max_val = max(values)
        normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
        normalized_emissions = zip(emissions.keys(), normalized_values)
        return normalized_emissions

    def get_current_emissions(self):
        netuid = 5
        request_dict = {
            "SubspaceModule": [
                ("Name", [netuid]),
                ("Emission", []),
                ("Incentive", []),
                ("Dividends", []),
            ],
        }
        emission_dict = {}
        result = self.client.query_batch_map(request_dict)

        emission = result["Emission"]
        netuid_emission = emission[netuid]
        validator_uids = self.get_validator_uids()
        names = result["Name"]
        highest_uid = max(names.keys())
        for uid_int in range(highest_uid + 1):
            uid = str(uid_int)
            if uid not in validator_uids:
                emission_dict[uid] = netuid_emission[uid]
        return emission_dict

    def get_validator_uids(self):
        modules = cast(dict[str, Any], get_map_modules(self.client, netuid=5))
        modules = [value for _, value in modules.items()]
        validator_uids = []
        for module in modules:
            if not (module["incentive"] == module["dividends"] == 0 or module["incentive"] > module["dividends"]):
                validator_uids.append(str(module['uid']))
        return validator_uids

    def reset(self):
        self.reset_running_weights()
        self.reset_queue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="transaction validator")
    parser.add_argument("--config", type=str, default=None, help="config file path")
    args = parser.parse_args()

    logger.info("Loading validator config ... ")
    if args.config is None:
        default_config_path = 'env/config.ini'
        config_file = default_config_path
    else:
        config_file = args.config
    config = Config(config_file=config_file)

    use_testnet = True if config.validator.get("testnet") == "1" else False
    if use_testnet:
        logger.info("Connecting to TEST network ... ")
    else:
        logger.info("Connecting to Main network ... ")
    c_client = CommuneClient(get_node_url(use_testnet=use_testnet))
    subnet_uid = get_netuid(use_testnet)
    keypair = classic_load_key(config.validator.get("keyfile"))

    validator = TranslateValidator(
        keypair,
        subnet_uid,
        c_client,
        call_timeout=20,
    )
    validator.reset()
    logger.info("Running validator ... ")
    validator.validation_loop(config)
