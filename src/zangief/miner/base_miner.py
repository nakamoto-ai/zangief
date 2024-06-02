import os
import time
import argparse
import uvicorn
from os.path import dirname, realpath
from urllib.parse import urlparse
from abc import abstractmethod
from loguru import logger
from communex.module import endpoint, Module
from communex.module.server import ModuleServer
from communex.compat.key import classic_load_key
from keylimiter import TokenBucketLimiter
from typing import Optional, Any, Union

from src.zangief.miner.config import Config


def get_netuid(is_testnet):
    return 23 if is_testnet else 1


class BaseMiner(Module):

    model_name: Optional[Union[str, Any]]
    device: Optional[Union[str, Any]]
    max_length: Optional[Union[int, Any]]
    do_sample: Optional[Union[bool, Any]]
    temperature: Optional[Union[float, Any]]
    top_k: Optional[Union[int, Any]]
    no_repeat_ngram_size: Optional[Union[int, Any]]
    num_beams: Optional[Union[int, Any]]
    
    @endpoint
    def generate(self, prompt: str, source_language: str, target_language: str) -> dict[str, str]:
        start_time = time.time()
        logger.info("Generating translation... ")

        logger.info(f"Source ({source_language})")
        logger.info(f"{prompt}")

        translation = self.generate_translation(prompt, source_language, target_language)

        logger.info(f"Translation ({target_language})")
        logger.info(translation)

        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Responded in {execution_time} seconds")

        return {"answer": str(translation)}

    @staticmethod
    def get_config():
        config_file = os.path.join(f'{dirname(dirname(dirname(dirname(realpath(__file__)))))}', 'env/config.ini')
        return Config(config_file=config_file)

    @abstractmethod
    def generate_translation(self, prompt: str, source_language: str, target_language: str):
        pass

    @staticmethod
    def start_miner_server(miner):
        config_file = os.path.join(f'{dirname(dirname(dirname(dirname(realpath(__file__)))))}', 'env/config.ini')
        config = Config(config_file=config_file)
        key = classic_load_key(str(config.get_value("keyfile")))
        url = config.get_value("url")
        parsed_url = urlparse(url)

        refill_rate = 1 / 100

        use_testnet = config.get_value("isTestnet") == "1"
        if use_testnet:
            logger.info("Connecting to TEST network ... ")
        else:
            logger.info("Connecting to main network ... ")

        netuid = get_netuid(is_testnet=use_testnet)
        bucket = TokenBucketLimiter(20, refill_rate)
        server = ModuleServer(miner, key, limiter=bucket, subnets_whitelist=[netuid], use_testnet=use_testnet)
        app = server.get_fastapi_app()

        uvicorn.run(app, host=str(parsed_url.hostname), port=int(str(parsed_url.port)))
