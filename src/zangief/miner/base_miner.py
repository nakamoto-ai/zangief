import os
import argparse
import time
from communex.module import Module, endpoint
from communex.key import generate_keypair
from keylimiter import TokenBucketLimiter
from communex.module.server import ModuleServer
import uvicorn
from communex.compat.key import classic_load_key
from config import Config
from loguru import logger
from os.path import dirname, realpath
from urllib.parse import urljoin, urlparse
from abc import abstractmethod


class BaseMiner(Module):

    @endpoint
    def generate(self, prompt: str, source_language: str, target_language: str) -> dict[str, str]:
        start_time = time.time()
        logger.info(f"Generating translation... ")

        logger.info(f"Source ({source_language})")
        logger.info(f"{prompt}")

        translation = self.generate_translation(prompt, source_language, target_language)

        logger.info(f"Translation ({target_language})")
        logger.info(translation)

        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Responded in {execution_time} seconds")

        return {"answer": translation}

    @staticmethod
    def get_config():
        config_file = os.path.join(f'{dirname(dirname(dirname(dirname(realpath(__file__)))))}', 'env/config.ini')
        config = Config(config_file=config_file)
        return config

    @abstractmethod
    def generate_translation(self, prompt: str, source_language: str, target_language: str):
        pass

    @staticmethod
    def start_miner_server(miner):
        config_file = os.path.join(f'{dirname(dirname(dirname(dirname(realpath(__file__)))))}', 'env/config.ini')
        config = Config(config_file=config_file)
        key = classic_load_key(config.get_value("keyfile"))
        url = config.get_value("url")
        parsed_url = urlparse(url)

        refill_rate = 1 / 100

        use_testnet = True if config.get_value("isTestnet") == "1" else False
        if use_testnet:
            logger.info("Connecting to TEST network ... ")
        else:
            logger.info("Connecting to main network ... ")

        bucket = TokenBucketLimiter(20, refill_rate)
        server = ModuleServer(miner, key, limiter=bucket, subnets_whitelist=[23], use_testnet=use_testnet)
        app = server.get_fastapi_app()

        uvicorn.run(app, host=parsed_url.hostname, port=parsed_url.port)
