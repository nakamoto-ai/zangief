import time
from communex.module import Module, endpoint
from keylimiter import TokenBucketLimiter
from communex.module.server import ModuleServer
import uvicorn
from communex.compat.key import classic_load_key
from loguru import logger
from urllib.parse import urlparse
from abc import abstractmethod
from contextlib import contextmanager
from typing import Dict, Type


@contextmanager
def log_execution_time(description: str):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{description} completed in {execution_time:.2f} seconds")


def get_test_or_main(use_testnet: bool) -> str:
    if use_testnet:
        return "TEST"
    return "Main"


class BaseMiner(Module):

    @endpoint
    def score(self, bert: float, comet: float, composite: float) -> Dict[str, bool]:
        print(f"Your Scores:\n\nBERT: {bert}\nCOMET: {comet}")
        print(f"Composite Score (50% BERT + 50% COMET): {composite}")
        logger.info(f"Your Scores:\n\nBERT: {bert}\nCOMET: {comet}")
        logger.info(f"Composite Score (50% BERT + 50% COMET): {composite}")
        return {"answer": True}

    @endpoint
    def generate(self, prompt: str, source_language: str, target_language: str) -> Dict[str, str]:
        with log_execution_time("Translation Generation"):
            logger.info(f"Generating translation...")

            translation = self.generate_translation(prompt, source_language, target_language)

            logger.info(f"Source ({source_language})\n{prompt}\nTranslation ({target_language})\n{translation}")

        return {"answer": translation}

    @abstractmethod
    def generate_translation(self, prompt: str, source_language: str, target_language: str):
        pass

    @staticmethod
    def start_miner_server(miner):  # 'miner' is a child class of BaseMiner (i.e. OpenAIMiner, M2MMiner)
        key = classic_load_key(
            miner.config.get_value("keyfile"),
            password=miner.config.get_value("key_password")
        )
        url = miner.config.get_value("url")
        parsed_url = urlparse(url)

        refill_rate = 1 / 100

        use_testnet = True if miner.config.get_value("isTestnet") == "1" else False
        logger.info(f"Connecting to {get_test_or_main(use_testnet)} network ... ")

        bucket = TokenBucketLimiter(1000, refill_rate)
        server = ModuleServer(miner, key, limiter=bucket, subnets_whitelist=[23], use_testnet=use_testnet)
        app = server.get_fastapi_app()

        uvicorn.run(app, host=parsed_url.hostname, port=parsed_url.port)

    @staticmethod
    def _start_server():
        pass
