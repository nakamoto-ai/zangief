import time
from communex.module import Module, endpoint
from keylimiter import TokenBucketLimiter
from communex.module.server import ModuleServer
import uvicorn
from communex.compat.key import classic_load_key
from loguru import logger
from urllib.parse import urlparse
from abc import abstractmethod


class BaseMiner(Module):

    @endpoint
    def score(self, bert: float, comet: float, composite: float):
        logger.info("Your Scores:\n")
        logger.info(f"BERT: {bert}")
        logger.info(f"COMET: {comet}")
        logger.info(f"Composite Score (50% BERT + 50% COMET): {composite}")

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

    @abstractmethod
    def generate_translation(self, prompt: str, source_language: str, target_language: str):
        pass

    @staticmethod
    def start_miner_server(miner):
        key = classic_load_key(miner.config.get_value("keyfile"))
        url = miner.config.get_value("url")
        parsed_url = urlparse(url)

        refill_rate = 1 / 100

        use_testnet = True if miner.config.get_value("isTestnet") == "1" else False
        if use_testnet:
            logger.info("Connecting to TEST network ... ")
        else:
            logger.info("Connecting to main network ... ")

        bucket = TokenBucketLimiter(1000, refill_rate)
        server = ModuleServer(miner, key, limiter=bucket, subnets_whitelist=[13], use_testnet=use_testnet)
        app = server.get_fastapi_app()

        uvicorn.run(app, host=parsed_url.hostname, port=parsed_url.port)
