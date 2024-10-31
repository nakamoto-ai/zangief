import time
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from loguru import logger
from communex.module.server import ModuleServer
from communex.compat.key import classic_load_key
from keylimiter import TokenBucketLimiter


from typing import Optional, Any, Tuple, Union, List
import argparse
from base_miner import BaseMiner
from config import Config
from translator import SeamlessTranslator


def get_netuid(use_testnet: bool) -> int:
    if use_testnet:
        return 23
    return 13


class TranslateMiner(BaseMiner):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.translator = SeamlessTranslator()

    def generate_translation(self, prompt: str, source_language: str, target_language: str) -> str:
        """Generate translation using SeamlessTranslator."""
        return self.translator.translation_inference(prompt)

    @staticmethod
    def start_miner_server(miner: BaseMiner):
        keyname = miner.config.get_value("keyname")
        host = miner.config.get_value("host", "0.0.0.0")
        port = miner.config.get_value("port", 8080)
        miner._start_server(keyname, host, port)

    def _start_server(self, keyname: str, host, port):
        key_password = self.config.get_value("key_password")
        key = classic_load_key(name=keyname, password=key_password) if key_password else classic_load_key(name=keyname)
        refill_rate = 1 / 1000
        use_testnet = self.config.get_value("isTestnet") == "1"

        logger.info("Connecting to TEST network..." if use_testnet else "Connecting to main network...")

        bucket = TokenBucketLimiter(refill_rate=refill_rate, bucket_size=1000, time_func=time.time)
        server = ModuleServer(module=self, key=key, limiter=bucket, subnets_whitelist=[get_netuid(use_testnet)],
                              use_testnet=use_testnet)

        app = server.get_fastapi_app()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        uvicorn.run(app, host=host, port=port)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyname", type=str, default="eden.Miner_2")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    miner = TranslateMiner()

    miner.start_miner_server(args.keyname, args.host, args.port)
