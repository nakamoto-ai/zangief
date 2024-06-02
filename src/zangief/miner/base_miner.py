import os
import time
from fastapi import FastAPI
from substrateinterface.keypair import Keypair
import uvicorn
from os.path import dirname, realpath
from pathlib import Path
from urllib.parse import ParseResult, urlparse
from abc import abstractmethod
from loguru import logger
from communex.module import endpoint, Module
from communex.module.server import ModuleServer
from communex.compat.key import classic_load_key
from keylimiter import TokenBucketLimiter


from typing import Optional, Any, Tuple, Union, List

from src.zangief.miner.translator import SeamlessTranslator
from src.zangief.miner.config import Config


def get_netuid(is_testnet) -> int:
    return 23 if is_testnet else 1


class BaseMiner(Module):
    config: Optional[Union[Config, Any]]
    translator: Optional[Union[SeamlessTranslator, Any]]
    model_name: Optional[Union[str, Any]]
    device: Optional[Union[str, Any]]
    max_length: Optional[Union[int, Any]]
    do_sample: Optional[Union[bool, Any]]
    temperature: Optional[Union[float, Any]]
    top_k: Optional[Union[int, Any]]
    no_repeat_ngram_size: Optional[Union[int, Any]]
    num_beams: Optional[Union[int, Any]]

    @endpoint
    def generate(
        self,
        prompt: str,
        source_language: str,
        target_language: str,
    ) -> dict[str, str]:
        self.config = self.get_config()
        start_time: float = time.time()
        logger.info("Generating translation... ")

        logger.info(f"Source ({source_language})")
        logger.info(f"{prompt}")
        self.translator = SeamlessTranslator()

        translation: Tuple[Path] | Any = self.generate_translation(
            prompt=prompt,
            source_language=source_language,
            target_language=target_language,
        )

        logger.info(f"Translation ({target_language})")
        logger.info(translation)

        end_time: float = time.time()
        execution_time: float = end_time - start_time
        logger.info(f"Responded in {execution_time} seconds")

        return {"answer": str(object=translation)}

    def get_config(self) -> Config:
        config_file = os.path.join(os.path.dirname(__file__), "../../../env/config.ini")
        return Config(config_file=config_file)

    def generate_translation(
        self, prompt: str, source_language: str, target_language: Union[List[str], str]
    ) -> Tuple[Path, Path] | Any:
        logger.info("Generating translation... ")
        input_file = Path("input/input_file.txt")
        if not prompt:
            logger.error("Prompt cannot be empty")
            raise ValueError("Prompt cannot be empty")
        input_file.write_text(data=prompt, encoding="utf-8")
        if not source_language:
            logger.error("Source language cannot be empty")
            raise ValueError("Source language cannot be empty")
        if not self.translator:
            self.translator = SeamlessTranslator()
        if not target_language:
            target_langauge: str = self.translator.target_languages["eng"]
        tgt_language: List[str] = []
        if isinstance(target_langauge, str):
            tgt_language.append(str(object=target_language))

        return self.translator.translation_inference(
            in_file=input_file, task_string="t2tt", target_languages=tgt_language
        )

    def start_miner_server(self, keyname, host, port) -> None:
        config: Config = self.get_config()
        key: Keypair = classic_load_key(name=str(keyname))

        refill_rate: float = 1 / 1000
        use_testnet: bool = config.get_value(option="isTestnet") == "1"
        if use_testnet:
            logger.info("Connecting to TEST network ... ")
        else:
            logger.info("Connecting to main network ... ")
        bucket = TokenBucketLimiter(
            refill_rate=refill_rate,
            bucket_size=1000,
            time_func=time.time,
        )
        netuid: int = get_netuid(is_testnet=use_testnet)
        server = ModuleServer(
            module=self,
            key=key,
            limiter=bucket,
            subnets_whitelist=[netuid],
            use_testnet=use_testnet,
        )
        app: FastAPI = server.get_fastapi_app()

        uvicorn.run(
            app=app,
            host=str(object=host),
            port=int(str(object=port)),
        )
