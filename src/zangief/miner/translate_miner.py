import time
from communex.module import endpoint
from loguru import logger
from translate import Translator
from src.zangief.miner.base_miner import BaseMiner


class TranslateMiner(BaseMiner):

    def __init__(self):
        super().__init__()
        config = self.get_config()

    def generate_translation(self, prompt: str, source_language: str, target_language: str):
        translator = Translator(from_lang=source_language, to_lang=target_language)
        return translator.translate(prompt)


if __name__ == "__main__":
    miner = TranslateMiner()
    TranslateMiner.start_miner_server(miner)
