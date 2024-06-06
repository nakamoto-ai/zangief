import os
from dotenv import load_dotenv

class Config:
    def __init__(self, config_file=None, ignore_config=False):
        # Load environment variables from the .env file if it exists. Does not override
        # environment variables that are already set.
        if (ignore_config == False):
            load_dotenv(dotenv_path=config_file if config_file else 'env/.env', override=True)
        
        self.validator = {
            "keyfile": os.getenv("KEY_FILE"),
            "interval": os.getenv("VALIDATOR_INTERVAL"),
            "testnet": os.getenv("IS_TESTNET"),
        }