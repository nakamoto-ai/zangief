import configparser


class Config:
    def __init__(self, config_file):
        if config_file is None:
            config_file = 'env/config.ini'        
        config = configparser.ConfigParser()
        config.read(config_file)
        self.validator = {
            "name": config.get("validator","name"),
            "keyfile": config.get("validator", "keyfile"),
            "interval": config.get("validator", "interval"),
            "testnet": config.get("validator", "isTestnet"),
        }
