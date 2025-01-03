import argparse
from config import Config


def get_miner_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="transaction validator")
    parser.add_argument("--miner", type=str, default="openai", help="miner type")
    parser.add_argument("--config", type=str, default="env/config.ini", help="config file path")
    args = parser.parse_args()
    return args


class Miner:
    def __init__(self, miner_type: str, config: Config, miner_classes=None):
        """
        Initialize MinerRunner with miner type, config, and optional miner classes for testing.

        :param miner_type: Type of the miner ("m2m" or "openai").
        :param config: Configuration object for the miner.
        :param miner_classes: Optional dictionary of miner classes for testing.
                              Expected format: {"m2m": M2MMiner, "openai": OpenAIMiner}
        """
        self.miner_type = miner_type
        self.config = config
        self.miner_classes = miner_classes or {
            "m2m": self._import_miner_class("m2m_miner", "M2MMiner"),
            "openai": self._import_miner_class("openai_miner", "OpenAIMiner")
        }
        self.miner = None

    def _import_miner_class(self, module_name, class_name):
        module = __import__(module_name)
        return getattr(module, class_name)

    def initialize(self):
        if self.miner_type in self.miner_classes:
            miner_class = self.miner_classes[self.miner_type]
            self.miner = miner_class(config=self.config)
        else:
            raise ValueError("Unsupported miner")

    def start(self):
        self.initialize()
        if self.miner:
            self.miner.start_miner_server(miner=self.miner)
        else:
            print("Miner not initialized")


def start_miner():
    args = get_miner_args()
    config = Config(config_file=args.config)
    miner = Miner(args.miner, config)
    miner.start()


if __name__ == "__main__":
    start_miner()
