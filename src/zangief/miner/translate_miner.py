import argparse
from overrides import override
from src.zangief.miner.base_miner import endpoint
from src.zangief.miner.base_miner import BaseMiner


class TranslateMiner(BaseMiner):

    def __init__(self):
        super().__init__()
        self.config = self.get_config()
        self.get_endpoints()

def parse_arugments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyname", type=str, default="eden.Miner_2")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arugments()
    miner = TranslateMiner()

    miner.start_miner_server(args.keyname, args.host, args.port)
