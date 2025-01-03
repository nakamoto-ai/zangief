
from communex.module.client import ModuleClient
from substrateinterface import Keypair


class ModuleClientFactory:
    def __init__(self, key: Keypair):
        self.key = key

    def create_client(self, module_ip: str, module_port: int) -> ModuleClient:
        module_client = ModuleClient(module_ip, module_port, self.key)
        return module_client
