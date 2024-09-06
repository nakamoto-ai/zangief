from abc import abstractmethod

from communex.client import CommuneClient, Keypair
from communex.compat.key import classic_load_key
from communex.types import SubnetParamsWithEmission, ModuleInfoWithOptionalBalance
from communex.misc import get_map_modules, get_map_subnets_params
from communex._common import get_node_url, ComxSettings
from communex.module.client import ModuleClient
from communex.types import Ss58Address

from .interface import ComxInterface, ModClientInterface
from loguru import logger
import random
import time
from typing import Any


class ComxClient(ComxInterface):
    """
    A client implementation of the ComxInterface for communication with Communex.

    Attributes:
        client (CommuneClient):
            The client used to interact with the Communex network.

    Methods:
        __init__(client: CommuneClient):
            Initializes the ComxClient with a given CommuneClient.
        get_map_modules(
            netuid: int = 0,
            include_balances: bool = False
        ) -> dict[str, ModuleInfoWithOptionalBalance]:
            Retrieves a mapping of registered modules for a given subnet netuid.
        get_subnet_params(
            block_hash: str | None = None,
            key: int = 0
        ) -> SubnetParamsWithEmission | None:
            Retrieves the parameters of a subnet given a block hash and key.
        get_current_block() -> int:
            Retrieves the current block number.
    """

    def __init__(self, client: CommuneClient):
        """
        Initializes the ComxClient with a given CommuneClient.

        Args:
            client (CommuneClient):
                The client used to interact with the Communex network.
        """
        self.client = client

    def get_map_modules(
        self,
        netuid: int = 0,
        include_balances: bool = False
    ) -> dict[str, ModuleInfoWithOptionalBalance]:
        """
        Retrieves a mapping of registered modules for a given subnet netuid.

        Args:
            netuid (int, optional):
                The subnet ID for which to retrieve module information. Defaults to 0.
            include_balances (bool, optional):
                Whether to include balance information in the returned module information.
                Defaults to False.

        Returns:
            dict[str, ModuleInfoWithOptionalBalance]:
                A dictionary mapping module keys to their corresponding information,
                potentially including balance information.
        """
        return get_map_modules(self.client, netuid, include_balances)

    def get_subnet_params(
        self,
        block_hash: str | None = None,
        key: int = 0
    ) -> SubnetParamsWithEmission | None:
        """
        Retrieves the parameters of a subnet given a block hash and key. Not specifying a
        block_hash will return the current subnet parameters.

        Args:
            block_hash (str | None, optional):
                The block hash for which to retrieve subnet parameters. If None, the latest
                parameters are retrieved.
            key (int, optional):
                An additional key parameter for the query. Defaults to 0.

        Returns:
            SubnetParamsWithEmission | None:
                The parameters of the subnet, or None if no parameters are found for the given
                block hash and key.
        """
        subnets = get_map_subnets_params(self.client, block_hash)
        subnet = subnets.get(key, None)
        return subnet

    def get_current_block(self) -> int:
        """
        Retrieves the current block number.

        Returns:
            int: The current block number.
        """
        current_block = self.client.get_block()["header"]["number"]
        return int(current_block)

    def vote(
        self,
        key: Keypair,
        uids: list[int],
        weights: list[int],
        netuid: int = 0,
        use_testnet: bool = False
    ):
        """
        Args:
            key: The keypair used for signing the vote transaction.
            uids: A list of module UIDs to vote on.
            weights: A list of weights corresponding to each UID.
            netuid: The network identifier.
            use_testnet: Whether testnet is being used
        """
        try:
            self.client.vote(key=key, uids=uids, weights=weights, netuid=netuid)
        except Exception as e:
            logger.error(f"WARNING: Failed to set weights with exception: {e}. Will retry.")
            sleepy_time = random.uniform(1, 2)
            time.sleep(sleepy_time)

    @staticmethod
    def get_node_url(
        comx_settings: ComxSettings | None = None, *args, use_testnet: bool = False
    ) -> str:
        """
        Args:
            comx_settings: instance of class 'ComxSettings' that inherits 'BaseSettings', includes list of different node urls
            use_testnet: Bool using testnet if True, mainnet if False
        """
        return get_node_url(comx_settings=comx_settings, *args, use_testnet=use_testnet)

    @staticmethod
    def classic_load_key(
        name: str, password: str | None = None
    ) -> Keypair:
        """
        Args:
            name: str name of local key/wallet
            password: optional str/None that is set only if the key is password encrypted
        """
        if password is not None:
            return classic_load_key(name=name, password=password)
        else:
            return classic_load_key(name=name)

    def query_map_key(self, netuid: int = 0, extract_value: bool = False) -> dict[int, Ss58Address]:
        return self.client.query_map_key(netuid=netuid, extract_value=extract_value)

    def retry_vote(
        self,
        key: Keypair,
        uids: list[int],
        weights: list[int],
        netuid: int = 0,
        use_testnet: bool = False,
        comx_settings: ComxSettings | None = None,
        *args
    ):
        self.client = CommuneClient(self.get_node_url(use_testnet=use_testnet, *args, comx_settings=comx_settings))
        self.vote(key=key, uids=uids, weights=weights, netuid=netuid, use_testnet=use_testnet)

    def module_call(
        self,
        host: str,
        port: int,
        key: Keypair,
        fn: str,
        target_key: Ss58Address,
        params: Any = {},
        timeout: int = 16
    ):
        """
        Args:
            host: string ip of module
            port: integer port of module
            key: miner module keypair object
            fn: miner module endpoint name
            target_key: ss58 address of miner module
            params: values to be sent to miner module, probably dict
            timeout: int value for time allowed before stopping connection attempt
        """
        modx_client = ModXClient(host=host, port=port, key=key)
        return modx_client.call(fn=fn, target_key=target_key, params=params, timeout=timeout)


class ModXClient(ModClientInterface):

    def __init__(self, host: str, port: int, key: Keypair):
        self.client = ModuleClient(host=host, port=port, key=key)

    def call(
        self, fn: str, target_key: Ss58Address, params: Any = {}, timeout: int = 16
    ) -> Any:
        """
        Args:
            fn: str module endpoint name
            target_key: str ss58 address of module
            params: dict of values to be sent to module
            timeout: wait time before connection attempt stops
        """
        return self.client.call(
            fn=fn,
            target_key=target_key,
            params=params,
            timeout=timeout
        )

