from abc import ABC, abstractmethod

from communex.types import SubnetParamsWithEmission, ModuleInfoWithOptionalBalance, Ss58Address
from communex._common import ComxSettings
from typing import Any


class ComxInterface(ABC):
    """

    Interface for 'CommuneClient'

    An abstract base class that defines the interface for communication with Communex.

    Methods:
        get_map_modules(
            netuid: int = 0,
            include_balances: bool = False
        ) -> dict[str, ModuleInfoWithOptionalBalance]:
            Retrieves a mapping of registered modules for a given subnet netuid.
        get_subnet_params(
            block_hash: str | None = None,
            key: int = 0
        ) -> SubnetParamsWithEmission | None:
            Retrieves the parameters of a subnet given a block hash and key. Not
            specifying a block_hash will return the current subnet parameters.
        get_current_block() -> int:
            Retrieves the current block number.
    """

    @abstractmethod
    def get_map_modules(
        self,
        netuid: int = 0,
        include_balances: bool = False
    ) -> dict[str, ModuleInfoWithOptionalBalance]:
        """
        Retrieves a mapping of registered modules for a given subnet netuid.

        Args:
            netuid (int, optional):
                The network ID for which to retrieve module information. Defaults to 0.
            include_balances (bool, optional):
                Whether to include balance information in the returned module information.
                Defaults to False.

        Returns:
            dict[str, ModuleInfoWithOptionalBalance]: A dictionary mapping module keys to their
            corresponding information, potentially including balance information.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_current_block(self) -> int:
        """
        Retrieves the current block number.

        Returns:
            int: The current block number.
        """
        pass

    @abstractmethod
    def vote(self):
        """
        Sets weights for miners
        """
        pass

    @abstractmethod
    def get_node_url(self, comx_settings: ComxSettings | None = None, *, use_testnet: bool = False):
        """
        Gets url of commune node to connect to
        """
        pass

    @abstractmethod
    def classic_load_key(self, name: str, password: str | None = None):
        """
        Loads local commune key/wallet
        """
        pass


class ModClientInterface(ABC):

    """
    Interface for communex 'ModuleClient'
    -- not to be confused with 'CommuneClient' or 'Module' Class --
    """

    @abstractmethod
    def call(
        self,
        fn: str,
        target_key: Ss58Address,
        params: Any = {},
        timeout: int = 16
    ):
        """
        Creates connection object with miner module
        """
        pass


class AsyncIOInterface(ABC):

    @abstractmethod
    async def run(self, main, *, debug=None):
        """
        Run the provided coroutine called 'main'
        """
        pass
