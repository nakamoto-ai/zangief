import os
from dotenv import load_dotenv

ENV_KEY_NAME = "KEY_NAME"
ENV_TESTNET = "TESTNET"
ENV_NETUID = "NETUID"


class BaseConfig:
    """
    A base configuration class for loading environment variables.

    Attributes:
        env_path (str): 
            Path to the environment file. Defaults to '.env'.
        ignore_config_file (bool): 
            Flag to ignore loading the environment file. Defaults to False.

    Methods:
        __init__(env_path='.env', ignore_config_file=False):
            Initializes the BaseConfig instance and loads the environment file if not ignored.
        _get(key, default=None):
            Retrieves the value of an environment variable with a given key.
        get_key_name() -> str:
            Retrieves the KEY_NAME environment variable.
        get_testnet() -> bool:
            Retrieves the TESTNET environment variable as a boolean.
        get_netuid() -> int:
            Retrieves the NETUID environment variable as an integer.
    """

    def __init__(self, env_path='.env', ignore_config_file=False):
        """
        Initializes the BaseConfig instance and loads the environment file if not ignored.

        Args:
            env_path (str, optional): 
                Path to the environment file. Defaults to '.env'.
            ignore_config_file (bool, optional): 
                Flag to ignore loading the environment file. Defaults to False.
        """
        if ignore_config_file is False:
            load_dotenv(dotenv_path=env_path, override=True)

    def _get(self, key, default=None):
        """
        Retrieves the value of an environment variable with a given key.

        Args:
            key (str): 
                The environment variable key.
            default: 
                The default value to return if the environment variable is not set. 
                Defaults to None.

        Returns:
            The value of the environment variable, or the default value if not set.
        """
        value = os.getenv(key, default)

        if value is None or value == "":
            value = default

        return value

    def get_key_name(self) -> str:
        """
        Retrieves the KEY_NAME environment variable.

        Returns:
            str: The value of the KEY_NAME environment variable.

        Raises:
            ValueError: If the KEY_NAME environment variable is not set or is empty.
        """
        key = self._get(ENV_KEY_NAME)

        if not key:
            raise ValueError(
                f"The environment variable '{ENV_KEY_NAME}' is required but not set or is empty.")

        return str(key)

    def get_testnet(self) -> bool:
        """
        Retrieves the TESTNET environment variable as a boolean.

        Returns:
            bool: True if the TESTNET environment variable is set to '1', False otherwise.
        """
        value = self._get(ENV_TESTNET, '0')
        return value == '1'

    def get_netuid(self) -> int:
        """
        Retrieves the NETUID environment variable as an integer.

        Returns:
            int: The value of the NETUID environment variable.

        Raises:
            ValueError: 
                If the NETUID environment variable is not set, is empty, or contains 
                non-digit characters.
        """
        netuid = self._get(ENV_NETUID, None)

        if not netuid:
            raise ValueError(
                f"The environment variable '{ENV_NETUID}' is required but not set or is empty.")

        if not netuid.isdigit():
            raise ValueError(f"The environment variable '{ENV_NETUID}' should only contain digits.")

        return int(netuid)
