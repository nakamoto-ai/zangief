from .base import BaseConfig

ENV_VALIDATOR_INTERVAL = "VALIDATOR_INTERVAL"
ENV_VALIDATOR_CALL_TIMEOUT = "VALIDATOR_CALL_TIMEOUT"


class ValidatorConfig(BaseConfig):
    """
    A configuration class for retrieving validator-specific settings from environment variables.

    Methods:
        get_validator_interval() -> int:
            Retrieves the VALIDATOR_INTERVAL environment variable as an integer.
    """

    def get_validator_interval(self) -> int:
        """
        Retrieves the VALIDATOR_INTERVAL environment variable as an integer.

        Returns:
            int: 
                The value of the VALIDATOR_INTERVAL environment variable, or 10 if not set.

        Raises:
            ValueError: 
                If the VALIDATOR_INTERVAL environment variable contains non-digit characters.
        """
        interval = self._get(ENV_VALIDATOR_INTERVAL, '10')

        if not interval.isdigit():
            raise ValueError(
                f"The environment variable '{ENV_VALIDATOR_INTERVAL}' should only contain digits.")

        return int(interval)

    def get_validator_call_timeout(self) -> int:
        """
        Retrieves the VALIDATOR_CALL_TIMEOUT environment variable as an integer.

        Returns:
            int: 
                The value of the VALIDATOR_CALL_TIMEOUT environment variable, or 20 if not set.

        Raises:
            ValueError: 
                If the VALIDATOR_CALL_TIMEOUT environment variable contains non-digit characters.
        """
        timeout = self._get(ENV_VALIDATOR_CALL_TIMEOUT, '20')

        if not timeout.isdigit():
            raise ValueError(
                f"The environment variable '{ENV_VALIDATOR_CALL_TIMEOUT}' should only contain digits.")

        return int(timeout)
