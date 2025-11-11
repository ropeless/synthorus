from typing import Dict, Optional

from synthorus.error import SynthorusError
from synthorus.utils.config_help import config


def resolve_connection(
        connection_params: Optional[Dict[str, Optional[str]]],
        *,
        prefix: str = 'DB_',
        connection_config: str = 'DB_CONNECTION',
) -> Dict[str, str]:
    """
    Use local configuration (`config_help`) to resolve the value of
    any key marked as `None`. If an entry with value `None` cannot be resolved,
    then an exception is raised. If connection_params is `None` local configuration will be checked
    for a connection dictionary.

    Args:
        connection_params: A dictionary.
        prefix: the prefix to use to identify a local configuration variable from a dictionary key.
        connection_config: the config variable to use if connection_params is `None`.

    Returns:
        A dictionary with no value `None`.
    """
    if connection_params is None:
        connection_params = config.get(connection_config)
        if connection_params is None:
            raise SynthorusError(f'cannot resolve connection parameter: {connection_config!r}')

    filtered_connection_params: Dict[str, str] = {}
    for param, value in connection_params.items():
        if value is not None:
            filtered_connection_params[param] = value
        else:
            value = config.get(f'{prefix}{param}')
            if value is not None:
                filtered_connection_params[param] = str(value)
            else:
                raise SynthorusError(f'cannot resolve connection parameter: {param!r}')
    return filtered_connection_params


def connection_str(connection_params: Dict[str, str], delim: str = ';') -> str:
    """
    Return a string representation of a connection parameters.

    Args:
        connection_params: a dictionary of connection parameters.
        delim: the delimiter to separate parameters.

    Returns:
        a connection string, with parameters sorted for string stability.
    """
    return delim.join(f'{k}={v}' for k, v in sorted(connection_params.items()))
