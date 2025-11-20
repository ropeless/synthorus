"""
A module to help access local configuration.

Example usage:
```
    from synthorus.utils.config_help import config

    # Assuming config.py contains:
    #    X = 2
    #    Y = 'yes'
    #
    # And the OS environment contains:
    #    X=1
    #    Z=abc
    #
    # The following will all be true:

    'X' in config
    'Y' in config
    'Z' in config
    'A' not in config

    config.X == 2
    config['Y'] == 'yes'
    config.get('Z') == 'abc'
    config.get('A') is None
    config.get('A', 'no') == 'no'

    len(config) == 3
```

"""
import os
from typing import Dict, Any, Mapping, KeysView, ValuesView, Iterator, ItemsView

from synthorus.utils.const import Const

try:
    # try to import the user's config.py
    import config as _config

    _CONFIG: Dict[str, Any] = {
        var: value
        for var, value in _config.__dict__.items()
        if len(var) > 0 and not var.startswith('_')
    }
except ImportError:
    # if not, no problem
    _config = None
    _CONFIG: Dict[str, Any] = {}

_Nil = Const('_Nil')


def config_loaded() -> bool:
    """
    Is config.py loaded?

    Return:
        True if config.py was loaded, False otherwise.
    """
    return _config is not None


class Config(Mapping[str, Any]):
    def __init__(self):
        self._config: Dict[str, Any] = {
            var: value
            for var, value in os.environ.items()
            if len(var) > 0 and not var.startswith('_')
        }
        self._config.update(_CONFIG)

    def keys(self) -> KeysView[str]:
        return self._config.keys()

    def values(self) -> ValuesView[Any]:
        return self._config.values()

    def items(self) -> ItemsView[str, Any]:
        return self._config.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def __iter__(self) -> Iterator[str]:
        return iter(self._config)

    def __contains__(self, key: str, /) -> bool:
        return key in self._config

    def __getitem__(self, key, /) -> Any:
        return self._config[key]

    def __len__(self) -> int:
        return len(self._config)

    def __getattr__(self, name: str) -> Any:
        if name in ('', 'get', 'keys', 'values', 'items',) or name.startswith('_'):
            raise AttributeError(f'illegal config attribute: {name!r}')

        got = self._config.get(name, _Nil)
        if got is _Nil:
            raise AttributeError(f'attribute not defined in config: {name!r}')
        return got


config: Config = Config()
"""
This is the variable to import.
It is a mapping from config variable name to value.
"""
