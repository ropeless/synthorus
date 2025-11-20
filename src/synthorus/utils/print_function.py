from __future__ import annotations

from pathlib import Path
from typing import Callable, Union, Protocol, TypeAlias

PrintFunction: TypeAlias = Callable[..., None]
"""
A PrintFunction operates just like the Python builtin `print`
but does not use a `file` parameter as the destinations
are defined by the print function.

The function signature used by Synthorus is actually:
```
(self, *args, sep: str = ' ', end: str = '\n') -> None
```
"""


class Writable(Protocol):
    def write(self, value: str) -> int:
        ...


Destination = Union[Path, str, PrintFunction, Writable, None]


def NO_LOG(*_, **__) -> None:
    """
    A print function that does nothing.
    """
    pass


class Print(PrintFunction, Writable):
    """
    A print function to send printed text to multiple destinations.
    Example usage:
    ```
    with Print(destination) as _print:
        _print('value of x', x, sep=' is ')
    ```
    """

    def __init__(
            self,
            *destination: Destination,
            default=None,
            encoding=None
    ):
        """
        Where 'destination' can be:
            * a Path object,
            * a str representing a file name,
            * a PrintFunction object,
            * a file-like object with a 'write' method,
            * a callable object accepting arguments like print (excluding 'file' parameter).
        
        Args:
            destination: when to send print output to, multiple destinations may be provided.
            default: destination to use if destination=None. If default=None,
                it means /dev/null.
            encoding: pass to builtin open() method.
        """
        self._destinations = []
        self._destinations.extend(
            _Destination(d, default, encoding)
            for d in destination
        )

    def __call__(self, *args, sep: str = ' ', end: str = '\n') -> None:
        for destination in self._destinations:
            destination.call(*args, sep=sep, end=end)

    def write(self, value: str) -> int:
        self(value, end='')
        return len(value)

    def __del__(self):
        self.close()

    def close(self):
        destinations = self._destinations
        self._destinations = ()
        errors = []
        for destination in destinations:
            try:
                destination.close()
            except Exception as err:
                errors.append(err)
        if len(errors) > 0:
            raise errors[0]

    def __enter__(self) -> Print:
        # All the work done in the constructor
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return exc_val is None


class _Destination:
    __slots__ = ('file', 'to_close', 'call')

    def __init__(
            self,
            destination: Destination,
            default,
            encoding
    ):
        if destination is None:
            destination = default

        self.file = None
        self.to_close = None

        if destination is None:
            self.call = NO_LOG
        elif isinstance(destination, (Path, str)):  # Openable
            self.call = self._print_to_file
            self.file = open(destination, 'w', encoding=encoding)
            self.to_close = self.file
        elif callable(destination):  # Printable
            self.call = destination
        elif hasattr(destination, 'write'):  # Writable
            self.call = self._print_to_file
            self.file = destination
        else:
            raise RuntimeError(f'unknown print destination type: {type(destination)}')

    def _print_to_file(self, *args, **kwargs):
        print(*args, file=self.file, **kwargs)

    def __del__(self):
        self.close()

    def close(self):
        to_close = self.to_close
        self.to_close = None
        self.file = None
        self.call = NO_LOG
        if to_close is not None:
            to_close.close()
