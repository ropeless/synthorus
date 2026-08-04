"""
This module facades some of Python's file management because of some type
inference issues with Path, Traversable, PathLike, etc
"""
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TypeAlias, Union

DataPath: TypeAlias = Union[Path, Traversable]
DataPathLike: TypeAlias = Union[Path, Traversable, str]


def stem(file_path: DataPathLike) -> str:
    if isinstance(file_path, Path):
        return file_path.stem
    elif isinstance(file_path, str):
        return Path(file_path).stem
    else:
        return Path(file_path.name).stem


class open_text:
    """
    Context manager for opening text files.
    """

    def __init__(self, file_path: DataPathLike, *, encoding: str = 'utf-8') -> None:
        self._file_path = file_path
        self._encoding = encoding
        self._file = None

    def __enter__(self):
        if isinstance(self._file_path, Path | str):
            self._file = open(self._file_path, mode='r', encoding=self._encoding)
        else:
            self._file = self._file_path.open(mode='r', encoding=self._encoding)
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
            self._file = None
        return False


class open_binary:
    """
    Context manager for opening text files.
    """

    def __init__(self, file_path: DataPathLike) -> None:
        self._file_path = file_path
        self._file = None

    def __enter__(self):
        if isinstance(self._file_path, Path | str):
            self._file = open(self._file_path, mode='rb')
        else:
            self._file = self._file_path.open(mode='rb')
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
            self._file = None
        return False
