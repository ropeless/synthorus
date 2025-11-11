"""
This is a module for creating a temporary directory and making it the working directory.

Usage:
    from tmp_dir import tmp_dir

    with tmp_dir():
        # do some stuff
"""

import os as _os
import shutil as _shutil
import tempfile as _tempfile
from pathlib import Path
from typing import Union


class tmp_dir:
    __slots__ = ('_new_dir', '_old_cwd', '_chdir', '_delete_when_done')

    def __init__(
            self,
            name: Union[Path, str, None] = None,
            chdir: bool = True,
            delete_when_done: bool = True
    ):
        """
        Create a temporary directory, make it the working
        directory (unless chdir is False), clean up when done
        (unless delete_when_done is False).

        :param name: the name of the temporary directory. If name
            is None, then tempfile.mkdtemp() is used.

        :param chdir: if True, then chdir to the temporary directory.

        :param delete_when_done: if True, then the temporary directory is
            deleted when done.
        """

        if name is None:
            self._new_dir = Path(_tempfile.mkdtemp())
        else:
            self._new_dir = Path(name).absolute()
            self._new_dir.mkdir(exist_ok=True, parents=True)

        self._old_cwd = Path.cwd()
        self._chdir = chdir
        if chdir:
            _os.chdir(self._new_dir)
        self._delete_when_done = delete_when_done

    def __del__(self):
        """
        Calls self.done().
        """
        self.done()

    @property
    def path(self) -> Path:
        """
        What is the path to the temporary directory.
        """
        return self._new_dir

    def done(self) -> None:
        """
        Change back to the old working directory and delete the temporary
        directory, along with all its contents.

        Subsequent calls to done() take no further action.
        """
        if self._new_dir is not None:
            if self._chdir:
                _os.chdir(self._old_cwd)
            if self._delete_when_done:
                _shutil.rmtree(self._new_dir)
        self._old_cwd = None
        self._new_dir = None

    def __enter__(self):
        # nothing to do - already created at __init__
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done()
        return exc_val is None

    def __str__(self):
        return str(self._new_dir)
