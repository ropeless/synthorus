import os as _os
import shutil as _shutil
import tempfile as _tempfile
from pathlib import Path
from typing import Union


class tmp_dir(Path):
    """
    This is a context for creating a temporary directory and making it the working directory.

    Usage:
    ```
        from tmp_dir import tmp_dir

        with tmp_dir():
            # do some stuff in the current working directory
    ```
    """

    __slots__ = ('_new_dir', '_old_cwd', '_chdir', '_delete_when_done')

    def __init__(
            self,
            *,
            chdir: bool = True,
            delete_when_done: bool = True
    ):
        """
        Create a temporary directory, make it the working
        directory (unless chdir is False), clean up when done
        (unless delete_when_done is False).

        Args:
            chdir: if True, then chdir to the temporary directory.
            delete_when_done: if True, then the temporary directory is
                deleted when done.
        """
        self._new_dir: Path
        self._old_cwd: Path
        self._delete_when_done: bool

        self._new_dir = Path(_tempfile.mkdtemp())
        self._old_cwd = Path.cwd()
        self._chdir = chdir
        if chdir:
            _os.chdir(self._new_dir)
        self._delete_when_done = delete_when_done

        super().__init__(self._new_dir)

    @property
    def old_cwd(self) -> Path:
        return self._old_cwd

    def cleanup(self) -> None:
        """
        Change back to the old working directory and delete the temporary
        directory, along with all its contents.

        Subsequent calls to cleanup() take no further action.
        """
        if self._new_dir is not None:
            if self._chdir:
                _os.chdir(self._old_cwd)
            if self._delete_when_done:
                _shutil.rmtree(self._new_dir)
        self._old_cwd = None
        self._new_dir = None

    def __del__(self):
        self.cleanup()

    def __enter__(self):
        # nothing to do - already created at __init__
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return exc_val is None

    def with_segments(self, *pathsegments):
        # Stop `Path` trying to recursively create `output_directory` objects.
        return Path(*pathsegments)
