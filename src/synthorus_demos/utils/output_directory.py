import shutil
import tempfile
from pathlib import Path

from synthorus.utils.config_help import config


class output_directory(Path):
    """
    THIS IS ONLY USED TO SUPPORT DEMONSTRATION SCRIPTS. IT IS NOT NEEDED FOR SYNTHORUS.

    A managed directory path for demo output files.

    A managed directory is a named subdirectory in the configured demo output directory.
    The demo output directory is configured in config.py or OS environment, or may
    be a temporary directory if not configured.

    Example usage:
    ```
        with demo_directory('my_demo', allow_tmp=True) as demo_dir:
            with open(demo_dir / 'demo_file.txt', 'w') as f:
                f.write('This is an output file.')
    ```
    """

    def __init__(self, demo_name: str, overwrite: bool = True, allow_tmp: bool = True):
        """
        Prepare an output directory for output files of a demo script.

        This will create an empty directory called DEMO_OUT/demo_name, where
        DEMO_OUT is the value of:
            config.DEMO_OUT (i.e., config.py or OS environment),
            or a temporary directory if config.DEMO_OUT is not configured and allow_tmp is true.

        Ensures:
            `self.name == demo_name`
            `self.is_dir()`
            the directory is empty, i.e., `any(self.iterdir())` is False.

        Args:
            demo_name:  The name of the demo session, used to create a subdirectory in the configured
                demo output directory.
            overwrite: If true, an existing demo subdirectory will be overwritten (default is True).
            allow_tmp: If True, allows the use of a temporary directory when 'DEMO_OUT' is not configured
                in config.py or OS environment (default is True).

        Raises:
            RuntimeError: If "DEMO_OUT" is not in config.py or OS environment (unless allow_tmp is True).
            FileNotFoundError: If "DEMO_OUT" is configured, but the directory does not exist.
            RuntimeError: If the demo subdirectory already exists (unless overwrite is True).
            RuntimeError: If `demo_name` contains a path separator.

        Ensures:
            self.name == demo_name
        """
        self._tmp_dir = None
        demo_out: Path
        if 'DEMO_OUT' not in config:
            if allow_tmp:
                self._tmp_dir = tempfile.TemporaryDirectory()
                demo_out = Path(self._tmp_dir.name)
            else:
                raise RuntimeError('DEMO_OUT not in config.py or OS environment')
        else:
            demo_out = Path(config.DEMO_OUT)

        if not demo_out.is_dir():
            raise FileNotFoundError(f'the demo output directory does not exist: {demo_out}')

        sub_dir: Path = demo_out / demo_name
        if sub_dir.name != demo_name:
            raise RuntimeError(f'cannot have nested demo directories: {demo_name!r}')
        if sub_dir.exists():
            if not sub_dir.is_dir():
                raise RuntimeError(f'not a directory: {sub_dir}')
            if overwrite:
                shutil.rmtree(sub_dir)
            elif any(sub_dir.iterdir()):
                raise RuntimeError(f'not empty: {sub_dir}')
        sub_dir.mkdir(exist_ok=True, parents=False)

        super().__init__(sub_dir)

    def cleanup(self) -> None:
        """
        If a temporary directory was created, then delete it.

        Subsequent calls to cleanup() take no further action.
        """
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None

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
