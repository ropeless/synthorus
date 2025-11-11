import sys as _sys
from os import devnull as _devnull
from pathlib import Path as _Path


class redirect_out:
    """
    Temporarily redirect sys.stdout (and maybe sys.stderr too).

    Usage:

    with redirect_out():
        do_something_that_is_console_noisy()

    with redirect_out('tmp/output.txt'):
        do_something_that_is_console_noisy()

    output = io.StringIO()
    with redirect_out(output):
        do_something_that_is_console_noisy()
    capture = output.getvalue()

    """
    __slots__ = ('_out', '_and_stderr', '_old_stdout', '_old_stderr', '_close_out')

    def __init__(self, out=None, and_stderr=False):
        """
        :param out: output stream to redirect sys.stdout to, or None to send to devnull.
            Default is None. If a str or pathlib.Path, then it will be opened and closed
            automatically.
        :param and_stderr: if True, then sys.stdout is also redirected.
        """
        self._out = out
        self._and_stderr = and_stderr
        self._old_stdout = None
        self._old_stderr = None
        self._close_out = None

    def __enter__(self):
        self._old_stdout = _sys.stdout
        self._old_stderr = _sys.stderr

        try:
            if self._out is None:
                _sys.stdout = self._close_out = open(_devnull, 'w')
            elif isinstance(self._out, (str, _Path)):
                _sys.stdout = self._close_out = open(self._out, 'w')
            else:
                _sys.stdout = self._out
        except (OSError, IOError) as e:
            self._revert_()
            raise e

        if self._and_stderr:
            _sys.stderr = _sys.stdout

        return _sys.stdout

    def __exit__(self, type, value, traceback):
        self._revert_()

    def _revert_(self):
        _sys.stdout = self._old_stdout
        _sys.stderr = self._old_stderr
        self._close_out = _safe_close_(self._close_out)


def _safe_close_(me) -> object:
    try:
        if me is not None:
            me.close()
    except (OSError, IOError):
        pass
    return None
