import os as _os
import traceback as _traceback
import warnings as _warnings
from dataclasses import dataclass
from multiprocessing.pool import Pool
from types import TracebackType
from typing import Union, Iterable, Callable, List, Optional, TypeVar, Protocol, Generic, TypeAlias
from warnings import WarningMessage

from synthorus.utils.print_function import NO_LOG

# Type of object indicating a number of processes.
# See function num_processes(...).
#
NumProcesses: TypeAlias = Union[int, float, str, None]


def num_processes(processes: NumProcesses, assume_cpu_count: Optional[int] = None) -> int:
    """
    Convert 'processes' to an integer >= 1.

    None or 'single' or 'min' is converted to 1.
    'max' or 'all' is converted to os.cpu_count().
    'half' is converted to os.cpu_count() // 2.

    After the above conversions, the following steps are done.
    1. Floating point numbers with -1 < processes < 1 are converted
       to round(os.cpu_count() * processes).
    2. Negative values are converted to os.cpu_count() + processes.
    3. Values <= 1 are converted to 1.

    Args:
        processes: an integer representing number of process,
            or an expected str value.
        assume_cpu_count: override os.cpu_count().
    
    Returns:
        an integer >= 1.
    """
    the_cpu_count = _os.cpu_count() if assume_cpu_count is None else assume_cpu_count

    # Try to interpret non-numeric input
    if processes in (None, 'single', 'min'):
        processes = 1
    elif processes in ('max', 'all'):
        processes = the_cpu_count
    elif processes == 'half':
        processes = the_cpu_count // 2
    elif not isinstance(processes, (float, int)):
        try:
            processes_f = float(processes)
            processes_i = int(processes)
            if processes_i == processes_f:
                processes = processes_i
            else:
                processes = processes_f
        except ValueError:
            pass

    # Check floating point interpretation
    if isinstance(processes, float) and -1.0 < processes < 1.0:
        sgn = 1 if processes >= 0.0 else -1
        processes = sgn * int(the_cpu_count * abs(processes) + 0.5)

    # Confirm we now have an integer
    if not isinstance(processes, int):
        raise ValueError(f'number of processes not understood: {processes!r}')

    # Check bounds
    if processes < 0:
        processes += the_cpu_count
    if processes <= 1:
        processes = 1

    return processes


# Generic type for type hints
_TRIAL_RESULT = TypeVar('_TRIAL_RESULT')
_COLLECTOR_RESULT = TypeVar('_COLLECTOR_RESULT')


class TrialLogger(Protocol[_TRIAL_RESULT]):
    """
    A trial logger is supplied by the caller when running trial processes.
    """

    def log_begin(self, trial_number: int, trial_name: Optional[str]):
        """
        Send a message to the log to indicate beginning to run a trial.

        You can assume that the next call to this logger is either
        log_end or log_error for the given trial. Therefore,
        it is okay to not end any log message with a new line,
        as that may be performed by log_end() or log_error().

        Args:
            trial_number: current trial number (to indicate progress).
            trial_name: name of the trial, if available.
        """
        ...

    def log_end(self, result: _TRIAL_RESULT, warning_messages: List[WarningMessage]):
        """
        Send a message to the log to indicate completion of a trial.
        This will be called after log_begin() for the same trial.

        Args:
            result: a return value from running the trial (for custom loggers).
            warning_messages: a list of warning messages from running the trial.
        """
        ...

    def log_error(self, exception: Exception):
        """
        Send a message to the log to indicate failing a run of a trial.
        This will be called after log_begin() for the same trial.

        Args:
            exception: the offending Exception.
        """
        ...


def run_trial_processes(
        trials: Iterable[Callable[[], _TRIAL_RESULT]],
        collector: Callable[[Iterable[_TRIAL_RESULT]], _COLLECTOR_RESULT],
        trial_logger: TrialLogger,
        processes: Union[NumProcesses, Pool],
        discard_results_with_warning: bool = False
) -> _COLLECTOR_RESULT:
    """
    This method will run the given iterable of trials.
    Progress messages, warnings and errors are automatically managed and logged
    using the given trial logger. Trials that successfully complete are
    processed by the given collector.

    The collector is given an iterable of trial results, as provided by each trial.
    The order of results may be different to the order of trials, and may not
    include results where running the trial raised an exception.

    This is essentially map-reduce with error management and logging.

    Args:
        trials: an iterable of Trial objects.
            A Trial object is callable with no arguments, returning a generic TrialResult object.
            A Trial object may have a property/field called trial_name (for logging).
        collector: takes an iterable of generic TrialResult objects and
            returns the result for this function.
        trial_logger: an object managing logging the beginning and ending of a trial.
            The trial logger must conform to the TrialLogger protocol.
        processes: number of processes, as interpretable by num_processes(...)
            or a process Pool.
        discard_results_with_warning: Normally only results that raise an exception are filtered out
            before passing to the collector. If this flag is True, then results that issued warnings are also
            filtered out before passing to the collector.
    
    Returns:
        whatever the given collector returns.
    """
    if isinstance(processes, Pool):
        return _run_multi_process(trials, collector, trial_logger, discard_results_with_warning, processes)
    else:
        n_jobs = num_processes(processes)
        if n_jobs <= 1:
            return _run_single_process(trials, collector, trial_logger, discard_results_with_warning)
        else:
            with Pool(n_jobs) as pool:
                return _run_multi_process(trials, collector, trial_logger, discard_results_with_warning, pool)


class DefaultTrialLogger(TrialLogger[_TRIAL_RESULT]):
    """
    A standard trial logger.
    This can be used either as is, or can form the base
    class for a custom trial logger.
    """

    def __init__(self, log=print, num_trials: Optional[int] = None):
        """
        Construct a default logger, printing to the given log print function.

        Args:
            log: print function to print to, or None.
            num_trials: total number of trials (to indicate progress), or None.
        """
        self.log = log if log is not None else NO_LOG
        self.num_trials = num_trials

    def log_begin(self, trial_number: int, trial_name: Optional[str]):
        """
        Prints:
            'trial {trial_number} of {num_trials}, {trial_name}: '
        or similar depending on availability of {num_trials} and {trial_name}
        to the log, with no trailing newline.
        Assumes the next method call is log_end or log_error.
        """
        if self.num_trials is None:
            self.log(f'trial {trial_number}', end='')
        else:
            self.log(f'trial {trial_number} of {self.num_trials}', end='')

        if trial_name is None:
            self.log(': ', end='')
        else:
            self.log(f', {trial_name}: ', end='')

    def log_end(self, result: _TRIAL_RESULT, warning_messages: List[WarningMessage]):
        """
        Prints:
            'completed'
        or
            'completed with warnings'
            followed by the warning messages
        to the log.
        """
        if len(warning_messages) == 0:
            self.log('completed')
        else:
            self.log('completed (with warnings)')
            self.log_warnings(warning_messages)

    def log_warnings(self, warning_messages: List[WarningMessage]):
        """
        Support for log_end method.
        """
        for warning_message in warning_messages:
            # Filter out blank lines
            warning_lines = [
                line
                for line in str(warning_message.message).split('\n')
                if len(line.strip()) != 0
            ]
            # Print the lines, formatted to line up.
            self.log(f'warning: {warning_lines[0]}')
            for line in warning_lines[1:]:
                self.log(f'       : {line}')

    def log_error(self, exception: Exception):
        """
        Prints:
            'exception raised'
            followed by a stack trace
        to the log.
        """
        self.log(f'exception raised: {exception!r}')
        traceback: Optional[TracebackType] = exception.__traceback__
        if traceback is not None:
            for line in _traceback.format_exception(exception):
                self.log(line.rstrip())


# ============================================================================
#  Private implementation
# ============================================================================


def _run_single_process(
        trials: Iterable[Callable[[], _TRIAL_RESULT]],
        collector: Callable[[Iterable[_TRIAL_RESULT]], _COLLECTOR_RESULT],
        trial_logger: TrialLogger,
        discard_results_with_warning: bool
) -> _COLLECTOR_RESULT:
    """
    Run the given trials in the current process.
    Parameters as per _run_trial_processes.
    """
    return collector(
        _unwrap(
            discard_results_with_warning,
            (
                _log_wrap_run(trial_num, trial, trial_logger)
                for trial_num, trial in enumerate(trials, start=1)
            )
        )
    )


def _run_multi_process(
        trials: Iterable[Callable[[], _TRIAL_RESULT]],
        collector: Callable[[Iterable[_TRIAL_RESULT]], _COLLECTOR_RESULT],
        trial_logger: TrialLogger,
        discard_results_with_warning: bool,
        pool: Pool
) -> _COLLECTOR_RESULT:
    """
    Run the given trials in the given process pool.
    Other parameters as per _run_trial_processes.
    """
    return collector(
        _unwrap(
            discard_results_with_warning,
            (
                _log_wrapped_result(trial_num, wrapped_result, trial_logger)
                for trial_num, wrapped_result in enumerate(
                    pool.imap_unordered(_wrap_run, trials),
                    start=1,
                )
            )
        )
    )


@dataclass
class _WrappedResult(Generic[_TRIAL_RESULT]):
    result: _TRIAL_RESULT
    trial_name: str
    warnings: List[WarningMessage]
    exception: Optional[Exception]

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def has_exception(self) -> bool:
        return self.exception is not None


def _trial_name(trial) -> Optional[str]:
    """
    Try to get a name for the given trial.
    """
    if hasattr(trial, 'trial_name'):
        return str(trial.trial_name)
    else:
        return None


def _wrap_run(
        trial: Callable[[], _TRIAL_RESULT],
) -> _WrappedResult[_TRIAL_RESULT]:
    """
    Support for running a trial.
    Warnings and errors are captured and recorded in the
    returned _WrappedResult.
    No logging is performed.

    Args:
        trial: a Callable taking no arguments, returning a trial-result.
    
    Return:
        a _WrappedResult.
    """
    trial_name = _trial_name(trial)
    try:
        with _warnings.catch_warnings(record=True) as the_warnings:
            _warnings.simplefilter('always')
            result = trial()
            return _WrappedResult(result, trial_name, list(the_warnings), None)

    except Exception as exception:
        return _WrappedResult(None, trial_name, [], exception)


def _unwrap(
        discard_results_with_warning: bool,
        wrapped_results: Iterable[_WrappedResult[_TRIAL_RESULT]]
) -> Iterable[_TRIAL_RESULT]:
    """
    Yields unwrapped results, only where each wrapped result did not record an error.
    No logging is performed.
    """
    for wrapped_result in wrapped_results:
        if not wrapped_result.has_exception:
            if not (discard_results_with_warning and wrapped_result.has_warnings):
                yield wrapped_result.result


def _log_wrap_run(
        trial_num: int,
        trial: Callable[[], _TRIAL_RESULT],
        trial_logger: TrialLogger,
) -> _WrappedResult[_TRIAL_RESULT]:
    """
    Log the beginning of the trial, run it using _wrap_run, then log
    its completion.
    Returns the wrapped result.
    """
    trial_name = _trial_name(trial)
    trial_logger.log_begin(trial_num, trial_name)
    wrapped_result = _wrap_run(trial)
    _finish_log(wrapped_result, trial_logger)
    return wrapped_result


def _log_wrapped_result(
        trial_num: int,
        wrapped_result: _WrappedResult[_TRIAL_RESULT],
        trial_logger: TrialLogger,
) -> _WrappedResult[_TRIAL_RESULT]:
    """
    Log the beginning and completion of a trial (after it was run).
    Returns the given wrapped_result.
    """
    trial_logger.log_begin(trial_num, wrapped_result.trial_name)
    _finish_log(wrapped_result, trial_logger)
    return wrapped_result


def _finish_log(
        wrapped_result: _WrappedResult,
        trial_logger: TrialLogger,
):
    """
    Log a completed trial.
    """
    exception = wrapped_result.exception
    if exception is None:
        trial_logger.log_end(wrapped_result.result, wrapped_result.warnings)
    else:
        trial_logger.log_error(exception)
