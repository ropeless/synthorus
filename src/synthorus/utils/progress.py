import timeit as _timeit
from typing import TypeVar, Iterable, Iterator, Optional

_PROG_ITEM = TypeVar('_PROG_ITEM')


class ProgressCheck:
    """
    A class to support simple progress checking in a loop.

    For a more integrated approach to logging progress,
    see ck.utils.logger.progress.
    """

    def __init__(self, reporting_seconds: float, first_check: Optional[float] = None):
        """
        Args:
            reporting_seconds: how often (in seconds) should 'check' return True.
                This is the minimum time between 'check' returning True.
            first_check: when should 'check' first return True (in seconds).
                If first_check is None the default value is reporting_seconds.
        """
        self._reporting_seconds = reporting_seconds
        self._next_check = reporting_seconds if first_check is None else first_check
        self._start = _timeit.default_timer()

    def check(self):
        """
        Returns True only if it has been long enough since the last True check.
        """
        seconds = _timeit.default_timer() - self._start
        if seconds > self._next_check:
            self._next_check = seconds + self._reporting_seconds
            return True
        else:
            return False

    def __bool__(self) -> bool:
        return self.check()


def progress(
        iterable: Iterable[_PROG_ITEM], *,
        counter_start=1,
        label='progress',
        iter_message='{label}: {counter:,} of {total}',
        end_message='{label}: finished {counter:,}',
        always_log_end=False,
        iter_frequency=None,
        iter_seconds=None,
        total=None,
        log=print
) -> Iterator[_PROG_ITEM]:
    """
    This is a loop progress logger, similar in spirit to tqdm, but is
    designed to work well with experiment loggers.

    Message strings (iter_message and end_message) may refer to variables:
        label (the value of the 'label' parameter),
        counter (the current counter value),
        total (the expected total number of iterations),
        it (the current loop value, or None for the end message).

    If both iter_frequency and iter_seconds is None, then the default
    of iter_frequency = 1 is used.

    Args:
        iterable: The things we are iterating over. The items from iterable
            are available in messages as {it}.
        counter_start: Each loop iteration is associated with a counter
            value, starting at counter_start and incrementing by 1 each iteration.
            The counter value is available im messages as {counter}.
        label: a label available in messages as {label}.
        iter_message: A format string to use for messages
            within the iteration loop (or None to disable).
        end_message: A format string to use for a message
            after the iteration loop completes (or None to disable).
        always_log_end: if False, then the end message will only be logged
            if an iter message was logged.
        iter_frequency: log progress every time this number of iterations is performed (or None).
        iter_seconds: log progress every time this number of seconds have passed (or None).
        total: The expected number of iterations. The total is available
            to progress messages as {total}. If this parameter is None, then it is inferred using
            len and shape, as is possible.
        log: is the print function used to log the progress messages.
    """
    if iter_frequency is not None and iter_frequency <= 0:
        raise ValueError('iter_frequency must be positive')
    if iter_seconds is not None and iter_seconds <= 0:
        raise ValueError('iter_seconds must be positive')

    # Infer total.
    if total is None:
        try:
            total = iterable.shape[0]  # type: ignore
        except AttributeError:
            pass
    if total is None:
        try:
            total = len(iterable)  # type: ignore
        except TypeError:
            pass
    if isinstance(total, int):
        total = f'{total:,}'
    elif total is None:
        total = '?'

    # Construct a checking function 'should_log(counter)'
    if iter_message is None:
        def should_log(_):
            return False
    elif iter_frequency is not None:
        log_mod = (counter_start - 1) % iter_frequency
        if iter_seconds is not None:
            progress_check = ProgressCheck(iter_seconds)

            def should_log(_counter):
                return progress_check.check() or (_counter % iter_frequency == log_mod)
        else:
            def should_log(_counter):
                return _counter % iter_frequency == log_mod
    else:  # iter_frequency is None
        if iter_seconds is not None:
            progress_check = ProgressCheck(iter_seconds)

            def should_log(_):
                return progress_check.check()
        else:
            def should_log(_):
                return True

    # ------------------
    #  This is the loop
    # ------------------
    counter = None
    showed_iter_message = False
    for counter, it in enumerate(iterable, start=counter_start):
        if should_log(counter):
            showed_iter_message = True
            log(iter_message.format(
                label=label,
                counter=counter,
                total=total,
                it=it
            ))
        yield it

    if end_message is not None and (showed_iter_message or always_log_end):
        log(end_message.format(
            label=label,
            counter=counter,
            total=total,
            it=None
        ))
