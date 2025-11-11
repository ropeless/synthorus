"""
A simple code execution timer.

Example usage:
```
    time = StopWatch()
    # Do some work
    time.stop()

    print('time:', time)
```
Alternate usage:
```
    with timer('stuff'):
        # do some stuff
```
Usage of ProgressCheck:
```
    check = ProgressCheck(60)
    for iteration in range(max_iterations):
        # Do one iteration.
        ...

        if check:
            print(f'progress: {iteration=} time={check}')
```
"""
from __future__ import annotations

import timeit as _timeit
from typing import Tuple, Dict, Any, Optional


class StopWatch:
    __slots__ = ('start_time', 'stop_time', 'offset_seconds', 'multiplier')

    def __init__(self, offset_seconds: float = 0, multiplier: float = 1, running: bool = True):
        """
        Create a StopWatch to start timing, by using timeit.default_timer().
        A StopWatch will be created in the running state.
        Call self.stop() to stop (or pause) the StopWatch.

        Args:
            offset_seconds: is an initial time offset.
            multiplier: is an initial time multiplier (also applied to offset_seconds).
            running: is a Boolean flag to set the stopwatch running (default True).
        """
        assert multiplier > 0, 'multiplier must be positive'
        self.start_time = _timeit.default_timer()
        self.stop_time = None if running else self.start_time
        self.offset_seconds = offset_seconds
        self.multiplier = multiplier

    def copy(self, running: Optional[bool] = None) -> StopWatch:
        """
        Return a copy of this stop watch.

        Args:
            running: controls the running state of the copy.
                If True, the copy will be running (continued),
                if False, the copy will be stopped,
                if None, the copy will be in the same state as this stop watch.
        """
        result = StopWatch(
            offset_seconds=self.offset_seconds,
            multiplier=self.multiplier,
        )
        result.start_time = self.start_time
        result.stop_time = self.stop_time

        if running is not None:
            if running:
                if self.stop_time is not None:
                    # starting
                    result.continu()
            else:
                if self.stop_time is None:
                    # stopping
                    result.stop()
        return result

    def start(self, offset_seconds: float = 0, multiplier: float = 1) -> None:
        """
        Mark the start time for the timer as now.
        Cancels any previous start and stop.

        Args:
            offset_seconds: is an initial time offset.
            multiplier: is an initial time multiplier (also applied to offset_seconds).
        """
        assert multiplier > 0, 'multiplier must be positive'
        self.start_time = _timeit.default_timer()
        self.stop_time = None
        self.offset_seconds = offset_seconds
        self.multiplier = multiplier

    def stop(self) -> None:
        """
        Mark the stop time for the timer as now.
        If the stop watch was already stopped, then this overrides the previous stop.
        """
        self.stop_time = _timeit.default_timer()

    def continu(self) -> None:
        """
        Continue the timer, cancelling any previous stop.
        Any 'pause' time between a stop and continu is not included in the elapsed time.
        """
        if self.stop_time is not None:
            paused_seconds = _timeit.default_timer() - self.stop_time
            self.offset_seconds -= paused_seconds
            self.stop_time = None

    @property
    def running(self) -> bool:
        """
        Is this stopwatch running?
        
        Returns:
            true if the stopwatch is running, false otherwise.
        """
        return self.stop_time is None

    def set(self, seconds: float, multiplier: float = 1) -> None:
        """
        Set the stopwatch to the given number of seconds.
        This stops the stopwatch and resets the time multiplier.

        Args:
            seconds: is the value to set the stop watch to.
            multiplier: is reset the time multiplier (also applied to seconds).
        """
        self.start_time = _timeit.default_timer()
        self.stop_time = self.start_time
        self.offset_seconds = seconds
        self.multiplier = multiplier

    def add(self, seconds: float | StopWatch) -> None:
        """
        Add the given number of seconds to the stopwatch.
        The number of seconds added is not affected by the time multiplier.
        """
        if isinstance(seconds, StopWatch):
            seconds = seconds.seconds()
        self.offset_seconds += seconds / self.multiplier

    def subtract(self, seconds: float | StopWatch) -> None:
        """
        Subtract the given number of seconds from the stopwatch.
        The number of seconds subtracted is not affected by the time multiplier.
        """
        if isinstance(seconds, StopWatch):
            seconds = seconds.seconds()
        self.offset_seconds -= seconds / self.multiplier

    def multiply(self, multiplier: float) -> None:
        """
        Multiply the rate of time by the given multiplier.
        Multiplication is accumulative.
        """
        assert multiplier > 0, 'multiplier must be positive'
        self.multiplier *= multiplier

    def seconds(self) -> float:
        """Number of seconds of elapsed time."""
        if self.stop_time is None:
            time = _timeit.default_timer() - self.start_time
        else:
            time = self.stop_time - self.start_time
        return (time + self.offset_seconds) * self.multiplier

    def minutes(self) -> float:
        """Number of minutes elapsed."""
        return self.seconds() / 60.0

    def hours(self) -> float:
        """Number of hours elapsed."""
        return self.seconds() / 3600.0

    def hms(self) -> Tuple[int, int, float]:
        """
        (hours, minutes, seconds) of elapsed time.
        Hours and minutes will always be integers.
        Only the absolute value of time will be reported
        (i.e., if negative time offsets are used).
        """
        elapsed = abs(self.seconds())
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        return int(hours), int(minutes), seconds

    def __str__(self) -> str:
        (hours, minutes, seconds) = self.hms()
        if hours > 0:
            return f'{hours:}:{minutes:0>2}:{seconds:06.3f}'
        elif minutes > 0:
            return f'{minutes:}:{seconds:06.3f}'
        elif seconds >= 0.1:
            return f'{seconds:.3f}'
        elif seconds >= 0.01:
            return f'{seconds:.4f}'
        elif seconds >= 0.001:
            return f'{seconds:.5f}'
        elif seconds >= 0.0001:
            return f'{seconds:.6f}'
        else:
            return str(seconds)

    def __repr__(self) -> str:
        offset_seconds = self.seconds()
        multiplier = self.multiplier
        running = self.running
        name = self.__class__.__name__
        return f'{name}(offset_seconds={offset_seconds}, multiplier={multiplier}, running={running})'

    def __float__(self) -> float:
        return self.seconds()

    def __add__(self, other: float | StopWatch) -> StopWatch:
        """The returned stop watch will be stopped."""
        s = self.copy(running=False)
        s.add(other)
        return s

    def __iadd__(self, other: float | StopWatch) -> StopWatch:
        self.add(other)
        return self

    def __sub__(self, other: float | StopWatch) -> StopWatch:
        """The returned stop watch will be stopped."""
        s = self.copy(running=False)
        s.subtract(other)
        return s

    def __isub__(self, other: float | StopWatch) -> StopWatch:
        self.subtract(other)
        return self

    def __mul__(self, multiplier: float) -> StopWatch:
        """The returned stop watch will be stopped."""
        s = self.copy(running=False)
        s.multiply(multiplier)
        return s

    def __imul__(self, multiplier: float) -> StopWatch:
        self.multiply(multiplier)
        return self

    def __eq__(self, other: StopWatch) -> bool:
        return self.seconds() == other.seconds()

    def __ne__(self, other: StopWatch) -> bool:
        return self.seconds() != other.seconds()

    def __lt__(self, other: StopWatch) -> bool:
        return self.seconds() < other.seconds()

    def __gt__(self, other: StopWatch) -> bool:
        return self.seconds() > other.seconds()

    def __le__(self, other: StopWatch) -> bool:
        return self.seconds() <= other.seconds()

    def __ge__(self, other: StopWatch) -> bool:
        return self.seconds() >= other.seconds()

    def __hash__(self):
        return hash(self.seconds())


class ProgressCheck(StopWatch):
    """
    A class to support simple progress checking in a loop.
    """

    def __init__(self, reporting_seconds: float, first_check: Optional[float] = None):
        """
        Args:
            reporting_seconds: how often (in seconds) should 'check' return True.
                This is the minimum time between 'check' returning True.
            first_check: when should 'check' first return True (in seconds).
                If first_check is None the default value is reporting_seconds.
        """
        self.reporting_seconds = reporting_seconds
        self.next_check = reporting_seconds if first_check is None else first_check
        super().__init__()

    def check(self) -> bool:
        """
        Returns:
            True only if it has been long enough since the last True check.
        """
        seconds = self.seconds()
        if seconds > self.next_check:
            self.next_check = seconds + self.reporting_seconds
            return True
        else:
            return False

    def __bool__(self) -> bool:
        return self.check()


class timer(StopWatch):

    def __init__(
            self,
            label: str = 'a',
            start_message: str = '{label} timer started',
            stop_message: str = '{label} timer stopped: {time}',
            file=None,
            logger=None
    ):
        """
        Create a timer that will use a stop watch to time a section of code within a 'with' statement.
        The timer label will be printed on entering the 'with' statement.
        The timer label and time taken will be printed on exiting the 'with' statement.

        Args:
            label: A text string to label the timer.
            start_message: How the 'enter' message will be formatted, including format fields.
            stop_message: How the 'exit' message will be formatted, including format fields.
            file: Where messages should be printed - an output stream.
            logger: Where messages should be printed - a print function.

        Available format fields:
            {label} the label parameter as passed at construction time.
            {time} the time rendered as per StopWatch.__str__.
            {seconds} the time, in seconds.
            {minutes} the time, in minutes.
            {hours} the time, in hours.

        Either file or logger may be specified, not both. If neither,
        then the standard output is used.
        """
        super().__init__(running=False)
        self._label = '' if label is None else label
        self._start_message = start_message
        self._stop_message = stop_message
        self._file = file
        self._logger = logger

        if self._file is not None:
            if self._logger is not None:
                raise RuntimeError('cannot specify both file and logger')
            self._print = self._print_file
        elif self._logger is not None:
            self._print = self._print_logger
        else:
            self._print = self._print_stdout

    def _print(self, *args):
        pass  # dynamically set at construction time

    def _format_fields(self) -> Dict[str, Any]:
        return {
            'label': self._label,
            'time': str(self),
            'seconds': self.seconds(),
            'minutes': self.minutes(),
            'hours': self.hours(),
        }

    @staticmethod
    def _print_stdout(*args):
        print(*args)

    def _print_file(self, *args):
        print(*args, file=self._file)

    def _print_logger(self, *args):
        self._logger(*args)

    def __enter__(self):
        if self._start_message is not None:
            self._print(self._start_message.format(**self._format_fields()))
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self._stop_message is not None:
            self._print(self._stop_message.format(**self._format_fields()))
        return exc_val is None
