from io import StringIO
from pathlib import Path
from typing import Union

import pandas as pd
from pandas._typing import ReadCsvBuffer

from synthorus.error import SynthorusError

SEP = ','  # The field separator
SENTINEL = '"Total"' + SEP  # Start of line indicating no more records
KEEP_ZEROS = False  # Keep records with zero counts?


class SourceWrapper(ReadCsvBuffer[str]):

    def __init__(self, source):
        self.source = source
        self.prev = None

        # Skip three blocks of comments
        self._skip_block()
        self._skip_block()
        self._skip_block()

        # Read up to and including the header
        line = source.readline()
        while line == '\n':
            line = source.readline()

        # Field read_buff holds a line read from source,
        # but not yet provided by this stream.
        self.read_buff = line

    @property
    def mode(self):
        return 'r'

    def readline(self, n: int = -1) -> str:
        if self.read_buff != '':
            line = self.read_buff
            self.read_buff = ''
        else:
            while True:
                line = self.source.readline()

                # Check for end-of-file conditions
                if line == '' or line == '\n' or line.startswith(SENTINEL):
                    return ''

                cur = line.strip().split(SEP)
                # For some reason, TableBuilder adds a stray field separator
                # at the end of each record. We need to remove this or pandas
                # will create an extra column.
                cur.pop(-1)

                if KEEP_ZEROS or int(cur[-1]) > 0:
                    break

            for i, val in enumerate(cur):
                if val == '':
                    cur[i] = self.prev[i]
            self.prev = cur

            line = SEP.join(cur) + '\n'

        if 0 < n < len(line):
            # The line is longer than the requested limit
            self.read_buff = line[n:]
            line = line[:n]

        return line

    def read(self, n: Union[int, None] = ...) -> str:
        if self.read_buff == '':
            self.read_buff = self.readline()

        if n < len(self.read_buff):
            result = self.read_buff[:n]
            self.read_buff = self.read_buff[n:]
        else:
            result = self.read_buff
            self.read_buff = ''

        return result

    def __iter__(self):
        while True:
            yield self.readline()

    @property
    def closed(self):
        return False

    def _skip_block(self):
        state = 0
        while True:
            line = self.source.readline()
            if line == '':
                # End of file
                return
            if state == 0:
                if line == '\n':
                    # Empty line and in state 0
                    # Stay in state 0
                    pass
                else:
                    # Non-empty line and in state 0
                    # Transition to state 1
                    state = 1
            elif state == 1 and line == '\n':
                # Empty line and in state 1
                # Finished reading a block
                return


def read_table_builder(source: Union[StringIO, Path]) -> pd.DataFrame:
    """
    Read a CSV file created by ABS TableBuilder.
    """
    if isinstance(source, Path):
        with open(source, 'r') as file:
            return pd.read_csv(SourceWrapper(file), sep=SEP)
    elif isinstance(source, StringIO):
        return pd.read_csv(SourceWrapper(source), sep=SEP)
    else:
        # noinspection PyUnreachableCode
        raise SynthorusError(f'unexpected source {type(source)}')
