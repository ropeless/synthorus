from __future__ import annotations

import json
from abc import abstractmethod, ABC
from io import StringIO
from typing import MutableMapping, Iterator, Dict, Iterable, List, Mapping, Sequence, Tuple, Set

import numpy as np
import pandas as pd

# Typedef for record values.
ValueType = object


class DataCatcher(ABC):
    """
    A data catcher is an object to easily capture ad-hoc data records.

    Records and columns can be added at any time. In general, columns are added as needed.
    """

    def __init__(self, columns: Iterable[str] = ()):
        self._columns: Tuple[str, ...] = ()
        self._column_set: Set[str] = set()
        self.new_column(*columns)

    def new_column(self, *column: str) -> None:
        for col in column:
            if self.has_column(col):
                raise ValueError(f'column already exists: {col!r}')
            self._columns += (col,)
            self._column_set.add(col)

    def ensure_column(self, *column: str) -> None:
        for col in column:
            if not self.has_column(col):
                self._columns += (col,)
                self._column_set.add(col)

    def has_column(self, column: str) -> bool:
        return column in self._column_set

    @property
    def columns(self) -> Sequence[str]:
        return self._columns

    @abstractmethod
    def append(self) -> Record:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, i):
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Record]:
        ...

    @abstractmethod
    def __delitem__(self, i):
        ...

    def get_column_list(self, column: str, default=None) -> List[ValueType]:
        """
        Construct a list of column values from the records.

        Args:
            column: is the name of the column to construct.
            default: is a default value to use when not defined in a record.
        
        Returns:
            a list of values.
        """
        return [
            record.get(column, default)
            for record in self
        ]

    def get_column_array(self, column: str, default=None, dtype=None) -> np.ndarray:
        """
        Construct a numpy array from the records.

        Args:
            column: is the name of the column to construct.
            default: is a default value to use when not defined in a record.
            dtype: is a numpy dtype to use.
        
        Returns:
            a numpy array.
        """
        try:
            # Fragile but efficient if it works :-(
            return np.fromiter(
                (
                    record.get(column, default)
                    for record in self
                ),
                count=len(self),
                dtype=dtype
            )
        # noinspection PyBroadException
        except Exception:
            return np.array(self.get_column_list(column, default), dtype=dtype)

    def get_column_series(self, column: str, default=None, dtype=None) -> pd.Series:
        """
        Construct a Pandas Series from the records.

        Args:
            column: is the name of the column to construct.
            default: is a default value to use when not defined in a record.
            dtype: is a numpy dtype to use.
        
        Returns:
            a Pandas Series.
        """
        return pd.Series(
            data=self.get_column_array(column, default, dtype),
            name=column,
            dtype=dtype
        )

    def as_json(self, indent: int = 4) -> str:
        dent: str = ' ' * indent
        string_io = StringIO()

        def _print(*args) -> None:
            print(*args, file=string_io)

        _print('{')
        field_names: str = ', '.join(json.dumps(column) for column in self.columns)
        columns_name: str = json.dumps('columns')
        records_name: str = json.dumps('records')
        _print(f'{dent}{columns_name}: [{field_names}],')
        _print(f'{dent}{records_name}: [')

        last_i: int = len(self) - 1
        for i, record in enumerate(self):
            sep = ',' if i < last_i else ''
            record_str: str = ', '.join(json.dumps(record.get(column)) for column in self.columns)
            _print(f'{dent}{dent}[{record_str}]{sep}')
        _print(f'{dent}]')
        _print('}')

        return string_io.getvalue()

    def as_dataframe(self, default=None, dtype=None) -> pd.DataFrame:
        """
        Construct a Pandas DataFrame from the records.

        Args:
            default: is a default value to use when not defined in a record, or
                is a dictionary mapping a column name to a default value.
            dtype: is a numpy dtype to use, or is a dictionary mapping a column
                name to a dtype.
        
        Returns:
            a Pandas DataFrame.
        """
        if len(self) == 0:
            return pd.DataFrame(data=[], columns=tuple(self.columns))

        if isinstance(default, dict):
            _default = lambda col: default.get(col)
        else:
            _default = lambda col: default
        if isinstance(dtype, dict):
            _dtype = lambda col: dtype.get(col)
        else:
            _dtype = lambda col: dtype

        all_series = [
            self.get_column_series(col, default=_default(col), dtype=_dtype(col))
            for col in self.columns
        ]
        return pd.concat(all_series, axis=1)

    def to_csv(
            self,
            path_or_buff,
            *,
            default=None,
            dtype=None,
            index=False,
            lineterminator='\n',
            **kwargs
    ):
        """
        Write to CSV file, using Pandas.

        Args:
            path_or_buff: Where to write to.
            default: is a default value to use when not defined in a record, or
                is a dictionary mapping a column name to a default value.
            dtype: is a numpy dtype to use, or is a dictionary mapping a column
                name to a dtype.
            index: whether to include Pandas index or not, passed to Pandas to_csv.
            lineterminator: passed to Pandas to_csv. The newline character or character
                sequence to use in the output file.
            kwargs: other arguments passed to Pandas to_csv.
        """
        df = self.as_dataframe(default, dtype)
        df.to_csv(
            path_or_buff,
            index=index,
            lineterminator=lineterminator,
            **kwargs
        )


class Record(MutableMapping[str, ValueType]):

    def __init__(self, catcher: DataCatcher):
        self._catcher = catcher

    @abstractmethod
    def clear(self):
        ...

    @abstractmethod
    def __setitem__(self, column: str, value: ValueType):
        ...

    @abstractmethod
    def __delitem__(self, column: str):
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, column: str) -> ValueType:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        ...

    @abstractmethod
    def items(self) -> Iterable[Tuple[str, ValueType]]:
        ...

    @abstractmethod
    def keys(self) -> Iterable[str]:
        ...

    @abstractmethod
    def values(self) -> Iterable[ValueType]:
        ...

    @abstractmethod
    def get(self, column: str, default=None):
        ...

    def set(self, data: Mapping[str, ValueType]):
        for col, val in data.items():
            self[col] = val

    def set_kwargs(self, **data):
        self.set(data)

    def set_values(self, *values: ValueType):
        if len(values) > len(self.columns):
            raise ValueError('more values than columns')
        self.clear()
        for col, val in zip(self.columns, values):
            self[col] = val

    def get_column_values(self) -> Sequence[ValueType]:
        return tuple(
            self.get(col)
            for col in self.columns
        )

    @property
    def columns(self) -> Sequence[str]:
        return self._catcher.columns

    def __getattr__(self, column: str) -> ValueType:
        return self.__getitem__(column)


# ============================================================================
#  Implementations
# ============================================================================


class RamDataCatcher(DataCatcher):
    """
    A DataCatcher that just keeps data in RAM.
    This is a reference implementation of `DataCatcher`.
    """

    def __init__(self, columns: Iterable[str] = ()):
        super().__init__(columns)
        self._records: List[Record] = []

    def append(self) -> Record:
        record = RamRecord(self)
        self._records.append(record)
        return record

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, i):
        return self._records[i]

    def __iter__(self) -> Iterator[Record]:
        return iter(self._records)

    def __delitem__(self, i):
        del self._records[i]


class RamRecord(Record):

    def __init__(self, catcher: DataCatcher):
        super().__init__(catcher)
        self._data: Dict[str, ValueType] = {}

    def clear(self):
        self._data.clear()

    def __setitem__(self, column: str, value: ValueType):
        self._catcher.ensure_column(column)
        self._data.__setitem__(column, value)

    def __delitem__(self, column: str):
        self._data.__delitem__(column)

    def __len__(self) -> int:
        return self._data.__len__()

    def __getitem__(self, column: str) -> ValueType:
        return self._data.__getitem__(column)

    def __iter__(self) -> Iterator[str]:
        return self._data.__iter__()

    def items(self) -> Iterable[Tuple[str, ValueType]]:
        return self._data.items()

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def values(self) -> Iterable[ValueType]:
        return self._data.values()

    def get(self, column: str, default=None):
        return self._data.get(column, default)
