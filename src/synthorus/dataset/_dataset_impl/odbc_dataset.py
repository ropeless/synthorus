"""
An implementation of DataSource based on querying a database via ODBC.

Two implementations are available:
1) OdbcDataSource queries the database each time a cross-table is needed.
2) OdbcPandasDataSource queries the database once to read a primary cross-table.
"""

from os import environ as config
from typing import Optional, Iterable, Tuple, Dict, TypeAlias, Any

import pandas as pd

from synthorus.dataset import Dataset
from synthorus.dataset._dataset_impl._connection_params import connection_str, resolve_connection
from synthorus.dataset._dataset_impl._query import query
from synthorus.dataset._dataset_impl.pandas_dataset import PandasDataset
from synthorus.error import SynthorusError

try:
    from pyodbc import Connection, connect

except ImportError:
    # ODBC not installed in the OS
    Connection: TypeAlias = Any


    def connect(_connection_str: str) -> Connection:
        raise SynthorusError('ODBC not installed')


class OdbcDataset(Dataset):
    """
    A dataset that queries a Postgres database each time
    a cross-table is requested.
    """

    _CONNECTIONS: Dict[str, Connection] = {}

    def __init__(
            self,
            table_name: str,
            *,
            connection_params: Optional[Dict[str, Optional[str]]] = None,
            column_names: Optional[Iterable[str]] = None,
            schema_name: Optional[str] = None,
    ):
        """
        Make an OdbcDataSource.

        Args:
            table_name: The name of the source table (or view).
            connection_params: A dictionary with entries PARAMETER:value for the connection string. If a value
                is None, then it is replaced with a config entry with the name DB_{PARAMETER}.
            column_names: Optional collection of column names. If provided, the names must be
                a subset of the table's available columns. If not provided, all available columns are used.
                schema_name: Optional schema name that includes the given table, default is from config.DB_SCHEMA.
        """
        # Resolve default values
        if schema_name is None:
            schema_name = config.get('DB_SCHEMA')

        self._connection = self._get_connection(connection_params)
        if schema_name is None:
            self._table = _make_identifier(table_name)
        else:
            self._table = _make_identifier(schema_name, table_name)
        self._dtype_map = None
        self._value_set = {}  # Cache of value sets
        self._number_of_records = None  # Cache of record count

        # Get the available columns
        rvs = self._get_rvs(column_names)
        super().__init__(rvs)

    def _get_rvs(self, column_names: Optional[Iterable[str]] = None) -> Tuple:
        available_columns = self._query_columns()
        if column_names is None:
            rvs = tuple(available_columns)
        else:
            rvs = tuple(column_names)
            missing_columns = set(rvs).difference(available_columns)
            if len(missing_columns) > 0:
                raise SynthorusError(f'not all columns are available - missing: {missing_columns}')
        return rvs

    def _query_columns(self) -> pd.Index:
        """
        Returns:
            the columns (as a Pandas Index object).
        """
        sql = f'select * from {self._table} limit 1'
        df = self._query(sql)
        return df.columns

    def _query(
            self,
            sql: str,
            variables: Tuple = ()
    ) -> pd.DataFrame:
        """
        Query the database and collect the results.

        Args:
            sql: The SQL as a string or Composable.
            variables: As per 'execute(sql, variables)'.
        """
        return query(self._connection, sql, variables)

    @staticmethod
    def _get_connection(connection_params: Optional[Dict[str, Optional[str]]]):
        """
        Args:
            connection_params: a dictionary with entries PARAMETER:value for the connection string. If a value
                is None, then it is replaced with a config entry with the name DB_{PARAMETER}.
        """
        conn_str = connection_str(resolve_connection(connection_params))

        connections = OdbcDataset._CONNECTIONS
        conn = connections.get(conn_str)
        if conn is None:
            conn = connect(conn_str)
            connections[conn_str] = conn
        return conn

    def number_of_records(self) -> int:
        if self._number_of_records is None:
            sql = f'select count(*) from {self._table}'
            df = self._query(sql)
            self._number_of_records = int(df.iloc[0, 0])
        return self._number_of_records

    def value_maybe_none(self, rv: str) -> bool:
        return None in self.value_set(rv)

    def value_min(self, rv: str):
        return min(val for val in self.value_set(rv) if val is not None)

    def value_max(self, rv: str):
        return max(val for val in self.value_set(rv) if val is not None)

    def value_set(self, rv: str):
        if rv not in self.rvs:
            available = '[' + ','.join(self.rvs) + ']'
            raise SynthorusError(
                f'lost rv: {rv!r}, available: {available}'
            )
        values = self._value_set.get(rv)
        if values is None:
            sql = f'select distinct {_make_identifier(rv)} from {self._table}'
            df = self._query(sql)
            values = df[rv].unique()
            self._value_set[rv] = values
        return values

    def crosstab(self, rvs: Iterable[str]) -> pd.DataFrame:
        col_vars = ', '.join(_make_identifier(rv) for rv in rvs)
        sql = f'select {col_vars}, count(*) from {self._table} group by {col_vars}'
        crosstab = self._query(sql)

        # Weight column must be numeric
        weight_col = crosstab.columns[-1]
        weights = crosstab[weight_col]
        crosstab[weight_col] = pd.to_numeric(weights)

        return crosstab


class OdbcPandasDataset(PandasDataset):
    """
    A dataset that queries a Postgres database once to read
    a primary cross-table which is then projected as needed.
    """

    def __init__(
            self,
            table_name: str,
            *,
            connection_params: Dict,
            column_names: Optional[Iterable[str]] = None,
            schema_name: Optional[str],
    ):
        """
        Make an OdbcPandasDataSource.

        Args:
            table_name: The name of the source table (or view).
            connection_params: A dictionary with entries PARAMETER:value for the connection string. If a value
                is None, then it is replaced with a config entry with the name DB_{PARAMETER}.
            column_names: Optional collection of column names. If provided, the names must be
                a subset of the table's available columns. If not provided, all available columns are used.
            schema_name: An optional schema name that includes the given table, the default is from config.DB_SCHEMA.
        """
        datasource = OdbcDataset(
            table_name=table_name,
            connection_params=connection_params,
            schema_name=schema_name,
            column_names=column_names,
        )
        df = datasource.crosstab(datasource.rvs)
        super().__init__(df, -1)


def _make_identifier(*ids: str) -> str:
    """
    Make a safe identifier (for table or field).
    """
    return '.'.join(_quote_identifier(identifier) for identifier in ids)


def _quote_identifier(identifier: str) -> str:
    # Replace each double quote character with two consecutive double quote characters.
    identifier = identifier.replace('"', '""')
    # Wrap in double quotes.
    return '"' + identifier + "."
