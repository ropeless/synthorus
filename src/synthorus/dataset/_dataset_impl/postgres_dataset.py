"""
An implementation of DataSource based on querying a Postgres database.

Two implementations are available:
1) PostgresDataSource queries the database each time a cross-table is needed.
2) PostgresPandasDataSource queries the database once to read a primary cross-table.
"""

from typing import Optional, Iterable, Tuple, Union, Dict

import pandas as pd

from synthorus.dataset import Dataset
from synthorus.dataset._dataset_impl._connection_params import resolve_connection, connection_str
from synthorus.dataset._dataset_impl._query import query
from synthorus.dataset._dataset_impl.pandas_dataset import PandasDataset
from synthorus.error import SynthorusError

# Try to import psycopg package.
try:
    # noinspection PyUnresolvedReferences
    from psycopg import connect
    # noinspection PyUnresolvedReferences
    from psycopg.sql import Identifier, SQL, Composable

    _PG_VER_2 = False
except ImportError:
    # noinspection PyUnresolvedReferences
    from psycopg2 import connect
    # noinspection PyUnresolvedReferences
    from psycopg2.sql import Identifier, SQL, Composable

    _PG_VER_2 = True


class PostgresDataset(Dataset):
    """
    A dataset that queries a Postgres database each time
    a cross-table is requested.
    """

    _CONNECTIONS = {}

    def __init__(
            self,
            table_name: str,
            *,
            connection_params: Optional[Dict[str, Optional[str]]] = None,
            column_names: Optional[Iterable[str]] = None,
            schema_name: Optional[str] = None,
    ):
        """
        Make a PostgresDataSource.

        Args:
            table_name: The name of the source table (or view).
            connection_params: A dictionary with entries PARAMETER:value for the connection string. If a value
                is None, then it is replaced with a config entry with the name DB_{PARAMETER}.
            column_names: Optional collection of column names. If provided, the names must be
                a subset of the table's available columns. If not provided, all available columns are used.
            schema_name: An optional schema name that includes the given table
        """
        self._connection = self._get_connection(connection_params)
        if schema_name is None:
            self._table = Identifier(table_name)
        else:
            self._table = Identifier(schema_name, table_name)
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
        sql = SQL('select * from {} limit 1').format(self._table)
        df = self._query(sql)
        return df.columns

    def _query(
            self,
            sql: Union[str, Composable],
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
        connection_params: Dict[str, str] = resolve_connection(connection_params)
        connection_string = connection_str(connection_params, delim=' ')
        connections = PostgresDataset._CONNECTIONS

        conn = connections.get(connection_string)
        if conn is None:
            if _PG_VER_2:
                conn = connect(
                    host=connection_params.get('host'),
                    port=connection_params.get('port'),
                    database=connection_params.get('dbname'),
                    user=connection_params.get('user'),
                    password=connection_params.get('password')
                )
            else:
                conn = connect(connection_string)
            connections[connection_string] = conn
        return conn

    # def _get_dtype(self, type_code):
    #     """
    #     Convert a psycopg type code to a numpy dtype,
    #     checking for None in data_col.
    #     """
    #     if self._dtype_map is None:
    #         cursor = self._connection.cursor()
    #         cursor.execute("""
    #             select
    #                 oid,
    #                 case
    #                     when typname = 'float4' then 'float32'
    #                     when typname = 'float8' then 'float64'
    #                     when typname = 'int2' then 'int16'
    #                     when typname = 'int4' then 'int32'
    #                     when typname = 'int8' then 'int64'
    #                     when typname = 'bool' then 'bool'
    #                     else 'object'
    #                 end as dtype_name
    #             from pg_type
    #         """)
    #         self._dtype_map = {
    #             int(oid): np.dtype(dtype_name)
    #             for oid, dtype_name in cursor.fetchall()
    #             if dtype_name != 'object'
    #         }
    #
    #     return self._dtype_map.get(type_code, np.object_)

    def value_maybe_none(self, rv: str) -> bool:
        return None in self.value_set(rv)

    def value_min(self, rv: str):
        return min(val for val in self.value_set(rv) if val is not None)

    def value_max(self, rv: str):
        return max(val for val in self.value_set(rv) if val is not None)

    def number_of_records(self) -> int:
        if self._number_of_records is None:
            sql = SQL('select count(*) from {}').format(
                self._table
            )
            df = self._query(sql)
            self._number_of_records = int(df.iloc[0, 0])
        return self._number_of_records

    def value_set(self, rv: str):
        if rv not in self.rvs:
            available = '[' + ','.join(self.rvs) + ']'
            raise SynthorusError(
                f'lost rv: {rv!r}, available: {available}'
            )
        values = self._value_set.get(rv)
        if values is None:
            sql = SQL('select distinct {} from {}').format(
                Identifier(rv), self._table
            )
            df = self._query(sql)
            values = df[rv].unique()
            self._value_set[rv] = values
        return values

    def crosstab(self, rvs: Iterable[str]) -> pd.DataFrame:
        rv_ids = tuple(Identifier(rv) for rv in rvs)
        col_vars = ', '.join('{}' for _ in rv_ids)
        sql = SQL(
            'select ' + col_vars + ', count(*) from {} group by ' + col_vars,
        ).format(*(rv_ids + (self._table,) + rv_ids))
        crosstab = self._query(sql)

        # Weight column must be numeric
        weight_col = crosstab.columns[-1]
        weights = crosstab[weight_col]
        crosstab[weight_col] = pd.to_numeric(weights)

        return crosstab


class PostgresPandasDataset(PandasDataset):
    """
    A dataset that queries a Postgres database once to read
    a primary cross-table which is then projected as needed.
    """

    def __init__(
            self,
            table_name: str,
            *,
            connection_params: Optional[Dict[str, Optional[str]]] = None,
            column_names: Optional[Iterable[str]] = None,
            schema_name: Optional[str] = None,
    ):
        """
        Make a PostgresPandasDataSource.

        Args:
            table_name: The name of the source table (or view).
            schema_name: An optional schema name that includes the given table.
            column_names: Optional collection of column names. If provided, the names must be
                a subset of the table's available columns. If not provided, all available columns are used.
            connection_params: A dictionary with entries PARAMETER:value for the connection string. If a value
                is None, then it is replaced with a config entry with the name DB_{PARAMETER}.
        """
        datasource = PostgresDataset(
            table_name=table_name,
            schema_name=schema_name,
            column_names=column_names,
            connection_params=connection_params,
        )
        df = datasource.crosstab(datasource.rvs)
        super().__init__(df, -1)
