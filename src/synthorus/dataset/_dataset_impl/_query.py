from typing import Tuple, Union

import numpy as np
import pandas as pd
from psycopg.sql import Composable


def query(
        connection,
        sql: Union[str, Composable],
        variables: Tuple = ()
) -> pd.DataFrame:
    """
    Query the database and collect the results.

    Args:
        connection: A database connection object that can provide a cursor.
        sql: The SQL as a string or Composable.
        variables: As per 'execute(sql, variables)'.
    """
    cursor = connection.cursor()
    cursor.execute(sql, variables)

    description = cursor.description
    if description is None:
        df = pd.DataFrame()
    else:
        rows = cursor.fetchall()
        data_cols = zip(*rows)
        series = {
            desc[0]: pd.Series(data_col, name=desc[0], dtype=np.object_)
            for desc, data_col in zip(description, data_cols)
        }
        df = pd.DataFrame(series)

    return df
