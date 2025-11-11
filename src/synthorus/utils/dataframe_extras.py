"""
Functions for creating and manipulating cross-tables.
"""

from typing import Iterable, Union, Callable, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from pandas._typing import Dtype

from synthorus.error import SynthorusError


def make_crosstab(
        source_data: pd.DataFrame,
        rvs: Iterable[str],
        weights: Union[pd.Series, str, int, None]
) -> pd.DataFrame:
    """
    Make a cross-table from the given datasource for the give rvs.
    """
    # Pandas really needs a list, so we ensure it!
    columns = list(rvs)

    if weights is None:
        weight_col = None
    elif isinstance(weights, pd.Series):
        weight_col = weights.name
    elif isinstance(weights, str):
        weight_col = weights
        rv_cols = [col for col in source_data.columns if col != weight_col]
        weights = source_data[weight_col]
        source_data = source_data[rv_cols]
    elif isinstance(weights, int):
        weight_col = source_data.columns[weights]
        rv_cols = [col for col in source_data.columns if col != weight_col]
        weights = source_data[weight_col]
        source_data = source_data[rv_cols]
    else:
        # noinspection PyUnreachableCode
        raise SynthorusError(f'weights not understood: {weights!r}')

    try:
        if len(columns) == 0:
            if weights is not None:
                total_weight = weights.sum()
            else:
                total_weight = source_data.shape[0]
            crosstab = pd.DataFrame({weight_col: [total_weight]})
        else:
            cols = source_data[columns]
            if weights is not None:
                cols = pd.concat([cols, weights], axis=1)
                crosstab = cols.groupby(columns, observed=True, dropna=False).sum().reset_index()
            else:
                crosstab = cols.groupby(columns, observed=True, dropna=False).size().reset_index()

    except KeyError as key_error:
        available = '[' + ','.join(rvs) + ']'
        raise SynthorusError(
            f'lost rv: {columns!r}, available: {available}, error: {key_error}'
        )

    # Internal consistency - check the counts column is numeric
    weights = crosstab.iloc[:, -1]
    assert pd.api.types.is_numeric_dtype(weights)

    # Ensure crosstab weight has no name
    crosstab = crosstab.rename(columns={crosstab.columns[-1]: ''})

    return crosstab


def adjust_cross_table(
        cross_table: pd.DataFrame,
        cond_cross_table: pd.DataFrame,
        log=print
) -> pd.DataFrame:
    """
    Group cross_table by the random variables in cond_cross_table.
    Normalise the weight in each group to sum to 1.
    Multiply the weight of each row in the group by the matching entry in cond_cross_table.

    :param cross_table: is the cross-table to be conditioned.
    :param cond_cross_table: is the conditioning cross-table.
    :param log: is a print function to receive warning messages.
    """
    # Check the trivial case, either cross-table is empty
    if cross_table.shape[0] == 0 or cond_cross_table.shape[0] == 0:
        return cross_table

    cond_rv_names = list(cond_cross_table.columns[:-1])
    weight_col = cross_table.columns[-1]

    # Unfortunately Pandas forces us to treat the singleton case differently.
    # "FutureWarning: In a future version of pandas, a length 1 tuple will be
    # returned when iterating over a groupby with a grouper equal to a list
    # of length 1. Don't supply a list with a single grouper to avoid this
    # warning."
    if len(cond_rv_names) == 1:
        weight_dict = {
            row[0]: row[-1]
            for row in cond_cross_table.itertuples(index=False)
        }
        cond_rv_name = cond_rv_names[0]
        groups = cross_table.groupby(cond_rv_name, observed=True, dropna=False)
        for key, group in groups:
            weights = group.iloc[:, -1]
            group_sum = weights.sum()

            if group_sum > 0:
                if key in weight_dict:
                    # Normal condition
                    weight_dict[key] /= group_sum
                else:
                    log(
                        f'Warning: non-zero condition value for {cond_rv_name!r} = {key!r} '
                        'cannot be applied to a cross-table'
                    )
            elif weight_dict.get(key, 0) > 0:
                log(
                    f'Warning: zero condition value for {cond_rv_name!r} = {key!r} '
                    'applied to a cross-table'
                )
                weight_dict[key] = 0

        def calculate_weight(_row):
            _key = _row[cond_rv_names[0]]
            return _row[weight_col] * weight_dict.get(_key, 0)

    else:
        weight_dict: Dict[Tuple] = {
            row[:-1]: row[-1]
            for row in cond_cross_table.itertuples(index=False)
        }
        groups = cross_table.groupby(cond_rv_names, observed=True, dropna=False)
        key: Tuple
        for key, group in groups:
            weights = group.iloc[:, -1]
            group_sum = weights.sum()

            if group_sum > 0:
                if key in weight_dict:
                    # Normal condition
                    weight_dict[key] /= group_sum
                else:
                    log(
                        f'Warning: non-zero condition value for {cond_rv_names} = {key!r} '
                        'cannot be applied to a cross-table'
                    )
            elif weight_dict.get(key, 0) > 0:
                log(
                    f'Warning: zero condition value for {cond_rv_names} = {key!r} '
                    'applied to a cross-table'
                )
                weight_dict[key] = 0

        def calculate_weight(_row):
            _key = tuple(_row[attr] for attr in cond_rv_names)
            return _row[weight_col] * weight_dict.get(_key, 0)

    new_weights = cross_table.apply(calculate_weight, axis=1)
    cross_table[weight_col] = new_weights
    return cross_table


def project_crosstab(crosstab: pd.DataFrame, rv_names: Iterable[str]) -> pd.DataFrame:
    """
    Get a cross-table, projected onto the given rvs.

    Ensures the column of the returned data frame are exactly rv_names + weight column.
    """
    rv_names = list(rv_names)
    rv_names_set = set(rv_names)
    weight_col = crosstab.columns[-1]

    if weight_col in rv_names_set:
        raise SynthorusError('rv_names cannot include the weight column')
    if len(rv_names_set) != len(rv_names):
        raise SynthorusError('duplicate rv names no permitted')
    del rv_names_set

    available_cols = {col for col in crosstab.columns[:-1]}
    for rv_name in rv_names:
        if rv_name not in available_cols:
            raise SynthorusError(f'rv name {rv_name!r} not available in cross-table')

    to_keep = rv_names + [weight_col]

    if len(rv_names) == 0:
        # Summing over all rvs
        return pd.DataFrame({
            weight_col: [crosstab[weight_col].sum()]
        })
    elif len(rv_names) == len(crosstab.columns) - 1:
        # No need to group
        return crosstab[to_keep]
    else:
        return crosstab[to_keep].groupby(rv_names, observed=True, dropna=False).sum().reset_index()


def functional_series_from_dataframe(
        dataframe: pd.DataFrame,
        input_column_names: Union[Iterable[str], str],
        column_function: Callable,
        dtype: Optional[Dtype] = None
) -> pd.Series:
    """
    Make a Pandas Series that is a function of other columns.
    The passed column_function is called for each row in the dataframe, where the arguments
    passed to column_function(...) are co-indexed with the columns of input_column_names.

    :param dataframe: is a Panda dataframe supplying the function input values.
    :param input_column_names: indicates columns in 'dataframe' whose values are arguments to column_function.
    :param column_function: is a function called to generate values for the new column.
    :param dtype: optional numpy datatype for the new series (generally more efficient if provided).
    :returns: the series containing computed values.
    """
    # Pandas really needs a list, so we ensure it!
    if isinstance(input_column_names, str):
        input_column_names = [input_column_names]
    else:
        input_column_names = list(input_column_names)

    series_length = dataframe.shape[0]

    if len(input_column_names) == 0:
        value = column_function()
        array = np.full(series_length, fill_value=value, dtype=dtype)
        return pd.Series(array)

    else:
        input_series = [dataframe[col] for col in input_column_names]

        if dtype is not None:
            # Try using np.fromiter.
            # np.fromiter can be a bit finicky with undocumented assumptions.
            # If it fails, it fails immediately, but if works it's efficient.
            try:
                array = np.fromiter(
                    (column_function(*args) for args in zip(*input_series)),
                    count=series_length,
                    dtype=dtype
                )
                return pd.Series(array)
            # noinspection PyBroadException
            except Exception:
                pass

        # General method (of last resort)
        return pd.Series(
            [column_function(*args) for args in zip(*input_series)],
            dtype=dtype
        )


def functional_series_from_series(
        series: pd.Series,
        column_function: Callable,
        dtype: Optional[Dtype] = None
) -> pd.Series:
    """
    Make a Pandas Series that is a function of other columns.
    The passed column_function is called for each row in the dataframe, where the arguments
    passed to column_function(...) are co-indexed with the columns of input_column_names.

    :param series: is a Panda series supplying the function input values.
    :param column_function: is a function, taking one argument.
    :param dtype: optional numpy datatype for the new series (generally more efficient if provided).
    :returns: the series containing computed values.
    """
    series_length = series.shape[0]

    if dtype is not None:
        # Try using np.fromiter.
        # np.fromiter can be a bit finicky with undocumented assumptions.
        # If it fails, it fails immediately, but if works it's efficient.
        try:
            array = np.fromiter(
                (column_function(arg) for arg in series),
                count=series_length,
                dtype=dtype
            )
            return pd.Series(array)
        # noinspection PyBroadException
        except Exception:
            pass

    # General method (of last resort)
    return pd.Series(
        [column_function(arg) for arg in series],
        dtype=dtype
    )
