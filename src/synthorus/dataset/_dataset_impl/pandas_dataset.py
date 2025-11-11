"""
An implementation of DataSource wrapping a Pandas DataFrame.
"""

from typing import Iterable, Union

import pandas as pd

from synthorus.dataset import Dataset
from synthorus.error import SynthorusError
from synthorus.utils.dataframe_extras import make_crosstab


class PandasDataset(Dataset):
    """
    A Pandas dataset is defined by a Pandas DataFrame object.
    """

    def __init__(
            self,
            dataframe: pd.DataFrame,
            weights: Union[pd.Series, str, int, None]
    ):
        if weights is None:
            pass
        elif isinstance(weights, pd.Series):
            weights = pd.to_numeric(weights)
        elif isinstance(weights, str):
            weight_col = weights
            rv_cols = [col for col in dataframe.columns if col != weight_col]
            weights = pd.to_numeric(dataframe[weight_col])
            dataframe = dataframe[rv_cols]
        elif isinstance(weights, int):
            weight_col = dataframe.columns[weights]
            rv_cols = [col for col in dataframe.columns if col != weight_col]
            weights = pd.to_numeric(dataframe[weight_col])
            dataframe = dataframe[rv_cols]
        else:
            # noinspection PyUnreachableCode
            raise SynthorusError(f'weights not understood: {weights!r}')

        super().__init__(dataframe.columns)
        self._dataframe = dataframe
        self._weights = weights

    @property
    def dataframe(self):
        return self._dataframe

    @property
    def has_weight(self) -> bool:
        return self._weights is not None

    def number_of_records(self) -> int:
        return self._dataframe.shape[0]

    def value_maybe_none(self, rv: str) -> bool:
        series = self._dataframe[rv]
        return series.isnull().values.any()

    def value_min(self, rv: str):
        series = self._dataframe[rv]
        return series.min()

    def value_max(self, rv: str):
        series = self._dataframe[rv]
        return series.max()

    def value_set(self, rv: str):
        series = self._dataframe[rv]
        return series.unique()

    def crosstab(self, rvs: Iterable[str]) -> pd.DataFrame:
        return make_crosstab(self._dataframe, rvs, self._weights)
