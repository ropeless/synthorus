from abc import ABC, abstractmethod
from typing import Sequence, Iterable, Collection

import pandas as pd
from ck.pgm import State


class Dataset(ABC):
    """
    A Synthorus dataset is a collection of reference data for building a probabilistic model.

    A Dataset is an abstract class. Every Dataset has:
    * a list of random variables (which are just names, i.e., strings)
    * basic dataset metadata
    * a method to make cross-tables.
    """

    def __init__(self, rvs: Sequence[str]):
        self._rvs = tuple(rvs)

    @property
    def rvs(self) -> Sequence[str]:
        """
        Returns:
            a tuple of all random variable names.
        """
        return self._rvs

    @abstractmethod
    def number_of_records(self) -> int:
        """
        Returns:
            number of records in the dataset.
        """

    @abstractmethod
    def value_maybe_none(self, rv: str) -> bool:
        """
        Does the named random variable have None (i.e. missing)
        as a possible value?
        """

    @abstractmethod
    def value_min(self, rv: str) -> State:
        """
        The minimum value of the random variable.
        """

    @abstractmethod
    def value_max(self, rv: str) -> State:
        """
        The maximum value of the random variable.
        """

    @abstractmethod
    def value_set(self, rv: str) -> Collection[str]:
        """
        The set of all possible values of the random variable.
        """

    @abstractmethod
    def crosstab(self, rvs: Iterable[str]) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame representing the crosstab
        of the given rvs. The rvs of the result are
        as provided as columns, with an additional column (crosstab.iloc[:, -1])
        containing the numeric weights.

        This method does not add noise to crosstab. If noise is required,
        it needs to be added subsequently.
        """
