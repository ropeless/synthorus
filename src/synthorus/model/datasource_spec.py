from __future__ import annotations

from importlib.abc import Traversable
from pathlib import Path
from typing import List, Sequence, Self, Set

from pydantic import BaseModel, NonNegativeFloat, model_validator

from synthorus.dataset import Dataset
from synthorus.model.dataset_spec import DatasetSpec


class DatasourceSpec(BaseModel):
    """
    A DataSpec is a serializable object defining a synthorus dataset.
    It does not directly hold the data. The data is defined by a
    DatasetSpec object which can be used to create a Dataset
    object that provides an API to the actual data.
    """

    sensitivity: NonNegativeFloat  # Differential privacy parameter for the dataset.
    rvs: List[str]  # All rvs that are provided by the datasource
    dataset_spec: DatasetSpec  # The definition for getting the data
    non_distribution_rvs: List[str] = []  # The rvs that should _not_ be considered as providing a distribution

    def dataset(self, roots: Sequence[Path | Traversable] = ()) -> Dataset:
        """
        Load the dataset for this datasource.

        Delegate to self.dataset_spec.dataset(roots)

        Args:
            roots: list of root paths passed to the dataset spec.
        """
        return self.dataset_spec.dataset(roots)

    @model_validator(mode='after')
    def validate_data_spec(self) -> Self:
        # Ensure rvs are unique.
        rvs_set: Set[str] = set(self.rvs)
        if len(rvs_set) != len(self.rvs):
            raise ValueError('dataset rvs must be unique within a dataset')

        # Ensure all non-distribution rvs are in rvs.
        if any(rv not in rvs_set for rv in self.non_distribution_rvs):
            raise ValueError('non-distribution rvs must be a subset of dataset rvs')

        return self
