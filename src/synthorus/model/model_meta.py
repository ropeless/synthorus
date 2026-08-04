from __future__ import annotations

from typing import Dict

from ck.pgm import State
from pydantic import BaseModel


class ModelMeta(BaseModel):
    """
    A record of model metadata calculated from the datasources of a models data sources.
    """
    rvs: Dict[str, RVMeta]
    crosstabs: Dict[str, CrosstabMeta]


class RVMeta(BaseModel):
    """
    A random variable in a ModelMeta.
    """
    name: str  # The name of the random variable

    clean_distribution: Dict[State, float]


class CrosstabMeta(BaseModel):
    """
    A cross-table in a ModelMeta.
    """
    name: str  # The name of the cross-table

    number_of_states: int  # The total number of possible states of the cross-table

    # Cross-table statistics, before noise is added
    clean_num_rows: int
    clean_min_weight: float
    clean_max_weight: float
    clean_total_weight: float

    # Cross-table statistics, after noise is added
    noisy_num_rows: int
    noisy_min_weight: float
    noisy_max_weight: float
    noisy_total_weight: float
    rows_lost: int
    rows_added: int

    @property
    def clean_num_suppressed(self) -> int:
        return self.number_of_states - self.clean_num_rows

    @property
    def noisy_num_suppressed(self) -> int:
        return self.number_of_states - self.noisy_num_rows
