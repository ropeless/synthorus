from __future__ import annotations

from enum import Enum
from typing import Sequence

from ck.dataset.cross_table import CrossTable
from ck.pgm import RandomVariable
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.probability import divergence
from ck.probability.cross_table_probability_space import CrossTableProbabilitySpace
from ck.probability.probability_space import ProbabilitySpace, Condition


class Measure(Enum):
    HI = ('histogram-intersection', divergence.hi)
    KL = ('KL-divergence', divergence.kl)
    FHI = ('factorised-histogram-intersection', divergence.fhi)
    PSEUDO_KL = ('pseudo-KL-divergence', divergence.pseudo_kl)

    def full_name(self) -> str:
        return self.value[0]

    def __call__(self, crosstab: CrossTable, wmc: WMCProgram) -> float:
        crosstab_pr = CrossTableProbabilitySpace(crosstab)
        wmc_pr = _ProjectedWMC(wmc, crosstab.rvs)
        return self.value[1](crosstab_pr, wmc_pr)


class _ProjectedWMC(ProbabilitySpace):
    def __init__(self, wmc: WMCProgram, rvs: Sequence[RandomVariable]):
        """
        Project a WMCProgram onto a smaller probability space.

        Assumes:
            `rvs` is a subset of `wmc.rvs`
        """
        self._rvs: Sequence[RandomVariable] = tuple(rvs)
        self._wmc: WMCProgram = wmc

    @property
    def rvs(self) -> Sequence[RandomVariable]:
        return self._rvs

    def wmc(self, *condition: Condition) -> float:
        """
        Assumes:
            `conditions` only relate to `self.rvs`.
        """
        return self._wmc.wmc(*condition)

    @property
    def z(self) -> float:
        return self._wmc.z
