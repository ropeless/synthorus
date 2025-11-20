from itertools import chain
from typing import Dict, Tuple, Sequence

from ck.dataset.cross_table import CrossTable
from ck.pgm import PGM
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import DEFAULT_PGM_COMPILER

from synthorus.workflows.utility_measures import Measure
from tests.helpers.unittest_fixture import Fixture, test_main


class DataCatcherTest(Fixture):

    def test_hi(self) -> None:
        rvs = ['A']
        p = {
            (0,): 0.1,
            (1,): 0.2,
            (2,): 0.3,
            (3,): 0.4,
        }
        q = {
            (0,): 0.3,
            (1,): 0.3,
            (2,): 0.2,
            (3,): 0.2,
        }

        got: float = compute_measure(Measure.HI, rvs, p, p)
        self.assertAlmostEqual(got, 1.0)

        got: float = compute_measure(Measure.HI, rvs, p, p)
        self.assertAlmostEqual(got, 1.0)

        got: float = compute_measure(Measure.HI, rvs, p, q)
        self.assertAlmostEqual(got, 0.7)

        got: float = compute_measure(Measure.HI, rvs, q, p)
        self.assertAlmostEqual(got, 0.7)

    def test_kl(self) -> None:
        rvs = ['A']
        p = {
            (0,): 0.1,
            (1,): 0.2,
            (2,): 0.3,
            (3,): 0.4,
        }
        q = {
            (0,): 0.3,
            (1,): 0.3,
            (2,): 0.2,
            (3,): 0.2,
        }

        got: float = compute_measure(Measure.KL, rvs, p, p)
        self.assertAlmostEqual(got, 0.0)

        got: float = compute_measure(Measure.KL, rvs, p, p)
        self.assertAlmostEqual(got, 0.0)

        got: float = compute_measure(Measure.KL, rvs, p, q)
        self.assertAlmostEqual(got, 0.3)

        got: float = compute_measure(Measure.KL, rvs, q, p)
        self.assertAlmostEqual(got, 0.333985)

    def test_fhi(self) -> None:
        rvs = ['A']
        p = {
            (0,): 0.1,
            (1,): 0.2,
            (2,): 0.3,
            (3,): 0.4,
        }
        q = {
            (0,): 0.3,
            (1,): 0.3,
            (2,): 0.2,
            (3,): 0.2,
        }

        got: float = compute_measure(Measure.FHI, rvs, p, p)
        self.assertAlmostEqual(got, 1.0)

        got: float = compute_measure(Measure.FHI, rvs, p, p)
        self.assertAlmostEqual(got, 1.0)

        got: float = compute_measure(Measure.FHI, rvs, p, q)
        self.assertAlmostEqual(got, 0.7)

        got: float = compute_measure(Measure.FHI, rvs, q, p)
        self.assertAlmostEqual(got, 0.7)

    def test_pseudo_kl(self) -> None:
        rvs = ['A']
        p = {
            (0,): 0.1,
            (1,): 0.2,
            (2,): 0.3,
            (3,): 0.4,
        }
        q = {
            (0,): 0.3,
            (1,): 0.3,
            (2,): 0.2,
            (3,): 0.2,
        }

        got: float = compute_measure(Measure.PSEUDO_KL, rvs, p, p)
        self.assertAlmostEqual(got, 0.0)

        got: float = compute_measure(Measure.PSEUDO_KL, rvs, p, p)
        self.assertAlmostEqual(got, 0.0)

        got: float = compute_measure(Measure.PSEUDO_KL, rvs, p, q)
        self.assertAlmostEqual(got, 0.3)

        got: float = compute_measure(Measure.PSEUDO_KL, rvs, q, p)
        self.assertAlmostEqual(got, 0.333985)


def compute_measure(
        measure: Measure,
        rv_names: Sequence[str],
        p: Dict[Tuple[int, ...], float],
        q: Dict[Tuple[int, ...], float],
) -> float:
    rv_max_state: Dict[str, int] = {
        rv_name: max(instance[i] for instance in chain(p.keys(), q.keys()))
        for i, rv_name in enumerate(rv_names)
    }

    pgm = PGM()
    for rv_name in rv_names:
        pgm.new_rv(rv_name, rv_max_state[rv_name] + 1)

    crosstab_p = CrossTable(pgm.rvs, update=p.items())
    crosstab_q = CrossTable(pgm.rvs, update=q.items())

    f = pgm.new_factor(*pgm.rvs).set_dense()
    for instance, weight in crosstab_q.items():
        f[instance] = weight
    wmc_q = WMCProgram(DEFAULT_PGM_COMPILER(pgm))

    return measure(crosstab_p, wmc_q)


if __name__ == '__main__':
    test_main()
