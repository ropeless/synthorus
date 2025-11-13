from collections import OrderedDict
from typing import List

import pandas as pd

from synthorus.model.cross_table.noise import NoNoise, BasicLaplaceNoise, LaplaceNoise
from synthorus.model.cross_table.safe_random import SafeRandom
from synthorus.model.model_index import RVIndex
from synthorus.utils.print_function import NO_LOG
from tests.helpers.unittest_fixture import Fixture, test_main


class NoiseTest(Fixture):

    @staticmethod
    def sort_cross_table(crosstab: pd.DataFrame) -> pd.DataFrame:
        rvs = sorted(crosstab.columns[:-1])
        crosstab = crosstab[rvs + [crosstab.columns[-1]]]
        crosstab = crosstab.sort_values(rvs)
        return crosstab

    @staticmethod
    def rvs_from_crosstab(crosstab: pd.DataFrame, datasource: str) -> List[RVIndex]:
        rvs = [
            RVIndex(
                name=rv_name,
                states=list(crosstab[rv_name].unique()),
                primary_datasource=datasource,
                all_datasources=[datasource],
            )
            for rv_name in crosstab.columns[:-1]
        ]
        return rvs

    def assertEqualCrossTables(self, crosstab_1: pd.DataFrame, crosstab_2: pd.DataFrame):
        self.assertEqual(crosstab_1.shape, crosstab_2.shape, 'same shape')

        # Permutation of rows and columns are allowed, so we sort them.
        crosstab_1 = self.sort_cross_table(crosstab_1)
        crosstab_2 = self.sort_cross_table(crosstab_2)

        # Check rvs are equal (ignore weight column)
        rvs_1 = crosstab_1.columns[:-1].to_list()
        rvs_2 = crosstab_2.columns[:-1].to_list()
        self.assertEqual(rvs_1, rvs_2)

        # Check rows are equal
        row_pairs = zip(
            crosstab_1.itertuples(index=False),
            crosstab_2.itertuples(index=False)
        )
        for row_1, row_2 in row_pairs:
            self.assertEqual(row_1, row_2)

    def test_no_noise(self):
        data = OrderedDict()
        data['x'] = [0, 1, 0, 1, 0, 1, 0, 1]
        data['y'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['z'] = [0, 0, 1, 1, 0, 0, 1, 1]
        data['w'] = [1, 2, 3, 4, 5, 6, 7, 8]
        cross_table = pd.DataFrame(data)

        sensitivity = 1
        epsilon = 1
        min_cell_size = 5
        no_noise = NoNoise()
        noise_result = no_noise(cross_table.copy(), sensitivity, epsilon, min_cell_size)
        noisy_cross_table = noise_result.cross_table
        self.assertEqualCrossTables(noisy_cross_table, cross_table)
        self.assertEqual(noise_result.rows_original, 8)
        self.assertEqual(noise_result.rows_added, 0)
        self.assertEqual(noise_result.rows_lost, 0)
        self.assertEqual(noise_result.rows_final, 8)

    def test_basic_laplace_no_change(self):
        data = OrderedDict()
        data['x'] = [0, 1, 0, 1, 0, 1, 0, 1]
        data['y'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['z'] = [0, 0, 1, 1, 0, 0, 1, 1]
        data['w'] = [1, 2, 3, 4, 5, 6, 7, 8]
        cross_table = pd.DataFrame(data)

        epsilon = 1.0
        min_cell_size = 0
        sensitivity = 0

        noiser = BasicLaplaceNoise(SafeRandom(n=4))

        noise_result = noiser(cross_table.copy(), sensitivity, epsilon, min_cell_size)
        noisy_cross_table = noise_result.cross_table
        self.assertEqualCrossTables(cross_table, noisy_cross_table)
        self.assertEqual(noise_result.rows_original, 8)
        self.assertEqual(noise_result.rows_added, 0)
        self.assertEqual(noise_result.rows_lost, 0)
        self.assertEqual(noise_result.rows_final, 8)

    def test_basic_laplace_min_cell_size(self):
        data = OrderedDict()
        data['x'] = [0, 1, 0, 1, 0, 1, 0, 1]
        data['y'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['z'] = [0, 0, 1, 1, 0, 0, 1, 1]
        data['w'] = [1, 2, 3, 4, 5, 6, 7, 8]
        cross_table = pd.DataFrame(data)

        epsilon = 1.0
        min_cell_size = 5
        sensitivity = 0

        noiser = BasicLaplaceNoise(SafeRandom(n=4))

        noise_result = noiser(cross_table.copy(), sensitivity, epsilon, min_cell_size)
        noisy_cross_table = noise_result.cross_table

        self.assertEqual(noisy_cross_table.shape, (4, 4))
        self.assertEqual(noise_result.rows_original, 8)
        self.assertEqual(noise_result.rows_added, 0)
        self.assertEqual(noise_result.rows_lost, 4)
        self.assertEqual(noise_result.rows_final, 4)

        cross_table = self.sort_cross_table(cross_table)
        noisy_cross_table = self.sort_cross_table(noisy_cross_table)

        iter_1 = iter(cross_table.itertuples(index=False))
        iter_2 = iter(noisy_cross_table.itertuples(index=False))

        for row_1 in iter_1:
            weight = row_1[-1]
            if weight >= min_cell_size:
                row_2 = next(iter_2)
                self.assertEqual(row_1, row_2)

    def test_basic_laplace_weight_change(self):
        data = OrderedDict()
        data['x'] = [0, 1, 0, 1, 0, 1, 0, 1]
        data['y'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['z'] = [0, 0, 1, 1, 0, 0, 1, 1]
        data['w'] = [1, 2, 3, 4, 5, 6, 7, 8]
        cross_table = pd.DataFrame(data)

        epsilon = 10  # this will add very little noise
        min_cell_size = 0
        sensitivity = 1

        noiser = BasicLaplaceNoise(SafeRandom(n=4))

        noise_result = noiser(cross_table.copy(), sensitivity, epsilon, min_cell_size)
        noisy_cross_table = noise_result.cross_table

        self.assertEqual(noisy_cross_table.shape, cross_table.shape)
        self.assertEqual(noise_result.rows_original, 8)
        self.assertEqual(noise_result.rows_added, 0)
        self.assertEqual(noise_result.rows_lost, 0)
        self.assertEqual(noise_result.rows_final, 8)

        cross_table = self.sort_cross_table(cross_table)
        noisy_cross_table = self.sort_cross_table(noisy_cross_table)

        # Check rows are equal, excluding weight
        # Check that weights changed
        row_pairs = zip(
            cross_table.itertuples(index=False),
            noisy_cross_table.itertuples(index=False)
        )
        weight_change = 0
        for row_1, row_2 in row_pairs:
            self.assertEqual(row_1[:-1], row_2[:-1])
            weight_change += abs(row_1[-1] - row_2[-1])
        self.assertTrue(weight_change > 0)

    def test_laplace_no_change(self):
        data = OrderedDict()
        data['x'] = [0, 1, 0, 1, 0, 1, 0, 1]
        data['y'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['z'] = [0, 0, 1, 1, 0, 0, 1, 1]
        data['w'] = [1, 2, 3, 4, 5, 6, 7, 8]
        cross_table = pd.DataFrame(data)

        epsilon = 1.0
        min_cell_size = 0
        sensitivity = 0

        rvs = self.rvs_from_crosstab(cross_table, datasource='my_datasource')
        noiser = LaplaceNoise(safe_random=SafeRandom(n=4), rvs=rvs, max_add_rows=1000, log=NO_LOG)

        noise_result = noiser(cross_table.copy(), sensitivity, epsilon, min_cell_size)
        noisy_cross_table = noise_result.cross_table
        self.assertEqualCrossTables(noisy_cross_table, cross_table)
        self.assertEqual(noise_result.rows_original, 8)
        self.assertEqual(noise_result.rows_added, 0)
        self.assertEqual(noise_result.rows_lost, 0)
        self.assertEqual(noise_result.rows_final, 8)

    def test_laplace_min_cell_size(self):
        data = OrderedDict()
        data['x'] = [0, 1, 0, 1, 0, 1, 0, 1]
        data['y'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['z'] = [0, 0, 1, 1, 0, 0, 1, 1]
        data['w'] = [1, 2, 3, 4, 5, 6, 7, 8]
        cross_table = pd.DataFrame(data)

        epsilon = 1.0
        min_cell_size = 5
        sensitivity = 0

        rvs = self.rvs_from_crosstab(cross_table, datasource='my_datasource')
        noiser = LaplaceNoise(safe_random=SafeRandom(n=4), rvs=rvs, max_add_rows=1000, log=NO_LOG)

        noise_result = noiser(cross_table.copy(), sensitivity, epsilon, min_cell_size)
        noisy_cross_table = noise_result.cross_table

        self.assertEqual(noisy_cross_table.shape, (4, 4))
        self.assertEqual(noise_result.rows_original, 8)
        self.assertEqual(noise_result.rows_added, 0)
        self.assertEqual(noise_result.rows_lost, 4)
        self.assertEqual(noise_result.rows_final, 4)

        cross_table = self.sort_cross_table(cross_table)
        noisy_cross_table = self.sort_cross_table(noisy_cross_table)

        iter_1 = iter(cross_table.itertuples(index=False))
        iter_2 = iter(noisy_cross_table.itertuples(index=False))

        for row_1 in iter_1:
            weight = row_1[-1]
            if weight >= min_cell_size:
                row_2 = next(iter_2)
                self.assertEqual(row_1, row_2)

    def test_laplace_weight_change_full_support(self):
        # Test LaplaceNoise where the cross table has no suppressed rows
        # in this case, the basic Laplace method gets called.

        data = OrderedDict()
        data['x'] = [0, 1, 0, 1, 0, 1, 0, 1]
        data['y'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['z'] = [0, 0, 1, 1, 0, 0, 1, 1]
        data['w'] = [99, 99, 99, 99, 99, 99, 99, 99]
        cross_table = pd.DataFrame(data)

        epsilon = 1.0
        min_cell_size = 0
        sensitivity = 1

        rvs = self.rvs_from_crosstab(cross_table, datasource='my_datasource')
        noiser = LaplaceNoise(safe_random=SafeRandom(n=4), rvs=rvs, max_add_rows=1000, log=NO_LOG)

        noise_result = noiser(cross_table.copy(), sensitivity, epsilon, min_cell_size)
        noisy_cross_table = noise_result.cross_table

        self.assertEqual(noisy_cross_table.shape, cross_table.shape)
        self.assertEqual(noise_result.rows_original, 8)
        self.assertEqual(noise_result.rows_added, 0)
        self.assertEqual(noise_result.rows_lost, 0)
        self.assertEqual(noise_result.rows_final, 8)

        cross_table = self.sort_cross_table(cross_table)
        noisy_cross_table = self.sort_cross_table(noisy_cross_table)

        # Check rows are equal, excluding weight
        # Check that weights changed
        row_pairs = zip(
            cross_table.itertuples(index=False),
            noisy_cross_table.itertuples(index=False)
        )
        weight_change = 0
        for row_1, row_2 in row_pairs:
            self.assertEqual(row_1[:-1], row_2[:-1])
            weight_change += abs(row_1[-1] - row_2[-1])
        self.assertTrue(weight_change > 0)

    def test_laplace_weight_change_suppressed_1(self):
        # Test LaplaceNoise where the cross table has one suppressed row
        # in this case, the all_rows_method gets called.

        data = OrderedDict()
        data['x'] = [0, 1, 0, 1, 0, 1, 0]
        data['y'] = [0, 0, 0, 0, 1, 1, 1]
        data['z'] = [0, 0, 1, 1, 0, 0, 1]
        data['w'] = [99, 99, 99, 99, 99, 99, 99]
        cross_table = pd.DataFrame(data)

        epsilon = 1.0
        min_cell_size = 0
        sensitivity = 1

        rvs = self.rvs_from_crosstab(cross_table, datasource='my_datasource')
        noiser = LaplaceNoise(safe_random=SafeRandom(n=4), rvs=rvs, max_add_rows=1000, log=NO_LOG)

        noise_result = noiser(cross_table.copy(), sensitivity, epsilon, min_cell_size)
        noisy_cross_table = noise_result.cross_table

        # One new row may be added
        orig_rows = cross_table.shape[0]
        new_rows = noisy_cross_table.shape[0]
        self.assertIn(new_rows, {orig_rows, orig_rows + 1})
        self.assertEqual(noisy_cross_table.shape[1], cross_table.shape[1])

        if new_rows == 8:
            self.assertEqual(noise_result.rows_original, 7)
            self.assertEqual(noise_result.rows_added, 1)
            self.assertEqual(noise_result.rows_lost, 0)
            self.assertEqual(noise_result.rows_final, 8)
        else:
            self.assertEqual(noise_result.rows_original, 7)
            self.assertEqual(noise_result.rows_added, 0)
            self.assertEqual(noise_result.rows_lost, 0)
            self.assertEqual(noise_result.rows_final, 7)

        cross_table = self.sort_cross_table(cross_table)
        noisy_cross_table = self.sort_cross_table(noisy_cross_table)

        # Check rows are equal, excluding weight and the added row
        # Check that weights changed
        iter_1 = iter(cross_table.itertuples(index=False))
        iter_2 = iter(noisy_cross_table.itertuples(index=False))

        weight_change = 0
        for row_2 in iter_2:
            if row_2[:3] == (1, 1, 1):
                # This is the added row
                pass
            else:
                row_1 = next(iter_1)
                self.assertEqual(row_1[:-1], row_2[:-1])
                weight_change += abs(row_1[-1] - row_2[-1])

        self.assertTrue(weight_change > 0)

    def test_laplace_weight_change_suppressed_7(self):
        # Test LaplaceNoise where the cross table has 25 suppressed rows of 27.
        # Explicitly calls the decomposition_method.

        data = OrderedDict()
        data['x'] = [0, 1, 2]
        data['y'] = [0, 1, 2]
        data['z'] = [0, 1, 2]
        data['w'] = [7, 8, 9]
        cross_table = pd.DataFrame(data)

        epsilon = 1.0
        min_cell_size = 1
        sensitivity = 1

        rvs = self.rvs_from_crosstab(cross_table, datasource='my_datasource')
        noiser = LaplaceNoise(safe_random=SafeRandom(n=4), rvs=rvs, max_add_rows=1000, log=print)

        noise_result = noiser.decomposition_method(cross_table.copy(), sensitivity, epsilon, min_cell_size)
        noisy_cross_table = noise_result.cross_table

        # Rows should be added
        self.assertTrue(noisy_cross_table.shape[0] > cross_table.shape[0])
        self.assertEqual(noisy_cross_table.shape[1], cross_table.shape[1])

        cross_table = self.sort_cross_table(cross_table)
        noisy_cross_table = self.sort_cross_table(noisy_cross_table)

        # Check rows are equal, excluding weight and the added row
        # Check that weights changed
        iter_1 = iter(cross_table.itertuples(index=False))
        iter_2 = iter(noisy_cross_table.itertuples(index=False))

        weight_change = 0
        row_change = 0
        for row_2 in iter_2:
            if row_2[:3] in [(0, 0, 0), (1, 1, 1), (2, 2, 2)]:
                # This is an existing row
                row_1 = next(iter_1)
                self.assertEqual(row_1[:-1], row_2[:-1])
                weight = row_2[-1]
                self.assertTrue(weight >= min_cell_size)
                weight_change += abs(row_1[-1] - weight)
            else:
                # This is a new row
                weight = row_2[-1]
                self.assertTrue(weight >= min_cell_size)
                row_change += 1

        expected_row_change = noisy_cross_table.shape[0] - cross_table.shape[0]
        self.assertTrue(weight_change > 0)
        self.assertEqual(row_change, expected_row_change)


if __name__ == '__main__':
    test_main()
