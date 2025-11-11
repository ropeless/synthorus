from collections import OrderedDict

import numpy as np
import pandas as pd

from synthorus.utils.dataframe_extras import adjust_cross_table, project_crosstab, functional_series_from_dataframe, \
    make_crosstab
from tests.helpers.unittest_fixture import Fixture, test_main


class TestDataframeExtras(Fixture):

    def test_adjust_cross_table_cond_1(self):
        # Test conditioning with a single conditioning rv.

        data = OrderedDict()
        data['y'] = [0, 1]
        data['w'] = [3, 7]
        cond_cross_table = pd.DataFrame(data)

        data = OrderedDict()
        data['x'] = [0, 1, 0, 1, 0, 1, 0, 1]
        data['y'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['z'] = [0, 0, 1, 1, 0, 0, 1, 1]
        data['w'] = [1, 2, 3, 4, 5, 6, 7, 8]
        cross_table = pd.DataFrame(data)

        adjusted_cross_table = adjust_cross_table(cross_table, cond_cross_table)

        adjusted_cross_table = adjusted_cross_table[['x', 'z', 'y', 'w']]
        weight_dict = {}
        for row in adjusted_cross_table.itertuples(index=False):
            key = row[:-1]
            weight = row[-1]
            weight_dict[key] = weight

        #                                   x  z  y    weight
        self.assertAlmostEqual(weight_dict[(0, 0, 0)], 1 / 10 * 3)
        self.assertAlmostEqual(weight_dict[(1, 0, 0)], 2 / 10 * 3)
        self.assertAlmostEqual(weight_dict[(0, 1, 0)], 3 / 10 * 3)
        self.assertAlmostEqual(weight_dict[(1, 1, 0)], 4 / 10 * 3)
        self.assertAlmostEqual(weight_dict[(0, 0, 1)], 5 / 26 * 7)
        self.assertAlmostEqual(weight_dict[(1, 0, 1)], 6 / 26 * 7)
        self.assertAlmostEqual(weight_dict[(0, 1, 1)], 7 / 26 * 7)
        self.assertAlmostEqual(weight_dict[(1, 1, 1)], 8 / 26 * 7)

    def test_adjust_cross_table_cond_2(self):
        # Test conditioning with a pair of conditioning rvs.

        data = OrderedDict()
        data['x'] = [0, 1, 0, 1]
        data['y'] = [0, 0, 1, 1]
        data['w'] = [3, 5, 7, 9]
        cond_cross_table = pd.DataFrame(data)

        data = OrderedDict()
        data['a'] = [0, 1, 0, 1, 0, 1, 0, 1]
        data['x'] = [0, 0, 1, 1, 0, 0, 1, 1]
        data['y'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['w'] = [1, 2, 1, 3, 1, 4, 1, 5]
        cross_table = pd.DataFrame(data)

        adjusted_cross_table = adjust_cross_table(cross_table, cond_cross_table)

        adjusted_cross_table = adjusted_cross_table[['a', 'x', 'y', 'w']]
        weight_dict = {}
        for row in adjusted_cross_table.itertuples(index=False):
            key = row[:-1]
            weight = row[-1]
            weight_dict[key] = weight

        #                                   a  x  y    weight
        self.assertAlmostEqual(weight_dict[(0, 0, 0)], 1 / 3 * 3)
        self.assertAlmostEqual(weight_dict[(1, 0, 0)], 2 / 3 * 3)
        self.assertAlmostEqual(weight_dict[(0, 1, 0)], 1 / 4 * 5)
        self.assertAlmostEqual(weight_dict[(1, 1, 0)], 3 / 4 * 5)
        self.assertAlmostEqual(weight_dict[(0, 0, 1)], 1 / 5 * 7)
        self.assertAlmostEqual(weight_dict[(1, 0, 1)], 4 / 5 * 7)
        self.assertAlmostEqual(weight_dict[(0, 1, 1)], 1 / 6 * 9)
        self.assertAlmostEqual(weight_dict[(1, 1, 1)], 5 / 6 * 9)

    def test_project_1(self):
        data = OrderedDict()
        data['a'] = [0, 1, 0, 1, 0, 1, 0, 1]
        data['x'] = [0, 0, 1, 1, 0, 0, 1, 1]
        data['y'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['w'] = [1, 2, 3, 3, 1, 4, 1, 5]
        cross_table = pd.DataFrame(data)

        projected = project_crosstab(cross_table, ['y'])
        self.assertEqual(projected.shape, (2, 2))
        self.assertEqual(projected.columns.tolist(), ['y', 'w'])

        projected = projected.sort_values(['y'])
        rows = iter(projected.itertuples(index=False))

        self.assertEqual(tuple(next(rows)), (0, 1 + 2 + 3 + 3))
        self.assertEqual(tuple(next(rows)), (1, 1 + 4 + 1 + 5))
        self.assertIterFinished(rows)

    def test_project_two(self):
        data = OrderedDict()
        data['a'] = [0, 1, 0, 1, 0, 1, 0, 1]
        data['x'] = [0, 0, 1, 1, 0, 0, 1, 1]
        data['y'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['w'] = [1, 2, 3, 3, 1, 4, 1, 5]
        cross_table = pd.DataFrame(data)

        projected = project_crosstab(cross_table, ['a', 'x'])
        self.assertEqual(projected.shape, (4, 3))
        self.assertEqual(projected.columns.tolist(), ['a', 'x', 'w'])

        projected = projected.sort_values(['a', 'x'])
        rows = iter(projected.itertuples(index=False))

        self.assertEqual(tuple(next(rows)), (0, 0, 2))
        self.assertEqual(tuple(next(rows)), (0, 1, 4))
        self.assertEqual(tuple(next(rows)), (1, 0, 6))
        self.assertEqual(tuple(next(rows)), (1, 1, 8))
        self.assertIterFinished(rows)

    def test_project_rearranged(self):
        data = OrderedDict()
        data['a'] = [0, 1, 0, 1, 0, 1, 0, 1]
        data['x'] = [0, 0, 1, 1, 0, 0, 1, 1]
        data['y'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['w'] = [1, 2, 3, 3, 1, 4, 1, 5]
        cross_table = pd.DataFrame(data)
        self.assertEqual(cross_table.columns.tolist(), ['a', 'x', 'y', 'w'])

        projected = project_crosstab(cross_table, ['y', 'x', 'a'])
        self.assertEqual(projected.shape, (8, 4))
        self.assertEqual(projected.columns.tolist(), ['y', 'x', 'a', 'w'])

        projected = projected.sort_values(['y', 'x', 'a'])
        rows = iter(projected.itertuples(index=False))

        self.assertEqual(tuple(next(rows)), (0, 0, 0, 1))
        self.assertEqual(tuple(next(rows)), (0, 0, 1, 2))
        self.assertEqual(tuple(next(rows)), (0, 1, 0, 3))
        self.assertEqual(tuple(next(rows)), (0, 1, 1, 3))
        self.assertEqual(tuple(next(rows)), (1, 0, 0, 1))
        self.assertEqual(tuple(next(rows)), (1, 0, 1, 4))
        self.assertEqual(tuple(next(rows)), (1, 1, 0, 1))
        self.assertEqual(tuple(next(rows)), (1, 1, 1, 5))
        self.assertIterFinished(rows)

    def test_functional_series_0(self):
        data = OrderedDict()
        data['x'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['y'] = ['a', 'a', 'b', 'b', 'a', 'a', 'b', 'b']
        data['z'] = ['y', 'n', 'y', 'n', 'y', 'n', 'y', 'n']
        df = pd.DataFrame(data)

        def f():
            return 'ok'

        series = functional_series_from_dataframe(df, [], f)

        self.assertEqual(series.shape, (8,))
        self.assertEqual(series.tolist(), ['ok'] * 8)

    def test_functional_series_1(self):
        data = OrderedDict()
        data['x'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['y'] = ['a', 'a', 'b', 'b', 'a', 'a', 'b', 'b']
        data['z'] = ['y', 'n', 'y', 'n', 'y', 'n', 'y', 'n']
        df = pd.DataFrame(data)

        def f(y):
            return 'q' if y == 'a' else 'r'

        series = functional_series_from_dataframe(df, ['y'], f)

        self.assertEqual(series.shape, (8,))
        self.assertEqual(series.tolist(), ['q', 'q', 'r', 'r', 'q', 'q', 'r', 'r'])

    def test_functional_series_1_with_dtype(self):
        data = OrderedDict()
        data['x'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['y'] = ['a', 'a', 'b', 'b', 'a', 'a', 'b', 'b']
        data['z'] = ['y', 'n', 'y', 'n', 'y', 'n', 'y', 'n']
        df = pd.DataFrame(data)

        def f(y):
            return 8 if y == 'a' else 9

        series = functional_series_from_dataframe(df, ['y'], f, dtype=np.intc)

        self.assertEqual(series.shape, (8,))
        self.assertEqual(series.tolist(), [8, 8, 9, 9, 8, 8, 9, 9])
        self.assertEqual(series.dtype, np.intc)

    def test_functional_series_2(self):
        data = OrderedDict()
        data['x'] = [0, 0, 0, 0, 1, 1, 1, 1]
        data['y'] = ['a', 'a', 'b', 'b', 'a', 'a', 'b', 'b']
        data['z'] = ['y', 'n', 'y', 'n', 'y', 'n', 'y', 'n']
        df = pd.DataFrame(data)

        def f(z, y):
            if (y, z) == ('a', 'y'):
                return 1
            if (y, z) == ('a', 'n'):
                return 2
            if (y, z) == ('b', 'y'):
                return 3
            if (y, z) == ('b', 'n'):
                return 4
            return None

        series = functional_series_from_dataframe(df, ['z', 'y'], f)

        self.assertEqual(series.shape, (8,))
        self.assertEqual(series.tolist(), [1, 2, 3, 4, 1, 2, 3, 4])

    def test_make_crosstab_no_rvs_weighted(self):
        data_source = pd.DataFrame({
            'x': ['a', 'b', 'c'],
            'w': [1.0, 2.3, 4.5],
        })

        crosstab = make_crosstab(data_source, [], weights='w')
        self.assertEqual(crosstab.shape, (1, 1))
        self.assertEqual(crosstab.iloc[0, 0], 7.8)

    def test_make_crosstab_no_rvs_unweighted(self):
        data_source = pd.DataFrame({
            'x': ['a', 'b', 'c'],
            'w': [1.0, 2.3, 4.5],
        })

        crosstab = make_crosstab(data_source, [], weights=None)
        self.assertEqual(crosstab.shape, (1, 1))
        self.assertEqual(crosstab.iloc[0, 0], 3)


if __name__ == '__main__':
    test_main()
