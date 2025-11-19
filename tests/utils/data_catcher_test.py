import json

import pandas as pd

from synthorus.utils.data_catcher import RamDataCatcher
from tests.helpers.unittest_fixture import Fixture, test_main


class DataCatcherTest(Fixture):

    def test_empty_ram_data_catcher(self):
        data = RamDataCatcher()
        self.assertEqual(0, len(data))
        self.assertEqual(0, len(data.columns))

    def test_simple_ram_data_catcher(self):
        data = RamDataCatcher()

        data.append().set({'a': 2, 'b': 5})
        data.append().set({'a': 3, 'c': 6})

        self.assertEqual(2, len(data))
        self.assertEqual(3, len(data.columns))

        self.assertEqual(data.get_column_list('a'), [2, 3])
        self.assertEqual(data.get_column_list('b'), [5, None])
        self.assertEqual(data.get_column_list('c'), [None, 6])

    def test_simple_ram_data_catcher_json(self):
        data = RamDataCatcher()

        data.append().set_kwargs(a=2, b=5)
        data.append().set_kwargs(a=3, c=6)

        as_json_dict = json.loads(data.as_json())

        self.assertEqual(set(as_json_dict.keys()), {'columns', 'records'})
        self.assertEqual(as_json_dict['columns'], ['a', 'b', 'c'])
        records = as_json_dict['records']
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0], [2, 5, None])
        self.assertEqual(records[1], [3, None, 6])

    def test_simple_ram_data_catcher_pandas(self):
        data = RamDataCatcher()

        data.append().set({'a': 2, 'b': 5})
        data.append().set({'a': 3, 'c': 6})

        df: pd.DataFrame = data.as_dataframe()

        NAN = float('nan')
        self.assertEqual(list(df.columns), ['a', 'b', 'c'])
        self.assertEqual(len(df), 2)
        self.assertArrayEqual(df.iloc[0, :], [2, 5, NAN], nan_equality=True)
        self.assertArrayEqual(df.iloc[1, :], [3, NAN, 6], nan_equality=True)


if __name__ == '__main__':
    test_main()
