from io import StringIO

import pandas as pd

from synthorus.dataset import PandasDataset
from synthorus.utils.string_extras import unindent
from tests.helpers.unittest_fixture import Fixture, test_main


class PandasDatasetTest(Fixture):

    def test_with_weights_as_series(self):
        source = StringIO(unindent("""
            "Age","Sex",
            "0","Male",47069
            "0","Female",44443
            "1","Male",47785
            "1","Female",45457
            "2","Male",48413
            "2","Female",45172
        """))
        raw_dataset: pd.DataFrame = pd.read_csv(source)

        dataset = PandasDataset(raw_dataset[['Age', 'Sex']], raw_dataset.iloc[:, 2])

        self.assertEqual(dataset.has_weight, True)
        self.assertEqual(dataset.rvs, ('Age', 'Sex'))
        self.assertEqual(dataset.number_of_records(), 6)

        self.assertEqual(dataset.value_maybe_none('Age'), False)
        self.assertEqual(dataset.value_maybe_none('Sex'), False)

        self.assertEqual(dataset.value_min('Age'), 0)
        self.assertEqual(dataset.value_min('Sex'), 'Female')

        self.assertEqual(dataset.value_max('Age'), 2)
        self.assertEqual(dataset.value_max('Sex'), 'Male')

        self.assertEqual(set(dataset.value_set('Age')), {0, 1, 2})
        self.assertEqual(set(dataset.value_set('Sex')), {'Female', 'Male'})

        crosstab_sex: pd.DataFrame = dataset.crosstab(['Sex'])

        self.assertEqual(crosstab_sex.shape, (2, 2))
        self.assertEqual(crosstab_sex.columns[0], 'Sex')
        self.assertEqual(list(crosstab_sex.iloc[0, :]), ['Female', 44443 + 45457 + 45172])
        self.assertEqual(list(crosstab_sex.iloc[1, :]), ['Male', 47069 + 47785 + 48413])

    def test_with_weights_as_str(self):
        source = StringIO(unindent("""
            "Age","Sex","weights"
            "0","Male",47069
            "0","Female",44443
            "1","Male",47785
            "1","Female",45457
            "2","Male",48413
            "2","Female",45172
        """))
        raw_dataset: pd.DataFrame = pd.read_csv(source)

        dataset = PandasDataset(raw_dataset, 'weights')

        self.assertEqual(dataset.has_weight, True)
        self.assertEqual(dataset.rvs, ('Age', 'Sex'))
        self.assertEqual(dataset.number_of_records(), 6)

        self.assertEqual(dataset.value_maybe_none('Age'), False)
        self.assertEqual(dataset.value_maybe_none('Sex'), False)

        self.assertEqual(dataset.value_min('Age'), 0)
        self.assertEqual(dataset.value_min('Sex'), 'Female')

        self.assertEqual(dataset.value_max('Age'), 2)
        self.assertEqual(dataset.value_max('Sex'), 'Male')

        self.assertEqual(set(dataset.value_set('Age')), {0, 1, 2})
        self.assertEqual(set(dataset.value_set('Sex')), {'Female', 'Male'})

        crosstab_sex: pd.DataFrame = dataset.crosstab(['Sex'])

        self.assertEqual(crosstab_sex.shape, (2, 2))
        self.assertEqual(crosstab_sex.columns[0], 'Sex')
        self.assertEqual(list(crosstab_sex.iloc[0, :]), ['Female', 44443 + 45457 + 45172])
        self.assertEqual(list(crosstab_sex.iloc[1, :]), ['Male', 47069 + 47785 + 48413])

    def test_with_weights_as_int(self):
        source = StringIO(unindent("""
            "Age","Sex", "weights"
            "0","Male",47069
            "0","Female",44443
            "1","Male",47785
            "1","Female",45457
            "2","Male",48413
            "2","Female",45172
        """))
        raw_dataset: pd.DataFrame = pd.read_csv(source)

        dataset = PandasDataset(raw_dataset, -1)

        self.assertEqual(dataset.has_weight, True)
        self.assertEqual(dataset.rvs, ('Age', 'Sex'))
        self.assertEqual(dataset.number_of_records(), 6)

        self.assertEqual(dataset.value_maybe_none('Age'), False)
        self.assertEqual(dataset.value_maybe_none('Sex'), False)

        self.assertEqual(dataset.value_min('Age'), 0)
        self.assertEqual(dataset.value_min('Sex'), 'Female')

        self.assertEqual(dataset.value_max('Age'), 2)
        self.assertEqual(dataset.value_max('Sex'), 'Male')

        self.assertEqual(set(dataset.value_set('Age')), {0, 1, 2})
        self.assertEqual(set(dataset.value_set('Sex')), {'Female', 'Male'})

        crosstab_sex: pd.DataFrame = dataset.crosstab(['Sex'])

        self.assertEqual(crosstab_sex.shape, (2, 2))
        self.assertEqual(crosstab_sex.columns[0], 'Sex')
        self.assertEqual(list(crosstab_sex.iloc[0, :]), ['Female', 44443 + 45457 + 45172])
        self.assertEqual(list(crosstab_sex.iloc[1, :]), ['Male', 47069 + 47785 + 48413])


if __name__ == '__main__':
    test_main()
