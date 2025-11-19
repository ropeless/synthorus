from io import StringIO

import pandas as pd

from synthorus.dataset import read_table_builder
from synthorus.utils.string_extras import unindent
from tests.helpers.unittest_fixture import Fixture, test_main


class TableBuilderDatasetTest(Fixture):

    def test_it(self):
        source = StringIO(unindent("""
            Australian Bureau of Statistics
            
            "2021 Census - employment, income and education"
            "AGEP Age and SEXP Sex by STATE (UR)"
            "Counting: Person Records"
            
            Filters:
            "Default Summation","Person Records"
            "STATE (UR)","New South Wales"
            
            
            "Age","Sex",
            "0","Male",47069,
            ,"Female",44443,
            "1","Male",47785,
            ,"Female",45457,
            "2","Male",48413,
            ,"Female",45172,
            
            
            "INFO","Cells in this table have been randomly adjusted to avoid the release of confidential data. No reliance should be placed on small cells."
            
            
            "Copyright notice"
            "Licence notice"
        """))
        dataset: pd.DataFrame = read_table_builder(source)

        self.assertEqual(dataset.shape, (6, 3))
        self.assertEqual(list(dataset.columns)[:2], ['Age', 'Sex'])
        self.assertEqual(list(dataset.iloc[0, :]), [0, 'Male', 47069])
        self.assertEqual(list(dataset.iloc[1, :]), [0, 'Female', 44443])
        self.assertEqual(list(dataset.iloc[2, :]), [1, 'Male', 47785])
        self.assertEqual(list(dataset.iloc[3, :]), [1, 'Female', 45457])
        self.assertEqual(list(dataset.iloc[4, :]), [2, 'Male', 48413])
        self.assertEqual(list(dataset.iloc[5, :]), [2, 'Female', 45172])


if __name__ == '__main__':
    test_main()
