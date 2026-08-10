"""
A script convert CSV & TSV data source files to other formats.
"""

from pathlib import Path
import pandas as pd

from synthorus.dataset import read_table_builder
from synthorus.utils.dataframe_extras import read_csv
from synthorus_demos.demo_files import DATASET_FILES

SKIP_TABLE_BUILDER = True


def main():
    if not isinstance(DATASET_FILES, Path) or not DATASET_FILES.is_dir():
        raise TypeError('DATASET_FILES must be a Path to a directory')

    root_path = Path(DATASET_FILES)
    for file_path in root_path.iterdir():
        dataframe: pd.DataFrame
        save_csv_file: bool = False
        if file_path.suffix == '.tablebuilder':
            if SKIP_TABLE_BUILDER:
                continue
            dataframe = read_table_builder(file_path)
            save_csv_file = True
        elif file_path.suffix == '.csv':
            dataframe = read_csv(file_path)
        elif file_path.suffix == '.tsv':
            dataframe = read_csv(file_path, sep='\t')
            save_csv_file = True
        else:
            continue

        print(file_path)
        stem = file_path.stem
        dataframe.to_pickle(str(root_path / (stem + '.pkl')))
        dataframe.to_parquet(root_path / (stem + '.parquet'))
        dataframe.to_feather(root_path / (stem + '.feather'))
        if save_csv_file:
            dataframe.to_csv(root_path / (stem + '.csv'), index=False)

    print('Done.')


if __name__ == '__main__':
    main()
