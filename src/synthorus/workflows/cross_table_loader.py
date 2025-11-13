from pathlib import Path
from typing import Dict

import pandas as pd


def save_cross_table(dataframe: pd.DataFrame, directory: Path, cross_table_name: str) -> None:
    """
    Save a cross-table to a directory, which can latter be
    loaded using load_cross_table or CrossTableLoader.

    Args:
        dataframe: The cross-table data.
        directory: where to save the cross-table.
        cross_table_name: name of the cross-table.
    """
    file_path: Path = directory / (cross_table_name + '.pkl')
    dataframe.to_pickle(file_path)


def load_cross_table(directory: Path, cross_table_name: str) -> pd.DataFrame:
    """
    Load a cross-table from a directory.

    Args:
        directory: where to load the cross-table.
        cross_table_name: name of the cross-table.
    """
    file_path: Path = directory / (cross_table_name + '.pkl')
    dataframe: pd.DataFrame = pd.read_pickle(file_path)
    return dataframe


class CrossTableLoader:
    """
    Lazily load cross-table files.
    """

    def __init__(self, directory: Path, keep_loaded: bool):
        """
        Save a cross-table to a directory, which can latter be loaded using CrossTableLoader.

        Args:
            directory: where to load the cross-tables.
            keep_loaded: whether to keep loaded cross-table.
        """
        self.directory: Path = directory
        self.keep_loaded: bool = keep_loaded
        self.cross_tables: Dict[str, pd.DataFrame] = {}

    def __getitem__(self, cross_table_name: str) -> pd.DataFrame:
        """
        Get a cross-table, loading it and caching it if needed.

        Args:
            cross_table_name: name of the cross-table.

        Returns:
            a cross-table dataframe.
        """
        crosstab = self.cross_tables.get(cross_table_name)
        if crosstab is None:
            crosstab: pd.DataFrame = load_cross_table(self.directory, cross_table_name)
            if self.keep_loaded:
                self.cross_tables[cross_table_name] = crosstab

        return crosstab
