"""
Functions to help make model specs
"""
from io import StringIO
from pathlib import Path
from typing import Tuple, Sequence, Dict

import pandas as pd

from synthorus.model.dataset_spec import DatasetSpec
from synthorus.model.dataset_spec_impl import DatasetSpecCsv, TextInputSpecInline, TextInputSpecLocation, \
    DatasetSpecFeather
from synthorus.model.datasource_spec import DatasourceSpec
from synthorus.model.model_spec import ModelSpec
from synthorus.utils.string_extras import strip_lines


def make_dataset_csv_inline() -> Tuple[DatasetSpec, Sequence[str]]:
    """
    Returns:
        A dataset spec and the rvs that it defines.
    """
    rvs = ['A', 'C', 'X']
    dataset_spec = DatasetSpecCsv(
        input=TextInputSpecInline(
            inline=strip_lines("""
                A,C,X
                y,n,n
                n,n,y
                y,n,y
                y,y,y
                y,y,n
                n,y,y
                y,y,y
                n,n,y
                n,n,n
                y,n,n
            """)
        )
    )
    return dataset_spec, rvs


def make_dataset_csv_file(filename: Path) -> Tuple[DatasetSpec, Sequence[str]]:
    """
    Returns:
        A dataset spec and the rvs that it defines.
    """
    rvs = ['A', 'C', 'E']
    data = strip_lines("""
        A,C,E
        y,n,n
        n,n,y
        y,n,y
        y,y,y
        y,y,n
        n,y,y
        y,y,y
        n,n,y
        n,n,n
        y,n,n
    """)
    with open(filename, 'w') as f:
        f.write(data)
    dataset_spec = DatasetSpecCsv(
        input=TextInputSpecLocation(location=str(filename))
    )
    return dataset_spec, rvs


def make_dataset_feather_file(filename: Path) -> Tuple[DatasetSpec, Sequence[str]]:
    """
    Returns:
        A dataset spec and the rvs that it defines.
    """
    rvs = ['A', 'B', 'C']
    data = strip_lines("""
        A,B,C
        y,n,n
        n,n,y
        y,n,y
        y,y,y
        y,y,n
        n,y,y
        y,y,y
        n,n,y
        n,n,n
        y,n,n
    """)
    dataframe = pd.read_csv(StringIO(data))
    dataframe.to_feather(filename)
    dataset_spec = DatasetSpecFeather(location=str(filename))
    return dataset_spec, rvs


def make_model_spec(
        dataset_specs: Dict[str, Tuple[DatasetSpec, Sequence[str]]],
        roots: Sequence[str] = (),
        sensitivity: int = 0,
) -> ModelSpec:
    """
    Given a dictionary mapping  `datasource_name` to `(dataset_spec, rvs)`, construct
    and return a simple ModelSpec.

    Args:
        dataset_specs: dictionary mapping  `datasource_name` to `(dataset_spec, rvs)`
        roots: the root directories for finding dataset files.
        sensitivity: sensitivity value for each datasource.

    Returns:
        a simple ModelSpec object.
    """
    datasources: Dict[str, DatasourceSpec] = {
        name: DatasourceSpec(
            sensitivity=sensitivity,
            rvs=list(rvs),
            dataset_spec=dataset_spec,
        )
        for name, (dataset_spec, rvs) in dataset_specs.items()
    }
    return ModelSpec(
        name=f'test_model_spec',
        datasources=datasources,
        roots=list(roots),
        rvs={},
        crosstabs={},
        entities={},
    )
