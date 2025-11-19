import pickle
import warnings
from importlib.abc import Traversable
from io import StringIO
from pathlib import Path
from typing import Literal, Optional, Dict, TypeAlias, Annotated, Union, List, Tuple, Callable, Sequence, Set

import numpy as np
import pandas as pd
from ck.pgm import State
from pydantic import BaseModel, Field, field_validator, PositiveInt

from synthorus.dataset import Dataset, PandasDataset, MathDataset, MathRV, OdbcDataset, \
    PostgresDataset, read_table_builder
from synthorus.error import SynthorusError
from synthorus.utils import dataframe_extras
from synthorus.utils.string_extras import unindent
from synthorus.utils.config_help import config
from synthorus.utils.parse_formula import parse_formula
from synthorus.utils.validate_inputs import validate_inputs

ColumnSpec: TypeAlias = int | str  # a pandas column name


class ColumnDefinitionSpecFunction(BaseModel):
    type: Literal['function'] = 'function'  # for JSON round trip
    inputs: List[str]  # List of rvs that are used for function inputs
    function: str  # Python expression representing the body of a function
    delete_input: bool = False  # delete the input columns (default is False)

    # noinspection PyNestedDecorators
    @field_validator('inputs', mode='before')
    @classmethod
    def validate_inputs(cls, inputs: List[str]) -> List[str]:
        return validate_inputs(inputs)


class ColumnDefinitionSpecGroup(BaseModel):
    type: Literal['group_cut', 'group_qcut', 'group_normalise']
    # 'group_cut' create groups from a single column using Pandas 'cut'.
    # 'group_qcut' create groups from a single column using Pandas 'qcut'.
    # 'group_normalise' group values just as categories (multiple input columns permitted)
    inputs: List[str]  # List of rvs that are used for function inputs
    size: PositiveInt  # how many groups
    delete_input: bool = False  # delete the input columns (default is False)

    # noinspection PyNestedDecorators
    @field_validator('inputs', mode='before')
    @classmethod
    def validate_inputs(cls, inputs: List[str]) -> List[str]:
        return validate_inputs(inputs)


ColumnDefinitionSpec: TypeAlias = Annotated[
    Union[
        ColumnDefinitionSpecFunction,
        ColumnDefinitionSpecGroup,
    ],
    Field(discriminator='type')
]


class TextInputSpecLocation(BaseModel):
    type: Literal['location'] = 'location'  # for JSON round trip
    location: str


class TextInputSpecInline(BaseModel):
    type: Literal['inline'] = 'inline'  # for JSON round trip
    inline: str


TextInputSpec: TypeAlias = Annotated[
    Union[
        TextInputSpecLocation,
        TextInputSpecInline,
    ],
    Field(discriminator='type')
]


class DatasetSpecCsv(BaseModel):
    """
    A DatasetSpec to reference a CSV file or inline text
    (with generalized separators and line spacing).
    """

    type: Literal['csv'] = 'csv'

    weight: Optional[ColumnSpec] = None
    rv_map: Optional[Dict[str, ColumnSpec]] = None
    rv_define: Dict[str, ColumnDefinitionSpec] = {}
    input: TextInputSpec

    sep: str = ','  # separator
    header: bool = True  # is the first line a header line (default is True)
    skip_blank_lines: bool = True  # as per pandas.read_csv
    skip_initial_space: bool = False  # as per pandas.read_csv

    def dataset(self, roots: Sequence[Path | Traversable] = ()) -> Dataset:
        """
        Get a dataset object for this DatasetSpec.

        Args:
            roots: List of directory paths for finding data resource files.

        Returns:
            a Dataset object for accessing data.
        """
        return _make_dataset_text(self, roots)


class DatasetSpecTableBuilder(BaseModel):
    type: Literal['table_builder'] = 'table_builder'

    rv_map: Optional[Dict[str, ColumnSpec]] = None
    rv_define: Dict[str, ColumnDefinitionSpec] = {}
    input: TextInputSpec

    def dataset(self, roots: Sequence[Path | Traversable] = ()) -> Dataset:
        """
        Get a dataset object for this DatasetSpec.

        Args:
            roots: List of directory paths for finding data resource files.

        Returns:
            a Dataset object for accessing data.
        """
        return _make_dataset_text(self, roots)


class DatasetSpecPickle(BaseModel):
    type: Literal['pickle'] = 'pickle'  # for JSON round trip

    weight: Optional[ColumnSpec] = None
    rv_map: Optional[Dict[str, ColumnSpec]] = None
    rv_define: Dict[str, ColumnDefinitionSpec] = {}
    location: str

    def dataset(self, roots: Sequence[Path | Traversable] = ()) -> Dataset:
        """
        Get a dataset object for this DatasetSpec.

        Args:
            roots: List of directory paths for finding data resource files.

        Returns:
            a Dataset object for accessing data.
        """
        return _make_dataset_pickle(self, roots)


class DatasetSpecParquet(BaseModel):
    type: Literal['parquet'] = 'parquet'  # for JSON round trip

    weight: Optional[ColumnSpec] = None
    rv_map: Optional[Dict[str, ColumnSpec]] = None
    rv_define: Dict[str, ColumnDefinitionSpec] = {}
    location: str

    def dataset(self, roots: Sequence[Path | Traversable] = ()) -> Dataset:
        """
        Get a dataset object for this DatasetSpec.

        Args:
            roots: List of directory paths for finding data resource files.

        Returns:
            a Dataset object for accessing data.
        """
        return _make_dataset_parquet(self, roots)


class DatasetSpecFeather(BaseModel):
    type: Literal['feather'] = 'feather'  # for JSON round trip

    weight: Optional[ColumnSpec] = None
    rv_map: Optional[Dict[str, ColumnSpec]] = None
    rv_define: Dict[str, ColumnDefinitionSpec] = {}
    location: str

    def dataset(self, roots: Sequence[Path | Traversable] = ()) -> Dataset:
        """
        Get a dataset object for this DatasetSpec.

        Args:
            roots: List of directory paths for finding data resource files.

        Returns:
            a Dataset object for accessing data.
        """
        return _make_dataset_feather(self, roots)


class DatasetSpecFunction(BaseModel):
    type: Literal['function'] = 'function'  # for JSON round trip

    rvs: Dict[str, int | List[State]]
    output_rv: str
    function: str  # Python expression representing the body of a function

    def dataset(self, _: Sequence[Path | Traversable] = ()) -> Dataset:
        """
        Get a dataset object for this DatasetSpec.

        Returns:
            a Dataset object for accessing data.
        """
        return _make_dataset_function(self)


class DatasetSpecDBMS(BaseModel):
    type: Literal['odbc', 'postgres']

    schema_name: Optional[str] = None  # optional schema where to find the table, default taken from config.DB_SCHEMA
    table_name: str  # name of table in the database
    rvs: Optional[List[str]] = None  # optional restriction of the columns to query, default is all table columns

    # connection values of None will be resolved using local `config` via `synthorus.utils.config_help`
    # E.g.,
    # `connection=None` the None value will be replaced by `config.DB_CONX`,
    # `connection={password: None}` the None value will be replaced by `config.DB_CONX_password`,
    connection: Optional[Dict[str, Optional[str | int]]] = None

    def dataset(self, _: Sequence[Path | Traversable] = ()) -> Dataset:
        """
        Get a dataset object for this DatasetSpec.

        Returns:
            a Dataset object for accessing data.
        """
        return _make_dataset_dbms(self)


def _make_dataset_text(
        dataset_spec: DatasetSpecCsv | DatasetSpecTableBuilder,
        roots: Sequence[Path | Traversable],
) -> Dataset:
    input_spec: TextInputSpec = dataset_spec.input
    if isinstance(input_spec, TextInputSpecLocation):
        file_path: Path = _find_file(input_spec.location, roots)
        return _make_dataset_text_from_io(dataset_spec, file_path)
    elif isinstance(input_spec, TextInputSpecInline):
        io = StringIO(unindent(input_spec.inline))
        return _make_dataset_text_from_io(dataset_spec, io)
    else:
        raise SynthorusError(f'unsupported dataset input spec: {type(input_spec)}')


def _make_dataset_pickle(dataset_spec: DatasetSpecPickle, roots: Sequence[Path | Traversable]) -> Dataset:
    file_path: Path = _find_file(dataset_spec.location, roots)

    with open(file_path, 'rb') as file:
        dataframe: pd.DataFrame = pickle.load(file)

    datasource: PandasDataset = _finish_make_dataframe(
        dataframe,
        dataset_spec.weight,
        dataset_spec.rv_map,
        dataset_spec.rv_define,
    )
    return datasource


def _make_dataset_parquet(dataset_spec: DatasetSpecParquet, roots: Sequence[Path | Traversable]) -> Dataset:
    file_path: Path = _find_file(dataset_spec.location, roots)

    dataframe: pd.DataFrame = pd.read_parquet(file_path)

    datasource: PandasDataset = _finish_make_dataframe(
        dataframe,
        dataset_spec.weight,
        dataset_spec.rv_map,
        dataset_spec.rv_define,
    )
    return datasource


def _make_dataset_feather(dataset_spec: DatasetSpecFeather, roots: Sequence[Path | Traversable]) -> Dataset:
    file_path: Path = _find_file(dataset_spec.location, roots)

    dataframe: pd.DataFrame = pd.read_feather(file_path)

    datasource: PandasDataset = _finish_make_dataframe(
        dataframe,
        dataset_spec.weight,
        dataset_spec.rv_map,
        dataset_spec.rv_define,
    )
    return datasource


def _make_dataset_function(dataset_spec: DatasetSpecFunction) -> Dataset:
    func: str = dataset_spec.function
    rv_states: Dict[str, int | List[State]] = dataset_spec.rvs
    output_rv: str = dataset_spec.output_rv

    if output_rv in rv_states.keys():
        raise SynthorusError(f'output rv included in input rvs: {output_rv!r}')

    input_rvs: List[MathRV] = [
        MathRV(
            rv_name,
            list(range(rv_states)) if isinstance(rv_states, int) else rv_states,
        )
        for rv_name, rv_states in rv_states.items()
    ]

    return MathDataset(
        input_rvs=input_rvs,
        output_name=output_rv,
        func=func,
    )


def _make_dataset_dbms(dataset_spec: DatasetSpecDBMS) -> Dataset:
    api: Literal['odbc', 'postgres'] = dataset_spec.type

    schema_name: Optional[str] = dataset_spec.schema_name
    table_name: str = dataset_spec.table_name
    column_names: Optional[List[str]] = dataset_spec.rvs
    connection: Optional[Dict[str, Optional[str | int]]] = dataset_spec.connection

    # Resolve the schema name
    if schema_name is None:
        schema_name = config.get('DB_SCHEMA')

    # Resolve the connection credentials
    if connection is None:
        connection = config.get('DB_CONX')
    if isinstance(connection, dict):
        connection: Dict[str, str | int] = {
            _clean_connection_key(key): _clean_connection_val(key, val)
            for key, val in connection.items()
        }

    if api == 'odbc':
        if not isinstance(connection, dict):
            raise SynthorusError(f'connection must be a dictionary for ODBC')
        return OdbcDataset(
            table_name=table_name,
            column_names=column_names,
            schema_name=schema_name,
            connection_params=connection,
        )
    elif api == 'postgres':
        return PostgresDataset(
            table_name=table_name,
            column_names=column_names,
            schema_name=schema_name,
            connection_params=connection,
        )
    else:
        raise SynthorusError(f'unknown database api: {api!r}')


def _clean_connection_key(key: str) -> str:
    if not key.isidentifier():
        raise SynthorusError(f'invalid connection key: {key!r}')
    return key


def _clean_connection_val(key: str, val: str | int | None) -> Union[str, int]:
    if isinstance(val, (str, int)):
        return val
    if val is None:
        val = config.get(f'DB_CONX_{key}')
        if isinstance(val, (str, int)):
            return val
    raise SynthorusError(f'invalid connection value: {val!r}')


def _find_file(location: str, roots: Sequence[Path | Traversable]) -> Path:
    """
    Get a file path from a spec file location string.

    If `location` is not an absolute path, then look in all `roots` for `location`.

    Raises:
        SynthorusError if not exactly one path found to exist.
    """
    location_as_path = Path(location)
    if location_as_path.is_absolute():
        if not location_as_path.exists():
            raise SynthorusError(f'absolute file location but file not found: {location!r}')
        return location_as_path
    found = None
    for root in roots:
        location_as_path = root / location
        if location_as_path.exists():
            if found is not None:
                raise SynthorusError(f'multiple source files found: {location!r}')
            found = location_as_path
    if found is None:
        raise SynthorusError(f'could not resolve file location: {location!r}')
    return found


def _make_dataset_text_from_io(
        dataset_spec: DatasetSpecCsv | DatasetSpecTableBuilder,
        io: Path | StringIO,
) -> Dataset:
    data_format: Literal['csv', 'table_builder'] = dataset_spec.type

    dataframe: pd.DataFrame
    weight: Optional[ColumnSpec]
    try:
        if data_format == 'csv':
            sep = dataset_spec.sep
            header = dataset_spec.header
            skip_blank_lines = dataset_spec.skip_blank_lines
            skip_initial_space = dataset_spec.skip_initial_space

            # Pass low_memory=False to ensure consistent type inference.
            dataframe = pd.read_csv(
                io,
                sep=sep,
                header=(0 if header else None),
                skip_blank_lines=skip_blank_lines,
                skipinitialspace=skip_initial_space,
                low_memory=False
            )
            weight = dataset_spec.weight
        elif data_format == 'table_builder':
            dataframe = read_table_builder(io)
            weight = -1
        else:
            raise SynthorusError(f'unsupported data format: {data_format!r}')

    except SynthorusError as err:
        raise err
    except (ValueError, IOError) as err:
        raise SynthorusError('error loading data') from err

    datasource: PandasDataset = _finish_make_dataframe(
        dataframe,
        weight,
        dataset_spec.rv_map,
        dataset_spec.rv_define,
    )
    return datasource


def _finish_make_dataframe(
        dataframe: pd.DataFrame,
        weight: Optional[ColumnSpec],
        rv_map: Optional[Dict[str, ColumnSpec]],
        rv_define: Dict[str, ColumnDefinitionSpec],
) -> PandasDataset:
    """
    Given the load Pandas dataframe for a dataset,
    process any random variable renaming, selecting,
    nominated weight column, and defined columns.

    Defined columns are computed AFTER column renaming.
    """
    # Identify and separate any "weights" series
    weights: Optional[pd.Series] = None
    if weight is not None:
        if isinstance(weight, int):
            weights = dataframe.iloc[:, weight]
            dataframe = dataframe.drop(columns=dataframe.columns[weight])
        elif isinstance(weight, str):
            weights = dataframe[weight]
            dataframe = dataframe.drop(columns=weight)
        else:
            raise SynthorusError(f'cannot understand dataset weight: {weight!r}')

    dataframe = _apply_rv_map_and_define(dataframe, weights, rv_map, rv_define)

    return PandasDataset(
        dataframe=dataframe,
        weights=weights
    )


def _apply_rv_map_and_define(
        dataframe: pd.DataFrame,
        weights: Optional[pd.Series],
        rv_map: Optional[Dict[str, ColumnSpec]],
        rv_define: Dict[str, ColumnDefinitionSpec],
) -> pd.DataFrame:
    """
    Rename, functionally define, and drop columns in the given dataframe
    where `rv_map` is a dict from new_name to old_name.

    """
    # First rename columns
    if rv_map is not None:
        to_cols: List[str] = []
        from_cols: List = []
        for new_name, old_name in rv_map.items():
            if not (new_name is None or new_name == ''):
                if isinstance(old_name, str):
                    to_cols.append(new_name)
                    from_cols.append(old_name)
                elif isinstance(old_name, int):
                    to_cols.append(new_name)
                    from_cols.append(dataframe.columns[old_name])
                else:
                    raise SynthorusError(f'source column not understood: {old_name!r}')
        dataframe[to_cols] = dataframe[from_cols]

    # Next, add functionally defined columns
    for new_name, definition in rv_define.items():
        if isinstance(definition, ColumnDefinitionSpecFunction):
            dataframe[new_name] = _function_column(dataframe, definition)
        elif isinstance(definition, ColumnDefinitionSpecGroup):
            dataframe[new_name] = _group_column(dataframe, weights, definition)

    # Delete columns as inferred
    deleted_inputs: Set[str] = {
        rv
        for definition in rv_define.values()
        if definition.delete_input
        for rv in definition.inputs
    }
    if rv_map is not None:
        to_keep: List[str] = [
            rv
            for rv in set(rv_map.keys()).union(rv_define.keys())
            if rv not in deleted_inputs
        ]
        dataframe = dataframe[to_keep]
    elif len(deleted_inputs) > 0:
        dataframe = dataframe.drop(columns=list(deleted_inputs))

    return dataframe


def _function_column(dataframe: pd.DataFrame, definition: ColumnDefinitionSpecFunction) -> pd.Series:
    function: str = definition.function
    inputs: List[str] = definition.inputs
    column_function: Callable = parse_formula(function, inputs)

    # Return a series that is a function of other series.
    # The given function should take n arguments, where n = len(input_series).
    column_set = set(dataframe.columns)
    for input_col in inputs:
        if input_col not in column_set:
            raise SynthorusError(f'unknown input column: {input_col!r}')
    try:
        return dataframe_extras.functional_series_from_dataframe(
            dataframe=dataframe,
            input_column_names=inputs,
            column_function=column_function,
            dtype=None
        )
    except Exception as err:
        raise SynthorusError(f'Pandas cannot apply provided function: {function!r}') from err


def _group_column(
        dataframe: pd.DataFrame,
        weights: pd.Series,
        definition: ColumnDefinitionSpecGroup,
) -> pd.Series:
    grouping: Literal['group_cut', 'group_qcut', 'group_normalise'] = definition.type
    inputs: List[str] = definition.inputs
    size: int = definition.size

    if grouping == 'group_cut':
        return _create_group_column_cut(dataframe, inputs, size)
    elif grouping == 'group_qcut':
        return _create_group_column_qcut(dataframe, inputs, size)
    elif grouping == 'group_normalise':
        return _create_group_column_normalise(dataframe, weights, inputs, size)
    else:
        raise SynthorusError(f'unknown grouping method: {grouping!r}')


def _create_group_column_cut(
        dataframe: pd.DataFrame,
        inputs: List[str],
        number_of_groups: int,
) -> pd.Series:
    if len(inputs) != 1:
        raise SynthorusError('group_cut: input must be exactly one column')
    input_series = dataframe[inputs[0]]

    try:
        groups = pd.cut(input_series, number_of_groups)
    except Exception as err:
        raise SynthorusError('Pandas cannot apply "cut" method') from err
    return _make_groups_integers(groups)


def _create_group_column_qcut(
        dataframe: pd.DataFrame,
        inputs: List[str],
        number_of_groups: int
) -> pd.Series:
    if len(inputs) != 1:
        raise SynthorusError('group_qcut: input must be exactly one column')
    input_series = dataframe[inputs[0]]

    try:
        groups = pd.qcut(input_series, number_of_groups)
    except Exception as err:
        raise SynthorusError('Pandas cannot apply "qcut" method') from err
    return _make_groups_integers(groups)


def _create_group_column_normalise(
        dataframe: pd.DataFrame,
        weights: Optional[pd.Series],
        inputs: List[str],
        number_of_groups: int
) -> pd.Series:
    crosstab = dataframe_extras.make_crosstab(
        dataframe,
        inputs,
        weights=weights
    )
    num_rows = crosstab.shape[0]

    if num_rows <= number_of_groups:
        warnings.warn(
            f'insufficient categories ({num_rows}) to group ({number_of_groups})'
        )
        groups = [
            [tuple(row[:-1])]
            for row in crosstab.itertuples(index=False)
        ]

    else:
        # Simple algorithm to construct groups
        # Greedily combine rows, aiming for an approximately uniform distribution over the groups.

        # Sort crosstab rows from largest to smallest weight
        weight_col = crosstab.columns[-1]
        crosstab = crosstab.sort_values(weight_col, ascending=False)

        # Add row values, in turn, to the group with the lowest weight
        groups = [[] for _ in range(number_of_groups)]
        group_weights = [(0, i) for i in range(number_of_groups)]
        for row in crosstab.itertuples(index=False):
            values = tuple(row[:-1])
            weight = row[-1]
            group_w, group = group_weights.pop()
            groups[group].append(values)
            group_w += weight
            group_weights.append((group_w, group))
            group_weights.sort(reverse=True)

    value_map = {
        row_values: i
        for i, group in enumerate(groups)
        for row_values in group
    }
    group_function = _GroupFunction(value_map)

    return dataframe_extras.functional_series_from_dataframe(
        dataframe=dataframe,
        input_column_names=inputs,
        column_function=group_function,
        dtype=np.intc
    )


def _make_groups_integers(groups: pd.Series) -> pd.Series:
    value_map = {
        (val,): i
        for i, val in enumerate(groups.unique())
    }
    group_function = _GroupFunction(value_map)
    return dataframe_extras.functional_series_from_series(groups, group_function, dtype=np.intc)


class _GroupFunction:

    def __init__(self, value_map: Dict[Tuple, int]):
        """
        The value_map must map from a tuple of values to a group number.
        The keys must be tuples, even if being applied to a single series.
        """
        self.value_map = value_map

    def __call__(self, *label):
        return self.value_map[label]
