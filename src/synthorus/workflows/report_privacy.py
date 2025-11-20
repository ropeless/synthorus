import getpass
from pathlib import Path
from typing import Optional, List, Sequence, Iterable, Any

from ck.pgm import State

from synthorus.dataset import Dataset
from synthorus.model.dataset_cache import DatasetCache
from synthorus.model.datasource_spec import DatasourceSpec
from synthorus.model.model_index import ModelIndex, CrosstabIndex, RVIndex
from synthorus.model.model_spec import ModelSpec, ModelCrosstabSpec
from synthorus.utils.clean_num import clean_num
from synthorus.utils.math_extras import p_log_p
from synthorus.utils.print_function import PrintFunction, Destination, Print
from synthorus.utils.time_extras import timestamp
from synthorus.workflows.file_names import REPORTS, PRIVACY_REPORT_FILE_NAME, MODEL_SPEC_NAME, MODEL_INDEX_NAME

DEFAULT_INDENT = '    '


def make_privacy_report(
        model_directory_path: Path,
        *,
        cwd: Optional[Path] = None,
        overwrite: bool = False,
        report_author: Optional[str] = None,
) -> None:
    """
    Create a privacy report on an existing model.

    Note that this will read the model datasets.

    Args:
        model_directory_path: Directory where to find model cross-tables and other information.
        cwd: The working directory to use for resolving relative roots for accessing datasets.
        overwrite: if True, then any previous report will be overwritten.
        report_author: Optional name of the report author (default is system username).
    """
    report_path: Path = model_directory_path / REPORTS / PRIVACY_REPORT_FILE_NAME
    if overwrite:
        report_path.unlink(missing_ok=True)
    elif report_path.exists():
        raise RuntimeError(f'report already exists: {report_path}')

    with open(model_directory_path / MODEL_SPEC_NAME, 'r') as file:
        model_spec: ModelSpec = ModelSpec.model_validate_json(file.read())

    with open(model_directory_path / MODEL_INDEX_NAME, 'r') as file:
        model_index: ModelIndex = ModelIndex.model_validate_json(file.read())

    dataset_cache = DatasetCache(model_spec, cwd)

    report_privacy(
        model_spec=model_spec,
        model_index=model_index,
        dataset_cache=dataset_cache,
        destination=report_path,
        report_author=report_author,
    )


def report_privacy(
        model_spec: ModelSpec,
        model_index: ModelIndex,
        dataset_cache: DatasetCache,
        destination: Destination = None,
        *,
        report_author: Optional[str] = None,
        prefix: str = '',
        indent: str = DEFAULT_INDENT,
) -> None:
    """
    Generate a full privacy report on the given model specification.

    Args:
        model_spec: The synthetic data model specification.
        model_index: The cached relationships between model components.
        dataset_cache: Object to access to model datasets.
        destination: Where to write the report, as per `Print`.
        report_author: Optional name of the report author (default is system username).
        prefix: A prefix for each report line.
        indent: The additional prefix for indentation.
    """
    with Print(destination) as _print:
        next_prefix = prefix + indent

        if report_author is None:
            report_author = getpass.getuser()

        _print(f'{prefix}Privacy Report')
        _print(f'{prefix}==============')
        _print()
        _print(f'{prefix}Report date: {timestamp()}')
        _print(f'{prefix}Report author: {report_author}')
        _print()
        _print(f'{prefix}Model name: {model_spec.name}')
        _print(f'{prefix}Model author: {model_spec.author}')
        _print()
        _print(f'{prefix}PGM cross-tables: {model_spec.pgm_crosstabs}')
        _print(f'{prefix}Random number generator security level: {model_spec.rng_n} ({_rng_n_str(model_spec.rng_n)})')

        privacy_budget: float = _calculate_privacy_budget(model_spec, model_index)
        _print(f'{prefix}Privacy budget: {clean_num(privacy_budget)} ({_budget_str(privacy_budget)})')
        _print()

        # Get the datasources in a stable order, and identify those with zero sensitivity
        sorted_datasources: List[str] = sorted(model_spec.datasources.keys())
        zero_sensitivity_datasources: List[str] = [
            datasource_name
            for datasource_name in sorted_datasources
            if model_spec.datasources[datasource_name].sensitivity <= 0
        ]

        if len(zero_sensitivity_datasources) > 0:
            _print(
                f'{prefix}Datasources with zero sensitivity ({len(zero_sensitivity_datasources)} of {len(sorted_datasources)}):')
            for data_source_name in zero_sensitivity_datasources:
                _print(f'{next_prefix}{data_source_name}')
            _print()

        _print(f'{prefix}Datasources ({len(sorted_datasources)}):')
        for datasource_name in sorted_datasources:
            _privacy_report_datasource(
                datasource_name,
                model_spec.datasources[datasource_name],
                dataset_cache[datasource_name],
                _print,
                next_prefix,
            )
            _print()

        crosstab_names: List[str] = sorted(model_spec.crosstabs.keys())
        _print(f'{prefix}Cross-tables ({len(crosstab_names)}):')
        for crosstab_name in crosstab_names:
            crosstab_spec: ModelCrosstabSpec = model_spec.crosstabs[crosstab_name]
            crosstab_index: CrosstabIndex = model_index.crosstabs[crosstab_name]
            datasource_name: str = crosstab_index.datasource
            datasource: DatasourceSpec = model_spec.datasources[datasource_name]
            sensitivity: float = datasource.sensitivity
            dataset: Dataset = dataset_cache[datasource_name]
            _privacy_report_crosstab(crosstab_spec, crosstab_index, dataset, sensitivity, _print, next_prefix)
            _print()

        rv_names: List[str] = sorted(model_spec.rvs.keys())
        _print(f'{prefix}Random variables ({len(rv_names)}):')
        for rv_name in rv_names:
            rv_index: RVIndex = model_index.rvs[rv_name]
            _privacy_report_rv(rv_name, rv_index, dataset_cache, _print, next_prefix, state_limit=5)
            _print()

        _print(f'{prefix}{"-" * 80}')


def _privacy_report_rv(
        rv_name: str,
        rv_index: RVIndex,
        dataset_cache: DatasetCache,
        _print: PrintFunction,
        prefix: str,
        state_limit: int
):
    """
    Generate a privacy report on one model random variable.
    """
    primary_datasource: str = rv_index.primary_datasource
    states: List[State] = rv_index.states
    dataset: Dataset = dataset_cache[primary_datasource]

    distribution, total_weight, entropy = _analyse_crosstab_data(dataset, [rv_name])

    _print(f'{prefix}Random variable name: {rv_name!r}')
    _print(f'{prefix}Primary datasource: {primary_datasource!r}')
    _print(f'{prefix}Number of values: {len(states):,}')
    _print(f'{prefix}Number of support values: {clean_num(distribution.shape[0])}')
    show_all_states = len(states) <= state_limit * 2
    if show_all_states:
        _print(f'{prefix}Values: {_join(states)}')
    else:
        _print(
            f'{prefix}Values: {_join(states[:state_limit])}, '
            f'[...] '
            f'{_join(states[-state_limit:])}'
        )

    _print(f'{prefix}Total weight: {total_weight:,}')
    _print(f'{prefix}Entropy: {entropy}')
    if show_all_states:
        distribution_str = _join(
            f'{row[0]} = {row[-1] / total_weight:.4}'
            for row in distribution.itertuples(index=False)
        )
        _print(f'{prefix}Distribution: {distribution_str}')
    else:
        weights = distribution[distribution.columns[-1]]
        min_val = weights.min() / total_weight
        max_val = weights.max() / total_weight
        _print(f'{prefix}Distribution: {min_val:.4} to {max_val:.4} (range)')


def _privacy_report_datasource(
        datasource_name: str,
        datasource_spec: DatasourceSpec,
        dataset: Dataset,
        _print: PrintFunction,
        prefix: str,
) -> None:
    """
    Report on one datasource.

    Args:
        dataset: The datasource to report on.
        _print: The print function to use.
        prefix: The prefix for each report line.
    """
    distribution, total_weight, entropy = _analyse_crosstab_data(dataset)

    rv_names: str = ', '.join(datasource_spec.rvs)

    _print(f'{prefix}Datasource name: {datasource_name}')
    _print(f'{prefix}Sensitivity: {clean_num(datasource_spec.sensitivity)}')
    _print(f'{prefix}Random variables: {rv_names}')
    _print(f'{prefix}Distinct rows: {clean_num(distribution.shape[0])}')
    _print(f'{prefix}Total weight: {clean_num(total_weight)}')
    _print(f'{prefix}Entropy: {entropy}')


def _privacy_report_crosstab(
        crosstab_spec: ModelCrosstabSpec,
        crosstab_index: CrosstabIndex,
        dataset: Dataset,
        sensitivity: float,
        _print: PrintFunction,
        prefix: str,
):
    rv_names: str = ', '.join(crosstab_spec.rvs)

    epsilon: float = 0 if sensitivity == 0 else crosstab_spec.epsilon

    _print(f'{prefix}Random variables: {rv_names}')
    _print(f'{prefix}State space size: {crosstab_index.number_of_states:,}')
    _print(f'{prefix}Datasource: {crosstab_index.datasource}')
    _print(f'{prefix}Sensitivity: {clean_num(sensitivity)}')
    _print(f'{prefix}Epsilon: {clean_num(epsilon)}')
    _print(f'{prefix}Min cell size: {clean_num(crosstab_spec.min_cell_size)}')

    distribution, total_weight, entropy = _analyse_crosstab_data(dataset)
    _print(f'{prefix}Distinct rows: {clean_num(distribution.shape[0])}')
    _print(f'{prefix}Total weight: {clean_num(total_weight)}')
    _print(f'{prefix}Entropy: {entropy}')


def _calculate_privacy_budget(model_spec: ModelSpec, model_index: ModelIndex) -> float:
    """
    Calculate the total privacy budget for a model. This is
    the sum of cross-table epsilon values, for cross-tables of a
    datasource with sensitivity > 0.

    Args:
        model_spec: synthetic data model specification.
        model_index: cached relationships between model components.

    Returns:
        sum of cross-table epsilon using sources with sensitivity > 0.
    """
    total: float = 0
    crosstab_name: str
    crosstab_spec: ModelCrosstabSpec
    for crosstab_name, crosstab_spec in model_spec.crosstabs.items():
        crosstab_index: CrosstabIndex = model_index.crosstabs[crosstab_name]
        datasource_name: str = crosstab_index.datasource
        sensitivity: float = model_spec.datasources[datasource_name].sensitivity
        if sensitivity > 0:
            total += crosstab_spec.epsilon
    return total


def _analyse_crosstab_data(dataset: Dataset, rvs: Optional[Sequence[str]] = None):
    if rvs is None:
        rvs = dataset.rvs
    distribution = dataset.crosstab(rvs)
    total_weight = distribution[distribution.columns[-1]].sum()
    entropy = sum(
        - p_log_p(row[-1] / total_weight)
        for row in distribution.itertuples(index=False)
    )
    return distribution, total_weight, entropy


def _rng_n_str(rng_n: int) -> str:
    """
    Interpret the random number generator security level as a
    human-readable string.

    See class SafeRandom in package modelling.noise.

    For details on the interpretation, see:
    Holohan, N., & Braghin, S. (2021, October). Secure random sampling in differential privacy.
    In European Symposium on Research in Computer Security (pp. 523-542). Springer, Cham.
    """
    if rng_n < 4:
        return 'lower than AES128 - unverified security'
    if rng_n == 4:
        return 'equivalent to AES128 - adequate security'
    if rng_n == 5:
        return 'equivalent to AES192 - good security'
    if rng_n == 6:
        return 'equivalent to AES256 - very good security'
    if rng_n > 6:
        return 'better than AES256 - excellent security'
    return 'no interpretation'


def _budget_str(privacy_budget: float) -> str:
    """
    Interpret the privacy_budget as a human-readable string.
    """
    if privacy_budget <= 0:
        return 'no privacy risk'
    if privacy_budget < 0.01:
        return 'very good privacy protection'
    if privacy_budget < 1.0:
        return 'good privacy protection'
    if privacy_budget < 10.0:
        return 'low privacy protection'
    return 'effectively no privacy protection'


def _join(elements: Iterable[Any], sep: str = ', ') -> str:
    return sep.join(str(element) for element in elements)
