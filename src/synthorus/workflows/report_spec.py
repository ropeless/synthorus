"""
Module for generating reports on a model specification.
"""

import getpass
import math
from pathlib import Path
from typing import List, Optional

import pandas as pd

from synthorus.noise.noiser import LaplaceNoise
from synthorus.model.dataset_cache import DatasetCache
from synthorus.model.model_index import ModelIndex, CrosstabIndex
from synthorus.model.model_spec import ModelSpec, ModelCrosstabSpec
from synthorus.utils.clean_num import clean_num
from synthorus.utils.print_function import PrintFunction, Destination, Print
from synthorus.utils.time_extras import timestamp
from synthorus.workflows.file_names import REPORTS, MODEL_SPEC_REPORT_FILE_NAME, MODEL_SPEC_NAME, MODEL_INDEX_NAME

DEFAULT_INDENT = '    '


def make_model_spec_report(
        model_directory_path: Path,
        *,
        cwd: Optional[Path] = None,
        overwrite: bool = False,
        report_author: Optional[str] = None,
) -> None:
    """
    Create a privacy report on an existing model.

    Args:
        model_directory_path: Directory where to find model cross-tables and other information.
        cwd: The working directory to use for resolving relative roots for accessing datasets.
        overwrite: if True, then any previous report will be overwritten.
        report_author: Optional name of the report author (default is system username).
    """
    report_path: Path = model_directory_path / REPORTS / MODEL_SPEC_REPORT_FILE_NAME
    if overwrite:
        report_path.unlink(missing_ok=True)
    elif report_path.exists():
        raise RuntimeError(f'report already exists: {report_path}')

    with open(model_directory_path / MODEL_SPEC_NAME, 'r') as file:
        model_spec: ModelSpec = ModelSpec.model_validate_json(file.read())

    with open(model_directory_path / MODEL_INDEX_NAME, 'r') as file:
        model_index: ModelIndex = ModelIndex.model_validate_json(file.read())

    dataset_cache = DatasetCache(model_spec, cwd)

    report_model_spec(
        model_spec=model_spec,
        model_index=model_index,
        dataset_cache=dataset_cache,
        destination=report_path,
        report_author=report_author,
    )


def report_model_spec(
        model_spec: ModelSpec,
        model_index: ModelIndex,
        dataset_cache: DatasetCache,
        destination: Destination = None,
        *,
        prefix: str = '',
        indent: str = DEFAULT_INDENT,
        report_author: Optional[str] = None
):
    """
    Generate a report on the given model specification.

    This is a report based on exclusively analysing the model specification
    and extracted cross-table data. No PGMs are constructed or analysed.

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
        if report_author is None:
            report_author = f'user "{getpass.getuser()}"'

        _print(f'{prefix}Model Spec Report')
        _print(f'{prefix}=================')
        _print()
        _print(f'{prefix}Report date: {timestamp()}')
        _print(f'{prefix}Report author: {report_author}')
        _print()
        _print(f'{prefix}Model name: {model_spec.name}')
        _print(f'{prefix}Model author: {model_spec.author}')
        _print()

        crosstab_issues = {}
        _print(f'{prefix}Cross-tables:')
        next_prefix = prefix + indent
        crosstab_names: List[str] = sorted(model_spec.crosstabs.keys(), key=lambda _name: _name.lower())
        for crosstab_name in crosstab_names:
            crosstab: ModelCrosstabSpec = model_spec.crosstabs[crosstab_name]
            issues = _report_on_crosstab(
                crosstab_name,
                crosstab,
                model_spec,
                model_index,
                dataset_cache,
                _print,
                next_prefix,
            )
            crosstab_issues[crosstab_name] = issues
            _print()

        _print(f'{prefix}Summary:')
        if model_spec.pgm_crosstabs != 'noisy':
            _print(f'{next_prefix}WARNING PGM cross-tables: {model_spec.pgm_crosstabs!r}')
        num_crosstab_issues: int = sum(len(issues) for issues in crosstab_issues.values())
        if num_crosstab_issues == 0:
            _print(f'{next_prefix}No cross-table issues.')
        else:
            for crosstab_name, issues in crosstab_issues.items():
                for issue in issues:
                    _print(f'{next_prefix}WARNING cross-table: {crosstab_name!r}: {issue}')

        _print(f'{prefix}{"-" * 80}')


def _report_on_crosstab(
        crosstab_name: str,
        crosstab: ModelCrosstabSpec,
        model_spec: ModelSpec,
        model_index: ModelIndex,
        dataset_cache: DatasetCache,
        _print: PrintFunction,
        prefix: str = '',
        indent: str = '    ',
) -> List[str]:
    """
    Print a report with '_print' on the given crosstab.
    Note that this will make a call to crosstab.extract_crosstab() to
    analyse the cross-table data.
    A list of issues are returned.
    """
    crosstab_index: CrosstabIndex = model_index.crosstabs[crosstab_name]

    next_prefix = prefix + indent

    rv_names = ', '.join(crosstab.rvs)
    num_states = crosstab_index.number_of_states
    datasource_name: str = crosstab.datasource
    sensitivity = model_spec.datasources[datasource_name].sensitivity
    min_cell_size = crosstab.min_cell_size
    epsilon = 0 if sensitivity == 0 else crosstab.epsilon

    _print(f'{prefix}Cross-table: {crosstab_name}')

    _print(f'{next_prefix}Random variables: {rv_names}')
    _print(f'{next_prefix}Datasource: {datasource_name}')
    _print(f'{next_prefix}Sensitivity: {clean_num(sensitivity)}')
    _print(f'{next_prefix}Epsilon: {clean_num(epsilon)}')
    _print(f'{next_prefix}Min cell size: {clean_num(min_cell_size)}')

    crosstab_data: pd.DataFrame = dataset_cache[datasource_name].crosstab(crosstab.rvs)
    weights = crosstab_data.iloc[:, -1]
    num_rows = crosstab_data.shape[0]
    total_weight = weights.sum()
    min_weight = weights.min()
    max_weight = weights.max()
    num_suppressed = num_states - num_rows

    _print(f'{next_prefix}State space size: {num_states:,}')
    _print(f'{next_prefix}Number of rows: {clean_num(num_rows)}')
    _print(f'{next_prefix}Number of suppressed rows: {clean_num(num_suppressed)}')
    _print(f'{next_prefix}Min weight: {clean_num(min_weight)}')
    _print(f'{next_prefix}Max weight: {clean_num(max_weight)}')
    _print(f'{next_prefix}Total weight: {clean_num(total_weight)}')

    issues = []
    recommendations = []

    # Multiple condition tables are an issue as adjusting a cross-table
    # weights may distort previous condition adjustments made to the
    # cross-table.
    #
    # This issue may be improved by restructuring the datasources.
    #
    # if len(crosstab.conditions) > 1:
    #     condition_names = ', '.join(repr(cond.crosstab.name) for cond in crosstab.conditions)
    #     issues.append(f'multiple conditioning cross-tables: {condition_names}')

    # Unsatisfied condition rvs are rvs of datasources that are
    # inferred as condition rv but are not available as a distribution
    # rv in any other cross-table.
    #
    # Not having these rvs available as a distribution rv in some other cross
    # table means this cross-table may have reduced accuracy for the joint
    # distribution over the cross-table rvs.
    #
    # This issues can normally be rectified by adding new cross-tables
    # to cover the needed distribution rvs.
    #
    # if len(crosstab.unsatisfied_condition_rvs) > 0:
    #     rv_names = ', '.join(rv.name for rv in crosstab.unsatisfied_condition_rvs)
    #     issues.append(f'unsatisfied condition rvs: {rv_names}')

    # An unused condition rv is a rv mentioned as a condition rv in a
    # datasource of the cross-table, but is not in the cross-table.
    #
    # This represents a potential opportunity to grow the scope of the cross-table.
    #
    # if len(crosstab.unused_condition_rvs) > 0:
    #     rv_names = ', '.join(rv.name for rv in crosstab.unused_condition_rvs)
    #     issues.append(f'unused condition rvs: {rv_names}')

    # Report cross-tables where the addition of Laplace noise and min cell size
    # causes many rows to be created or lost.
    #
    if sensitivity > 0 or min_cell_size > 0:
        threshold = 0.5  # Lost rows and lost weight proportion threshold

        # Analyse potential new rows introduced by adding noise to suppressed rows
        recommended_min_cell_size_add_rows = 0
        if sensitivity > 0:
            alpha = 0.5 * math.exp(-min_cell_size * sensitivity / epsilon)
            expected_new_rows = math.ceil(num_suppressed * alpha)
            # Can never add more rows than is suppressed
            expected_new_rows = min(expected_new_rows, num_suppressed)

            if num_rows > 0 and num_suppressed > 0:
                recommended_min_cell_size_add_rows = LaplaceNoise.recommended_min_cell_size(
                    epsilon,
                    sensitivity,
                    num_suppressed,
                    target_rows=num_rows
                )

            new_rows_percent = int(expected_new_rows / num_rows * 100 + 0.5)
            _print(f'{next_prefix}Alpha: {clean_num(alpha)}')
            _print(f'{next_prefix}Expected new rows: {clean_num(expected_new_rows)} ({new_rows_percent}%)')
            recommendations.append(f'minimum min-cell-size: {clean_num(recommended_min_cell_size_add_rows)}')
            if expected_new_rows > num_rows:
                issues.append('high expected new rows')

        # Analyse potential lost rows
        if min_cell_size > 0:
            # TODO: These expectations are not right because it neglects injected noise
            #  which can cause a weight to drop below min_cell_size.
            lost_rows = weights[weights < min_cell_size]
            expected_lost_rows = lost_rows.count()
            expected_lost_weight = lost_rows.sum()

            # The proportion of rows lost will always be higher than the proportion of weight lost.

            # Work out the min cell size that leads to the threshold being met
            row_threshold = num_rows * threshold
            sorted_weights = weights.sort_values()
            recommended_min_cell_size_lost_rows = 0
            accumulated = 0
            for w in sorted_weights:
                accumulated += 1
                if accumulated >= row_threshold:
                    break
                recommended_min_cell_size_lost_rows = w

            row_loss_percent = int(expected_lost_rows / num_rows * 100 + 0.5)
            weight_loss_percent = int(expected_lost_weight / total_weight * 100 + 0.5)
            _print(f'{next_prefix}Expected lost rows: {clean_num(expected_lost_rows)} ({row_loss_percent}%)')
            _print(f'{next_prefix}Expected lost weight: {clean_num(expected_lost_weight)} ({weight_loss_percent}%)')
            recommendations.append(f'maximum min-cell-size: {clean_num(recommended_min_cell_size_lost_rows)}')

            if expected_lost_rows / num_rows > threshold:
                issues.append('high expected lost rows')

            if recommended_min_cell_size_lost_rows < recommended_min_cell_size_add_rows:
                issues.append(
                    f'no suitable min-cell-size: '
                    f'recommended maximum (for lost rows) = {clean_num(recommended_min_cell_size_lost_rows)}, '
                    f'recommended minimum (for added rows) = {clean_num(recommended_min_cell_size_add_rows)}'
                )

    for recommendation in recommendations:
        _print(next_prefix + 'Recommended:', recommendation)
    for issue in issues:
        _print(next_prefix + 'WARNING:', issue)

    return issues


def _print_issues(section_name, issues, prefix, indent, _print):
    if len(issues) > 0:
        _print(f'{prefix}{section_name}:')
        next_prefix = prefix + indent
        for issue in issues:
            _print(f'{next_prefix}{issue}')
    else:
        _print(f'{prefix}{section_name}: None')
