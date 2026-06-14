"""
Module for generating reports on a model specification.
"""

import getpass
import math
from importlib.abc import Traversable
from os import PathLike
from pathlib import Path
from typing import List, Optional, Dict, Any, Sequence

import dominate
import pandas as pd
from ck.pgm import State
from dominate.tags import p, i, h1, details, summary, a, div, html_tag

from synthorus.model.dataset_cache import DatasetCache
from synthorus.model.dataset_spec import DatasetSpec
from synthorus.model.dataset_spec_impl import DatasetSpecCsv, TextInputSpecLocation, TextInputSpecInline, \
    DatasetSpecTableBuilder, DatasetSpecPickle, DatasetSpecParquet, DatasetSpecFeather, DatasetSpecFunction, \
    DatasetSpecDBMS
from synthorus.model.datasource_spec import DatasourceSpec
from synthorus.model.model_index import ModelIndex, CrosstabIndex, RVIndex
from synthorus.model.model_spec import ModelSpec, ModelCrosstabSpec, ModelEntitySpec, ModelFieldSpec, \
    ModelFieldSpecSample, ModelFieldSpecSum, ModelFieldSpecFunction, ModelRVSpec
from synthorus.model.noiser_spec import NoiserSpec, NoiserSpecBasicLaplace
from synthorus.noise.noiser import recommended_min_cell_size
from synthorus.simulator.condition_spec import ConditionSpec, ConditionSpecFixedLimit, ConditionSpecVariableLimit, \
    ConditionSpecStates
from synthorus.utils.clean_num import clean_num
from synthorus.utils.time_extras import timestamp
from synthorus.workflows.file_names import REPORTS, MODEL_SPEC_REPORT_FILE_NAME, MODEL_SPEC_NAME, MODEL_INDEX_NAME
from synthorus.workflows.report_helpers import dict_table, rng_n_str, calculate_privacy_budget, budget_str, \
    render_comment, render_inline_data, render_code, add_head_styles


def make_model_spec_report(
        model_directory_path: Path,
        *,
        cwd: Optional[PathLike | Traversable] = None,
        overwrite: bool = False,
        report_author: Optional[str] = None,
) -> None:
    """
    Create a general report the model in the given directory.

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
        destination: Optional[PathLike] = None,
        *,
        report_author: Optional[str] = None
) -> None:
    """
    Generate a report on the given model specification.

    This is a report based on exclusively analysing the model specification
    and extracted cross-table data. No PGMs are constructed or analysed.

    Args:
        model_spec: The synthetic data model specification.
        model_index: The cached relationships between model components.
        dataset_cache: Object to access to model datasets.
        destination: Where to write the report, None for stdout.
        report_author: Optional name of the report author (default is system username).
    """
    if report_author is None:
        report_author = f'user "{getpass.getuser()}"'
    doc = _report_model_spec(model_spec, model_index, dataset_cache, report_author)
    if destination is None:
        print(doc.render())
    else:
        with open(destination, 'w') as file:
            print(doc.render(), file=file)


def _report_model_spec(
        model_spec: ModelSpec,
        model_index: ModelIndex,
        dataset_cache: DatasetCache,
        report_author: str,
) -> dominate.document:
    """
    Generate a report on the given model specification.

    This is a report based on exclusively analysing the model specification
    and extracted cross-table data. No PGMs are constructed or analysed.

    Args:
        model_spec: The synthetic data model specification.
        model_index: The cached relationships between model components.
        dataset_cache: Object to access to model datasets.
        report_author: name of the report author.

    Returns:
        a `dominate` document object.
    """
    privacy_budget: float = calculate_privacy_budget(model_spec, model_index)
    report_name: str = f'Model Spec Report for {model_spec.name!r}'

    doc = dominate.document(title=report_name)

    with doc.head:
        add_head_styles()

    with doc.body:
        h1(report_name)

        # Report metadata
        with details(open=False, cls='tree-menu'):
            summary('Report metadata')
            dict_table({
                'Report date': timestamp(),
                'Report author': report_author,
            })
        p()

        # Model metadata
        with details(open=False, cls='tree-menu'):
            summary('Model metadata')
            dict_table({
                'Model name': model_spec.name,
                'Model author': model_spec.author,
                'Random number generator security level': f'{model_spec.rng_n} ({rng_n_str(model_spec.rng_n)})',
                'Privacy budget': f'{clean_num(privacy_budget)} ({budget_str(privacy_budget)})',
                'PGM cross-table source': model_spec.pgm_crosstabs,
            })
            with details():
                summary('Comment')
                if len(model_spec.comment.strip()) == 0:
                    p(i('no comment'))
                else:
                    render_comment(model_spec.comment)
            with details():
                summary(f'Roots ({len(model_spec.roots)})')
                for root in model_spec.roots:
                    p(root)
        p()

        # Model parameters
        with details(open=False, cls='tree-menu'):
            summary(f'Model parameters ({len(model_spec.parameters)})')
            dict_table(model_spec.parameters)
        p()

        # Datasources
        with details(open=False, cls='tree-menu'):
            summary(f'Datasources ({len(model_spec.datasources)})')
            datasource_names: List[str] = sorted(model_spec.datasources.keys(), key=lambda _name: _name.lower())
            for datasource_name in datasource_names:
                datasource: DatasourceSpec = model_spec.datasources[datasource_name]
                _report_on_datasource(datasource_name, datasource)
        p()

        # Random variables
        with details(open=False, cls='tree-menu'):
            summary(f'Random variables ({len(model_spec.rvs)})')
            rv_names: List[str] = sorted(model_spec.rvs.keys(), key=lambda _name: _name.lower())
            for rv_name in rv_names:
                rv_spec: ModelRVSpec = model_spec.rvs[rv_name]
                rv_index: RVIndex = model_index.rvs[rv_name]
                _report_on_rv(rv_name, rv_spec, rv_index)
        p()

        # Cross-tables
        crosstab_issues = {}
        with details(open=False, cls='tree-menu'):
            crosstab_names: List[str] = sorted(model_spec.crosstabs.keys(), key=lambda _name: _name.lower())
            summary(f'Cross-tables ({len(crosstab_names)})')
            for crosstab_name in crosstab_names:
                crosstab: ModelCrosstabSpec = model_spec.crosstabs[crosstab_name]
                issues = _report_on_crosstab(
                    crosstab_name,
                    crosstab,
                    model_spec,
                    model_index,
                    dataset_cache,
                )
                crosstab_issues[crosstab_name] = issues
        p()

        # Cross-table issues
        with details(open=False, cls='tree-menu'):
            number_of_issues: int = sum(len(issues) for issues in crosstab_issues.values())
            summary(f'Cross-table issues ({number_of_issues})')
            dict_table({
                crosstab_name: issue
                for crosstab_name, issues in crosstab_issues.items()
                for issue in issues
            })
        p()

        # Entities
        with details(open=False, cls='tree-menu'):
            summary(f'Entities ({len(model_spec.entities)})')
            entity_names: List[str] = sorted(model_spec.entities.keys(), key=lambda _name: _name.lower())
            for entity_name in entity_names:
                entity: ModelEntitySpec = model_spec.entities[entity_name]
                _report_on_entity(entity_name, entity)
        p()

    return doc


def _report_on_entity(
        entity_name: str,
        entity: ModelEntitySpec,
) -> None:
    """
    Use `dominate` to make a report on the given entity.
    """
    with details():
        summary(entity_name)
        with div(id=f'ENTITY_{entity_name}'):
            summary_dict: Dict[str, Any] = ({
                'ID field name': entity.id_field_name,
                'Count field name': entity.count_field_name,
            })
            if entity.parent is None:
                summary_dict.update({
                    'Parent': None,
                })
            else:
                summary_dict.update({
                    'Parent': _entity_link(entity.parent),
                    'Foreign field name': entity.foreign_field_name,
                })
            dict_table(summary_dict)

            with details():
                summary(f'Fields ({len(entity.fields)})')
                for field_name, field_spec in entity.fields.items():
                    with details():
                        summary(f'{field_name} ({field_spec.type})')
                        _report_on_field(field_spec)

            with details():
                summary(f'Cardinality ({len(entity.cardinality)})')
                for cardinality_spec in entity.cardinality:
                    _report_on_cardinality(cardinality_spec)


def _report_on_field(field_spec: ModelFieldSpec) -> None:
    if isinstance(field_spec, ModelFieldSpecSample):
        _report_on_field_sample(field_spec)
    elif isinstance(field_spec, ModelFieldSpecSum):
        _report_on_field_sum(field_spec)
    elif isinstance(field_spec, ModelFieldSpecFunction):
        _report_on_field_function(field_spec)
    else:
        p(i('Field specification type is unknown!', f'type={field_spec.type}'))


def _report_on_field_sample(field_spec: ModelFieldSpecSample) -> None:
    dict_table({
        'Field type': field_spec.type,
        'PGM random variable': field_spec.rv_name,
    })


def _report_on_field_sum(field_spec: ModelFieldSpecSum) -> None:
    dict_table({
        'Field type': field_spec.type,
        'Initial value': field_spec.initial_value,
        'Offset value': field_spec.offset,
    })
    with details():
        summary(f'Summation fields ({len(field_spec.sum)})')
        for field_name in field_spec.sum:
            p(field_name)


def _report_on_field_function(field_spec: ModelFieldSpecFunction) -> None:
    dict_table({
        'Field type': field_spec.type,
        'Initial value': field_spec.initial_value,
    })
    with details():
        summary('Function')
        render_code(field_spec.function)
    with details():
        summary(f'Input fields ({len(field_spec.inputs)})')
        for field_name in field_spec.inputs:
            p(field_name)


def _report_on_cardinality(cardinality_spec: ConditionSpec) -> None:
    if isinstance(cardinality_spec, ConditionSpecFixedLimit):
        render_code(f'Field({cardinality_spec.field}) {cardinality_spec.op} {cardinality_spec.limit!r}')
    elif isinstance(cardinality_spec, ConditionSpecVariableLimit):
        render_code(f'Field({cardinality_spec.field}) {cardinality_spec.op} Field({cardinality_spec.limit_field})')
    elif isinstance(cardinality_spec, ConditionSpecStates):
        states_str: str = ', '.join(repr(state) for state in cardinality_spec.states)
        render_code(f'Field({cardinality_spec.field}) in {{{states_str}}}')
    else:
        p(i('Cardinality specification type is unknown!', f'type={cardinality_spec.type}'))


def _report_on_datasource(
        datasource_name: str,
        datasource: DatasourceSpec,
) -> None:
    """
    Use `dominate` to make a report on the given datasource.
    """
    with details():
        summary(f'{datasource_name} ({datasource.dataset_spec.type})')
        with div(id=f'DATASOURCE_{datasource_name}'):
            dict_table({
                'Sensitivity': datasource.sensitivity,
            })
            with details():
                summary(f'Dataset Specification')
                _report_on_dataset_spec(datasource.dataset_spec)

            _link_to_rvs('Random variables', datasource.rvs)


def _report_on_dataset_spec(dataset_spec: DatasetSpec) -> None:
    """
    Use `dominate` to make a report on the given dataset spec.
    """
    if isinstance(dataset_spec, DatasetSpecCsv):
        _report_on_dataset_spec_csv(dataset_spec)
    elif isinstance(dataset_spec, DatasetSpecTableBuilder):
        _report_on_dataset_spec_table_builder(dataset_spec)
    elif isinstance(dataset_spec, DatasetSpecPickle):
        _report_on_dataset_spec_binary(dataset_spec, 'Pickle')
    elif isinstance(dataset_spec, DatasetSpecParquet):
        _report_on_dataset_spec_binary(dataset_spec, 'Parquet')
    elif isinstance(dataset_spec, DatasetSpecFeather):
        _report_on_dataset_spec_binary(dataset_spec, 'Feather')
    elif isinstance(dataset_spec, DatasetSpecFunction):
        _report_on_dataset_spec_function(dataset_spec)
    elif isinstance(dataset_spec, DatasetSpecDBMS):
        _report_on_dataset_spec_dbms(dataset_spec)
    else:
        p(i('Dataset specification type is unknown!', f'type={dataset_spec.type}'))


def _report_on_dataset_spec_dbms(dataset_spec: DatasetSpecDBMS) -> None:
    summary_dict: Dict[str, Any] = {
        'Format': f'DBMS {dataset_spec.type}',
        'Schema': dataset_spec.schema_name,
        'Table': dataset_spec.table_name,
    }
    if dataset_spec.rvs is None:
        summary_dict.update({
            'RVs': 'all in table',
        })
    if dataset_spec.connection is None:
        summary_dict.update({
            'Connection': None,
        })

    dict_table(summary_dict)

    if dataset_spec.rvs is not None:
        with details():
            summary(f'RVs ({len(dataset_spec.rvs)})')
            for rv_name in dataset_spec.rvs:
                p(rv_name)

    if dataset_spec.connection is not None:
        with details():
            summary(f'Connection')
            dict_table(dataset_spec.connection)


def _report_on_dataset_spec_function(dataset_spec: DatasetSpecFunction) -> None:
    dict_table({
        'Format': 'Function',
        'Output RV': dataset_spec.output_rv,
    })
    with details():
        summary('Function')
        render_code(dataset_spec.function)

    _link_to_rvs('Input RVs', dataset_spec.rvs)


def _report_on_dataset_spec_binary(
        dataset_spec: DatasetSpecPickle | DatasetSpecParquet | DatasetSpecFeather,
        format_name: str,
) -> None:
    summary_dict: Dict[str, Any] = {
        'Format': format_name,
        'Data location': dataset_spec.location,
        'Weight column': dataset_spec.weight,
    }

    render_rv_map: bool = True
    if dataset_spec.rv_map is None or len(dataset_spec.rv_map) == 0:
        summary_dict.update({
            'RV map': None,
        })
        render_rv_map = False

    render_rv_define: bool = True
    if dataset_spec.rv_define is None or len(dataset_spec.rv_define) == 0:
        summary_dict.update({
            'RV define': None,
        })
        render_rv_define = False

    dict_table(summary_dict)

    if render_rv_map:
        assert (dataset_spec.rv_map is not None)
        with details():
            summary('RV map')
            dict_table(dataset_spec.rv_map)

    if render_rv_define:
        assert (dataset_spec.rv_define is not None)
        with details():
            summary('RV define')
            dict_table(dataset_spec.rv_define)


def _report_on_dataset_spec_table_builder(dataset_spec: DatasetSpecTableBuilder) -> None:
    summary_dict: Dict[str, Any] = {
        'Format': 'Table Builder',
    }
    if isinstance(dataset_spec.input, TextInputSpecLocation):
        summary_dict.update({
            'Data location': dataset_spec.input.location,
        })

    render_rv_map: bool = True
    if dataset_spec.rv_map is None or len(dataset_spec.rv_map) == 0:
        summary_dict.update({
            'RV map': None,
        })
        render_rv_map = False

    render_rv_define: bool = True
    if dataset_spec.rv_define is None or len(dataset_spec.rv_define) == 0:
        summary_dict.update({
            'RV define': None,
        })
        render_rv_define = False

    dict_table(summary_dict)

    if isinstance(dataset_spec.input, TextInputSpecInline):
        with details():
            summary('Data inline')
            render_inline_data(dataset_spec.input.inline)

    if render_rv_map:
        assert (dataset_spec.rv_map is not None)
        with details():
            summary('RV map')
            dict_table(dataset_spec.rv_map)

    if render_rv_define:
        assert (dataset_spec.rv_define is not None)
        with details():
            summary('RV define')
            dict_table(dataset_spec.rv_define)


def _report_on_dataset_spec_csv(dataset_spec: DatasetSpecCsv) -> None:
    summary_dict: Dict[str, Any] = {
        'Format': 'CSV',
        'Separator': repr(dataset_spec.sep),
        'Header': repr(dataset_spec.header),
        'Skip blank lines': repr(dataset_spec.skip_blank_lines),
        'Skip initial space': repr(dataset_spec.skip_initial_space),
        'Weight column': dataset_spec.weight,
    }
    if isinstance(dataset_spec.input, TextInputSpecLocation):
        summary_dict.update({
            'Data location': dataset_spec.input.location,
        })

    render_rv_map: bool = True
    if dataset_spec.rv_map is None or len(dataset_spec.rv_map) == 0:
        summary_dict.update({
            'RV map': None,
        })
        render_rv_map = False

    render_rv_define: bool = True
    if dataset_spec.rv_define is None or len(dataset_spec.rv_define) == 0:
        summary_dict.update({
            'RV define': None,
        })
        render_rv_define = False

    dict_table(summary_dict)

    if isinstance(dataset_spec.input, TextInputSpecInline):
        with details():
            summary('Data inline')
            render_inline_data(dataset_spec.input.inline)

    if render_rv_map:
        assert (dataset_spec.rv_map is not None)
        with details():
            summary('RV map')
            dict_table(dataset_spec.rv_map)

    if render_rv_define:
        assert (dataset_spec.rv_define is not None)
        with details():
            summary('RV define')
            dict_table(dataset_spec.rv_define)


def _report_on_crosstab(
        crosstab_name: str,
        crosstab: ModelCrosstabSpec,
        model_spec: ModelSpec,
        model_index: ModelIndex,
        dataset_cache: DatasetCache,
) -> List[str]:
    """
    Use `dominate` to make a report on the given crosstab.
    Note that this will make a call to crosstab.extract_crosstab() to
    analyse the cross-table data.
    A list of issues is returned.
    """
    crosstab_index: CrosstabIndex = model_index.crosstabs[crosstab_name]

    num_states: int = crosstab_index.number_of_states
    datasource_name: str = crosstab.datasource
    sensitivity: float = model_spec.datasources[datasource_name].sensitivity
    min_cell_size: float = crosstab.min_cell_size
    epsilon: float = 0 if sensitivity == 0 else crosstab.epsilon

    crosstab_data: pd.DataFrame = dataset_cache[datasource_name].crosstab(crosstab.rvs)
    weights: pd.Series = crosstab_data.iloc[:, -1]

    crosstab_summary: Dict[str, Any] = {
        'Datasource': _datasource_link(datasource_name),
        'Sensitivity': clean_num(sensitivity),
        'Epsilon': clean_num(epsilon),
        'Min cell size': clean_num(min_cell_size),
        'State space size': f'{num_states:,}',

        'Clean number of rows': clean_num(crosstab_index.clean_num_rows),
        'Clean number of suppressed rows': clean_num(crosstab_index.clean_num_suppressed),
        'Clean min weight': clean_num(crosstab_index.clean_min_weight),
        'Clean max weight': clean_num(crosstab_index.clean_max_weight),
        'Clean total weight': clean_num(crosstab_index.clean_total_weight),

        'Lost rows': clean_num(crosstab_index.rows_lost),
        'Added rows': clean_num(crosstab_index.rows_added),
        'Final rows': clean_num(crosstab_index.noisy_num_rows),
        'Final number of suppressed rows': clean_num(crosstab_index.noisy_num_suppressed),
        'Final min weight': clean_num(crosstab_index.noisy_min_weight),
        'Final max weight': clean_num(crosstab_index.noisy_max_weight),
        'Final total weight': clean_num(crosstab_index.noisy_total_weight),
    }

    issues = []
    recommendations = []

    # Multiple condition tables are an issue as adjusting cross-table
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
    # Not having these rvs available as a distribution rv in some other cross-table
    # means this cross-table may have reduced accuracy for the joint
    # distribution over the cross-table rvs.
    #
    # This issue can normally be rectified by adding new cross-tables
    # to cover the necessary distribution rvs.
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
        num_rows: int = crosstab_index.noisy_num_rows
        num_suppressed: int = num_states - num_rows
        total_weight: float = crosstab_index.noisy_total_weight

        threshold = 0.5  # Lost rows and lost weight proportion threshold

        # Analyse potential new rows introduced by adding noise to suppressed rows
        recommended_min_cell_size_add_rows = 0
        if sensitivity > 0:
            alpha = 0.5 * math.exp(-min_cell_size * sensitivity / epsilon)
            expected_new_rows = math.ceil(num_suppressed * alpha)
            # Can never add more rows than is suppressed
            expected_new_rows = min(expected_new_rows, num_suppressed)

            if num_rows > 0 and num_suppressed > 0:
                recommended_min_cell_size_add_rows = recommended_min_cell_size(
                    epsilon,
                    sensitivity,
                    num_suppressed,
                    target_rows=num_rows
                )

            new_rows_percent = int(expected_new_rows / num_rows * 100 + 0.5)
            crosstab_summary.update({
                'Alpha': clean_num(alpha),
                'Expected new rows': f'{clean_num(expected_new_rows)} ({new_rows_percent}%)',

            })
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
            crosstab_summary.update({
                'Expected lost rows': f'{clean_num(expected_lost_rows)} ({row_loss_percent}%)',
                'Expected lost weight': f'{clean_num(expected_lost_weight)} ({weight_loss_percent}%)',
            })
            recommendations.append(f'maximum min-cell-size: {clean_num(recommended_min_cell_size_lost_rows)}')

            if expected_lost_rows / num_rows > threshold:
                issues.append('high expected lost rows')

            if recommended_min_cell_size_lost_rows < recommended_min_cell_size_add_rows:
                issues.append(
                    f'no suitable min-cell-size: '
                    f'recommended maximum (for lost rows) = {clean_num(recommended_min_cell_size_lost_rows)}, '
                    f'recommended minimum (for added rows) = {clean_num(recommended_min_cell_size_add_rows)}'
                )

    with details():
        summary(crosstab_name)
        with div(id=f'CROSSTAB_{crosstab_name}'):
            noiser: NoiserSpec = crosstab.noiser
            if isinstance(noiser, NoiserSpecBasicLaplace):
                crosstab_summary.update({
                    'Noiser': noiser.type,
                })

            dict_table(crosstab_summary)

            if not isinstance(noiser, NoiserSpecBasicLaplace):
                with details():
                    summary(f'Noiser: {noiser.type}')
                    dict_table({
                        'Max add rows': noiser.max_add_rows,
                    })

            _link_to_rvs('Random variables', crosstab.rvs)

            with details(open=(len(issues) > 0)):
                summary(f'Issues ({len(issues)})')
                for issue in issues:
                    p(issue)

            with details():
                summary(f'Recommendations ({len(recommendations)})')
                for recommendation in recommendations:
                    p(recommendation)

    return issues


def _link_to_rvs(section_label: str, rv_names: Sequence[str]) -> None:
    with details():
        summary(f'{section_label} ({len(rv_names)})')
        for rv_name in rv_names:
            p(_rv_link(rv_name))


def _report_on_rv(rv_name: str, rv_spec: ModelRVSpec, rv_index: RVIndex) -> None:
    """
    Use `dominate` to make a report on random variable states.
    """
    states_spec = rv_spec.states
    states: List[State] = rv_index.states

    with details():
        summary(rv_name)
        with div(id=f'RV_{rv_name}'):

            summary_dict: Dict[str, Any] = {}

            if isinstance(states_spec, str):
                summary_dict.update({'States specified': states_spec})
            elif isinstance(states_spec, int):
                if states_spec <= 0:
                    summary_dict.update({'States specified': f'invalid value: {states_spec}'})
                elif states_spec < 5:
                    summary_dict.update({'States specified': ', '.join(repr(state) for state in range(states_spec))})
                else:
                    summary_dict.update({'States specified': f'0, 1, 2, 3, ... {states_spec - 2}, {states_spec - 1}'})
            else:
                # A general list of states
                if len(states_spec) == 0:
                    summary_dict.update({'States specified': 'invalid empty list'})
                else:
                    summary_dict.update({'States specified': f'list of {len(states_spec)} values'})

            summary_dict.update({
                'Ensure None': repr(rv_spec.ensure_none),
                'Primary datasource': _datasource_link(rv_index.primary_datasource),
            })
            dict_table(summary_dict)

            with details():
                summary(f'States ({len(states)})')
                for state in states:
                    p(repr(state))

            with details():
                summary(f'All datasources ({len(rv_index.all_datasources)})')
                for datasource_name in rv_index.all_datasources:
                    p(_datasource_link(datasource_name))

            with details():
                summary(f'All distribution crosstabs ({len(rv_index.all_distribution_crosstabs)})')
                for crosstab_name in rv_index.all_distribution_crosstabs:
                    p(_crosstab_link(crosstab_name))

            with details():
                summary(f'All sampling entities ({len(rv_index.all_sampling_entities)})')
                for entity_name in rv_index.all_sampling_entities:
                    p(_entity_link(entity_name))


def _datasource_link(datasource_name: str) -> html_tag:
    return a(datasource_name, href=f'#DATASOURCE_{datasource_name}')


def _rv_link(rv_name: str) -> html_tag:
    return a(rv_name, href=f'#RV_{rv_name}')


def _crosstab_link(crosstab_name: str) -> html_tag:
    return a(crosstab_name, href=f'#CROSSTAB_{crosstab_name}')


def _entity_link(entity_name: str) -> html_tag:
    return a(entity_name, href=f'#ENTITY_{entity_name}')
