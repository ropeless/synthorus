"""
Module for analysing and reporting on utility.
"""
from __future__ import annotations

import getpass
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Iterable, Tuple, Iterator, List, Set, Union, Collection, FrozenSet

from ck.dataset.cross_table import CrossTable
from ck.pgm import PGM, RandomVariable
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import DEFAULT_PGM_COMPILER
from ck.utils.iter_extras import powerset

from synthorus.model.defaults import DEFAULT_ENTITY_NAME
from synthorus.model.model_index import ModelIndex
from synthorus.model.model_spec import ModelSpec
from synthorus.utils.multiprocessing_extras import run_trial_processes, NumProcesses, DefaultTrialLogger, WarningMessage
from synthorus.utils.print_function import PrintFunction, Print
from synthorus.utils.time_extras import timestamp
from synthorus.workflows.cross_table_loader import CrossTableLoader
from synthorus.workflows.file_names import MODEL_SPEC_NAME, MODEL_INDEX_NAME, REPORTS, UTILITY_REPORT_FILE_NAME, \
    CLEAN_CROSS_TABLES, UTILITY_RESULTS_FILE_NAME, ENTITY_MODELS
from synthorus.workflows.load_entity_pgm import load_entity_pgm
from synthorus.workflows.make_pgms import EntityCrossTableMaker
from synthorus.workflows.utility_measures import Measure
from synthorus.utils.config_help import config

DEFAULT_EVAL_LIMIT = 10
DEFAULT_MEASURE_CROSSTABS = True
DEFAULT_INDENT = '    '
CHART_MAX_X_SIZE = 50
CHART_SOURCE_COLOUR = '#008000'
CHART_MODEL_COLOUR = '#800000'

DEFAULT_MEASURES: Collection[Measure] = (Measure.HI, Measure.KL)

# How many concurrent processes to use during evaluation
DEFAULT_PROCESSES: NumProcesses = config.get('DEFAULT_PROCESSES', 1)

# Keep cross-tables loaded while making PGMs
CACHE_LOADED_CROSSTABS: bool = config.get('CACHE_LOADED_CROSSTABS', True)


def make_utility_report(
        model_directory_path: Path,
        *,
        overwrite: bool = False,
        report_author: Optional[str] = None,
        limit_variable_combinations: Optional[int] = DEFAULT_EVAL_LIMIT,
        measure_crosstabs: bool = DEFAULT_MEASURE_CROSSTABS,
        measure: Union[Measure, Collection[Measure]] = DEFAULT_MEASURES,
        processes: NumProcesses = DEFAULT_PROCESSES,
        prefix: str = '',
        indent: str = DEFAULT_INDENT,
        log: PrintFunction = print
) -> None:
    """
    Analyse and report on model utility.
    Assumes that 'make_cross_tables' and 'make_models' was run with
    the given directory path, and all model files are in place there.

    Module utility is evaluated based on clean cross-tables.

    Args:
        model_directory_path: Directory where to find model cross-tables and other information.
        overwrite: if True, then any previous report will be overwritten.
        report_author: Optional name of the report author (default is system username).
        limit_variable_combinations: is an optional limit on analysis of variable combinations.
        measure_crosstabs: if True then every whole crosstab will be measured.
        measure: what Measure or collection of Measures to report on.
        processes: number of parallel processes to use to calculate measurements.
        prefix: is a string to prefix every written line.
        indent: is a string used for indentation.
        log: a destination for log messages
    """
    report_path: Path = model_directory_path / REPORTS / UTILITY_REPORT_FILE_NAME
    results_path: Path = model_directory_path / REPORTS / UTILITY_RESULTS_FILE_NAME
    pgms_path: Path = model_directory_path / ENTITY_MODELS

    if overwrite:
        report_path.unlink(missing_ok=True)
        results_path.unlink(missing_ok=True)
    elif report_path.exists() or results_path.exists():
        raise RuntimeError(f'report already exists: {report_path}')

    with open(model_directory_path / MODEL_SPEC_NAME, 'r') as file:
        model_spec: ModelSpec = ModelSpec.model_validate_json(file.read())

    with open(model_directory_path / MODEL_INDEX_NAME, 'r') as file:
        model_index: ModelIndex = ModelIndex.model_validate_json(file.read())

    crosstab_loader = CrossTableLoader(model_directory_path / CLEAN_CROSS_TABLES, CACHE_LOADED_CROSSTABS)

    measures: List[Measure]
    if isinstance(measure, Measure):
        measures = [measure]
    else:
        measures = list(measure)
    del measure

    if len(measures) == 0:
        raise ValueError('must have at least one measure of utility')
    if limit_variable_combinations is not None and limit_variable_combinations < 1:
        raise ValueError('limit_variable_combinations must be > 1')

    with Print(report_path) as _print:
        log('generating report on utility')

        next_prefix = prefix + indent

        if report_author is None:
            report_author = f'user "{getpass.getuser()}"'

        _print(f'{prefix}Utility Report')
        _print(f'{prefix}==============')
        _print(prefix)
        _print(f'{prefix}Report date: {timestamp()}')
        _print(f'{prefix}Report author: {report_author}')
        _print(prefix)
        _print(f'{prefix}Model name: {model_spec.name}')
        _print(f'{prefix}Model author: {model_spec.author}')
        _print(prefix)

        analysis: Analysis = _analyse(
            model_index=model_index,
            crosstab_loader=crosstab_loader,
            pgms_path=pgms_path,
            limit_variable_combinations=limit_variable_combinations,
            measure_crosstabs=measure_crosstabs,
            measures=measures,
            processes=processes,
            log=log,
        )

        # Save detailed results
        with open(results_path, 'w') as file:
            def _write(*args):
                print(*args, file=file, sep=',')

            analysis.print_results(_write)

        _print(f'{prefix}Analysis summary:')
        _print(prefix)
        _print(f'{next_prefix}Number of entities: {analysis.number_of_entities}')
        _print(prefix)

        summary_measure: Measure = measures[0]
        if analysis.number_of_entities == 1:
            entity: EntityAnalysis = next(iter(analysis))
            if entity.name != DEFAULT_ENTITY_NAME:
                _print(f'{next_prefix}Entity: {entity.name}')
            _summarise_entity(next_prefix, indent, entity, summary_measure, _print)
            _print(prefix)
        else:
            next_next_prefix = next_prefix + indent

            # Sort the entities from worst to best
            entities = [
                (entity, entity.minimum(num_rvs=None, measure=summary_measure))
                for entity in analysis
            ]
            entities.sort(key=lambda x: x[1][0])  # sort on measurement value

            entity: EntityAnalysis
            for entity, _ in entities:
                _print(f'{next_prefix}Entity: {entity.name}')
                _summarise_entity(next_next_prefix, indent, entity, summary_measure, _print)
                _print(prefix)

        _print(f'{prefix}{"-" * 80}')


# def make_chart(
#         entity_name: str,
#         crosstab: pd.DataFrame,
#         model: PGM,
#         stats: Mapping[str, float],
#         utility_results_directory: Path
# ) -> None:
#     """
#     Create a set of  Excel charts, one for each random variable, each chart
#     comparing empirical and model probabilities.
#
#     Each chart is named: utility_results_directory / f'{entity_name}_{rv.name}.xlsx'
#
#     Args:
#         entity_name: name of the entity.
#         crosstab: a cross-table covering the random variables of the model (empirical probability).
#         model: a probabilistic model (model probabilities).
#         stats: A dictionary of statistics to include in the chart.
#         utility_results_directory: where to write the results.
#     """
#     pgm_rvs = model.pgm_rvs
#     wmc = model.wmc
#     assert len(pgm_rvs) == 1, 'expect a single random variable to plot'
#     rv = pgm_rvs[0]
#
#     # Convert the data probabilities to a dictionary from state to probability
#     total_df = crosstab.iloc[:, -1].sum()
#     data_pr = {
#         clean_state(row[0]): row[-1] / total_df
#         for row in crosstab.itertuples(index=False)
#     }
#
#     # Create an Excel workbook
#     file_path = utility_results_directory / f'{entity_name}_{rv.name}.xlsx'
#     workbook = xlsxwriter.Workbook(file_path)
#     heading_format = workbook.add_format()
#     heading_format.set_bold()
#     heading_format.set_align('center')
#
#     # Write the data to the worksheet 'data'
#     worksheet = workbook.add_worksheet('data')
#     worksheet.write_row(0, 0, [rv.name, 'data-pr', 'model-pr'], heading_format)
#     for ind in rv:
#         state = rv.states[ind.state_idx]
#         state_data_pr = data_pr.get(state, 0.0)
#         state_model_pr = wmc.probability(ind)
#         sheet_row = ind.state_idx + 1
#         state = _clean_cell(state)
#         state_data_pr = _clean_cell(state_data_pr)
#         state_model_pr = _clean_cell(state_model_pr)
#         worksheet.write_row(sheet_row, 0, [state, state_data_pr, state_model_pr])
#     data_end = crosstab.shape[0]
#
#     # Append any provided statistics
#     for i, stat_name in enumerate(stats.keys()):
#         sheet_row = data_end + 2 + i
#         stat_value = _clean_cell(stats[stat_name])
#         worksheet.write_row(sheet_row, 0, [stat_name, stat_value])
#
#     # Create a histogram chart for the data
#     chart = workbook.add_chart({
#         'type': 'column',
#     })
#     chart.set_title({
#         'name': 'Histogram Comparison',
#     })
#     if entity_name == DEFAULT_ENTITY_NAME:
#         x_axis_name = f'Variable: {rv.name}'
#     else:
#         x_axis_name = f'Entity: {entity_name}, Variable: {rv.name}'
#     chart.set_x_axis({
#         'name': x_axis_name,
#     })
#     chart.set_y_axis({
#         'name': 'Probability',
#     })
#     chart.set_legend({
#         'position': 'right',
#     })
#
#     col = 1
#     chart.add_series({
#         'name': ['data', 0, col],
#         'categories': ['data', 1, 0, data_end, 0],
#         'values': ['data', 1, col, data_end, col],
#         'fill': {'color': CHART_SOURCE_COLOUR},
#     })
#     col = 2
#     chart.add_series({
#         'name': ['data', 0, col],
#         'categories': ['data', 1, 0, data_end, 0],
#         'values': ['data', 1, col, data_end, col],
#         'fill': {'color': CHART_MODEL_COLOUR},
#     })
#
#     chart_sheet = workbook.add_chartsheet('chart')
#     chart_sheet.set_chart(chart)
#     workbook.close()


def _analyse(
        model_index: ModelIndex,
        crosstab_loader: CrossTableLoader,
        pgms_path: Path,
        limit_variable_combinations: Optional[int],
        measure_crosstabs: bool,
        measures: Collection[Measure],
        processes: NumProcesses,
        log
) -> Analysis:
    analysis = Analysis()

    trials = _make_trials(
        model_index=model_index,
        crosstab_loader=crosstab_loader,
        pgms_path=pgms_path,
        limit_variable_combinations=limit_variable_combinations,
        measure_crosstabs=measure_crosstabs,
        measures=measures,
        log=log,
    )

    # Sort the trials from largest to smallest.
    # This is a heuristic to avoid idle CPUs for trailing jobs.
    trials = sorted(trials, key=lambda t: -len(t.crosstab))
    num_trials = len(trials)

    log('running evaluation trials')
    run_trial_processes(
        trials,
        collector=_TrialCollector(analysis),
        trial_logger=_TrialLogger(log=log, num_trials=num_trials),
        processes=processes,
        discard_results_with_warning=False
    )

    return analysis


def _summarise_entity(prefix: str, indent: str, entity: EntityAnalysis, summary_measure: Measure, _print):
    if len(entity.warnings) > 0:
        for warning in entity.warnings:
            _print(f'{prefix}{warning}')
        return

    value, rvs_sets = entity.minimum(num_rvs=None, measure=summary_measure)
    if len(rvs_sets) == 0:
        _print(f'{prefix}No evaluation records')
    else:
        _print(f'{prefix}Worst {summary_measure.full_name()} overall: {value:3.2%}')
        next_prefix = prefix + indent
        sorted_rvs_sets = sorted(rvs_sets, key=lambda _rvs: -len(_rvs))  # sort largest to smallest set
        for rvs in sorted_rvs_sets:
            rvs_str = _rvs_str(rvs, ', ')
            _print(f'{next_prefix}random variable set: {{{rvs_str}}}')


@dataclass
class _TrialResult:
    entity_name: str
    rvs: Tuple[str, ...]
    crosstab_number_of_rows: int
    measure_name: str
    measurement: float


@dataclass
class _Trial:
    entity_name: str
    measure: Measure
    crosstab: CrossTable
    pgm_cct: PGMCircuit

    @property
    def trial_name(self) -> str:
        rvs: str = ', '.join(rv.name for rv in self.crosstab.rvs)
        return f'{self.entity_name} {self.measure.name} {rvs}'

    def __call__(self) -> _TrialResult:
        wmc = WMCProgram(self.pgm_cct)
        if wmc.z == 0:
            raise RuntimeError(f'PGM has z = 0, no probabilities are defined')

        measurement = self.measure(
            self.crosstab,
            wmc,
        )

        return _TrialResult(
            entity_name=self.entity_name,
            rvs=tuple(rv.name for rv in self.crosstab.rvs),
            crosstab_number_of_rows=len(self.crosstab),
            measure_name=self.measure.full_name(),
            measurement=measurement,
        )


@dataclass
class _TrialCollector:
    analysis: Analysis

    def __call__(self, results: Iterable[_TrialResult]) -> None:
        for result in results:
            self.analysis.add(
                result.entity_name,
                result.rvs,
                result.measure_name,
                result.measurement
            )


class _TrialLogger(DefaultTrialLogger):

    def __init__(self, log=print, num_trials: Optional[int] = None):
        super().__init__(log, num_trials)

    def log_end(self, result, warning_messages: List[WarningMessage]):
        if len(warning_messages) == 0:
            self.log(f'{result.measurement}')
        else:
            self.log(f'{result.measurement} (completed with warnings)')
            self.log_warnings(warning_messages)


def _make_trials(
        model_index: ModelIndex,
        crosstab_loader: CrossTableLoader,
        pgms_path: Path,
        limit_variable_combinations: Optional[int],
        measure_crosstabs: bool,
        measures: Collection[Measure],
        log
) -> Iterator[_Trial]:
    for entity_name, entity_index in model_index.entities.items():
        log(f'entity: {entity_name}')
        log('loading PGM')
        pgm: PGM = load_entity_pgm(pgms_path, entity_name)

        log('compiling PGM')
        pgm_cct: PGMCircuit = DEFAULT_PGM_COMPILER(pgm)

        log('constructing evaluation trials')
        crosstab: CrossTable
        for crosstab in _make_eval_crosstabs(
                entity_name,
                pgm,
                model_index,
                crosstab_loader,
                limit_variable_combinations,
                measure_crosstabs,
        ):
            for measure in measures:
                yield _Trial(
                    entity_name=entity_name,
                    measure=measure,
                    crosstab=crosstab,
                    pgm_cct=pgm_cct,
                )


def _make_eval_crosstabs(
        entity_name: str,
        pgm: PGM,
        model_index: ModelIndex,
        crosstab_loader: CrossTableLoader,
        limit_variable_combinations: Optional[int],
        measure_crosstabs: bool,
) -> Iterable[CrossTable]:
    # Get all the cross-tables used for the entity
    crosstab_maker = EntityCrossTableMaker(
        crosstab_loader=crosstab_loader,
        model_index=model_index,
        entity_index=model_index.entities[entity_name],
        pgm=pgm,
        add_rvs=False,
    )
    cross_tables: List[CrossTable] = crosstab_maker.get_cross_tables()

    # Whole cross-tables
    if measure_crosstabs:
        for cross_table in cross_tables:
            if len(cross_table.rvs) < 2:
                # Cross-tables with less than 2 rvs are dealt by the next loop.
                continue
            if limit_variable_combinations is None:
                yield cross_table
            elif len(cross_table.rvs) <= limit_variable_combinations:
                yield cross_table
            else:
                combos = powerset(
                    cross_table.rvs,
                    min_size=limit_variable_combinations,
                    max_size=limit_variable_combinations,
                )
                for rv_combo in combos:
                    yield cross_table.project(rv_combo)

    # Individual random variables
    rvs_seen: Set[RandomVariable] = set()
    for cross_table in sorted(cross_tables, key=lambda table: -len(table.rvs)):
        for rv in cross_table.rvs:
            if rv not in rvs_seen:
                rvs_seen.add(rv)
                yield cross_table.project([rv])


def _rvs_str(rvs, sep) -> str:
    return sep.join(rv for rv in sorted(rvs))


def _clean_cell(value):
    """
    Clean a cell value for Excel.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return 'nan'
        if math.isinf(value):
            return 'inf'
    return value


@dataclass
class EntityAnalysisRecord:
    rvs: FrozenSet[str]
    measure_name: str
    measurement: float


class EntityAnalysis:
    """
    Utility analysis results for one entity.
    """

    def __init__(self, entity_name: str):
        self._entity_name = entity_name
        self._all_data: List[EntityAnalysisRecord] = []
        self._seen_rvs = set()
        self._warnings = []

    @property
    def name(self) -> str:
        return self._entity_name

    @property
    def warnings(self) -> List[str]:
        return self._warnings

    def add_warning(self, warning: str):
        self._warnings.append(warning)

    def add(self, rvs: Iterable[str], measure_name: str, measurement: float):
        rvs = frozenset(rvs)
        self._all_data.append(EntityAnalysisRecord(rvs, measure_name, measurement))
        self._seen_rvs.add(rvs)

    def contains(self, rvs: Iterable[str]) -> bool:
        rvs = frozenset(rvs)
        return rvs in self._seen_rvs

    def __iter__(self) -> Iterator[EntityAnalysisRecord]:
        return iter(self._all_data)

    def combination_sizes(self) -> Set[int]:
        return {
            len(rv_combo)
            for rv_combo in self._seen_rvs
        }

    def minimum(self, num_rvs: Optional[int], measure: Measure) -> Tuple[float, List[frozenset[str]]]:
        """
        Which random variables had the minimum value for the given measure.

        Args:
            num_rvs: Optionally restrict the number of random variables being considered to a specific number.
            measure: What is the measure of interest.

        Returns:
            value: the smallest value of the measure of interest
            A list of sets of random variable resulting in the measure having that value.
        """
        measure_name: str = measure.full_name()

        if num_rvs is not None:
            selected = [
                (record.measurement, record.rvs)
                for record in self._all_data
                if record.measure_name == measure_name and len(record.rvs) == num_rvs
            ]
        else:
            selected = [
                (record.measurement, record.rvs)
                for record in self._all_data
                if record.measure_name == measure_name
            ]
        if len(selected) == 0:
            return float('nan'), []
        value, _ = min(selected)
        return value, [_rvs for _value, _rvs in selected if _value == value]


class Analysis:
    """
    Utility analysis results for all entities.
    """

    def __init__(self):
        self._entities: Dict[str, EntityAnalysis] = {}

    def add_warning(self, entity_name: str, warning: str):
        entity = self._entities.get(entity_name)
        if entity is None:
            entity = EntityAnalysis(entity_name)
            self._entities[entity_name] = entity
        entity.add_warning(warning)

    def add(self, entity_name: str, rvs: Iterable[str], measure_name: str, measurement: float):
        entity = self._entities.get(entity_name)
        if entity is None:
            entity = EntityAnalysis(entity_name)
            self._entities[entity_name] = entity
        entity.add(rvs, measure_name, measurement)

    def contains(self, entity_name: str, rvs: Iterable[str]) -> bool:
        """
        Has a result been recorded for the given entity and RVS?
        """
        entity = self._entities.get(entity_name)
        if entity is None:
            return False
        return entity.contains(rvs)

    @property
    def number_of_entities(self):
        return len(self._entities)

    def __iter__(self) -> Iterator[EntityAnalysis]:
        return iter(self._entities.values())

    def print_results(self, _print: PrintFunction = print) -> None:
        entity: EntityAnalysis
        entity_record: EntityAnalysisRecord

        _print('entity', 'number_of_rvs', 'rvs', 'measure_name', 'measurement')
        for entity in self:
            for entity_record in entity:
                rvs = entity_record.rvs
                number_of_rvs = len(rvs)
                rvs_as_str = _rvs_str(rvs, ' ')
                _print(
                    entity.name,
                    number_of_rvs,
                    rvs_as_str,
                    entity_record.measure_name,
                    entity_record.measurement,
                )
