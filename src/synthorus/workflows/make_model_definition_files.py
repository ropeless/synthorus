import shutil
from pathlib import Path
from typing import Optional, Sequence, List, Dict

import pandas as pd
from ck.pgm import State

from synthorus.dataset import Dataset
from synthorus.error import SynthorusError, NotReached
from synthorus.model.dataset_cache import DatasetCache
from synthorus.model.datasource_spec import DatasourceSpec
from synthorus.model.make_model_index import make_model_index
from synthorus.model.model_index import ModelIndex, CrosstabIndex
from synthorus.model.model_spec import ModelSpec, ModelCrosstabSpec
from synthorus.noise.noiser import NoiserResult, Noiser
from synthorus.noise.safe_random import SafeRandom
from synthorus.simulator.make_simulator_spec_from_model_spec import make_simulator_spec_from_model_spec
from synthorus.simulator.simulator_spec import SimulatorSpec
from synthorus.utils.config_help import config
from synthorus.utils.data_catcher import RamDataCatcher
from synthorus.utils.file_extras import DataPathLike
from synthorus.utils.print_function import PrintFunction
from synthorus.workflows.cross_table_loader import save_cross_table, CrossTableLoader
from synthorus.workflows.file_names import CLEAN_CROSS_TABLES, NOISY_CROSS_TABLES, REPORTS, \
    MODEL_SPEC_FILE_NAME, SIMULATOR_SPEC_FILE_NAME, CROSSTAB_REPORT_FILE_NAME, MODEL_INDEX_FILE_NAME, ENTITY_MODELS, \
    PRIVACY_REPORT_FILE_NAME, MODEL_SPEC_REPORT_FILE_NAME
from synthorus.workflows.make_pgms import make_entity_pgms
from synthorus.workflows.report_privacy import report_privacy
from synthorus.workflows.report_spec import report_model_spec

# Keep cross-tables loaded while making PGMs
CACHE_LOADED_CROSSTABS: bool = config.get('CACHE_LOADED_CROSSTABS', True)


def make_model_definition_files(
        model_spec: ModelSpec,
        model_definition_directory: Path | str,
        *,
        cwd: Optional[DataPathLike] = None,
        overwrite: bool = False,
        save_clean: bool = True,
        save_noisy: bool = True,
        make_privacy_report: bool = True,
        make_crosstab_report: bool = True,
        make_model_spec_report: bool = True,
        make_pgms: bool = True,
        log: PrintFunction = print
) -> None:
    """
    Given a model spec, create all model definition files, saving them to `model_definition_directory`.

    Extract the clean and noisy cross-tables from the given model spec, saving them to
    the nominated directory as a pickled Pandas DataFrame object (as flagged by
    parameters save_clean and save_noisy). Also creates reports to support the usage
    and auditing of the cross-tables in a synthetic data simulator.

    Importantly, the created cross-tables will not include any adjustment for
    condition variables in the datasources. That is done when the entity
    models are created.

    This method reads the given model spec and records in the `model_definition_directory`:
        (1) the given model spec
        (2) a model index
        (3) simulator specification
        (4) clean cross-tables
        (5) noisy cross-tables
        (6) entity pgms
        (7) a privacy report
        (8) a cross-table report.

    Args:
        model_spec: The model specification defining cross-tables, datasources, etc.
        cwd: working directory for interpreting roots in `model_spec`.
        model_definition_directory: Directory where to save cross-tables and other information.
        overwrite: if true, any existing output directory will first be emptied.
        save_clean: flag whether to save clean cross-tables or not.
        save_noisy: flag whether to save noisy cross-tables or not.
        make_privacy_report: flag whether to save a privacy report or not.
        make_crosstab_report: flag whether to save a cross-table report or not.
        make_model_spec_report: flag whether to save a model spec report or not.
        make_pgms: flag whether to create the PGMs or not.
        log: print function for log messages.

    Raises:
        SynthorusError: if there is an error interpreting the model spec.
        ValueError: if pgm models are requested, but the corresponding cross-tables are not saved.
    """
    model_directory_path: Path = _set_up_model_directory(model_definition_directory, overwrite)

    # Set up the subdirectory for reports
    report_path = model_directory_path / REPORTS
    report_path.mkdir(exist_ok=True)

    # Set up subdirectories for cross-tables
    clean_path: Optional[Path] = _cross_table_path(model_directory_path, CLEAN_CROSS_TABLES, save_clean)
    noisy_path: Optional[Path] = _cross_table_path(model_directory_path, NOISY_CROSS_TABLES, save_noisy)

    if clean_path is not None and noisy_path is not None and clean_path == noisy_path:
        raise SynthorusError('clean and noisy directories must be different')

    # Make model index
    dataset_cache = DatasetCache(model_spec, cwd)
    model_index: ModelIndex = make_model_index(model_spec, dataset_cache)

    # Infer the simulator from the model spec and save it
    simulator_spec: SimulatorSpec = make_simulator_spec_from_model_spec(model_spec)
    with open(model_directory_path / SIMULATOR_SPEC_FILE_NAME, 'w') as file:
        print(simulator_spec.model_dump_json(indent=2), file=file)

    # Extract and save cross-tables
    _extract_cross_tables(model_spec, model_index, dataset_cache, clean_path, noisy_path, log)
    # Make PGM models if requested
    if make_pgms:
        log(f'creating entity PGMs: {model_spec.pgm_crosstabs}')
        if model_spec.pgm_crosstabs == 'clean':
            if clean_path is None:
                raise ValueError('clean pgm models requested but clean cross-tables not saved.')
            crosstab_loader = CrossTableLoader(clean_path, CACHE_LOADED_CROSSTABS)
        elif model_spec.pgm_crosstabs == 'noisy':
            if noisy_path is None:
                raise ValueError('noisy pgm models requested but noisy cross-tables not saved.')
            crosstab_loader = CrossTableLoader(noisy_path, CACHE_LOADED_CROSSTABS)
        else:
            raise NotReached()
        entity_models_directory: Path = model_directory_path / ENTITY_MODELS
        entity_models_directory.mkdir()
        make_entity_pgms(model_index, crosstab_loader, entity_models_directory, log=log)

    # Save the model spec
    with open(model_directory_path / MODEL_SPEC_FILE_NAME, 'w') as file:
        print(model_spec.model_dump_json(indent=2), file=file)

    # Save the model index
    with open(model_directory_path / MODEL_INDEX_FILE_NAME, 'w') as file:
        print(model_index.model_dump_json(indent=2), file=file)

    # Save initial reports
    if make_crosstab_report:
        log('saving cross-tables report')
        crosstab_report: RamDataCatcher = _extract_crosstab_report(model_spec, model_index)
        crosstab_report.to_csv(report_path / CROSSTAB_REPORT_FILE_NAME)
    if make_privacy_report:
        log('saving privacy report')
        report_privacy(model_spec, model_index, dataset_cache, report_path / PRIVACY_REPORT_FILE_NAME)
    if make_model_spec_report:
        log('saving model spec report')
        report_model_spec(model_spec, model_index, dataset_cache, report_path / MODEL_SPEC_REPORT_FILE_NAME)

    log()
    log('make_cross_tables completed')


def _set_up_model_directory(model_definition_directory: Path | str, overwrite: bool) -> Path:
    model_directory: Path = Path(model_definition_directory)
    if model_directory.exists():
        if not model_directory.is_dir():
            raise SynthorusError(f'model directory is not a directory: {model_directory}')
        if overwrite:
            shutil.rmtree(model_directory)
        elif len([f for f in model_directory.iterdir() if not f.name.startswith('.')]) != 0:
            raise SynthorusError(f'model directory not empty: {model_directory}')

    model_directory.mkdir(exist_ok=True)
    return model_directory


def _extract_crosstab_report(model_spec: ModelSpec, model_index: ModelIndex) -> RamDataCatcher:
    """
    Extract a tabular report of the cross-tables.
    """
    crosstab_report = RamDataCatcher()
    for crosstab_name, crosstab_spec in model_spec.crosstabs.items():
        crosstab_index = model_index.crosstabs[crosstab_name]
        datasource_spec = model_spec.datasources[crosstab_index.datasource]

        crosstab_record = crosstab_report.append()
        crosstab_record['Cross-table'] = crosstab_name

        def _track(label, value, denominator=None):
            crosstab_record[label] = value
            if denominator is not None:
                percentage: float = value / denominator * 100
                crosstab_record[f'{label}%'] = percentage

        _track('Random variables', ' '.join(repr(rv) for rv in crosstab_index.rvs))
        _track('Number-of-rvs', len(crosstab_index.rvs))
        _track('Datasource', crosstab_index.datasource)
        _track('State space size', crosstab_index.number_of_states)

        _track('Clean number of rows', crosstab_index.clean_num_rows)
        _track('Clean number of suppressed rows', crosstab_index.clean_num_suppressed)
        _track('Clean min weight', crosstab_index.clean_min_weight)
        _track('Clean max weight', crosstab_index.clean_max_weight)
        _track('Clean total weight', crosstab_index.clean_total_weight)

        _track('Sensitivity', datasource_spec.sensitivity)
        _track('Epsilon', crosstab_spec.epsilon)
        _track('Min cell size', crosstab_spec.min_cell_size)
        _track('Noiser', crosstab_spec.noiser.model_dump_json())
        _track('Orig rows', crosstab_index.clean_num_rows)
        _track('Lost rows', crosstab_index.rows_lost, crosstab_index.clean_num_rows)
        _track('Added rows', crosstab_index.rows_added, crosstab_index.clean_num_rows)
        _track('Final rows', crosstab_index.noisy_num_rows, crosstab_index.clean_num_rows)
        _track('Final min weight', crosstab_index.noisy_min_weight)
        _track('Final max weight', crosstab_index.noisy_max_weight)
        _track('Final total weight', crosstab_index.noisy_total_weight)

    return crosstab_report


def _cross_table_path(model_directory_path: Path, sub_dir: str, save: bool) -> Optional[Path]:
    """
    Constructs a cross-table's subdirectory.
    """
    if save:
        sub_dir_path = model_directory_path / sub_dir
        sub_dir_path.mkdir()
        return sub_dir_path
    else:
        return None


def _extract_cross_tables(
        model_spec: ModelSpec,
        model_index: ModelIndex,
        dataset_cache: DatasetCache,
        clean_path: Optional[Path],
        noisy_path: Optional[Path],
        log: PrintFunction
) -> None:
    """
    This will extract all necessary cross-tables, not just the specified cross-tables, but also
    cross-tables needed to resolve datasource conditioning variables.

    Args:
        model_spec: model specification object.
        dataset_cache: access to datasources.
        clean_path: where to save clean cross-tables (or None).
        noisy_path: where to save noisy cross-tables (or None).
        log: print function for log messages.
    """
    # Create a safe random number for adding noise to cross-tables.
    random = SafeRandom(model_spec.rng_n)

    datasources: Dict[str, DatasourceSpec] = model_spec.datasources

    # Create each cross-table
    for crosstab_name in model_spec.crosstabs.keys():
        _extract_cross_table(
            crosstab_name,
            model_spec,
            model_index,
            dataset_cache,
            datasources,
            random,
            clean_path,
            noisy_path,
            log
        )
    log()


def _extract_cross_table(
        crosstab_name: str,
        model_spec: ModelSpec,
        model_index: ModelIndex,
        dataset_cache: DatasetCache,
        datasource_specs: Dict[str, DatasourceSpec],
        random: SafeRandom,
        clean_path: Optional[Path],
        noisy_path: Optional[Path],
        log: PrintFunction
):
    crosstab_spec: ModelCrosstabSpec = model_spec.crosstabs[crosstab_name]
    crosstab_index: CrosstabIndex = model_index.crosstabs[crosstab_name]

    rvs_names: List[str] = crosstab_index.rvs  # must use the index rvs as may include extras
    crosstab_rvs: Dict[str, Sequence[State]] = {rv_name: model_index.rvs[rv_name].states for rv_name in rvs_names}
    datasource_name: str = crosstab_index.datasource
    dataset: Dataset = dataset_cache[datasource_name]

    def _track(label, value, denominator=None):
        if denominator is not None:
            percentage: float = value / denominator * 100
            log(f'{label}: {value:,} ({percentage:.2f}%)')
        elif isinstance(value, (int, float)):
            log(f'{label}: {value:,}')
        else:
            log(f'{label}: {value}')

    log()
    log(f'making clean cross-table {crosstab_name!r}')

    _track('Random variables', ' '.join(repr(rv) for rv in rvs_names))
    _track('Number-of-rvs', len(rvs_names))
    _track('Datasource', datasource_name)

    crosstab: pd.DataFrame = dataset.crosstab(rvs_names)
    weights = crosstab.iloc[:, -1]
    num_rows = crosstab.shape[0]
    total_weight = weights.sum()
    min_weight = weights.min()
    max_weight = weights.max()
    num_states = crosstab_index.number_of_states
    num_suppressed = num_states - num_rows
    datasource_sensitivity: float = datasource_specs[datasource_name].sensitivity
    epsilon = 0.0 if datasource_sensitivity == 0 else crosstab_spec.epsilon

    _track('State space size', num_states)
    _track('Clean number of rows', num_rows)
    _track('Clean number of suppressed rows', num_suppressed)
    _track('Clean min weight', min_weight)
    _track('Clean max weight', max_weight)
    _track('Clean total weight', total_weight)

    crosstab_index.clean_num_rows = num_rows
    crosstab_index.clean_min_weight = min_weight
    crosstab_index.clean_max_weight = max_weight
    crosstab_index.clean_total_weight = total_weight

    if clean_path is not None:
        save_cross_table(crosstab, clean_path, crosstab_name)

    # Default noiser result, updated if noise is added.
    crosstab_index.noisy_num_rows = crosstab_index.clean_num_rows
    crosstab_index.noisy_min_weight = crosstab_index.clean_min_weight
    crosstab_index.noisy_max_weight = crosstab_index.clean_max_weight
    crosstab_index.noisy_total_weight = crosstab_index.clean_total_weight
    crosstab_index.rows_lost = 0
    crosstab_index.rows_added = 0

    if noisy_path is not None:
        log()
        log(f'making noisy cross-table {crosstab_name!r}')

        _track('Sensitivity', datasource_sensitivity)
        _track('Epsilon', epsilon)
        _track('Min cell size', crosstab_spec.min_cell_size)
        _track('Noiser', crosstab_spec.noiser.model_dump_json())

        noiser: Noiser = crosstab_spec.noiser.noiser()
        noiser_result: NoiserResult = noiser.add_noise(
            crosstab,
            crosstab_rvs,
            random,
            datasource_sensitivity,
            epsilon,
            crosstab_spec.min_cell_size,
            log
        )

        assert crosstab_index.clean_num_rows == noiser_result.rows_original, 'consistency check'

        noisy_cross_table: pd.DataFrame = noiser_result.cross_table
        weights: pd.Series = noisy_cross_table.iloc[:, -1]

        crosstab_index.noisy_num_rows = noiser_result.rows_final
        crosstab_index.noisy_min_weight = float(weights.min())
        crosstab_index.noisy_max_weight = float(weights.max())
        crosstab_index.noisy_total_weight = float(weights.sum())
        crosstab_index.rows_lost = noiser_result.rows_lost
        crosstab_index.rows_added = noiser_result.rows_added

        clean_num_rows = crosstab_index.clean_num_rows
        _track('Orig rows', clean_num_rows)
        _track('Lost rows', noiser_result.rows_lost, clean_num_rows)
        _track('Added rows', noiser_result.rows_added, clean_num_rows)
        _track('Final rows', noiser_result.rows_final, clean_num_rows)

        save_cross_table(noiser_result.cross_table, noisy_path, crosstab_name)
