from pathlib import Path

from synthorus.model.datasource_spec import DatasourceSpec
from synthorus.model.model_index import ModelIndex, CrosstabIndex
from synthorus.model.model_meta import ModelMeta, CrosstabMeta
from synthorus.model.model_spec import ModelSpec, ModelCrosstabSpec
from synthorus.utils.data_catcher import RamDataCatcher
from synthorus.utils.file_extras import open_text
from synthorus.workflows.file_names import REPORTS, CROSSTAB_REPORT_FILE_NAME, MODEL_META_FILE_NAME, \
    MODEL_INDEX_FILE_NAME, MODEL_SPEC_FILE_NAME


def make_crosstabs_report(
        model_directory_path: Path | str,
        *,
        overwrite: bool = False,
) -> None:
    """
    Create a crosstabs report on an existing model.

    Args:
        model_directory_path: Directory where to find model cross-tables and other information.
        overwrite: if True, then any previous report will be overwritten.
    """
    if isinstance(model_directory_path, str):
        model_directory_path = Path(model_directory_path)
    report_path: Path = model_directory_path / REPORTS / CROSSTAB_REPORT_FILE_NAME

    if overwrite:
        report_path.unlink(missing_ok=True)
    elif report_path.exists():
        raise RuntimeError(f'report already exists: {report_path}')

    with open_text(model_directory_path / MODEL_SPEC_FILE_NAME) as file:
        model_spec: ModelSpec = ModelSpec.model_validate_json(file.read())

    with open_text(model_directory_path / MODEL_INDEX_FILE_NAME) as file:
        model_index: ModelIndex = ModelIndex.model_validate_json(file.read())

    with open_text(model_directory_path / MODEL_META_FILE_NAME) as file:
        model_meta: ModelMeta = ModelMeta.model_validate_json(file.read())

    report_crosstabs(
        model_spec=model_spec,
        model_index=model_index,
        model_meta=model_meta,
        destination=report_path,
    )


def report_crosstabs(
        model_spec: ModelSpec,
        model_index: ModelIndex,
        model_meta: ModelMeta,
        destination: Path,
) -> None:
    """
    Generate a tabular cross-tables report on the given model specification.

    Args:
        model_spec: The model specification.
        model_index: Index of model specification.
        model_meta: The metadata of the model specification.
        destination: Where to write the report.
    """
    crosstab_report: RamDataCatcher = _extract_crosstab_report(model_spec, model_index, model_meta)
    crosstab_report.to_csv(destination)


def _extract_crosstab_report(
        model_spec: ModelSpec,
        model_index: ModelIndex,
        model_meta: ModelMeta,
) -> RamDataCatcher:
    """
    Extract a tabular report of the cross-tables.
    """
    crosstab_report = RamDataCatcher()
    crosstab_spec: ModelCrosstabSpec
    for crosstab_name, crosstab_spec in model_spec.crosstabs.items():
        crosstab_index: CrosstabIndex = model_index.crosstabs[crosstab_name]
        crosstab_meta: CrosstabMeta = model_meta.crosstabs[crosstab_name]
        datasource_spec: DatasourceSpec = model_spec.datasources[crosstab_index.datasource]

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
        _track('State space size', crosstab_meta.number_of_states)

        _track('Clean number of rows', crosstab_meta.clean_num_rows)
        _track('Clean number of suppressed rows', crosstab_meta.clean_num_suppressed)
        _track('Clean min weight', crosstab_meta.clean_min_weight)
        _track('Clean max weight', crosstab_meta.clean_max_weight)
        _track('Clean total weight', crosstab_meta.clean_total_weight)

        _track('Sensitivity', datasource_spec.sensitivity)
        _track('Epsilon', crosstab_spec.epsilon)
        _track('Min cell size', crosstab_spec.min_cell_size)
        _track('Noiser', crosstab_spec.noiser.model_dump_json())
        _track('Orig rows', crosstab_meta.clean_num_rows)
        _track('Lost rows', crosstab_meta.rows_lost, crosstab_meta.clean_num_rows)
        _track('Added rows', crosstab_meta.rows_added, crosstab_meta.clean_num_rows)
        _track('Final rows', crosstab_meta.noisy_num_rows, crosstab_meta.clean_num_rows)
        _track('Final min weight', crosstab_meta.noisy_min_weight)
        _track('Final max weight', crosstab_meta.noisy_max_weight)
        _track('Final total weight', crosstab_meta.noisy_total_weight)

    return crosstab_report
