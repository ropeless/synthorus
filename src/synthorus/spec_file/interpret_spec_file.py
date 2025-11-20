import warnings
from importlib.abc import Traversable
from itertools import count
from numbers import Integral, Real
from os import PathLike
from pathlib import Path
from types import MappingProxyType, NoneType
from typing import Optional, Dict, Any, Mapping, List, Literal, Sequence, Set, Iterable, Tuple

from ck.pgm import State

import synthorus.spec_file.keys as key
from synthorus.dataset import Dataset
from synthorus.error import NotReached
from synthorus.model.dataset_cache import interpret_roots
from synthorus.model.dataset_spec import DatasetSpec
from synthorus.model.dataset_spec_impl import DatasetSpecCsv, TextInputSpec, TextInputSpecLocation, TextInputSpecInline, \
    ColumnSpec, ColumnDefinitionSpec, ColumnDefinitionSpecFunction, ColumnDefinitionSpecGroup, DatasetSpecPickle, \
    DatasetSpecTableBuilder, DatasetSpecParquet, DatasetSpecFeather, DatasetSpecFunction, DatasetSpecDBMS
from synthorus.model.datasource_spec import DatasourceSpec
from synthorus.model.defaults import *
from synthorus.model.model_spec import ModelSpec, ModelRVSpec, ModelCrosstabSpec, ModelEntitySpec, ModelFieldSpecSample, \
    ModelFieldSpec, ModelFieldSpecFunction, ModelFieldSpecSum
from synthorus.simulator.condition_spec import ConditionSpec, ConditionSpecFixedLimit, ConditionSpecVariableLimit, \
    ConditionSpecStates
from synthorus.spec_file.spec_dict import SpecDict
from synthorus.utils import py_loader
from synthorus.utils.clean_state import clean_state, clean_states
from synthorus.utils.string_extras import strip_lines

# Application level default values for any spec file.
DEFAULTS: Mapping[str, Any] = MappingProxyType({
    key.name: DEFAULT_NAME,
    key.author: DEFAULT_AUTHOR,
    key.comment: DEFAULT_COMMENT,
    key.rng_n: DEFAULT_RNG_N,
    key.min_cell_size: DEFAULT_MIN_CELL_SIZE,
    key.epsilon: DEFAULT_EPSILON,
    key.id_field: DEFAULT_ID_FIELD,
    key.count_field: DEFAULT_COUNT_FIELD,
    key.max_add_rows: DEFAULT_MAX_ADD_ROWS
})

# These are the top-level section names in a spec dictionary.
# Specifically, these names merely introduce sub-dictionaries.
# These names are not inherited across each other. E.g., there
# is no need for the 'rvs' section to inherit the 'entities' section.
SECTIONS: List[str] = [
    key.datasources,
    key.rvs,
    key.crosstabs,
    key.parameters,
    key.entities,
]


def load_spec_file(
        filepath: PathLike | Traversable,
        *,
        variable: Optional[str] = None,
        defaults: Mapping[str, Any] = DEFAULTS,
        cwd: Optional[PathLike | Traversable] = None,
) -> ModelSpec:
    """
    Read a spec dictionary from a file then interpret it as a ModelSpec.

    Args:
        filepath: path to file to read.
        variable: is the name of the Python variable in the file
            that references the spec dictionary. If None, then the file
            should only have one such variable.
        cwd: file path for current working directory for resolving datasource filenames.
        defaults: an optional dictionary of default values.
    """
    module = py_loader.load(filepath)
    spec: Mapping[str, Any] = py_loader.get_object(module, object_type=dict, variable=variable)
    module_vars = vars(module)

    defaults: Dict[str, Any] = dict(defaults)

    def _check_module_default(_key: str, _value: Any, _alt_override: Any) -> None:
        """
        Add _key = _value to the `defaults` dictionary, if
        _value is not None and _key is not already present.
        """
        nonlocal defaults
        _cur_value = defaults.get(_key)
        if _cur_value in (None, _alt_override) and _value is not None:
            if isinstance(_value, str):
                _value = _value.strip('\n\r')
            defaults[_key] = _value

    _check_module_default(key.name, Path(filepath).stem, DEFAULT_NAME)
    _check_module_default(key.author, module_vars.get('__author__'), DEFAULT_AUTHOR)
    _check_module_default(key.comment, module_vars.get('__doc__'), DEFAULT_COMMENT)

    return interpret_spec_file(spec, defaults=defaults, cwd=cwd)


def interpret_spec_file(
        spec_file_dict: Mapping[str, Any],
        *,
        defaults: Mapping[str, Any] = DEFAULTS,
        cwd: Optional[PathLike | Traversable] = None,
) -> ModelSpec:
    """
    Create a ModelSpec object, using the given spec file dictionary.

    Args:
        spec_file_dict: is a dict conforming to the spec file format.
        defaults: an optional dictionary of default values.
        cwd: file path for current working directory for resolving datasource filenames.
    """
    # Start a SpecDict, including a hierarchy of default values.
    if key.name in spec_file_dict.keys():
        root_name = str(spec_file_dict[key.name])
    elif defaults is not None and key.name in defaults.keys():
        root_name = str(defaults[key.name])
    else:
        root_name = ''

    spec_dict: SpecDict = _wrap_with_spec_dict(root_name, spec_file_dict, defaults, DEFAULTS)

    name: str = spec_dict.get_string(key.name, DEFAULT_NAME)
    author: str = spec_dict.get_string(key.author, DEFAULT_AUTHOR)
    comment: str = spec_dict.get_string(key.comment, DEFAULT_COMMENT)
    rng_n: int = spec_dict.get_positive_int(key.rng_n, DEFAULT_RNG_N)
    roots: List[str] = spec_dict.get_string_list(key.roots, [])

    parameters: Dict[str, State] = _interpret_parameters(spec_dict)
    datasources: Dict[str, DatasourceSpec] = _interpret_datasources(spec_dict, interpret_roots(roots, cwd))
    rvs: Dict[str, ModelRVSpec] = _interpret_rvs(spec_dict, datasources)
    crosstabs: Dict[str, ModelCrosstabSpec] = _interpret_crosstabs(spec_dict, datasources)
    entities: Dict[str, ModelEntitySpec] = _interpret_entities(spec_dict, datasources)

    model_spec = ModelSpec(
        name=name,
        author=author,
        comment=comment,
        roots=roots,
        rng_n=rng_n,
        datasources=datasources,
        rvs=rvs,
        crosstabs=crosstabs,
        entities=entities,
        parameters=parameters,
    )

    return model_spec


def _interpret_parameters(spec_dict: SpecDict) -> Dict[str, State]:
    result: Dict[str, State] = {}
    parameters_spec: Optional[SpecDict] = spec_dict.get_dict_optional(key.parameters, dont_inherit=SECTIONS)
    if parameters_spec is not None:
        for field_id, value in parameters_spec.items():
            parameters_spec.check_is_id(field_id)
            parameters_spec.check_is_state(value)
            result[field_id] = value
    return result


def _interpret_entities(spec_dict: SpecDict, datasources: Dict[str, DatasourceSpec]) -> Dict[str, ModelEntitySpec]:
    if key.entities not in spec_dict.keys():
        # No 'entities', make a default one with all RVS
        all_rvs: Set[str] = _all_datasource_rvs(datasources.values())
        return {
            DEFAULT_ENTITY_NAME: ModelEntitySpec(
                fields={
                    rv_name: ModelFieldSpecSample(rv_name=rv_name)
                    for rv_name in all_rvs
                }
            )
        }
    entities_dict: SpecDict = spec_dict.get_dict(key.entities, dont_inherit=SECTIONS)
    result: Dict[str, ModelEntitySpec] = {
        entities_dict.check_is_id(entity_name): _interpret_entity(entities_dict.get_dict(entity_name))
        for entity_name in entities_dict.keys()
    }
    return result


def _interpret_entity(entity_dict: SpecDict) -> ModelEntitySpec:
    id_field_name: str = entity_dict.get_string(key.id_field, default=DEFAULT_ID_FIELD)
    count_field_name: str = entity_dict.get_string(key.count_field, default=DEFAULT_COUNT_FIELD)
    foreign_field_name: Optional[str] = entity_dict.get_string_optional(key.foreign_field)
    parent: Optional[str] = entity_dict.get_string_optional(key.parent)

    if parent is None:
        foreign_field_name = None  # just ignore it
    elif foreign_field_name is None:
        foreign_field_name = DEFAULT_FOREIGN_ID_FIELD_FORMAT.format(entity=parent)

    reserved_fields: Set[str] = {id_field_name, count_field_name}
    if foreign_field_name is not None:
        reserved_fields.add(foreign_field_name)

    if len({count_field_name, id_field_name, foreign_field_name}) != 3:
        raise entity_dict.error(f'count, id and foreign field names must be different')

    fields: Dict[str, ModelFieldSpec] = _interpret_fields(entity_dict, reserved_fields)

    cardinality: List[ConditionSpec] = _interpret_cardinality(entity_dict, count_field_name)

    return ModelEntitySpec(
        id_field_name=id_field_name,
        count_field_name=count_field_name,
        foreign_field_name=foreign_field_name,
        fields=fields,
        cardinality=cardinality,
        parent=parent,
    )


def _interpret_cardinality(entity_dict: SpecDict, count_field_name: str) -> List[ConditionSpec]:
    cardinality_spec = entity_dict.find(key.cardinality)
    return _interpret_cardinality_r(entity_dict, cardinality_spec, count_field_name)


def _interpret_cardinality_r(parent: SpecDict, cardinality_spec: Any, count_field_name: str) -> List[ConditionSpec]:
    # Default cardinality
    if cardinality_spec is None:
        return []

    # Fixed limit on count
    if isinstance(cardinality_spec, Integral):
        count_limit_fixed: int = parent.check_is_positive_int(int(cardinality_spec))
        return [ConditionSpecFixedLimit(
            field=count_field_name,
            limit=count_limit_fixed,
        )]

    # Variable limit on count
    if isinstance(cardinality_spec, str):
        limit_field: str = cardinality_spec
        return [ConditionSpecVariableLimit(
            field=count_field_name,
            limit_field=limit_field,
        )]

    # Collection provided
    if isinstance(cardinality_spec, (list, tuple, set, frozenset)):
        result: List[ConditionSpec] = []
        for sub_spec in cardinality_spec:
            others: List[ConditionSpec] = _interpret_cardinality_r(parent, sub_spec, count_field_name)
            result.extend(others)
        return result

    # A test field
    if isinstance(cardinality_spec, Mapping):
        field = cardinality_spec.get(key.field)
        limit = cardinality_spec.get(key.limit)
        state = cardinality_spec.get(key.state)

        if not isinstance(field, str):
            raise parent.error(f'expected a test field for: {key.field}', repr(field))

        if (limit is None) and (state is None):
            raise parent.error(f'must specify either {key.limit} or {key.state}')

        if limit is not None:
            if state is not None:
                raise parent.error(f'must specify either {key.limit} or {key.state}, not both')

            if isinstance(limit, Integral):
                count_limit_fixed: int = parent.check_is_positive_int(int(limit))
                return [ConditionSpecFixedLimit(
                    field=count_field_name,
                    limit=count_limit_fixed,
                )]

            elif isinstance(limit, str):
                limit_field: str = limit
                return [ConditionSpecVariableLimit(
                    field=count_field_name,
                    limit_field=limit_field,
                )]

            raise parent.error(f'{key.limit} not understood', repr(limit))

        elif state is not None:

            if isinstance(state, str):
                return [ConditionSpecStates(
                    field=field,
                    states=[state],
                )]

            elif isinstance(state, (list, tuple, set, frozenset)):
                states = clean_states(set(state))
                return [ConditionSpecStates(
                    field=field,
                    states=states,
                )]

            raise parent.error(f'{key.state} not understood', repr(state))

    # Nothing valid found
    raise parent.error('cannot interpret cardinality', repr(cardinality_spec))


def _interpret_fields(entity_dict: SpecDict, reserved_fields: Set[str]) -> Dict[str, ModelFieldSpec]:
    # Get the sampled random variable fields.
    rvs: List[str] = entity_dict.get_string_list(key.rvs, [])
    fields: Dict[str, ModelFieldSpec] = {
        rv_name: ModelFieldSpecSample(rv_name=rv_name)
        for rv_name in rvs
    }

    # Get other specified fields.
    fields_dict: Optional[SpecDict] = entity_dict.get_dict_optional(key.fields)
    if fields_dict is not None:
        duplicated_fields: Set[str] = set(fields.keys()).intersection(fields_dict.keys())
        if len(duplicated_fields) > 0:
            raise fields_dict.error('fields overlap with sampled random variables', repr(sorted(duplicated_fields)))
        for field_name in fields_dict.keys():
            fields[field_name] = _interpret_field(fields_dict.get_dict(field_name))

    clash_fields: Set[str] = reserved_fields.intersection(fields.keys())
    if len(clash_fields) > 0:
        raise fields_dict.error('field names overlap with special fields', repr(sorted(clash_fields)))

    return fields


def _interpret_field(field_dict: SpecDict) -> ModelFieldSpec:
    is_sum: bool = key.sum in field_dict.keys()
    is_function: bool = key.function in field_dict.keys()
    is_sample: bool = key.sample in field_dict.keys()
    is_value: bool = list(field_dict.keys()) == [key.value]

    if sum([is_sum, is_function, is_sample, is_value]) != 1:
        raise field_dict.error(f'a field must be either a value, sum, function or sample',
                               repr(sorted(field_dict.keys())))

    if is_value:
        value: State = clean_state(field_dict.get(key.value, 0))
        return ModelFieldSpecSum(
            initial_value=value,
            sum=[],
            offset=0,
        )

    elif is_sum:
        initial_value: State = clean_state(field_dict.get(key.value, 0))

        sum_list: List[Any]
        sum_spec = field_dict.get(key.sum)
        if isinstance(sum_spec, (list, tuple, set, frozenset)):
            sum_list = list(sum_spec)
        elif isinstance(sum_spec, (Integral, Real)):
            sum_list = [sum_spec]
        else:
            raise field_dict.error(f'cannot interpret sum', repr(sum_spec))

        sum_rvs: List[str] = []
        offset: State = 0
        for sum_item in sum_list:
            if isinstance(sum_item, str):
                sum_rvs.append(sum_item)
            elif isinstance(sum_item, (Integral, Real)):
                offset += clean_state(sum_item)
            else:
                raise field_dict.error(f'cannot sum item', repr(sum_item))

        return ModelFieldSpecSum(
            initial_value=initial_value,
            sum=sum_rvs,
            offset=offset,
        )
    elif is_sample:
        rv_name = field_dict.get_string(key.sample)
        return ModelFieldSpecSample(rv_name=rv_name)
    elif is_function:
        initial_value: State = clean_state(field_dict.get(key.value))
        function: str = field_dict.get_string(key.function)
        inputs: List[str] = field_dict.get_string_list(key.input)
        return ModelFieldSpecFunction(
            initial_value=initial_value,
            function=function,
            inputs=inputs,
        )
    else:
        raise NotReached()


def _interpret_crosstabs(spec_dict: SpecDict, datasources: Dict[str, DatasourceSpec]) -> Dict[str, ModelCrosstabSpec]:
    all_datasource_rvs: Set[str] = _all_datasource_rvs(datasources.values())

    crosstabs_spec = spec_dict.get(key.crosstabs)

    iterator: Iterable[Tuple[str, Any]]
    if crosstabs_spec is None:
        iterator = ()
    elif isinstance(crosstabs_spec, Mapping):
        iterator = crosstabs_spec.items()
    elif isinstance(crosstabs_spec, (list, tuple, set, frozenset)):
        specs = list(crosstabs_spec)
        names = []
        for i, spec in enumerate(specs):
            name = _make_crosstab_name(i, spec, names)
            names.append(name)
        iterator = zip(names, specs)
    else:
        raise spec_dict.error(f'cannot interpret section: {key.crosstabs}', repr(crosstabs_spec))

    # Make the specified crosstabs
    crosstabs: Dict[str, ModelCrosstabSpec] = {
        crosstab_name: _interpret_crosstab(spec_dict, crosstab_name, crosstab_spec, datasources, all_datasource_rvs)
        for crosstab_name, crosstab_spec in iterator
    }

    # Add a singleton cross-table for any random variable not already in a cross-table
    crosstab: ModelCrosstabSpec
    covered_rvs: Set[str] = {
        rv_name
        for crosstab in crosstabs.values()
        for rv_name in crosstab.rvs
    }
    singleton_rvs: Set[str] = all_datasource_rvs.difference(covered_rvs)
    epsilon: float = spec_dict.get_positive(key.epsilon)
    min_cell_size: float = spec_dict.get_non_neg(key.min_cell_size)
    max_add_rows: int = spec_dict.get_positive_int(key.max_add_rows)
    for rv_name in singleton_rvs:
        crosstab_name = _make_unique_id('_' + rv_name, crosstabs.keys())
        rvs = [rv_name]
        crosstabs[crosstab_name] = ModelCrosstabSpec(
            rvs=rvs,
            datasource=_find_datasource(spec_dict, rvs, datasources),
            epsilon=epsilon,
            min_cell_size=min_cell_size,
            max_add_rows=max_add_rows,
        )

    return crosstabs


def _interpret_crosstab(
        parent: SpecDict,
        crosstab_name: str,
        crosstab_spec: Any,
        datasources: Dict[str, DatasourceSpec],
        all_datasource_rvs: Set[str],
) -> ModelCrosstabSpec:
    datasource: str
    rvs: List[str]

    if isinstance(crosstab_spec, str):
        # The crosstab_spec is a string - assume it is a datasource id
        datasource = crosstab_spec
        crosstab_spec_dict = parent.sub_dict(crosstab_name)
        datasource_spec: Optional[DatasourceSpec] = datasources.get(datasource)
        if datasource_spec is None:
            raise crosstab_spec_dict.error('cannot find datasource', repr(datasource))
        rvs = datasource_spec.rvs

    elif isinstance(crosstab_spec, Mapping):
        # The crosstab_spec is a dict
        crosstab_spec_dict = parent.sub_dict(crosstab_name, crosstab_spec)

        rvs = crosstab_spec_dict.get_string_list(key.rvs)
        if key.datasource in crosstab_spec_dict.keys():
            datasource = crosstab_spec_dict.get_string(key.datasource)
        else:
            datasource = _find_datasource(crosstab_spec_dict, rvs, datasources)

    elif isinstance(crosstab_spec, (list, tuple, set, frozenset)):
        # The crosstab_spec is not a string or dict - assume it is a list of random variables
        rvs = list(crosstab_spec)
        for rv in rvs:
            if rv not in all_datasource_rvs:
                raise parent.error('random variable not in a datasource', repr(rv))

        crosstab_spec_dict = parent.sub_dict(crosstab_name)
        datasource = _find_datasource(crosstab_spec_dict, rvs, datasources)

    else:
        crosstab_spec_dict = parent.sub_dict(crosstab_name)
        raise crosstab_spec_dict.error('cannot interpret section', repr(crosstab_spec))

    epsilon: float = crosstab_spec_dict.get_positive(key.epsilon)
    min_cell_size: float = crosstab_spec_dict.get_non_neg(key.min_cell_size)
    max_add_rows: int = crosstab_spec_dict.get_positive_int(key.max_add_rows)

    return ModelCrosstabSpec(
        rvs=rvs,
        datasource=datasource,
        epsilon=epsilon,
        min_cell_size=min_cell_size,
        max_add_rows=max_add_rows,
    )


def _find_datasource(parent: SpecDict, rvs: List[str], datasources: Dict[str, DatasourceSpec]) -> str:
    rvs_set: Set[str] = set(rvs)
    candidates: List[str] = [
        datasource
        for datasource, datasource_spec in datasources.items()
        if rvs_set.issubset(datasource_spec.rvs)
    ]
    if len(candidates) == 0:
        raise parent.error('no datasource found for random variables', repr(rvs))
    if len(candidates) == 1:
        return candidates[0]

    # Prefer datasources where the fewest of given `rvs` are non-distribution.
    # Then break ties with the smallest datasource rvs.
    max_datasource_rvs_count: int = max(len(_datasource_spec.rvs) for _datasource_spec in datasources.values())

    def _sort_key(_datasource: str) -> float:
        nonlocal datasources, rvs_set, max_datasource_rvs_count

        _datasource_spec: DatasourceSpec = datasources[_datasource]
        non_distribution_count: int = len(rvs_set.intersection(_datasource_spec.non_distribution_rvs))
        datasource_rvs_count: int = len(_datasource_spec.rvs)

        return non_distribution_count + datasource_rvs_count / max_datasource_rvs_count

    return sorted(candidates, key=_sort_key)[0]


def _make_crosstab_name(i, spec, names) -> str:
    """
    Come up with a name for the ith cross tab, given its spec,
    and a collection of names already used.
    """
    if isinstance(spec, str):
        name = spec
    elif isinstance(spec, dict) and key.rvs in spec.keys():
        rvs = spec[key.rvs]
        if isinstance(rvs, str):
            name = rvs
        elif isinstance(rvs, Iterable):
            name = ','.join(str(rv) for rv in rvs)
        else:
            name = str(i)
    elif isinstance(spec, Iterable):
        name = ','.join(str(rv) for rv in spec)
    else:
        name = str(i)
    name = _make_unique_id(name, names)
    return name


def _make_unique_id(base_name: str, seen_names) -> str:
    """
    Make a name, based on base_name but not in seen_names.
    """
    if base_name not in seen_names:
        return base_name
    for i in count(start=1):
        check_name = f'{base_name}({i})'
        if check_name not in seen_names:
            return check_name
    raise NotReached()


def _all_datasource_rvs(datasources: Iterable[DatasourceSpec]) -> Set[str]:
    return {
        rv
        for datasource_spec in datasources
        for rv in datasource_spec.rvs
    }


def _interpret_rvs(spec_dict: SpecDict, datasources: Dict[str, DatasourceSpec]) -> Dict[str, ModelRVSpec]:
    all_datasource_rvs: Set[str] = _all_datasource_rvs(datasources.values())

    rvs_dict: SpecDict
    if key.rvs in spec_dict.keys():
        rvs_dict = spec_dict.get_dict(key.rvs, dont_inherit=SECTIONS)
    else:
        # If there is no `rvs_dict` then create a dummy one with all RVs seen in the datasources.
        dummy_rvs: Dict[str, Any] = {rv: {} for rv in all_datasource_rvs}
        rvs_dict = spec_dict.sub_dict(key.rvs, dummy_rvs, dont_inherit=SECTIONS)

    result: Dict[str, ModelRVSpec] = {}
    for rv_name in rvs_dict.keys():
        rvs_dict.check_is_id(rv_name)
        rv_dict = rvs_dict.get_dict(rv_name)
        result[rv_name] = _interpret_rv(rv_name, rv_dict, all_datasource_rvs)
    return result


def _interpret_rv(rv_name: str, rv_dict: SpecDict, all_datasource_rvs: Set[str]) -> ModelRVSpec:
    ensure_none: bool = rv_dict.get_bool(key.ensure_none, False)
    if rv_name not in all_datasource_rvs:
        raise rv_dict.error(f'random variable {rv_name!r} has no datasource')

    states: int | List[State] | Literal['infer_distinct', 'infer_range', 'infer_max']

    states_spec = rv_dict.check_exists(key.states)
    if states_spec == key.infer_distinct:
        states = 'infer_distinct'
    elif states_spec == key.infer_range:
        states = 'infer_range'
    elif states_spec == key.infer_max:
        states = 'infer_max'
    elif isinstance(states_spec, int):
        states = states_spec
    elif isinstance(states_spec, Mapping):
        range_dict: SpecDict = rv_dict.sub_dict(key.states, states_spec)
        states = _get_state_from_range_dict(range_dict)
    elif isinstance(states_spec, (list, tuple, set, frozenset)):
        states = list(states_spec)
    else:
        raise rv_dict.error(f'invalid {key.states} value', repr(states_spec))

    return ModelRVSpec(states=states, ensure_none=ensure_none)


def _interpret_datasources(spec_dict: SpecDict, roots: List[Path]) -> Dict[str, DatasourceSpec]:
    datasources_dict: SpecDict = spec_dict.get_dict(key.datasources, dont_inherit=SECTIONS)
    result: Dict[str, DatasourceSpec] = {}
    for datasource_name in datasources_dict.keys():
        datasources_dict.check_is_id(datasource_name)
        datasource_dict = datasources_dict.get_dict(datasource_name)
        result[datasource_name] = _interpret_datasource(datasource_name, datasource_dict, roots)
    return result


def _interpret_datasource(datasource_name: str, datasource_dict: SpecDict, roots: List[Path]) -> DatasourceSpec:
    # If one of these keys is present, then the others must not be.
    datasource_dict.check_mutually_exclusive(
        key.function,
        key.location,
        key.table,
    )

    # Infer a default data format value, in case no value for key data_format
    # is provided by the datasource spec.
    default_format = None
    if key.function in datasource_dict.keys():
        # It looks like a functional datasource
        default_format = key.function
    elif key.location in datasource_dict.keys():
        # It looks like a file datasource.
        # Infer default data format from the filename extension.
        location = datasource_dict.get_string_optional(key.location)
        parts = location.split('.')
        if len(parts) > 1:
            default_format = parts[-1].lower()
            default_format = {
                # Explicit file extension mapping (where not identity)
                'pk': key.pickle,
                'pkl': key.pickle,
                key.function: None,
            }.get(default_format, default_format)

    format_str = datasource_dict.get_string(key.data_format, default=default_format).lower()
    if format_str == key.csv:
        return _make_datasource_sv(datasource_dict, roots, ',')
    elif format_str == key.tsv:
        return _make_datasource_sv(datasource_dict, roots, '\t')
    elif format_str == key.table_builder:
        return _make_datasource_table_builder(datasource_dict, roots)
    elif format_str == key.pickle:
        return _make_datasource_pickle(datasource_dict, datasource_name, roots)
    elif format_str == key.parquet:
        return _make_datasource_parquet(datasource_dict, datasource_name, roots)
    elif format_str == key.feather:
        return _make_datasource_feather(datasource_dict, datasource_name, roots)
    elif format_str == key.function:
        return _make_datasource_function(datasource_dict, datasource_name)
    elif format_str == key.odbc:
        return _make_datasource_dbms(datasource_dict, 'odbc')
    elif format_str == key.postgres:
        return _make_datasource_dbms(datasource_dict, 'postgres')
    elif format_str is None:
        raise datasource_dict.error(
            f'missing {key.data_format}',
            'cannot infer a default value'
        )
    else:
        raise datasource_dict.error(
            'invalid value',
            f'{key.data_format}: {format_str!r} not understood'
        )


def _make_datasource_sv(
        datasource_dict: SpecDict,
        roots: List[Path],
        expected_sep: str,
) -> DatasourceSpec:
    """
    Construct a DatasourceSpec with a DatasetSpecCsv dataset.
    """
    sep = datasource_dict.get_string_optional(key.sep)
    sep = expected_sep if sep is None else sep

    header = datasource_dict.get_bool(key.header, default=True)
    skip_blank_lines = datasource_dict.get_bool(key.skip_blank_lines, default=True)
    skip_initial_space = True
    weight = datasource_dict.find(key.weight)
    input_spec: TextInputSpec = _get_text_input_spec(datasource_dict)
    rv_map: Optional[Dict[str, ColumnSpec]] = _get_rv_map(datasource_dict)
    rv_define: Dict[str, ColumnDefinitionSpec] = _get_rv_define(datasource_dict)

    dataset_spec = DatasetSpecCsv(
        weight=weight,
        rv_map=rv_map,
        rv_define=rv_define,
        input=input_spec,
        sep=sep,
        header=header,
        skip_blank_lines=skip_blank_lines,
        skip_initial_space=skip_initial_space,
    )

    rvs_list: List[str] = _get_rvs_list(datasource_dict, dataset_spec, rv_map, rv_define, roots)

    sensitivity = datasource_dict.get_non_neg(key.sensitivity)
    non_distribution_rvs = datasource_dict.get_string_list(key.condition, [])

    return DatasourceSpec(
        sensitivity=sensitivity,
        rvs=rvs_list,
        dataset_spec=dataset_spec,
        non_distribution_rvs=non_distribution_rvs,
    )


def _make_datasource_table_builder(
        datasource_dict: SpecDict,
        roots: List[Path],
) -> DatasourceSpec:
    """
    Construct a DatasourceSpec with a DatasetSpecTableBuilder dataset.
    """
    input_spec: TextInputSpec = _get_text_input_spec(datasource_dict)
    rv_map: Optional[Dict[str, ColumnSpec]] = _get_rv_map(datasource_dict)
    rv_define: Dict[str, ColumnDefinitionSpec] = _get_rv_define(datasource_dict)

    dataset_spec = DatasetSpecTableBuilder(
        rv_map=rv_map,
        rv_define=rv_define,
        input=input_spec,
    )

    rvs_list: List[str] = _get_rvs_list(datasource_dict, dataset_spec, rv_map, rv_define, roots)

    sensitivity = datasource_dict.get_non_neg(key.sensitivity)
    non_distribution_rvs = datasource_dict.get_string_list(key.condition, [])

    return DatasourceSpec(
        sensitivity=sensitivity,
        rvs=rvs_list,
        dataset_spec=dataset_spec,
        non_distribution_rvs=non_distribution_rvs,
    )


def _make_datasource_pickle(
        datasource_dict: SpecDict,
        datasource_name: str,
        roots: List[Path],
) -> DatasourceSpec:
    """
    Construct a DatasourceSpec with a DatasetSpecPickle dataset.
    """
    weight = datasource_dict.find(key.weight)
    location: str = datasource_dict.get_string(key.location, f'{datasource_name}.pkl')
    rv_map: Optional[Dict[str, ColumnSpec]] = _get_rv_map(datasource_dict)
    rv_define: Dict[str, ColumnDefinitionSpec] = _get_rv_define(datasource_dict)

    dataset_spec = DatasetSpecPickle(
        weight=weight,
        rv_map=rv_map,
        rv_define=rv_define,
        location=location,
    )

    rvs_list: List[str] = _get_rvs_list(datasource_dict, dataset_spec, rv_map, rv_define, roots)

    sensitivity = datasource_dict.get_non_neg(key.sensitivity)
    non_distribution_rvs = datasource_dict.get_string_list(key.condition, [])

    return DatasourceSpec(
        sensitivity=sensitivity,
        rvs=rvs_list,
        dataset_spec=dataset_spec,
        non_distribution_rvs=non_distribution_rvs,
    )


def _make_datasource_parquet(
        datasource_dict: SpecDict,
        datasource_name: str,
        roots: List[Path],
) -> DatasourceSpec:
    """
    Construct a DatasourceSpec with a DatasetSpecParquet dataset.
    """
    weight = datasource_dict.find(key.weight)
    location: str = datasource_dict.get_string(key.location, f'{datasource_name}.parquet')
    rv_map: Optional[Dict[str, ColumnSpec]] = _get_rv_map(datasource_dict)
    rv_define: Dict[str, ColumnDefinitionSpec] = _get_rv_define(datasource_dict)

    dataset_spec = DatasetSpecParquet(
        weight=weight,
        rv_map=rv_map,
        rv_define=rv_define,
        location=location,
    )

    rvs_list: List[str] = _get_rvs_list(datasource_dict, dataset_spec, rv_map, rv_define, roots)

    sensitivity = datasource_dict.get_non_neg(key.sensitivity)
    non_distribution_rvs = datasource_dict.get_string_list(key.condition, [])

    return DatasourceSpec(
        sensitivity=sensitivity,
        rvs=rvs_list,
        dataset_spec=dataset_spec,
        non_distribution_rvs=non_distribution_rvs,
    )


def _make_datasource_feather(
        datasource_dict: SpecDict,
        datasource_name: str,
        roots: List[Path],
) -> DatasourceSpec:
    """
    Construct a DatasourceSpec with a DatasetSpecFeather dataset.
    """
    weight = datasource_dict.find(key.weight)
    location: str = datasource_dict.get_string(key.location, f'{datasource_name}.feather')
    rv_map: Optional[Dict[str, ColumnSpec]] = _get_rv_map(datasource_dict)
    rv_define: Dict[str, ColumnDefinitionSpec] = _get_rv_define(datasource_dict)

    dataset_spec = DatasetSpecFeather(
        weight=weight,
        rv_map=rv_map,
        rv_define=rv_define,
        location=location,
    )

    rvs_list: List[str] = _get_rvs_list(datasource_dict, dataset_spec, rv_map, rv_define, roots)

    sensitivity = datasource_dict.get_non_neg(key.sensitivity)
    non_distribution_rvs = datasource_dict.get_string_list(key.condition, [])

    return DatasourceSpec(
        sensitivity=sensitivity,
        rvs=rvs_list,
        dataset_spec=dataset_spec,
        non_distribution_rvs=non_distribution_rvs,
    )


def _make_datasource_function(
        datasource_dict: SpecDict,
        datasource_name: str,
) -> DatasourceSpec:
    """
    Construct a DatasourceSpec with a DatasetSpecFunction dataset.
    """
    function: str = datasource_dict.get_string(key.function)
    input_dict: Dict[str, int | List[State]] = _get_rvs_dict(datasource_dict.get_dict(key.input))
    output_rv: str = datasource_dict.get_string(key.output, datasource_name)

    dataset_spec = DatasetSpecFunction(
        rvs=input_dict,
        output_rv=output_rv,
        function=function,
    )

    rvs_list: List[str] = list(input_dict.keys()) + [output_rv]

    sensitivity = datasource_dict.get_non_neg(key.sensitivity)
    non_distribution_rvs = datasource_dict.get_string_list(key.condition, [])

    return DatasourceSpec(
        sensitivity=sensitivity,
        rvs=rvs_list,
        dataset_spec=dataset_spec,
        non_distribution_rvs=non_distribution_rvs,
    )


def _make_datasource_dbms(
        datasource_dict: SpecDict,
        api: Literal['odbc', 'postgres']
) -> DatasourceSpec:
    """
    Construct a DatasourceSpec with a DatasetSpecDBMS dataset.
    """
    schema_name: Optional[str] = datasource_dict.get_string_optional(key.schema)
    table_name: str = datasource_dict.get_string(key.table)

    rvs: Optional[List[str]] = None
    if key.rvs in datasource_dict.keys():
        rvs = datasource_dict.get_string_list(key.rvs)

    connection: Optional[str | Dict[str, Optional[str]]] = None
    if key.connection in datasource_dict.keys():
        connection_dict: SpecDict = datasource_dict.get_dict(key.connection)
        connection = {}
        for conx_key, conx_value in connection_dict.items():
            if not isinstance(conx_key, str):
                raise connection_dict.error('invalid connection key', repr(conx_key))
            if not isinstance(conx_value, (str, int, NoneType)):
                raise connection_dict.error('invalid connection value', repr(conx_value))
            connection[conx_key] = conx_value

    dataset_spec = DatasetSpecDBMS(
        type=api,
        schema_name=schema_name,
        table_name=table_name,
        rvs=rvs,
        connection=connection,
    )

    if rvs is None:
        rvs = _get_rvs_from_dataset(datasource_dict, dataset_spec, ())

    sensitivity = datasource_dict.get_non_neg(key.sensitivity)
    non_distribution_rvs = datasource_dict.get_string_list(key.condition, [])

    return DatasourceSpec(
        sensitivity=sensitivity,
        rvs=rvs,
        dataset_spec=dataset_spec,
        non_distribution_rvs=non_distribution_rvs,
    )


def _get_rvs_dict(rvs_dict: SpecDict) -> Dict[str, int | List[State]]:
    result: Dict[str, int | List[State]] = {}
    for rv_name, states_def in rvs_dict.items():
        if isinstance(states_def, int):
            result[rv_name] = states_def
        elif isinstance(states_def, (list, tuple, set)):
            states = list(states_def)
            for state in states:
                if not isinstance(state, (int, str, bool, float, NoneType)):
                    raise rvs_dict.error('unexpected random variable state', repr(state))
            result[rv_name] = states
        elif isinstance(states_def, range):
            result[rv_name] = list(states_def)
        elif isinstance(states_def, dict):
            range_dict: SpecDict = rvs_dict.get_dict(rv_name)
            result[rv_name] = _get_state_from_range_dict(range_dict)
        else:
            raise rvs_dict.error('uninterpretable random variable states', repr(states_def))
    return result


def _get_state_from_range_dict(range_dict: SpecDict) -> List[int]:
    if not {key.start, key.stop, key.step}.issuperset(range_dict.keys()):
        raise range_dict.error(f'only {key.start}, {key.stop}, and {key.step} permitted')
    start: int = range_dict.get_int(key.start, default=0)
    stop: int = range_dict.get_int(key.stop)
    step: int = range_dict.get_int(key.step, default=1)
    return list(range(start, stop, step))


def _get_rvs_list(
        datasource_dict: SpecDict,
        dataset_spec: DatasetSpec,
        rv_map: Optional[Dict[str, ColumnSpec]],
        rv_define: Dict[str, ColumnDefinitionSpec],
        roots: Sequence[Path],
) -> List[str]:
    if rv_map is None:
        # Get random variables for the dataset directly
        return _get_rvs_from_dataset(datasource_dict, dataset_spec, roots)
    else:
        return list(set(rv_map.keys()).union(rv_define.keys()))


def _get_rvs_from_dataset(
        datasource_dict: SpecDict,
        dataset_spec: DatasetSpec,
        roots: Sequence[Path],
) -> List[str]:
    with warnings.catch_warnings(record=True) as the_warnings:
        warnings.simplefilter('always')
        dataset: Dataset = dataset_spec.dataset(roots)
    for w in the_warnings:
        datasource_dict.warn(
            f'datasource {datasource_dict.dict_id!r} load warning',
            w.message
        )
    return list(dataset.rvs)


def _get_rv_map(datasource_dict: SpecDict) -> Optional[Dict[str, ColumnSpec]]:
    rv_map = datasource_dict.get(key.rvs)

    if rv_map is None:
        return None

    if isinstance(rv_map, (list, tuple, set)):
        for rv in rv_map:
            if not isinstance(rv, str):
                raise datasource_dict.error(f'rv {rv!r} is not a string')
        return {rv: rv for rv in rv_map}

    if isinstance(rv_map, dict):
        result: Dict[str, ColumnSpec] = {}
        for rv_name, definition in rv_map.items():
            if not isinstance(rv_name, str):
                raise datasource_dict.error(f'rv {rv_name!r} is not a string')
            if isinstance(definition, (int, str)):
                result[rv_name] = definition
            else:
                raise datasource_dict.error('rvs entry not understood', repr(rv_name))
        return result

    raise datasource_dict.error('rvs not understood')


def _get_rv_define(datasource_dict: SpecDict) -> Dict[str, ColumnDefinitionSpec]:
    rv_define = datasource_dict.get_dict_optional(key.define)

    if rv_define is None:
        return {}
    else:
        return {
            rv_name: _get_column_definition_spec(rv_define.get_dict(rv_name))
            for rv_name in rv_define.keys()
        }


def _get_column_definition_spec(column_definition_dict: SpecDict) -> ColumnDefinitionSpec:
    """
    Args:
        column_definition_dict: a spec file definition of a
            ColumnDefinitionSpecFunction or ColumnDefinitionSpecGroup

    Returns:
        a ColumnDefinitionSpecFunction or ColumnDefinitionSpecGroup
    """
    is_function: bool = key.function in column_definition_dict.keys()
    is_grouping: bool = key.grouping in column_definition_dict.keys()
    if is_function:
        if is_grouping:
            raise column_definition_dict.error(f'cannot define both {key.function} and {key.grouping}')
        return _get_column_definition_spec_function(column_definition_dict)
    elif is_grouping:
        return _get_column_definition_spec_group(column_definition_dict)
    else:
        raise column_definition_dict.error(f'must define {key.function} or {key.grouping}')


def _get_column_definition_spec_function(column_definition_dict: SpecDict) -> ColumnDefinitionSpecFunction:
    inputs: List[str] = column_definition_dict.get_string_list(key.input)
    function: str = column_definition_dict.get_string(key.function)
    delete_input: bool = column_definition_dict.get_bool(key.delete_input, False)
    return ColumnDefinitionSpecFunction(
        inputs=inputs,
        function=function,
        delete_input=delete_input,
    )


def _get_column_definition_spec_group(column_definition_dict: SpecDict) -> ColumnDefinitionSpecGroup:
    group_type: Literal['group_cut', 'group_qcut', 'group_normalise']
    grouping: str = column_definition_dict.check_exists(key.grouping)
    if grouping in ('group_cut', 'group_qcut', 'group_normalise'):
        group_type = grouping
    else:
        raise column_definition_dict.error(f'grouping method not understood', repr(key.grouping))

    inputs: List[str] = column_definition_dict.get_string_list(key.input)
    size: int = column_definition_dict.get_positive_int(key.size)
    delete_input: bool = column_definition_dict.get_bool(key.delete_input, False)

    return ColumnDefinitionSpecGroup(
        type=group_type,
        inputs=inputs,
        size=size,
        delete_input=delete_input,
    )


def _get_text_input_spec(datasource_dict: SpecDict) -> TextInputSpec:
    """
    Return either a StringIO or a file Path that can be used as a source
    (e.g., to construct a Pandas dataframe).

    This method will look for a 'location' or an 'inline' key.
    """
    location = datasource_dict.get_string_optional(key.location)
    inline = datasource_dict.get_string_optional(key.inline)
    if location is not None:
        if inline is not None:
            raise datasource_dict.error('conflicting keys', f'cannot have both {key.location} and {key.inline}')
        return TextInputSpecLocation(location=location)
    elif inline is not None:
        return TextInputSpecInline(inline=strip_lines(inline))
    else:
        raise datasource_dict.error('missing key', f'must have either {key.location} or {key.inline}')


def _wrap_with_spec_dict(root_name: str, update: Mapping[str, Any], *defaults: Mapping[str, Any]) -> SpecDict:
    """
    Construct a SpecDict with local items from 'update', and default
    values from the additional parents.

    Args:
        root_name: name of the root section.
        update: A Mappings to initialise the new SpecDict.
        defaults: zero or more Mappings of default values.
    """
    defaults = [d for d in defaults if d is not None]
    return SpecDict(root_name, root_name, *defaults, update=update)
