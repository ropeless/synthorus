import warnings
from itertools import count
from numbers import Integral, Real
from types import MappingProxyType, NoneType
from typing import Optional, Dict, Any, Mapping, List, Literal, Sequence, Set, Iterable, Tuple, Collection

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
    ModelFieldSpec, ModelFieldSpecFunction, ModelFieldSpecSum, ForeignKeyField
from synthorus.model.noiser_spec import NoiserSpecLaplace, NoiserSpec, NoiserSpecBasicLaplace, NoiserSpecNaiveLaplace, \
    NoiserSpecDecompositionLaplace
from synthorus.simulator.condition_spec import ConditionSpec, ConditionSpecFixedLimit, ConditionSpecVariableLimit, \
    ConditionSpecStates
from synthorus.spec_file.spec_dict import SpecDict
from synthorus.utils import py_loader
from synthorus.utils.clean_state import clean_state, clean_states
from synthorus.utils.file_extras import DataPathLike, stem, DataPath
from synthorus.utils.iter_extras import duplicates
from synthorus.utils.string_extras import strip_lines

_SpecList = (list, tuple, set, frozenset)
_SpecIterable = (list, tuple, set, frozenset, range)

# Application level default values for any spec file.
DEFAULTS: Mapping[str, Any] = MappingProxyType({
    key.name: DEFAULT_NAME,
    key.author: DEFAULT_AUTHOR,
    key.comment: DEFAULT_COMMENT,
    key.rng_n: DEFAULT_RNG_N,
    key.sensitivity: DEFAULT_SENSITIVITY,
    key.epsilon: DEFAULT_EPSILON,
    key.min_cell_size: DEFAULT_MIN_CELL_SIZE,
    key.noise: key.laplace,
    key.max_add_rows: DEFAULT_MAX_ADD_ROWS,
    key.id_field: DEFAULT_ID_FIELD,
    key.count_field: DEFAULT_COUNT_FIELD,
    key.flatten_min_cell_size: DEFAULT_FLATTEN_MIN_CELL_SIZE,
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
        filepath: DataPathLike,
        *,
        variable: Optional[str] = None,
        defaults: Mapping[str, Any] = DEFAULTS,
        cwd: Optional[DataPathLike] = None,
) -> ModelSpec:
    """
    Read a spec dictionary from a file then interpret it as a ModelSpec.

    Args:
        filepath: path to file to read.
        variable: is the name of the Python variable in the file
            that references the spec dictionary. If None, then the file
            should only have one such variable.
        cwd: file path for the current working directory for resolving datasource filenames.
        defaults: an optional dictionary of default values.
    """
    module = py_loader.load(filepath)
    spec: Mapping[str, Any] = py_loader.get_object(module, object_type=dict, variable=variable)
    module_vars = vars(module)

    defaults_copy: Dict[str, Any] = dict(defaults)

    def _check_module_default(_key: str, _value: Any, _alt_override: Any) -> None:
        """
        Add _key = _value to the `defaults_copy` dictionary, if
        _value is not None and _key is not already present.
        """
        nonlocal defaults_copy
        _cur_value = defaults_copy.get(_key)
        if _cur_value in (None, _alt_override) and _value is not None:
            if isinstance(_value, str):
                _value = _value.strip('\n\r')
            defaults_copy[_key] = _value

    _check_module_default(key.name, stem(filepath), DEFAULT_NAME)
    _check_module_default(key.author, module_vars.get('__author__'), DEFAULT_AUTHOR)
    _check_module_default(key.comment, module_vars.get('__doc__'), DEFAULT_COMMENT)

    return interpret_spec_file(spec, defaults=defaults_copy, cwd=cwd)


def interpret_spec_file(
        spec_file_dict: Mapping[str, Any],
        *,
        defaults: Mapping[str, Any] = DEFAULTS,
        cwd: Optional[DataPathLike] = None,
) -> ModelSpec:
    """
    Create a ModelSpec object, using the given spec file dictionary.

    Args:
        spec_file_dict: is a dict conforming to the spec file format.
        defaults: a dictionary of default values.
        cwd: file path for the current working directory for resolving datasource filenames.
    """
    # Start a SpecDict, including a hierarchy of default values.
    if key.name in spec_file_dict.keys():
        root_name = str(spec_file_dict[key.name])
    elif key.name in defaults.keys():
        root_name = str(defaults[key.name])
    else:
        root_name = ''

    spec_dict: SpecDict = _wrap_with_spec_dict(root_name, spec_file_dict, defaults, DEFAULTS)

    name: str = spec_dict.get_string(key.name, DEFAULT_NAME)
    author: str = spec_dict.get_string(key.author, DEFAULT_AUTHOR)
    comment: str = spec_dict.get_string(key.comment, DEFAULT_COMMENT)
    rng_n: int = spec_dict.get_positive_int(key.rng_n, DEFAULT_RNG_N)
    roots: List[str] = spec_dict.get_string_list(key.roots, [])
    flatten_min_cell_size: bool = spec_dict.get_bool(key.flatten_min_cell_size, DEFAULT_FLATTEN_MIN_CELL_SIZE)

    parameters: Dict[str, State] = _interpret_parameters(spec_dict)
    datasources: Dict[str, DatasourceSpec] = _interpret_datasources(spec_dict, interpret_roots(roots, cwd))
    rvs: Dict[str, ModelRVSpec] = _interpret_rvs(spec_dict, datasources)
    crosstabs: Dict[str, ModelCrosstabSpec] = _interpret_crosstabs(spec_dict, rvs, datasources, flatten_min_cell_size)
    entities: Dict[str, ModelEntitySpec] = _interpret_entities(spec_dict, rvs)

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
            value = parameters_spec.check_is_state(value)
            result[field_id] = value
    return result


def _interpret_entities(spec_dict: SpecDict, rvs: Dict[str, ModelRVSpec]) -> Dict[str, ModelEntitySpec]:
    if key.entities not in spec_dict.keys():
        # No 'entities', make a default one with all RVS
        return {
            DEFAULT_ENTITY_NAME: ModelEntitySpec(
                fields={
                    rv_name: ModelFieldSpecSample(rv_name=rv_name)
                    for rv_name in rvs.keys()
                }
            )
        }
    entities_dict: SpecDict = spec_dict.get_dict(key.entities, dont_inherit=SECTIONS)
    result: Dict[str, ModelEntitySpec] = {
        entities_dict.check_is_id(entity_name): _interpret_entity(entities_dict.get_dict(entity_name))
        for entity_name in entities_dict.keys()
    }
    for entity_name in entities_dict.keys():
        entity_dict: SpecDict = entities_dict.get_dict(entity_name)
        model_entity_spec: ModelEntitySpec = result[entity_name]
        _interpret_foreign_keys(entity_dict, model_entity_spec, result)
    return result


def _interpret_entity(entity_dict: SpecDict) -> ModelEntitySpec:
    id_field_name: str = entity_dict.get_string(key.id_field, default=DEFAULT_ID_FIELD)
    count_field_name: str = entity_dict.get_string(key.count_field, default=DEFAULT_COUNT_FIELD)

    # id and count fields must be different to each other
    if count_field_name == id_field_name:
        raise entity_dict.error('count and id field names must be different', count_field_name)

    reserved_fields = {id_field_name, count_field_name}
    fields: Dict[str, ModelFieldSpec] = _interpret_fields(entity_dict, reserved_fields)

    cardinality: List[ConditionSpec] = _interpret_cardinality(entity_dict, count_field_name)

    return ModelEntitySpec(
        id_field_name=id_field_name,
        count_field_name=count_field_name,
        foreign_key_fields=[],  # filled in later
        fields=fields,
        cardinality=cardinality,
    )


def _interpret_foreign_keys(
        entity_dict: SpecDict,
        spec: ModelEntitySpec,
        entities: Dict[str, ModelEntitySpec],
) -> None:
    """
    Update `spec.foreign_key_fields`.
    """
    interpreted_foreign_keys: List[ForeignKeyField] = []

    foreign_keys = entity_dict.get(key.foreign_keys)
    if foreign_keys is None:
        return
    elif isinstance(foreign_keys, (str, Mapping)):
        interpreted_foreign_keys = [
            _interpret_one_foreign_key(foreign_keys, entity_dict, entities)
        ]
    elif isinstance(foreign_keys, _SpecList):
        for foreign_key in foreign_keys:
            if isinstance(foreign_key, (str, dict)):
                interpreted_foreign_keys.append(
                    _interpret_one_foreign_key(foreign_key, entity_dict, entities)
                )
            else:
                raise entity_dict.error('foreign key for entity not understood', foreign_key)
    else:
        raise entity_dict.error('foreign keys for entity not understood', foreign_keys)

    # Error checking
    dup_fields: str = ', '.join(duplicates(map(lambda x: x.foreign_key_field_name, interpreted_foreign_keys)))
    if len(dup_fields) > 0:
        raise entity_dict.error('duplicate foreign key fields', dup_fields)
    dup_entities: str = ', '.join(duplicates(map(lambda x: x.foreign_entity, interpreted_foreign_keys)))
    if len(dup_entities) > 0:
        raise entity_dict.error('duplicate foreign entities', dup_entities)
    foreign_field_name_set: Set[str] = set(map(lambda x: x.foreign_key_field_name, interpreted_foreign_keys))
    if spec.id_field_name in foreign_field_name_set:
        raise entity_dict.error('foreign field name same as id field', spec.id_field_name)
    if spec.count_field_name in foreign_field_name_set:
        raise entity_dict.error('foreign field name same as count field', spec.count_field_name)
    overlaps = ', '.join(foreign_field_name_set.intersection(spec.fields.keys()))
    if len(overlaps) > 0:
        raise entity_dict.error('field names and foreign field names overlap', overlaps)

    # Use the interpreted foreign keys and validate them
    spec.foreign_key_fields = interpreted_foreign_keys


def _interpret_one_foreign_key(
        foreign_key: str | Mapping,
        entity_dict: SpecDict,
        entities: Dict[str, ModelEntitySpec],
) -> ForeignKeyField:
    if isinstance(foreign_key, str):
        foreign_entity: str = foreign_key
        if foreign_entity not in entities.keys():
            raise entity_dict.error('foreign entity does not exist', foreign_entity)
        return ForeignKeyField(
            foreign_entity=foreign_entity,
            foreign_key_field_name=_foreign_key_field_name(foreign_entity, entities),
        )
    elif isinstance(foreign_key, Mapping):
        subdict: SpecDict = entity_dict.sub_dict(key.foreign_keys, foreign_key)
        foreign_entity: str = subdict.get_string(key.entity)
        if foreign_entity not in entities.keys():
            raise entity_dict.error('foreign entity does not exist', foreign_entity)
        foreign_key_field_name: str = subdict.get_string(
            key.field,
            default=_foreign_key_field_name(foreign_entity, entities)
        )
        return ForeignKeyField(
            foreign_entity=foreign_entity,
            foreign_key_field_name=foreign_key_field_name,
        )
    else:
        raise entity_dict.error('foreign key not understood', foreign_key)


def _foreign_key_field_name(entity_name: str, entities: Dict[str, ModelEntitySpec]) -> str:
    return f'_{entity_name}_{entities[entity_name].id_field_name}'


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
    if isinstance(cardinality_spec, _SpecList):
        result: List[ConditionSpec] = []
        for sub_spec in cardinality_spec:
            others: List[ConditionSpec] = _interpret_cardinality_r(parent, sub_spec, count_field_name)
            result.extend(others)
        return result

    # A test field
    if isinstance(cardinality_spec, Mapping):
        cardinality_dict: SpecDict = parent.sub_dict(key.cardinality, cardinality_spec)
        field = cardinality_dict.get(key.field)
        limit = cardinality_dict.get(key.limit)
        state = cardinality_dict.get(key.state)

        if not isinstance(field, str):
            raise parent.error('expected a test field', field)

        if (limit is None) and (state is None):
            raise parent.error(f'must specify either {key.limit} or {key.state}', field)

        if limit is not None:
            if state is not None:
                raise parent.error(f'must specify either {key.limit} or {key.state}, not both', field)

            if isinstance(limit, Integral):
                count_limit_fixed: int = parent.check_is_positive_int(int(limit))
                return [ConditionSpecFixedLimit(
                    field=field,
                    limit=count_limit_fixed,
                )]

            elif isinstance(limit, str):
                limit_field: str = limit
                return [ConditionSpecVariableLimit(
                    field=field,
                    limit_field=limit_field,
                )]

            raise parent.error(f'{key.limit} not understood', limit)

        elif state is not None:

            if state is None or isinstance(state, State):
                return [ConditionSpecStates(
                    field=field,
                    states=[state],
                )]

            elif isinstance(state, _SpecList):
                states = clean_states(set(state))
                return [ConditionSpecStates(
                    field=field,
                    states=states,
                )]

            raise parent.error(f'{key.state} not understood', state)

    # Nothing valid found
    raise parent.error(f'{key.cardinality} not understood', cardinality_spec)


def _interpret_fields(entity_dict: SpecDict, reserved_fields: Set[str]) -> Dict[str, ModelFieldSpec]:
    # Get the sampled random variable fields.
    rvs: List[str] = entity_dict.get_string_list(key.rvs, [])
    duplicated_rvs: str = ', '.join(duplicates(rvs))
    if len(duplicated_rvs) > 0:
        raise entity_dict.error('duplicated sample rvs', duplicated_rvs)
    fields: Dict[str, ModelFieldSpec] = {
        rv_name: ModelFieldSpecSample(rv_name=rv_name)
        for rv_name in rvs
    }

    error_context: SpecDict = entity_dict  # may be refined below

    # Get other specified fields.
    fields_dict: Optional[SpecDict] = entity_dict.get_dict_optional(key.fields)
    if fields_dict is not None:
        error_context = fields_dict
        duplicated_fields: str = ', '.join(set(fields.keys()).intersection(fields_dict.keys()))
        if len(duplicated_fields) > 0:
            raise error_context.error('fields overlap with sampled random variables', duplicated_fields)
        for field_name in fields_dict.keys():
            fields[field_name] = _interpret_field(fields_dict.get_dict(field_name))

    clash_fields: str = ', '.join(reserved_fields.intersection(fields.keys()))
    if len(clash_fields) > 0:
        raise error_context.error('field names overlap with special fields', clash_fields)

    return fields


def _interpret_field(field_dict: SpecDict) -> ModelFieldSpec:
    is_sum: bool = key.sum in field_dict.keys()
    is_function: bool = key.function in field_dict.keys()
    is_sample: bool = key.sample in field_dict.keys()
    is_value: bool = list(field_dict.keys()) == [key.value]

    if sum([is_sum, is_function, is_sample, is_value]) != 1:
        raise field_dict.error('a field must be either a value, sum, function, or sample')

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
        if isinstance(sum_spec, _SpecList):
            sum_list = list(sum_spec)
        elif isinstance(sum_spec, (Integral, Real, str)):
            sum_list = [sum_spec]
        else:
            raise field_dict.error(f'{key.sum} not understood', sum_spec)

        sum_rvs: List[str] = []
        offset: State = 0
        for sum_item in sum_list:
            if isinstance(sum_item, str):
                sum_rvs.append(sum_item)
            else:
                try:
                    # noinspection PyTypeChecker, PyUnresolvedReferences
                    offset += clean_state(sum_item)
                except (TypeError, AttributeError):
                    raise field_dict.error('cannot sum item', sum_item)

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
            function=strip_lines(function),
            inputs=inputs,
        )
    raise NotReached()


def _interpret_crosstabs(
        spec_dict: SpecDict,
        rvs: Dict[str, ModelRVSpec],
        datasources: Dict[str, DatasourceSpec],
        flatten_min_cell_size: bool,
) -> Dict[str, ModelCrosstabSpec]:
    all_datasource_rvs: Set[str] = _all_datasource_rvs(datasources.values())

    crosstabs_spec = spec_dict.get(key.crosstabs, ())
    context = spec_dict.sub_dict(key.crosstabs)

    iterator: Iterable[Tuple[str, Any]]
    if isinstance(crosstabs_spec, Mapping):
        # crosstabs_spec is a dictionary: crosstab name => crosstab spec
        iterator = crosstabs_spec.items()
    elif isinstance(crosstabs_spec, _SpecList):
        # crosstabs_spec is a list of crosstab spec
        specs: List[Any] = list(crosstabs_spec)
        names: List[str] = []
        for spec in specs:
            name = _make_crosstab_name(spec, names)
            names.append(name)
        iterator = zip(names, specs)
    else:
        raise context.error(f'{key.crosstabs} section not understood', crosstabs_spec)

    # Make the specified crosstabs
    crosstabs: Dict[str, ModelCrosstabSpec] = {
        crosstab_name: _interpret_crosstab(
            context,
            crosstab_name,
            crosstab_spec,
            datasources,
            all_datasource_rvs,
            flatten_min_cell_size,
        )
        for crosstab_name, crosstab_spec in iterator
    }

    # Add a singleton cross-table for any random variable not already in a cross-table
    covered_rvs: Set[str] = {
        rv_name
        for crosstab in crosstabs.values()
        for rv_name in crosstab.rvs
    }
    singleton_rvs: Set[str] = set(rvs.keys()).difference(covered_rvs)
    for rv_name in singleton_rvs:
        crosstab_name: str = _make_unique_id(f'_{rv_name}_', crosstabs.keys())
        crosstab_rvs: List[str] = [rv_name]
        datasource: str = _find_datasource(spec_dict, crosstab_rvs, datasources)
        sensitivity: float = datasources[datasource].sensitivity
        epsilon: float = (
            0 if (sensitivity == 0)
            else spec_dict.get_non_neg(key.epsilon)
        )
        min_cell_size: float = (
            0 if (sensitivity == 0 and flatten_min_cell_size)
            else spec_dict.get_non_neg(key.min_cell_size)
        )

        crosstabs[crosstab_name] = ModelCrosstabSpec(
            rvs=crosstab_rvs,
            datasource=datasource,
            epsilon=epsilon,
            min_cell_size=min_cell_size,
            noiser=_interpret_noiser(spec_dict),
        )

    return crosstabs


def _interpret_noiser(parent: SpecDict) -> NoiserSpec:
    noise: Any = parent.find(key.noise)

    if noise is None:
        raise parent.error(f'cannot find {key.noise}')
    elif isinstance(noise, str):
        return _make_noise_spec(noise, parent)
    elif isinstance(noise, Mapping):
        noise_spec_dict = parent.get_dict(key.noise)
        noise_type: str = noise_spec_dict.get_string(key.noise)
        return _make_noise_spec(noise_type, noise_spec_dict)
    else:
        raise parent.error(f'{key.noise} not understood', noise)


def _make_noise_spec(noise_type: str, context: SpecDict) -> NoiserSpec:
    if noise_type == key.basic_laplace:
        return NoiserSpecBasicLaplace()
    elif noise_type == key.laplace:
        max_add_rows: int = context.get_positive_int(key.max_add_rows)
        return NoiserSpecLaplace(max_add_rows=max_add_rows)
    elif noise_type == key.naive_laplace:
        max_add_rows: int = context.get_positive_int(key.max_add_rows)
        return NoiserSpecNaiveLaplace(max_add_rows=max_add_rows)
    elif noise_type == key.decomposition_laplace:
        max_add_rows: int = context.get_positive_int(key.max_add_rows)
        return NoiserSpecDecompositionLaplace(max_add_rows=max_add_rows)
    raise context.error('noise type not understood', noise_type)


def _interpret_crosstab(
        parent: SpecDict,
        crosstab_name: str,
        crosstab_spec: Any,
        datasources: Dict[str, DatasourceSpec],
        all_datasource_rvs: Set[str],
        flatten_min_cell_size: bool,
) -> ModelCrosstabSpec:
    datasource: str
    rvs: List[str]

    if isinstance(crosstab_spec, str):
        # The crosstab_spec is a string - interpret it is a datasource id
        datasource = crosstab_spec
        crosstab_spec_dict = parent.sub_dict(crosstab_name)
        datasource_spec: Optional[DatasourceSpec] = datasources.get(datasource)
        if datasource_spec is None:
            raise crosstab_spec_dict.error('cannot find datasource', datasource)
        rvs = datasource_spec.rvs
        if len(rvs) == 0:
            raise parent.error('no random variables in cross-table datasource', datasource)

    elif isinstance(crosstab_spec, Mapping):
        # The crosstab_spec is a dict
        crosstab_spec_dict = parent.sub_dict(crosstab_name, crosstab_spec)

        rvs = crosstab_spec_dict.get_string_list(key.rvs)
        if len(rvs) == 0:
            raise crosstab_spec_dict.error('no random variables in cross-table')
        if key.datasource in crosstab_spec_dict.keys():
            datasource = crosstab_spec_dict.get_string(key.datasource)
        else:
            datasource = _find_datasource(crosstab_spec_dict, rvs, datasources)

    elif isinstance(crosstab_spec, _SpecList):
        # The crosstab_spec is a list of random variables
        rvs = list(crosstab_spec)
        if len(rvs) == 0:
            raise parent.error('no random variables in cross-table')
        for rv in rvs:
            if rv not in all_datasource_rvs:
                raise parent.error('random variable not in a datasource', rv)

        crosstab_spec_dict = parent.sub_dict(crosstab_name)
        datasource = _find_datasource(crosstab_spec_dict, rvs, datasources)

    else:
        raise parent.sub_dict(crosstab_name).error('crosstab specification not understood', crosstab_spec)

    epsilon: float = crosstab_spec_dict.get_non_neg(key.epsilon)
    min_cell_size: float = crosstab_spec_dict.get_non_neg(key.min_cell_size)
    noiser: NoiserSpec = _interpret_noiser(crosstab_spec_dict)
    sensitivity: float = datasources[datasource].sensitivity
    if sensitivity > 0 and epsilon == 0:
        raise parent.error(
            f'cross-table {crosstab_name!r} has epsilon = 0 '
            f'but datasource {datasource!r} has sensitivity = {sensitivity}'
        )
    if sensitivity == 0 and epsilon > 0:
        parent.warn(
            f'cross-table {crosstab_name!r} has epsilon = {epsilon} '
            f'but datasource {datasource!r} has sensitivity = 0'
        )
    if flatten_min_cell_size and sensitivity == 0 and min_cell_size != 0:
        if key.min_cell_size in crosstab_spec_dict.keys():
            parent.warn(
                f'cross-table {crosstab_name!r} has min_cell_size = {min_cell_size} '
                f'and datasource {datasource!r} has sensitivity = 0'
                f'but flatten_min_cell_size is True'
            )
        min_cell_size = 0

    return ModelCrosstabSpec(
        rvs=rvs,
        datasource=datasource,
        epsilon=epsilon,
        min_cell_size=min_cell_size,
        noiser=noiser,
    )


def _find_datasource(parent: SpecDict, rvs: List[str], datasources: Dict[str, DatasourceSpec]) -> str:
    rvs_set: Set[str] = set(rvs)
    candidates: List[str] = [
        datasource
        for datasource, datasource_spec in datasources.items()
        if rvs_set.issubset(datasource_spec.rvs)
    ]
    if len(candidates) == 0:
        raise parent.error('no datasource found for random variables', ', '.join(rvs))
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


def _make_crosstab_name(crosstab_spec: Any, names: Collection[str]) -> str:
    """
    Come up with a name for a cross-table, because no name is in the
    spec file.

    Returns:
        The new name, which will be appended to `names`.
    """
    # First we select a preferred name for the cross-table, then
    # call `_make_unique_id` to ensure it is unique.

    name: str = 'crosstab'  # default preferred name if all else fails.
    if isinstance(crosstab_spec, str):
        # Looks like crosstab_spec is a datasource name
        name = crosstab_spec
    elif isinstance(crosstab_spec, Mapping) and key.rvs in crosstab_spec.keys():
        # Looks like crosstab_spec is a CROSSTAB_DICT
        rvs = crosstab_spec[key.rvs]
        if isinstance(rvs, str):
            name = rvs
        elif isinstance(rvs, _SpecList):
            name = ','.join(str(rv) for rv in rvs)
    elif isinstance(crosstab_spec, _SpecList):
        name = ','.join(str(rv) for rv in crosstab_spec)

    return _make_unique_id(name, names)


def _make_unique_id(base_name: str, seen_names) -> str:
    """
    Make a name, based on `base_name` but not in `seen_names`.
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
        raise rv_dict.error('random variable has no datasource', rv_name)

    states: int | List[State] | Literal['infer_distinct', 'infer_range', 'infer_max']

    states_spec = rv_dict.check_exists(key.states)
    if states_spec == key.infer_distinct:
        states = 'infer_distinct'
    elif states_spec == key.infer_range:
        states = 'infer_range'
    elif states_spec == key.infer_max:
        states = 'infer_max'
    elif isinstance(states_spec, Integral):
        states = int(states_spec)
    elif isinstance(states_spec, Mapping):
        states = _get_states_from_range_dict(rv_dict.sub_dict(key.states, states_spec))
    elif isinstance(states_spec, _SpecIterable):
        states = _get_states_from_iterable(rv_dict, states_spec)
    else:
        raise rv_dict.error(f'{key.states} not understood', states_spec)

    return ModelRVSpec(states=states, ensure_none=ensure_none)


def _get_states_from_iterable(context: SpecDict, iterable: Iterable[State]) -> List[State]:
    states = list(iterable)
    for state in states:
        if not isinstance(state, State):
            raise context.error('invalid random variable state', state)
    return states


def _interpret_datasources(spec_dict: SpecDict, roots: List[DataPath]) -> Dict[str, DatasourceSpec]:
    datasources_dict: SpecDict = spec_dict.get_dict(key.datasources, dont_inherit=SECTIONS, default={})
    result: Dict[str, DatasourceSpec] = {}
    for datasource_name in datasources_dict.keys():
        datasources_dict.check_is_id(datasource_name)
        datasource_dict = datasources_dict.get_dict(datasource_name)
        result[datasource_name] = _interpret_datasource(datasource_name, datasource_dict, roots)
    return result


def _interpret_datasource(datasource_name: str, datasource_dict: SpecDict, roots: List[DataPath]) -> DatasourceSpec:
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
        location: str = datasource_dict.get_string(key.location)
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
        return _make_datasource_csv(datasource_dict, roots, ',')
    elif format_str == key.tsv:
        return _make_datasource_csv(datasource_dict, roots, '\t')
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
        raise datasource_dict.error(f'cannot infer datasource {key.data_format}')
    else:
        raise datasource_dict.error(f'datasource {key.data_format} not understood', format_str)


def _make_datasource_csv(
        datasource_dict: SpecDict,
        roots: List[DataPath],
        default_sep: str,
) -> DatasourceSpec:
    """
    Construct a DatasourceSpec with a DatasetSpecCsv dataset.
    """
    sep = datasource_dict.get_string(key.sep, default=default_sep)
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
        roots: List[DataPath],
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
        roots: List[DataPath],
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
        roots: List[DataPath],
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
        roots: List[DataPath],
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
        function=strip_lines(function),
    )

    rvs_list: List[str] = list(input_dict.keys()) + [output_rv]

    sensitivity = datasource_dict.get_non_neg(key.sensitivity, 0)  # default is zero
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

    connection: Optional[Dict[str, Optional[str | int]]] = None
    if key.connection in datasource_dict.keys():
        connection_dict: SpecDict = datasource_dict.get_dict(key.connection)
        connection: Dict[str, Optional[str | int]] = {}
        for conx_key, conx_value in connection_dict.items():
            if not isinstance(conx_key, str):
                raise connection_dict.error('invalid connection key', conx_key)
            if not isinstance(conx_value, (str, int, NoneType)):
                raise connection_dict.error('invalid connection value', conx_value)
            connection[conx_key] = conx_value

    dataset_spec = DatasetSpecDBMS(
        type=api,
        schema_name=schema_name,
        table_name=table_name,
        rvs=rvs,
        connection=connection,
    )

    if rvs is None:
        rvs: List[str] = _get_rvs_from_dataset(datasource_dict, dataset_spec, ())

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
        elif isinstance(states_def, _SpecIterable):
            result[rv_name] = _get_states_from_iterable(rvs_dict, states_def)
        elif isinstance(states_def, dict):
            result[rv_name] = _get_states_from_range_dict(rvs_dict.get_dict(rv_name))
        else:
            raise rvs_dict.error('random variable states not understood', states_def)
    return result


def _get_states_from_range_dict(range_dict: SpecDict) -> List[float]:
    range_dict.check_restricted(
        [key.start, key.stop, key.step],
        message=f'only {key.start}, {key.stop}, and {key.step} permitted'
    )
    start: float = range_dict.get_numeric(key.start, default=0)
    stop: float = range_dict.get_numeric(key.stop)
    step: float = range_dict.get_numeric(key.step, default=1)
    if step == 0:
        raise range_dict.error('state range step cannot be zero')
    if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
        stop += 1 if step > 0 else -1
        return list(range(start, stop, step))
    # Need to calculate range of float values
    num = int((stop - start) / step) + 2  # deliberately start with more elements to manage rounding errors
    result = [start + i * step for i in range(num)]
    # trim extras
    if step > 0:
        # counting up
        while len(result) > 0 and result[-1] > stop:
            result.pop()
    elif step < 0:
        # counting down
        while len(result) > 0 and result[-1] < stop:
            result.pop()
    return result


def _get_rvs_list(
        datasource_dict: SpecDict,
        dataset_spec: DatasetSpec,
        rv_map: Optional[Dict[str, ColumnSpec]],
        rv_define: Dict[str, ColumnDefinitionSpec],
        roots: Sequence[DataPath],
) -> List[str]:
    if rv_map is None:
        # Get random variables for the dataset directly
        return _get_rvs_from_dataset(datasource_dict, dataset_spec, roots)
    else:
        return list(set(rv_map.keys()).union(rv_define.keys()))


def _get_rvs_from_dataset(
        datasource_dict: SpecDict,
        dataset_spec: DatasetSpec,
        roots: Sequence[DataPath],
) -> List[str]:
    with warnings.catch_warnings(record=True) as the_warnings:
        warnings.simplefilter('always')
        dataset: Dataset = dataset_spec.dataset(roots)
    if the_warnings is not None:
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
            datasource_dict.check_is_id(rv, 'invalid random variable name')
        return {rv: rv for rv in rv_map}

    if isinstance(rv_map, dict):
        result: Dict[str, ColumnSpec] = {}
        for rv_name, definition in rv_map.items():
            if not isinstance(rv_name, str):
                raise datasource_dict.error('rv is not a string', rv_name)
            if isinstance(definition, (int, str)):
                result[rv_name] = definition
            else:
                raise datasource_dict.error(f'{key.rvs} entry not understood', rv_name)
        return result

    raise datasource_dict.error(f'{key.rvs} not understood')


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
        function=strip_lines(function),
        delete_input=delete_input,
    )


def _get_column_definition_spec_group(column_definition_dict: SpecDict) -> ColumnDefinitionSpecGroup:
    group_type: Literal['group_cut', 'group_qcut', 'group_normalise']
    grouping: str = column_definition_dict.check_exists(key.grouping)
    if grouping in ('group_cut', 'group_qcut', 'group_normalise'):
        group_type = grouping
    else:
        raise column_definition_dict.error(f'{key.grouping} method not understood', grouping)

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
            raise datasource_dict.error(f'cannot have both {key.location} and {key.inline}')
        return TextInputSpecLocation(location=location)
    elif inline is not None:
        return TextInputSpecInline(inline=strip_lines(inline))
    else:
        raise datasource_dict.error(f'must have either {key.location} or {key.inline}')


def _wrap_with_spec_dict(root_name: str, update: Mapping[str, Any], *defaults: Mapping[str, Any]) -> SpecDict:
    """
    Construct a SpecDict with local items from 'update', and default
    values from the additional parents.

    Args:
        root_name: name of the root section.
        update: A Mappings to initialise the new SpecDict.
        defaults: zero or more Mappings of default values.
    """
    defaults = tuple(d for d in defaults if d is not None)
    return SpecDict(root_name, root_name, *defaults, update=update)
