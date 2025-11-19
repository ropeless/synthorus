from typing import Mapping, Dict, List, Optional, Tuple, Any

from synthorus.error import SynthorusError
from synthorus.simulator.condition_spec import FieldRef, ConditionSpec, ConditionSpecFixedLimit, \
    ConditionSpecVariableLimit, ConditionSpecStates
from synthorus.simulator.sim_entity import SimSampler, SimEntity
from synthorus.simulator.sim_field import SimField, SimFieldUpdate
from synthorus.simulator.sim_field_updaters import FunctionUpdate, SumUpdate
from synthorus.simulator.simulator import Simulator, DEFAULT_ID_FIELD, DEFAULT_COUNT_FIELD
from synthorus.simulator.simulator_spec import SimulatorSpec, SimEntitySpec, ValueSpec, FunctionSpec, \
    SampleSpec, ConstantSpec, SumSpec


def make_simulator_from_simulator_spec(spec: SimulatorSpec, samplers: Mapping[str, SimSampler]) -> Simulator:
    """
    Args:
        spec: a dict conforming to the SIM_SPEC format.
        samplers: a dict from sampler name (in simulator_spec) to a SimSampler object.
    """
    sim = Simulator()

    for parameter_name, value in spec.parameters.items():
        sim.add_parameter(parameter_name, value)

    for entity_name, entity_spec in _sort_entities(spec.entities):
        _add_entity(sim, samplers, entity_name, entity_spec)

    return sim


def _sort_entities(entities: Dict[str, SimEntitySpec]) -> List[Tuple[str, SimEntitySpec]]:
    """
    Returns:
         a total order over entities. Parents come before children.
    """
    sorted_entities: List[Tuple[str, SimEntitySpec]] = []
    entity_state: Dict[str, int] = {}
    for entity_name in entities.keys():
        _sort_entity_r(entity_name, entities, sorted_entities, entity_state)
    return sorted_entities


def _sort_entity_r(
        entity_name: str,
        entities: Dict[str, SimEntitySpec],
        sorted_entities: List[Tuple[str, SimEntitySpec]],
        entity_state: Dict[str, int],
) -> None:
    """
    Recursive support for `_sort_entities`.
    """
    state = entity_state.get(entity_name)
    if state is None:
        entity_state[entity_name] = 1
        entity_spec = entities[entity_name]
        parent: Optional[str] = entity_spec.parent
        if parent is not None:
            _sort_entity_r(parent, entities, sorted_entities, entity_state)
        sorted_entities.append((entity_name, entity_spec))
        entity_state[entity_name] = 2
    elif state == 1:
        raise SynthorusError(f'entity loop detected: {entity_name}')


def _default(val: Optional[str], default: str) -> str:
    return default if val is None else val


def _add_entity(
        sim: Simulator,
        samplers: Mapping[str, SimSampler],
        entity_name: str,
        entity_spec: SimEntitySpec
):
    """
    Add an entity to the simulator based on the given entity_spec (with the given
    entity name), and using the sampler as found in `samplers`.
    """
    kwargs: Dict[str, Any] = {
        'id_field_name': _default(entity_spec.id_field_name, DEFAULT_ID_FIELD),
        'count_field_name': _default(entity_spec.count_field_name, DEFAULT_COUNT_FIELD),
        'foreign_field_name': entity_spec.foreign_field_name,
    }

    parent_name: Optional[str] = entity_spec.parent
    if parent_name is not None:
        try:
            kwargs['parent'] = sim.entity(parent_name)
        except KeyError:
            raise SynthorusError(f'cannot find parent: {parent_name}')

    sampler_name: Optional[str] = entity_spec.sampler
    if sampler_name is not None:
        try:
            kwargs['sampler'] = samplers[sampler_name]
        except KeyError:
            raise SynthorusError(f'cannot find sampler: {sampler_name}')

    entity: SimEntity = sim.add_entity(entity_name, **kwargs)

    _add_fields(sim, entity, entity_spec.fields)
    _add_cardinality(sim, entity, entity_spec.cardinality)


def _add_fields(sim: Simulator, entity: SimEntity, fields_specs: Dict[str, ValueSpec]) -> None:
    # We need to add fields in the correct order to ensure updates
    # that refer to previous fields can work correctly.
    field_names = _get_field_order(entity, fields_specs)

    for field_name in field_names:
        if field_name in entity:
            raise SynthorusError(f'duplicate field name: {field_name}')
        _add_field(sim, entity, field_name, fields_specs)


def _get_field_order(entity: SimEntity, fields_specs: Dict[str, ValueSpec]) -> List[str]:
    states: Dict[str, int] = {}
    result: List[str] = []
    entity_name = entity.name
    for field_name in fields_specs.keys():
        _get_field_order_r(entity_name, fields_specs, field_name, states, result)
    return result


def _get_field_order_r(
        entity_name: str,
        fields_specs: Dict[str, ValueSpec],
        field_name: str,
        states: Dict[str, int],
        result: List[str]
):
    state = states.get(field_name)
    if state is None:
        states[field_name] = 1

        field_spec: ValueSpec = fields_specs[field_name]
        if isinstance(field_spec, (FunctionSpec, SumSpec)):
            field: FieldRef | str
            for input_field in field_spec.inputs:
                input_field_name: str
                input_field_entity: str
                if isinstance(input_field, FieldRef):
                    input_field_name = input_field.name
                    input_field_entity = input_field.entity
                else:
                    input_field_name = input_field
                    input_field_entity = entity_name

                if input_field_entity == entity_name:
                    _get_field_order_r(
                        input_field_entity,
                        fields_specs,
                        input_field_name,
                        states,
                        result
                    )
        result.append(field_name)
        states[field_name] = 2
    elif state == 1:
        raise SynthorusError(f'loop detected in field update references: {field_name}')


def _add_field(sim: Simulator, entity: SimEntity, field_name: str, fields_specs: Dict[str, ValueSpec]) -> None:
    field_spec: ValueSpec = fields_specs[field_name]

    if isinstance(field_spec, ConstantSpec):
        entity.add_field(field_name, value=field_spec.value)

    elif isinstance(field_spec, SampleSpec):
        entity.add_field_sampled(field_name, field_spec.rv_name)

    elif isinstance(field_spec, SumSpec):
        input_fields: List[SimField] = _get_sim_field_list(sim, entity, field_spec.inputs)
        update: SimFieldUpdate = SumUpdate(*input_fields, include_self=field_spec.add_self)
        entity.add_field(field_name, value=field_spec.initial_value, update=update)

    elif isinstance(field_spec, FunctionSpec):
        input_fields: List[SimField] = _get_sim_field_list(sim, entity, field_spec.inputs)
        update: SimFieldUpdate = FunctionUpdate(
            func=field_spec.function,
            fields=input_fields,
        )
        entity.add_field(field_name, value=field_spec.initial_value, update=update)
    else:
        raise SynthorusError(f'unknown field spec type: {field_name}: {field_spec!r}')


def _get_sim_field_list(sim: Simulator, entity: SimEntity, fields: List[FieldRef | str]) -> List[SimField]:
    field: FieldRef | str
    return [
        _get_sim_field(sim, entity, field)
        for field in fields
    ]


def _get_sim_field(sim: Simulator, entity: SimEntity, field: FieldRef | str) -> SimField:
    found: List[SimField] = []
    if isinstance(field, str):
        _find_sim_field_str(sim, entity, field, found)
        field_name = field
    else:
        _find_sim_field_ref(sim, entity, field, found)
        field_name = field.field

    if len(found) == 0:
        raise SynthorusError(f'field not found: {field_name!r}')
    if len(found) > 1:
        raise SynthorusError(f'multiple fields found: {field_name!r}')
    return found[0]


def _find_sim_field_str(sim: Simulator, entity: SimEntity, field: str, found: List[SimField]) -> None:
    while True:
        got: Optional[SimField] = entity.get(field)
        if got is not None:
            found.append(got)
        if entity.parent is None:
            _find_sim_field_param(sim, field, found)
            return
        entity = entity.parent


def _find_sim_field_ref(sim: Simulator, entity: SimEntity, field_ref: FieldRef, found: List[SimField]) -> None:
    if field_ref.entity == '':
        _find_sim_field_param(sim, field_ref.field, found)
        return

    while True:
        if field_ref.entity == entity.name:
            got: Optional[SimField] = entity.get(field_ref.field)
            if got is not None:
                found.append(got)
            return
        if entity.parent is None:
            return
        entity = entity.parent


def _find_sim_field_param(sim: Simulator, field: str, found: List[SimField]) -> None:
    got: Optional[SimField] = sim.parameters.get(field)
    if got is not None:
        found.append(got)


def _add_cardinality(sim: Simulator, entity: SimEntity, cardinality: List[ConditionSpec]) -> None:
    if len(cardinality) == 0:
        # Default cardinality - run once
        entity.add_cardinality_fixed_count(1)
    else:
        for condition in cardinality:
            _add_stopping_condition(sim, entity, condition)


def _add_stopping_condition(sim: Simulator, entity: SimEntity, condition) -> None:
    test_field_ref = FieldRef(entity=entity.name, field=condition.field)
    test_field: SimField = _get_sim_field(sim, entity, test_field_ref)

    if isinstance(condition, ConditionSpecFixedLimit):
        entity.add_cardinality_fixed_limit(test_field, condition.limit, condition.op)

    elif isinstance(condition, ConditionSpecVariableLimit):
        limit_field: SimField = _get_sim_field(sim, entity, condition.limit_field)
        entity.add_cardinality_variable_limit(test_field, limit_field, condition.op)

    elif isinstance(condition, ConditionSpecStates):
        entity.add_cardinality_field_state(test_field, *condition.states)

    else:
        raise SynthorusError(f'unknown condition spec type: {condition!r}')
