from typing import Dict, List, Iterable, Optional, Set

from synthorus.error import SynthorusError
from synthorus.model.model_spec import ModelSpec, ModelEntitySpec, ModelFieldSpecSum, ModelFieldSpecFunction, \
    ModelFieldSpecSample
from synthorus.simulator.condition_spec import FieldRef
from synthorus.simulator.simulator_spec import SimulatorSpec, SimEntitySpec, ValueSpec, SampleSpec, FunctionSpec, \
    SumSpec


def make_simulator_spec_from_model_spec(model_spec: ModelSpec) -> SimulatorSpec:
    """
    Constructs a SimulatorSpec from a ModelSpec.
    """
    simulator_spec = SimulatorSpec(
        parameters=model_spec.parameters.copy(),
        entities=_entities_from_model_spec(model_spec),
    )
    return simulator_spec


def _entities_from_model_spec(model_spec: ModelSpec) -> Dict[str, SimEntitySpec]:
    return {
        entity_name: _entity_from_entity_spec(entity_name, entity, model_spec.entities)
        for entity_name, entity in model_spec.entities.items()
    }


def _entity_from_entity_spec(
        entity_name: str,
        entity_spec: ModelEntitySpec,
        entities: Dict[str, ModelEntitySpec],
) -> SimEntitySpec:
    rvs: Set[str] = set(field.rv_name for _, field in entity_spec.sampled_fields())
    sampler: Optional[str] = entity_name if len(rvs) > 0 else None
    fields: Dict[str, ValueSpec] = _fields_from_entity_spec(entity_name, entity_spec, entities)

    result = SimEntitySpec(
        parent=entity_spec.parent,
        sampler=sampler,
        id_field_name=entity_spec.id_field_name,
        count_field_name=entity_spec.count_field_name,
        foreign_field_name=entity_spec.foreign_field_name,
        fields=fields,
        cardinality=entity_spec.cardinality.copy(),
    )
    return result


def _fields_from_entity_spec(
        entity_name: str,
        entity_spec: ModelEntitySpec,
        entities: Dict[str, ModelEntitySpec],
) -> Dict[str, ValueSpec]:
    fields = {
        field_name: _field_from_field_spec(field_name, field_spec, entity_name, entity_spec, entities)
        for field_name, field_spec in entity_spec.fields.items()
    }
    return fields


def _field_from_field_spec(
        field_name: str,
        field_spec: ModelFieldSpecSum | ModelFieldSpecFunction,
        entity_name: str,
        entity: ModelEntitySpec,
        entities: Dict[str, ModelEntitySpec],
) -> ValueSpec:
    if isinstance(field_spec, ModelFieldSpecSample):
        return SampleSpec(rv_name=field_spec.rv_name)
    elif isinstance(field_spec, ModelFieldSpecSum):
        add_fields_set = set(field_spec.sum)
        if len(add_fields_set) != len(field_spec.sum):
            raise SynthorusError(f'duplicate fields in field spec {field_spec.name}')
        add_self: bool = field_name in add_fields_set
        add_fields_set.discard(field_name)
        inputs: List[FieldRef | str] = _find_fields(add_fields_set, entity_name, entity, entities)
        return SumSpec(
            initial_value=field_spec.initial_value,
            inputs=inputs,
            add_self=add_self,
        )
    elif isinstance(field_spec, ModelFieldSpecFunction):
        return FunctionSpec(
            initial_value=field_spec.initial_value,
            inputs=_find_fields(field_spec.inputs, entity_name, entity, entities),
            function=field_spec.function,
        )
    else:
        raise SynthorusError(f'unexpected field spec type: {type(field_spec)}')


def _find_fields(
        field_names: Iterable[str],
        entity_name: str,
        entity: ModelEntitySpec,
        entities: Dict[str, ModelEntitySpec],
) -> List[FieldRef | str]:
    result: List[FieldRef | str] = []
    for field_name in field_names:
        candidate: List[str] = _find_field(field_name, entity_name, entity, entities)
        if len(candidate) == 0:
            raise SynthorusError(f'field not found: {field_name}')
        if len(candidate) > 1:
            raise SynthorusError(f'duplicate fields found: {field_name}')
        found_entity: str = candidate[0]
        if found_entity == entity_name:
            result.append(field_name)
        else:
            result.append(FieldRef(entity=found_entity, field=field_name))
    return result


def _find_field(
        field_name: str,
        entity_name: str,
        entity: ModelEntitySpec,
        entities: Dict[str, ModelEntitySpec],
) -> List[str]:
    """
    Check `entity` and its parents for the given field name.
    Return all entities that contain the field.
    Assumes there are no cycles in the entity hierarchy.
    """
    found: List[str] = []
    while True:
        if field_name in entity.fields:
            found.append(entity_name)
        if entity.parent is None:
            return found
        entity_name = entity.parent
        entity = entities[entity_name]
