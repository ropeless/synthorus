from typing import Dict, List, Iterable, Optional, Set

from synthorus.error import SynthorusError
from synthorus.model.model_index import ModelIndex, EntityIndex
from synthorus.model.model_spec import ModelFieldSpecSum, ModelFieldSpecFunction, \
    ModelFieldSpecSample, ModelFieldSpec
from synthorus.simulator.condition_spec import FieldRef
from synthorus.simulator.simulator_spec import SimulatorSpec, SimEntitySpec, ValueSpec, SampleSpec, FunctionSpec, \
    SumSpec


def make_simulator_spec_from_model_index(model_index: ModelIndex) -> SimulatorSpec:
    """
    Constructs a SimulatorSpec from a ModelIndex.
    """
    simulator_spec = SimulatorSpec(
        parameters=model_index.parameters,
        entities=_make_entities(model_index),
    )
    return simulator_spec


def _make_entities(model_index: ModelIndex) -> Dict[str, SimEntitySpec]:
    return {
        entity_name: _make_entity(entity_name, entity, model_index.entities)
        for entity_name, entity in model_index.entities.items()
    }


def _make_entity(
        entity_name: str,
        entity_index: EntityIndex,
        entities: Dict[str, EntityIndex],
) -> SimEntitySpec:
    rvs: Set[str] = entity_index.sample_rvs()
    sampler: Optional[str] = entity_name if len(rvs) > 0 else None
    fields: Dict[str, ValueSpec] = _make_fields(entity_name, entity_index, entities)

    result = SimEntitySpec(
        sampler=sampler,
        id_field_name=entity_index.model_entity.id_field_name,
        count_field_name=entity_index.model_entity.count_field_name,
        foreign_key_fields=entity_index.model_entity.foreign_key_fields,
        fields=fields,
        cardinality=entity_index.model_entity.cardinality,
    )
    return result


def _make_fields(
        entity_name: str,
        entity_index: EntityIndex,
        entities: Dict[str, EntityIndex],
) -> Dict[str, ValueSpec]:
    fields = {
        field_name: _make_field(field_name, field_spec, entity_name, entity_index, entities)
        for field_name, field_spec in entity_index.model_entity.fields.items()
    }
    return fields


def _make_field(
        field_name: str,
        field_spec: ModelFieldSpec,
        entity_name: str,
        entity_index: EntityIndex,
        entities: Dict[str, EntityIndex],
) -> ValueSpec:
    if isinstance(field_spec, ModelFieldSpecSample):
        return SampleSpec(rv_name=field_spec.rv_name)

    elif isinstance(field_spec, ModelFieldSpecSum):
        add_fields_set = set(field_spec.sum)
        if len(add_fields_set) != len(field_spec.sum):
            raise SynthorusError(f'duplicate fields in field spec {field_name!r}')
        add_self: bool = field_name in add_fields_set
        add_fields_set.discard(field_name)
        inputs: List[FieldRef | str] = _find_fields(add_fields_set, entity_name, entity_index, entities)
        return SumSpec(
            initial_value=field_spec.initial_value,
            inputs=inputs,
            add_self=add_self,
        )

    elif isinstance(field_spec, ModelFieldSpecFunction):
        return FunctionSpec(
            initial_value=field_spec.initial_value,
            inputs=_find_fields(field_spec.inputs, entity_name, entity_index, entities),
            function=field_spec.function,
        )

    else:
        raise SynthorusError(f'unexpected field spec type: {type(field_spec)}')


def _find_fields(
        field_names: Iterable[str],
        entity_name: str,
        entity_index: EntityIndex,
        entities: Dict[str, EntityIndex],
) -> List[FieldRef | str]:
    result: List[FieldRef | str] = []
    for field_name in field_names:
        candidate: List[str] = _find_field(field_name, entity_name, entity_index, entities)
        if len(candidate) == 0:
            raise SynthorusError(f'field not found: {field_name!r}')
        if len(candidate) > 1:
            raise SynthorusError(f'duplicate fields found: {field_name!r}')
        found_entity: str = candidate[0]
        if found_entity == entity_name:
            result.append(field_name)
        else:
            result.append(FieldRef(entity=found_entity, field=field_name))
    return result


def _find_field(
        field_name: str,
        entity_name: str,
        entity_index: EntityIndex,
        entities: Dict[str, EntityIndex],
) -> List[str]:
    """
    Check `entity` and its ancestors for the given field name.
    Return all entities that contain the field.
    """
    found: List[str] = []

    _find_field_one_entity(field_name, entity_name, entity_index, found)
    for ancestor_name in entity_index.ancestors:
        ancestor_index: EntityIndex = entities[ancestor_name]
        _find_field_one_entity(field_name, ancestor_name, ancestor_index, found)

    return found


def _find_field_one_entity(
        field_name: str,
        entity_name: str,
        entity_index: EntityIndex,
        found: List[str]
) -> None:
    """
    If the entity contains a field named `field_name` add the entity name to the `found` list.
    """
    foreign_key_fields: Iterable[str] = (
        foreign_key_field.foreign_key_field_name
        for foreign_key_field in entity_index.model_entity.foreign_key_fields
    )
    if (
            field_name in entity_index.model_entity.fields
            or field_name in foreign_key_fields
            or field_name == entity_index.model_entity.id_field_name
            or field_name == entity_index.model_entity.count_field_name
    ):
        found.append(entity_name)
