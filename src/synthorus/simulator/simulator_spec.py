"""
A Simulator Spec is a serializable (JSON) representation of a simulator.
"""
from __future__ import annotations

from typing import Dict, Optional, List, TypeAlias, Self, Set, Annotated, Union, Literal

from ck.pgm import State
from pydantic import BaseModel, Field, field_validator, model_validator

from synthorus.model.defaults import DEFAULT_ID_FIELD, DEFAULT_COUNT_FIELD
from synthorus.simulator.condition_spec import ConditionSpec, FieldRef


class SimulatorSpec(BaseModel):
    parameters: Dict[str, State]
    entities: Dict[str, SimEntitySpec]

    @model_validator(mode='after')
    def validate_simulator_spec(self) -> Self:
        # Ensure no parameter name is the empty string
        for param_name in self.parameters.keys():
            if param_name == '':
                raise ValueError('parameter name cannot be the empty string')

        # Ensure no entity name is the empty string
        for entity_name in self.entities.keys():
            if entity_name == '':
                raise ValueError('entity name cannot be the empty string')

        return self


class SimEntitySpec(BaseModel):
    parent: Optional[str] = None  # name of parent entity
    sampler: Optional[str] = None  # name of sampler
    id_field_name: str = DEFAULT_ID_FIELD  # name of the field holding row ID for the entity
    count_field_name: str = DEFAULT_COUNT_FIELD  # name of the field holding row count for the entity
    foreign_field_name: Optional[str] = None  # name of the field holding row ID for the _parent_ entity
    fields: Dict[str, ValueSpec] = {}
    cardinality: List[ConditionSpec] = []

    @model_validator(mode='after')
    def validate_model(self) -> Self:
        if (self.parent is None) != (self.foreign_field_name is None):
            raise ValueError(f'foreign field name required if and only if the entity has a parent')

        if self.id_field_name in self.fields:
            raise ValueError(f'id field name cannot be an explicit field: {self.id_field_name!r}')
        if self.count_field_name in self.fields:
            raise ValueError(f'count field name cannot be an explicit field: {self.count_field_name!r}')
        if self.foreign_field_name in self.fields:
            raise ValueError(f'foreign field name cannot be an explicit field: {self.foreign_field_name!r}')

        if len({self.count_field_name, self.id_field_name, self.foreign_field_name}) != 3:
            raise ValueError(f'count, id and foreign field names must be different')

        all_fields = list(self.fields.keys()) + [self.id_field_name, self.count_field_name]
        if self.foreign_field_name is not None:
            all_fields.append(self.foreign_field_name)

        for field_name in all_fields:
            if field_name == '':
                raise ValueError('field name cannot be the empty string')

        return self


class ConstantSpec(BaseModel):
    type: Literal['constant'] = 'constant'
    value: State


class SampleSpec(BaseModel):
    type: Literal['sample'] = 'sample'
    rv_name: str  # name of random variable to sample


class SumSpec(BaseModel):
    type: Literal['sum'] = 'sum'
    initial_value: State = 0
    inputs: List[FieldRef | str]
    add_self: bool


class FunctionSpec(BaseModel):
    type: Literal['function'] = 'function'
    initial_value: State = 0
    inputs: List[FieldRef | str]
    function: str  # Python expression representing the body of a function

    # noinspection PyNestedDecorators
    @field_validator('inputs', mode='before')
    @classmethod
    def validate_inputs(cls, inputs: List[FieldRef | str]) -> List[FieldRef | str]:
        seen: Set[str] = set()
        for field in inputs:
            name: str
            if isinstance(field, FieldRef):
                name = field.field
            else:
                name = field
            if not name.isidentifier():
                raise ValueError(f'invalid input identifier: {name!r}')
            if name in seen:
                raise ValueError(f'duplicated input field: {name!r}')
            seen.add(name)
        return inputs


ValueSpec: TypeAlias = Annotated[
    Union[
        ConstantSpec,
        SampleSpec,
        SumSpec,
        FunctionSpec,
    ],
    Field(discriminator='type')
]


def parameter(param_name: str) -> FieldRef:
    """
    A parameter is just a special field in the root entity,
    which always has name ''.
    """
    return FieldRef(entity='', field=param_name)
