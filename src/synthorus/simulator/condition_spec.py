"""
Specification structures for entity cardinality stopping conditions.
"""
from __future__ import annotations

from typing import TypeAlias, Literal, List, Annotated, Union

from ck.pgm import State
from pydantic import BaseModel, Field

OpSpec: TypeAlias = Literal['<', '<=', '>', '>=', '==', '!=']
"""
Comparison operators for entity cardinality stopping conditions.
"""

DEFAULT_OP: OpSpec = '>='


class ConditionSpecFixedLimit(BaseModel):
    """
    A stopping condition: stop if "{field} {op} {limit}".
    """
    type: Literal['fixed'] = 'fixed'  # for JSON round trip
    field: str  # name of a field in the current entity
    op: OpSpec = DEFAULT_OP
    limit: State


class ConditionSpecVariableLimit(BaseModel):
    """
    A stopping condition: stop if "{field} {op} {limit_field}".
    """
    type: Literal['variable'] = 'variable'  # for JSON round trip
    field: str  # name of a field in the current entity
    op: OpSpec = DEFAULT_OP
    limit_field: FieldRef | str  # limit is the value of the field.


class ConditionSpecStates(BaseModel):
    """
    A stopping condition: stop if "{field} in {states}".
    """
    type: Literal['states'] = 'states'  # for JSON round trip
    field: str  # name of a field in the current entity
    states: List[State]


ConditionSpec: TypeAlias = Annotated[
    Union[ConditionSpecFixedLimit, ConditionSpecVariableLimit, ConditionSpecStates],
    Field(discriminator='type')
]


class FieldRef(BaseModel):
    entity: str  # name of the entity with the filed
    field: str  # name of the field in the given entity
