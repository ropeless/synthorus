from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Callable, Dict, get_args, TypeAlias

from ck.pgm import State

from synthorus.simulator.condition_spec import OpSpec
from synthorus.simulator.sim_field import SimField

OpFunction: TypeAlias = Callable[[State, State], bool]

OPERATION: Dict[str, OpFunction] = {
    '<': lambda x, y: x < y,
    '<=': lambda x, y: x <= y,
    '>': lambda x, y: x > y,
    '>=': lambda x, y: x >= y,
    '==': lambda x, y: x == y,
    '!=': lambda x, y: x != y,
}

# Every literal in OpSpec must have a corresponding operation in OPERATION.
assert all(op in OPERATION.keys() for op in get_args(OpSpec))


class SimCondition(ABC):
    """
    Abstract method for testing whether to stop a simulation loop.
    """

    @abstractmethod
    def stop(self) -> bool:
        """
        Does this condition indicate a simulation loop should stop.
        """


class FixedLimitCondition(SimCondition):
    """
    Stop if check_field.value >= limit.
    """

    def __init__(self, check_field: SimField, limit: State, op: str):
        self.check_field = check_field
        self.limit = limit
        self.op = op
        self.op_function = OPERATION[op]

    def stop(self) -> bool:
        return self.op_function(self.check_field.value, self.limit)


class VariableLimitCondition(SimCondition):
    """
    Stop if check_field.value >= limit_field.value.
    """

    def __init__(self, check_field: SimField, limit_field: SimField, op: str):
        self.check_field = check_field
        self.limit_field = limit_field
        self.op = op
        self.op_function = OPERATION[op]

    def stop(self) -> bool:
        return self.op_function(self.check_field.value, self.limit_field.value)


class FieldStateCondition(SimCondition):
    """
    Stop if check_field.value in stopping_values.
    """

    def __init__(self, check_field: SimField, stopping_values: Iterable[State]):
        self.check_field = check_field
        self.stopping_values = set(stopping_values)

    def stop(self) -> bool:
        return self.check_field.value in self.stopping_values
