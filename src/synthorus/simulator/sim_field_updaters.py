from __future__ import annotations

from typing import Iterable, Union, Callable, Sequence, TypeAlias, Optional, List

from ck.pgm import State

from synthorus.error import SynthorusError
from synthorus.simulator.sim_field import SimFieldUpdate, SimField
from synthorus.utils.parse_formula import parse_formula


class NoUpdate(SimFieldUpdate):
    """
    This updater does nothing. I.e. the field remains constant.
    """

    def update(self, dest_field: SimField) -> None:
        pass


class IncrementUpdate(SimFieldUpdate):
    """
    Update a field by incrementing it.
    """

    def __init__(self, increment=1):
        self.increment = increment

    def update(self, dest_field: SimField) -> None:
        dest_field.value += self.increment


class CopyUpdate(SimFieldUpdate):
    """
    Update a field by copying a source field.
    """

    def __init__(self, source_field: SimField):
        self.source_field = source_field

    def update(self, dest_field: SimField) -> None:
        dest_field.value = self.source_field.value


class SumUpdate(SimFieldUpdate):
    """
    Update a field by summing source fields (and constant).

    This is a generalisation of IncrementUpdate and CopyUpdate.
    I.e.,
        `SumUpdate(dest_field, constant=x)` is equivalent to `IncrementUpdate(increment=x)`,
        `SumUpdate(source_field, constant=0)` is equivalent to `CopyUpdate(source_field)`.
    """

    def __init__(self, *source_fields: SimField, include_self=False, constant=0):
        self.constant = constant
        self.source_fields = source_fields
        self.include_self = include_self

    def update(self, dest_field: SimField) -> None:
        """
        dest_field.value := sum(source field values and the constant)
        """
        value = sum((field.value for field in self.source_fields), start=self.constant)
        if self.include_self:
            dest_field.value += value
        else:
            dest_field.value = value


UpdateFunction: TypeAlias = Callable[[State, ...], State]


class FunctionUpdate(SimFieldUpdate):
    """
    Update a field by executing a function, with arguments for source field values.

    This is a generalisation of SumUpdate.
    E.g.

    ```
    def f(a, b, c):
        return sum(a, b, c) + 123

    FunctionUpdate(f, [field_a, field_b, field_c])
    ```

    It is equivalent to `SumUpdate(field_a, field_b, field_c, constant=123)`.

    """

    def __init__(self, func: Union[str, UpdateFunction], fields: Iterable[SimField], prev_value: Optional[str] = None):
        """
        Construct a field update function that is a function of the given fields.

        Args:
            func: The function to calculate the value of the destination field. Can be a string or a callable.
                If a string, then `func` should be a python expression containing names of argument fields.
            fields: The fields to use as input arguments to the function.
            prev_value: The name of the function argument to use for the previous value of the destination field.
                'prev_value' must be different from any field name. If `func` is a callable with positional arguments,
                then the previous field value will be passed as the last argument, no matter what name is passed
                for `prev_value`.
        """
        self._fields: Sequence[SimField] = tuple(fields)
        self._prev_value: Optional[str] = prev_value

        in_names: List[str] = [field.name for field in self._fields]
        if len(in_names) != len(set(in_names)):
            raise SynthorusError(f'duplicate names in function input fields: {in_names}')
        if prev_value is not None:
            if prev_value in in_names:
                raise SynthorusError(f'self argument name {prev_value!r} cannot be the same as any field name')
            in_names.append(prev_value)

        self._func: UpdateFunction
        if isinstance(func, str):
            self._func = parse_formula(func, in_names)
        else:
            self._func = func

    def update(self, dest_field: SimField) -> None:
        input_values = tuple(field.value for field in self._fields)
        if self._prev_value is not None:
            input_values += dest_field.value,

        value = self._func(*input_values)
        dest_field.value = value


# Standard updaters
NO_UPDATE: SimFieldUpdate = NoUpdate()
INCR_UPDATE: SimFieldUpdate = IncrementUpdate()
