from __future__ import annotations

from abc import ABC, abstractmethod

from ck.pgm import State


class SimField:
    """
    A SimField relates to a single SimEntity.

    Each SimField has a name, a value, and an associated SimFieldUpdate object.

    The value can be updated (using the associated SimFieldUpdate object)
    by calling the `update` method.
    """

    def __init__(self, name: str, value: State, update: SimFieldUpdate):
        self._name: str = name
        self.value: State = value
        self._update: SimFieldUpdate = update

    @property
    def name(self) -> str:
        return self._name

    def update(self) -> State:
        """
        Update the filed value according to the associated SimFieldUpdate.

        Returns:
            the value of the field after being updated.
        """
        self._update.update(self)
        return self.value


class SimFieldUpdate(ABC):
    """
    Abstract method for updating a field of an entity during a simulation.

    A SimFieldUpdate is responsible for setting the value of a destination SimField.
    """

    @abstractmethod
    def update(self, dest_field: SimField, ) -> None:
        """
        Perform a field update.
        """
