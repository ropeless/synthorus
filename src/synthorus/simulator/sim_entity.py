from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Dict, List, Tuple, Mapping, Iterator

from ck.pgm import State

from synthorus.error import SynthorusError
from synthorus.simulator.condition_spec import DEFAULT_OP
from synthorus.simulator.sim_condition import SimCondition, FixedLimitCondition, VariableLimitCondition, \
    FieldStateCondition
from synthorus.simulator.sim_field import SimField, SimFieldUpdate
from synthorus.simulator.sim_field_updaters import CopyUpdate, INCR_UPDATE, NO_UPDATE
from synthorus.simulator.sim_record import SimRecord


class SimEntity(Mapping[str, SimField]):

    def __init__(
            self,
            name: str,
            parent: Optional[SimEntity],
            sampler: SimSampler,
            id_field_name: str,
            count_field_name: str,
            foreign_id_field_name: str,
    ):
        self._name: str = name
        self._parent: Optional[SimEntity] = parent
        self._sampler: SimSampler = sampler
        self._fields: Dict[str, SimField] = OrderedDict()
        self._field_names: Tuple[str, ...] = ()  # field names, in the order they were added
        self._conditions: List[SimCondition] = []  # stopping conditions when generating records for a parent record

        # Add fields (id field first)
        self._id_field: SimField = self.add_field(id_field_name, update=INCR_UPDATE)
        self._count_field: SimField = self.add_field(count_field_name, update=INCR_UPDATE)
        self._foreign_field: Optional[SimField] = None
        if parent is not None:
            self._foreign_field = self.add_field(foreign_id_field_name, update=CopyUpdate(parent.id_field))

        # These are set in the initialise method.
        self._reset_values: List[State] = []
        self._fields_to_reset: List[State] = []

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._fields)

    def __iter__(self) -> Iterator[str]:
        """
        Iterates over the field names of this entity, in the order in which they were added.
        """
        return iter(self._fields)

    def __getitem__(self, field_name: str) -> SimField:
        return self._fields[field_name]

    def get(self, field_name: str, default: Optional[SimField] = None) -> Optional[SimField]:
        return self._fields.get(field_name, default)

    @property
    def parent(self) -> Optional[SimEntity]:
        return self._parent

    @property
    def id_field(self) -> SimField:
        return self._id_field

    @property
    def count_field(self) -> SimField:
        return self._count_field

    @property
    def foreign_field(self) -> Optional[SimField]:
        return self._foreign_field

    def add_field(self, field_name: str, *, value=None, update: SimFieldUpdate = NO_UPDATE) -> SimField:
        if field_name in self._fields.keys():
            raise SynthorusError(f'field {field_name!r} already exists in entity {self.name!r}')
        field = SimField(field_name, value, update)
        self._fields[field_name] = field
        self._field_names += (field_name,)
        return field

    def add_field_sampled(self, field_name: str, rv_name: Optional[str] = None) -> SimField:
        """
        Add a field that is updated from the entity's sampler.
        If no rv_name is provided, the field_name is used.
        This calls:
            self.add_field(field_name, update=self.sampler.get_updater(rv_name))
        """
        if rv_name is None:
            rv_name = field_name
        return self.add_field(field_name, update=self._sampler.get_updater(rv_name))

    def add_cardinality_condition(self, condition: SimCondition) -> None:
        self._conditions.append(condition)

    def add_cardinality_fixed_count(self, limit: int) -> None:
        """
        Stop adding records when cardinality reaches limit.
        """
        condition = FixedLimitCondition(self._count_field, limit, DEFAULT_OP)
        self.add_cardinality_condition(condition)

    def add_cardinality_fixed_limit(self, test_field: SimField, limit: State, op: str = DEFAULT_OP) -> None:
        """
        Stop adding records when test_field reaches limit.
        """
        condition = FixedLimitCondition(test_field, limit, op)
        self.add_cardinality_condition(condition)

    def add_cardinality_variable_count(self, limit_field: SimField) -> None:
        """
        Stop adding records when cardinality reaches limit_field.
        """
        condition = VariableLimitCondition(self._count_field, limit_field, DEFAULT_OP)
        self.add_cardinality_condition(condition)

    def add_cardinality_variable_limit(self, test_field: SimField, limit_field: SimField, op: str = DEFAULT_OP) -> None:
        """
        Stop adding records when test_field reaches limit_field.
        """
        condition = VariableLimitCondition(test_field, limit_field, op)
        self.add_cardinality_condition(condition)

    def add_cardinality_field_state(self, test_field: SimField, *stopping_values: State) -> None:
        """
        The record cardinality is limited when test_field.value in stopping_values.
        """
        condition = FieldStateCondition(test_field, stopping_values)
        self.add_cardinality_condition(condition)

    # ===========================================================================
    # Simulation management methods...
    # ===========================================================================

    def initialise(self, id_field_value: int) -> None:
        """
        Called at the start of the simulation to prepare the entity.
        This records the initial values used by reset_values.
        """
        self.id_field.value = id_field_value
        self._count_field.value = 0

        # Fields for reset - all but the id field (first field in the ordered dict).
        self._fields_to_reset = list(self._fields.values())[1:]
        self._reset_values = [field.value for field in self._fields_to_reset]

    def reset_fields(self) -> None:
        """
        Reset all fields to their initial values, as registered when `initialise` is called.

        Assumes:
            `initialise` has been called.
        """
        for field, value in zip(self._fields_to_reset, self._reset_values):
            field.value = value

    def start(self) -> None:
        """
        Called at the start of a run of record generations.

        Assumes:
            `initialise` has been called.
        """
        # Reset fields that need to be reset
        self.reset_fields()

        # Prepare the sampler to issue samples.
        self._sampler.start_stream(self)

    def stop(self) -> bool:
        """
        Does any condition indicate a simulation loop should stop.
        If no conditions registered, then return True if record count
        is > 0, which is equivalent to cardinality one-one.
        """
        if len(self._conditions) == 0:
            return self._count_field.value > 0
        return any(condition.stop() for condition in self._conditions)

    def update(self) -> SimRecord:
        """
        Update all fields in this entity.

        This method will first call `next` on its sampler,
        then call `update` on each of its fields.

        Assumes:
            `start` has been called.

        Returns:
            the record of updated fields.
        """
        self._sampler.next()
        record: SimRecord = {
            field.name: field.update()
            for field in self._fields.values()
        }
        return record


class SimSampler(ABC):
    """
    Abstract base class for a sampler associated with an entity.
    """

    @abstractmethod
    def get_updater(self, rv_name: str) -> SimFieldUpdate:
        """
        Return a field updater that sets its destination field from a sample
        of the named random variable.
        """

    def start_stream(self, entity: SimEntity) -> None:
        """
        Prepare the sampler to provide samples for the given entity.
        This is called before a stream of samples will be drawn.
        If any ancestor entity is updated, this method will be called again
        to allow for setting up any sampler conditioning.
        """

    @abstractmethod
    def next(self) -> None:
        """
        Advance to the next sample.
        This will update any issued updaters so they know about the next sample.
        """
