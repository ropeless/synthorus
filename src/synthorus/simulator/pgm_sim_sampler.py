from itertools import repeat
from types import MappingProxyType
from typing import Mapping, List, Iterator, Sequence, Optional, Dict

from ck.pgm import RandomVariable, Indicator, Instance, State
from ck.pgm_circuit.wmc_program import WMCProgram

from synthorus.error import SynthorusError
from synthorus.simulator.sim_entity import SimEntity, SimSampler
from synthorus.simulator.sim_field import SimFieldUpdate, SimField

NO_CONDITIONS: Mapping[RandomVariable, str] = MappingProxyType({})


class PGMSimSampler(SimSampler):
    """
    A simulator sampler based on a CK PGM sampler.
    It also manages conditioning based on simulator fields.

    Defining which rvs are for conditioning is done at construction time.
    Defining the valued for conditioning is done in method `initialise`.
    Samples are provided by method `get_sample_stream`.
    """

    def __init__(
            self,
            wmc: WMCProgram,
            conditions: Mapping[RandomVariable, str] = NO_CONDITIONS
    ):
        """
        Make a SimSampler from a CK direct sampler (from a WMC program of a PGM).

        Optionally samples will be conditioned on rvs of the PGM, with condition values
        coming from fields of a simulation (i.e., SimFields).

        The parameter `condition` say which rvs of the PGM are used for conditioning
        and the names of the fields that should be used for the condition values.

        Condition rvs are not available for sampling.
        Condition fields must be available in entity parents.

        Args:
            wmc: the weighted model counter for a PGM.
            conditions: a mapping from a random variable to a field name. The random variable will have
                its value supplied prior to samping the PGM. The field is a field from an ancestor that
                that will supply a value for the random variable.
        """

        self._ck_wmc: WMCProgram = wmc
        self._ck_sample_rvs: List[RandomVariable] = []
        self._ck_samples: Iterator[Instance] = repeat(())
        self._ck_sample: Instance = ()
        self._name_to_rv: Dict[str, RandomVariable] = {rv.name: rv for rv in wmc.rvs if rv not in conditions.keys()}
        self._updaters: Dict[str, SimFieldUpdate] = {}
        self._conditions: Mapping[RandomVariable, str] = conditions
        self._conditioners: List[_Conditioner] = []
        self._entity: Optional[SimEntity] = None

        # Check that all condition rvs are actually part of the wmc
        all_rvs = set(wmc.rvs)
        for rv in conditions.keys():
            if rv not in all_rvs:
                raise SynthorusError(f'condition random variable not in given WMC: {rv}')

    def get_updater(self, rv_name: str) -> SimFieldUpdate:
        """
        The returned SimFieldUpdate object will update a simulator field value
        based on the value of the given named random variable.

        The returned SimFieldUpdate knows how to access the sample value from this sampler.
        """
        rv = self._name_to_rv.get(rv_name)
        if rv is None:
            raise SynthorusError(f'random variable not available for sampling: {rv_name}')
        updater = self._updaters.get(rv_name)
        if updater is None:
            idx = len(self._ck_sample_rvs)
            self._ck_sample_rvs.append(rv)
            updater = _SamplerUpdate(self, idx, rv.states)
            self._updaters[rv_name] = updater
        return updater

    def start_stream(self, entity: SimEntity) -> None:
        """
        Set up the values for the conditioning RVs, based on field
        values in the given entity (or its parents).
        """
        # Make sure we have the right conditioners for the given entity.
        if self._entity is not entity:
            self._conditioners = [
                _Conditioner(rv, field_name, entity)
                for rv, field_name in self._conditions.items()
            ]
            self._entity = entity

        # Apply the conditions
        condition = tuple(
            conditioner.get_condition()
            for conditioner in self._conditioners
        )

        # Get a conditioned sample stream
        sampler = self._ck_wmc.sample_direct(rvs=self._ck_sample_rvs, condition=condition)
        self._ck_samples = iter(sampler)

    def next(self) -> None:
        self._ck_sample = next(self._ck_samples)


class _SamplerUpdate(SimFieldUpdate):
    """
    Update a field by copying a sample value from a PGMSampler.
    """

    def __init__(self, sim_sampler: PGMSimSampler, sample_idx: int, rv_states: Sequence[State]):
        self._sim_sampler = sim_sampler
        self._sample_idx = sample_idx
        self._rv_states = rv_states

    def update(self, dest_field: SimField):
        # noinspection PyProtectedMember
        sample: Instance = self._sim_sampler._ck_sample
        dest_field.value = self._rv_states[sample[self._sample_idx]]


class _Conditioner:
    """
    Helper class for getting a conditioning indicator for a random variable.
    """

    def __init__(self, rv: RandomVariable, field_name: str, entity: SimEntity):
        parent: Optional[SimEntity] = entity.parent
        if parent is None:
            raise SynthorusError(f'the entity has no parent so cannot do conditioned sampling: {entity.name}')
        fields: List[SimField] = []
        while parent is not None:
            if field_name in parent:
                fields.append(parent[field_name])
            parent = parent.parent
        if len(fields) == 0:
            raise SynthorusError(f'cannot find conditioning field: {field_name}, rv: {rv.name}')
        if len(fields) > 1:
            raise SynthorusError(f'multiple matching conditioning fields: {field_name}, rv: {rv.name}')

        self.field: SimField = fields[0]  # The field providing a value
        self.rv: RandomVariable = rv  # The random variable being conditioned on the value

    def get_condition(self) -> Indicator:
        state: State = self.field.value
        idx: int = self.rv.state_idx(state)
        return self.rv[idx]
