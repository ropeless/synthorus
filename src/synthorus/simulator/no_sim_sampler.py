from synthorus.simulator.sim_entity import SimEntity, SimSampler
from synthorus.simulator.sim_field import SimFieldUpdate
from synthorus.simulator.sim_field_updaters import NO_UPDATE


class _NoSampler(SimSampler):
    """
    A dummy sampler that does not update any fields.
    """

    def get_updater(self, rv_name: str) -> SimFieldUpdate:
        return NO_UPDATE

    def start_stream(self, entity: SimEntity) -> None:
        return None

    def next(self) -> None:
        pass


NO_SIM_SAMPLER = _NoSampler()
