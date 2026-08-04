from synthorus.simulator.sim_field_updaters import FunctionUpdate
from synthorus.simulator.sim_recorder import DebugRecorder
from synthorus.simulator.simulator import Simulator


def make_simulator() -> Simulator:
    sim = Simulator()

    entity_1 = sim.add_entity('E1')
    entity_1.add_cardinality_fixed_count(5)  # emit multiple records within an ancestor context

    entity_1.add_field(
        'field',  # name of the field
        value=0,  # value prior to any updates (i.e., initial value for `prev_value`).
        update=FunctionUpdate(
            func='_count_ + prev_value',  # an expression representing the body of the function
            fields=[entity_1.count_field],  # argument to the function
            prev_value='prev_value',  # name in the `func` argument to use for the previous value of the field
        ),
    )

    return sim


def main() -> None:
    """
    Demonstrate how functional fields work.
    """
    sim: Simulator = make_simulator()
    sim.run(DebugRecorder(), iterations=2)


if __name__ == '__main__':
    main()
