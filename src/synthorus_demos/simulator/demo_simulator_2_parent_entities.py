from synthorus.simulator.sim_recorder import DebugRecorder
from synthorus.simulator.simulator import Simulator


def make_simulator() -> Simulator:
    # ===================================
    #  Create simulation
    # ===================================
    sim = Simulator()

    # Entity 1
    e1 = sim.add_entity('e1')
    e1.add_cardinality_fixed_count(3)

    # Entity 2
    e2 = sim.add_entity('e2')
    e2.add_cardinality_fixed_count(4)

    # Entity 3
    foreign_key_fields = [('_e1_id', e1), ('_e2_id', e2)]
    e3 = sim.add_entity('e3', foreign_key_fields=foreign_key_fields)
    e3.add_cardinality_fixed_count(5)

    return sim


def main() -> None:
    """
    Show an entity that has two foreign keys.
    """
    sim: Simulator = make_simulator()
    sim.run(DebugRecorder(blank_line_between_entities=True))


if __name__ == '__main__':
    main()
