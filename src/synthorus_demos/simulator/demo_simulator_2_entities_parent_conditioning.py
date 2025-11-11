from ck.pgm import PGM
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import DEFAULT_PGM_COMPILER

from synthorus.simulator.pgm_sim_sampler import PGMSimSampler
from synthorus.simulator.sim_recorder import DebugRecorder
from synthorus.simulator.simulator import Simulator


def make_simulator() -> Simulator:
    # ===================================
    #  Create probabilistic model
    # ===================================
    pgm = PGM()

    rv1 = pgm.new_rv('rv1', 2)
    rv2 = pgm.new_rv('rv2', 4)

    pgm.new_factor(rv1).set_dense().set_uniform()
    pgm.new_factor(rv2, rv1).set_cpt().set(
        #       0,   1,   2,   3
        ((0,), (0.5, 0.5, 0.0, 0.0)),  # rv1 = 0  ==>  rv2 is 0 or 1
        ((1,), (0.0, 0.0, 0.5, 0.5)),  # rv1 = 1  ==>  rv2 is 2 or 3
    )
    wmc = WMCProgram(DEFAULT_PGM_COMPILER(pgm))

    # ===================================
    #  Create simulation
    # ===================================
    sim = Simulator()

    # Samplers
    # In this demo, there are two entities, each with its own sampler, but they happen to share one PGM.
    # Normally each entity would have its own PGM to be more efficient, but not in this demo.
    #
    s1 = PGMSimSampler(wmc)  # sampler for entity 1, no conditioned random variables
    s2 = PGMSimSampler(wmc, conditions={rv1: 'f1'})  # sampler for entity 1, rv1 is conditioned on field f1

    # Entity 1
    e1 = sim.add_entity('e1', sampler=s1)
    e1.add_field_sampled(field_name='f1', rv_name='rv1')
    e1.add_cardinality_fixed_count(8)

    # Entity 2
    e2 = sim.add_entity('e2', parent=e1, foreign_field_name='_e1_id', sampler=s2)
    e2.add_field_sampled(field_name='f2', rv_name='rv2')
    e2.add_cardinality_fixed_count(8)

    return sim


def main() -> None:
    """
    Show very clearly how a child entity samples can be
    conditioned on a parent entity.
    """
    sim: Simulator = make_simulator()
    sim.run(DebugRecorder(blank_line_between_entities=True))


if __name__ == '__main__':
    main()
