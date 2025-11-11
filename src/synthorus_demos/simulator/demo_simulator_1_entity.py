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

    patient_age = pgm.new_rv('patient_age', ('young', 'middle_aged', 'old'))

    pgm.new_factor(patient_age).set_dense().set_uniform()

    wmc = WMCProgram(DEFAULT_PGM_COMPILER(pgm))

    # ===================================
    #  Create simulation
    # ===================================
    sim = Simulator()

    # Parameters
    number_of_patients = sim.add_parameter('number_of_patients', 5)  # SimField: name = number_of_patients, value = 5

    # Samplers
    # In this demo, there are two entities, each with its own sampler, but they happen to share one PGM.
    # Normally each entity would have its own PGM to be more efficient, but not in this demo.
    #
    patient_sampler = PGMSimSampler(wmc)  # no conditioned random variables

    # Entity - patient
    patient = sim.add_entity('patient', sampler=patient_sampler)
    patient.add_field_sampled(field_name='age', rv_name='patient_age')
    patient.add_cardinality_variable_count(number_of_patients)

    return sim


def main() -> None:
    """
    Demonstrate a simple Synthorus simulation by manually
    creating a Simulator with one entity.
    """
    sim: Simulator = make_simulator()

    # This will end up emitting 10 patients. The cardinality of the patient
    # entity is 5, and the simulation is run twice.
    sim.run(DebugRecorder(), iterations=2)


if __name__ == '__main__':
    main()
