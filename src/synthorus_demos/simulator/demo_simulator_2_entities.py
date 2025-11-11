from ck.pgm import PGM
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import DEFAULT_PGM_COMPILER

from synthorus.simulator.pgm_sim_sampler import PGMSimSampler
from synthorus.simulator.sim_field_updaters import SumUpdate
from synthorus.simulator.sim_recorder import DebugRecorder
from synthorus.simulator.simulator import Simulator


def make_simulator() -> Simulator:
    # ===================================
    #  Create probabilistic model
    # ===================================
    pgm = PGM()

    patient_age = pgm.new_rv('patient_age', ('young', 'middle_aged', 'old'))
    event_type = pgm.new_rv('event_type', ('ED', 'AP', 'GP', 'DEATH'))
    event_duration = pgm.new_rv('event_duration', range(1, 10))
    event_duration_since_last = pgm.new_rv('event_duration_since_last', range(1, 10))

    pgm.new_factor(patient_age).set_dense().set_uniform()
    pgm.new_factor(event_type, patient_age).set_cpt().set(
        # patient_age  ED,   AP,  GP,  DEATH
        ((0,), (0.0, 0.0, 1.0, 0.0)),  # young
        ((1,), (0.5, 0.0, 0.5, 0.0)),  # middle_aged
        ((2,), (0.3, 0.3, 0.3, 0.1)),  # old
    )
    pgm.new_factor(event_duration).set_dense().set_uniform()
    pgm.new_factor(event_duration_since_last).set_dense().set_uniform()

    wmc = WMCProgram(DEFAULT_PGM_COMPILER(pgm))

    # ===================================
    #  Create simulation
    # ===================================
    sim = Simulator()

    # Parameters
    number_of_patients = sim.add_parameter('number_of_patients', 10)  # SimField: name = number_of_patients, value = 10
    time_limit = sim.add_parameter('time_limit', 100)  # SimField: name = time_limit, value = 100

    # Samplers
    # In this demo, there are two entities, each with its own sampler, but they happen to share one PGM.
    # Normally each entity would have its own PGM to be more efficient, but not in this demo.
    #
    patient_sampler = PGMSimSampler(wmc)  # no conditioned random variables
    event_sampler = PGMSimSampler(wmc, conditions={patient_age: 'age'})  # rv patient_age is conditioned on field age

    # Entity - patient
    patient = sim.add_entity('patient', sampler=patient_sampler)
    patient.add_field_sampled(field_name='age', rv_name='patient_age')
    patient.add_cardinality_variable_count(number_of_patients)

    # Entity - event
    event = sim.add_entity('event', parent=patient, foreign_field_name='_patient_id', sampler=event_sampler)
    #
    field_event_type = event.add_field_sampled('type', 'event_type')
    field_duration = event.add_field_sampled('duration', 'event_duration')
    field_duration_since_last = event.add_field_sampled('duration_since_last', 'event_duration_since_last')
    #
    time_update = SumUpdate(field_duration, field_duration_since_last, include_self=True)
    field_time = event.add_field('time', value=0, update=time_update)
    #
    event.add_cardinality_variable_limit(field_time, time_limit)
    event.add_cardinality_field_state(field_event_type, 'DEATH')

    return sim


def main() -> None:
    """
    Demonstrate many of the components of a Synthorus simulation
    by manually creating a Simulator object with two related entities.
    """
    sim: Simulator = make_simulator()
    sim.run(DebugRecorder())


if __name__ == '__main__':
    main()
