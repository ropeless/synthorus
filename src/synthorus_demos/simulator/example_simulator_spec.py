from typing import Dict

from ck.pgm import State, PGM
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import DEFAULT_PGM_COMPILER

from synthorus.simulator.condition_spec import ConditionSpecVariableLimit, ConditionSpecFixedLimit, ConditionSpecStates
from synthorus.simulator.pgm_sim_sampler import PGMSimSampler
from synthorus.simulator.sim_entity import SimSampler
from synthorus.simulator.simulator_spec import (
    SimulatorSpec, SimEntitySpec, SampleSpec,
    ConstantSpec, SumSpec, FunctionSpec
)


def make_simulator_spec() -> SimulatorSpec:
    """
    Make a simulator spect that will exercise much of
    the available functionality.
    """
    parameters: Dict[str, State] = {
        'number_of_patients': 10,
        'time_limit': 100,
    }

    patient = SimEntitySpec(
        sampler='patient_sampler',
        fields={
            'age': SampleSpec(rv_name='patient_age'),
            'in_database': ConstantSpec(value=True),
            'decade': FunctionSpec(
                inputs=['age'],
                function='int(age / 10) + 1',
            ),
        },
        cardinality=[
            ConditionSpecVariableLimit(field='_count_', limit_field='number_of_patients'),
        ],
    )

    event = SimEntitySpec(
        sampler='event_sampler',
        parent='patient',
        foreign_field_name='_patient__id_',
        fields={
            'type': SampleSpec(rv_name='event_type'),
            'duration': SampleSpec(rv_name='event_duration'),
            'duration_since_last': SampleSpec(rv_name='event_duration_since_last'),
            'time': SumSpec(
                inputs=['duration', 'duration_since_last'],
                add_self=True,
            ),
        },
        cardinality=[
            ConditionSpecFixedLimit(field='time', limit=99),
            ConditionSpecVariableLimit(field='time', limit_field='time_limit'),
            ConditionSpecStates(field='type', states=['DEATH']),
        ],
    )

    entities: Dict[str, SimEntitySpec] = {
        'patient': patient,
        'event': event,
    }

    simulator_spec = SimulatorSpec(parameters=parameters, entities=entities)

    return simulator_spec


def make_samplers() -> Dict[str, SimSampler]:
    """
    This function will make samplers 'patient_sampler' and 'event_sampler'
    suitable for use with the above simulator spec.
    """

    # ===================================
    #  Create probabilistic model
    # ===================================
    pgm = PGM()

    patient_age = pgm.new_rv('patient_age', 100)
    patient_age_group = pgm.new_rv('patient_age_group', ('young', 'middle_aged', 'old'))
    event_type = pgm.new_rv('event_type', ('ED', 'AP', 'GP', 'DEATH'))
    event_duration = pgm.new_rv('event_duration', range(1, 10))
    event_duration_since_last = pgm.new_rv('event_duration_since_last', range(1, 10))

    def age_group_function(age: int) -> int:
        if age < 40:
            return 0  # young
        elif age < 60:
            return 1  # middle_aged
        else:
            return 2  # old

    pgm.new_factor(patient_age).set_dense().set_uniform()
    pgm.new_factor_functional(age_group_function, patient_age_group, patient_age)
    pgm.new_factor(event_type, patient_age_group).set_cpt().set(
        # patient_age  ED,   AP,  GP,  DEATH
        ((0,), (0.0, 0.0, 1.0, 0.0)),  # young
        ((1,), (0.5, 0.0, 0.5, 0.0)),  # middle_aged
        ((2,), (0.3, 0.3, 0.3, 0.1)),  # old
    )
    pgm.new_factor(event_duration).set_dense().set_uniform()
    pgm.new_factor(event_duration_since_last).set_dense().set_uniform()

    wmc = WMCProgram(DEFAULT_PGM_COMPILER(pgm))

    # ===================================
    # Map sampler names to SimSamplers
    # ===================================
    samplers = {
        'patient_sampler': PGMSimSampler(wmc),
        'event_sampler': PGMSimSampler(wmc, conditions={patient_age: 'age'}),
    }

    return samplers
