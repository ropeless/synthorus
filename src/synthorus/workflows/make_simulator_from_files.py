from os import PathLike
from pathlib import Path
from typing import Dict, Mapping

from ck.pgm import PGM, RandomVariable, RVMap
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import DEFAULT_PGM_COMPILER

from synthorus.model.model_index import ModelIndex, EntityIndex, AncestorConditionsIndex
from synthorus.simulator.make_simulator_from_simulator_spec import make_simulator_from_simulator_spec
from synthorus.simulator.pgm_sim_sampler import PGMSimSampler
from synthorus.simulator.sim_entity import SimSampler
from synthorus.simulator.simulator import Simulator
from synthorus.simulator.simulator_spec import SimulatorSpec, SimEntitySpec
from synthorus.utils.print_function import PrintFunction
from synthorus.utils.stop_watch import timer
from synthorus.workflows.file_names import SIMULATOR_SPEC_NAME, ENTITY_MODELS, MODEL_INDEX_NAME
from synthorus.workflows.load_entity_pgm import load_entity_pgm


def make_simulator_from_files(
        model_definition_directory: PathLike,
        *,
        log: PrintFunction = print,
) -> Simulator:
    """
    Construct an instance of Simulator from the given model definition directory.

    Args:
        model_definition_directory: where to read files.
        log: a destination for log messages
    """
    log(f'make_simulator_from_files started')

    model_definition_directory: Path = Path(model_definition_directory)

    with open(model_definition_directory / SIMULATOR_SPEC_NAME) as f:
        sim_spec: SimulatorSpec = SimulatorSpec.model_validate_json(f.read())

    with open(model_definition_directory / MODEL_INDEX_NAME) as f:
        model_index: ModelIndex = ModelIndex.model_validate_json(f.read())

    with timer('make samplers', logger=log):
        samplers = _make_samplers(sim_spec, model_index, model_definition_directory / ENTITY_MODELS, log)

    log('make simulator')
    simulator = make_simulator_from_simulator_spec(sim_spec, samplers)

    log('make_simulator_from_files completed')
    return simulator


def _make_samplers(
        sim_spec: SimulatorSpec,
        model_index: ModelIndex,
        pgms_path: Path,
        log: PrintFunction,
) -> Dict[str, SimSampler]:
    """
    Returns a dictionary mapping the entity name to a SimSampler object.
    """
    result: Dict[str, SimSampler] = {}

    entity_name: str
    entity_spec: SimEntitySpec

    for entity_name, entity_spec in sim_spec.entities.items():
        entity_index: EntityIndex = model_index.entities[entity_name]
        if len(entity_index.sampled_fields) > 0:
            result[entity_name] = _make_sampler(entity_name, entity_index, pgms_path, log)
    return result


def _make_sampler(
        entity_name: str,
        entity_index: EntityIndex,
        pgms_path: Path,
        log: PrintFunction,
) -> PGMSimSampler:
    """
    Make one PGMSimSampler from the model file.
    1) Load the model file (PGM),
    2) compile the model to a wmc,
    3) wrap the wmc in a PGMSimSampler.
    """
    log(f'loading PGM: {entity_name}')
    pgm: PGM = load_entity_pgm(pgms_path, entity_name)

    log(f'compiling PGM to cct')
    pgm_cct: PGMCircuit = DEFAULT_PGM_COMPILER(pgm)

    log(f'compiling cct to wmc')
    wmc = WMCProgram(pgm_cct)

    log(f'making PGMSimSampler')
    rv_map = RVMap(pgm)
    condition: AncestorConditionsIndex
    conditions: Mapping[RandomVariable, str] = {
        rv_map(condition.rv): condition.field
        for condition in entity_index.ancestor_conditions
    }
    return PGMSimSampler(wmc, conditions=conditions)
