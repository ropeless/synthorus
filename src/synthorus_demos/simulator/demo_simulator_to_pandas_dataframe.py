from typing import Dict

from synthorus.simulator.make_simulator_from_simulator_spec import make_simulator_from_simulator_spec
from synthorus.simulator.sim_entity import SimSampler
from synthorus.simulator.sim_recorder import PandasRecorder
from synthorus.simulator.simulator import Simulator
from synthorus.simulator.simulator_spec import SimulatorSpec
from synthorus_demos.simulator import example_simulator_spec


def main() -> None:
    # ===================================
    #  Create simulation
    # ===================================

    simulator_spec: SimulatorSpec = example_simulator_spec.make_simulator_spec()
    samplers: Dict[str, SimSampler] = example_simulator_spec.make_samplers()
    sim: Simulator = make_simulator_from_simulator_spec(simulator_spec, samplers)

    # ===================================
    #  Run simulation
    # ===================================
    dataframes = PandasRecorder()
    sim.run(dataframes)

    # ===================================
    #  Show results
    # ===================================
    for entity, dataframe in dataframes.items():
        print()
        print('------------------------------------')
        print(entity)
        print()
        print(dataframe.to_string())
    print('------------------------------------')
    print('Done.')


if __name__ == '__main__':
    main()
