from synthorus.model.model_spec import ModelSpec
from synthorus.simulator.make_simulator_from_simulator_spec import make_simulator_from_simulator_spec
from synthorus.simulator.make_simulator_spec_from_model_spec import make_simulator_spec_from_model_spec
from synthorus.simulator.sim_recorder import DebugRecorder
from synthorus.simulator.simulator import Simulator
from synthorus.simulator.simulator_spec import SimulatorSpec
from synthorus.spec_file.interpret_spec_file import interpret_spec_file


def main() -> None:
    spec_file_dict = {
        'datasources': {}
    }

    model_spec: ModelSpec = interpret_spec_file(spec_file_dict)

    print(model_spec.model_dump_json(indent=2))

    simulator_spec: SimulatorSpec = make_simulator_spec_from_model_spec(model_spec)
    simulator: Simulator = make_simulator_from_simulator_spec(simulator_spec, samplers={})

    print()
    print('-----------------------------------------------')
    simulator.run(DebugRecorder(), iterations=5)
    print('-----------------------------------------------')


if __name__ == '__main__':
    main()
