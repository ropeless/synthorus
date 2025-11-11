from typing import Dict

from synthorus.simulator.make_simulator_from_simulator_spec import make_simulator_from_simulator_spec
from synthorus.simulator.sim_entity import SimSampler
from synthorus.simulator.sim_recorder import DebugRecorder
from synthorus.simulator.simulator import Simulator
from synthorus.simulator.simulator_spec import (
    SimulatorSpec
)
from synthorus_demos.simulator import example_simulator_spec

# This example JSON data was taken from the output of
# `synthorus_demos.simulator.demo_simulator_spec`.
JSON: str = """
{
  "parameters": {
    "number_of_patients": 10,
    "time_limit": 100
  },
  "entities": {
    "patient": {
      "parent": null,
      "sampler": "patient_sampler",
      "id_field_name": "_id_",
      "count_field_name": "_count_",
      "foreign_field_name": null,
      "fields": {
        "age": {
          "type": "sample",
          "rv_name": "patient_age"
        },
        "in_database": {
          "type": "constant",
          "value": true
        },
        "decade": {
          "type": "function",
          "initial_value": 0,
          "inputs": [
            "age"
          ],
          "function": "int(age / 10) + 1"
        }
      },
      "cardinality": [
        {
          "type": "variable",
          "field": "_count_",
          "op": ">=",
          "limit_field": "number_of_patients"
        }
      ]
    },
    "event": {
      "parent": "patient",
      "sampler": "event_sampler",
      "id_field_name": "_id_",
      "count_field_name": "_count_",
      "foreign_field_name": "_patient__id_",
      "fields": {
        "type": {
          "type": "sample",
          "rv_name": "event_type"
        },
        "duration": {
          "type": "sample",
          "rv_name": "event_duration"
        },
        "duration_since_last": {
          "type": "sample",
          "rv_name": "event_duration_since_last"
        },
        "time": {
          "type": "sum",
          "initial_value": 0,
          "inputs": [
            "duration",
            "duration_since_last"
          ],
          "add_self": true
        }
      },
      "cardinality": [
        {
          "type": "fixed",
          "field": "time",
          "op": ">=",
          "limit": 99
        },
        {
          "type": "variable",
          "field": "time",
          "op": ">=",
          "limit_field": "time_limit"
        },
        {
          "type": "states",
          "field": "type",
          "states": [
            "DEATH"
          ]
        }
      ]
    }
  }
}
"""


def main() -> None:
    """
    Demonstrate many of the components of a Synthorus simulation
    using a simulator specification.
    """
    # ===================================
    #  Create simulation
    # ===================================

    simulator_spec: SimulatorSpec = SimulatorSpec.model_validate_json(JSON)

    samplers: Dict[str, SimSampler] = example_simulator_spec.make_samplers()

    sim: Simulator = make_simulator_from_simulator_spec(simulator_spec, samplers)

    # ===================================
    #  Run simulation
    # ===================================
    sim.run(DebugRecorder())


if __name__ == '__main__':
    main()
