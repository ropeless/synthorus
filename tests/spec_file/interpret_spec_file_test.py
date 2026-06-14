from typing import List

from synthorus.model.defaults import DEFAULT_ENTITY_NAME, DEFAULT_ID_FIELD, DEFAULT_COUNT_FIELD, DEFAULT_NAME, \
    DEFAULT_AUTHOR, DEFAULT_COMMENT, DEFAULT_RNG_N
from synthorus.model.model_spec import ModelSpec, ModelEntitySpec
from synthorus.simulator.condition_spec import ConditionSpec, ConditionSpecFixedLimit, ConditionSpecVariableLimit, \
    ConditionSpecStates
from synthorus.spec_file import keys
from synthorus.spec_file.interpret_spec_file import interpret_spec_file
from tests.helpers.unittest_fixture import Fixture, test_main


class InterpretSpecFileTest(Fixture):

    def test_minimal(self) -> None:
        spec = {
            keys.datasources: {}  # must at least define datasources
        }

        model_spec: ModelSpec = interpret_spec_file(spec)

        # Default metadata
        self.assertEqual(model_spec.name, DEFAULT_NAME)
        self.assertEqual(model_spec.author, DEFAULT_AUTHOR)
        self.assertEqual(model_spec.comment, DEFAULT_COMMENT)
        self.assertEqual(model_spec.roots, [])
        self.assertEqual(model_spec.rng_n, DEFAULT_RNG_N)

        # No datasources, random variables, cross-tables or parameters
        self.assertEqual(model_spec.datasources, {})
        self.assertEqual(model_spec.rvs, {})
        self.assertEqual(model_spec.crosstabs, {})
        self.assertEqual(model_spec.parameters, {})

        # The default entity with no parent and no fields
        self.assertEqual(list(model_spec.entities.keys()), [DEFAULT_ENTITY_NAME])
        entity_spec: ModelEntitySpec = model_spec.entities[DEFAULT_ENTITY_NAME]

        self.assertIsNone(entity_spec.parent)
        self.assertIsNone(entity_spec.foreign_field_name)
        self.assertEqual(entity_spec.id_field_name, DEFAULT_ID_FIELD)
        self.assertEqual(entity_spec.count_field_name, DEFAULT_COUNT_FIELD)
        self.assertEqual(entity_spec.fields, {})
        self.assertEqual(entity_spec.cardinality, [])

    def test_cardinalities(self) -> None:
        spec = {
            keys.data_format: keys.csv,
            keys.sensitivity: 0,
            keys.min_cell_size: 0,
            keys.states: keys.infer_distinct,
            keys.datasources: {
                'my_datasource': {
                    keys.inline: """
                    A, B, C
                    1, 2, 3
                    4, 5, 6
                    """
                }
            },
            keys.entities: {
                'my_entity': {
                    keys.rvs: ['A', 'B', 'C'],
                    keys.cardinality: [
                        10,
                        'A',
                        {keys.field: 'B', keys.limit: 3},
                        {keys.field: 'B', keys.limit: 'C'},
                        {keys.field: 'B', keys.state: 1},
                        {keys.field: 'C', keys.state: [0, 1]},
                    ],
                },
            }
        }

        model_spec: ModelSpec = interpret_spec_file(spec)

        cardinalities: List[ConditionSpec] = model_spec.entities['my_entity'].cardinality

        self.assertEqual(len(cardinalities), 6)

        test_cardinality = cardinalities[0]
        self.assertTrue(isinstance(test_cardinality, ConditionSpecFixedLimit))
        self.assertEqual(test_cardinality.field, DEFAULT_COUNT_FIELD)
        self.assertEqual(test_cardinality.op, '>=')
        self.assertEqual(test_cardinality.limit, 10)

        test_cardinality = cardinalities[1]
        self.assertTrue(isinstance(test_cardinality, ConditionSpecVariableLimit))
        self.assertEqual(test_cardinality.field, DEFAULT_COUNT_FIELD)
        self.assertEqual(test_cardinality.op, '>=')
        self.assertEqual(test_cardinality.limit_field, 'A')

        test_cardinality = cardinalities[2]
        self.assertTrue(isinstance(test_cardinality, ConditionSpecFixedLimit))
        self.assertEqual(test_cardinality.field, 'B')
        self.assertEqual(test_cardinality.op, '>=')
        self.assertEqual(test_cardinality.limit, 3)

        test_cardinality = cardinalities[3]
        self.assertTrue(isinstance(test_cardinality, ConditionSpecVariableLimit))
        self.assertEqual(test_cardinality.field, 'B')
        self.assertEqual(test_cardinality.op, '>=')
        self.assertEqual(test_cardinality.limit_field, 'C')

        test_cardinality = cardinalities[4]
        self.assertTrue(isinstance(test_cardinality, ConditionSpecStates))
        self.assertEqual(test_cardinality.field, 'B')
        self.assertEqual(test_cardinality.states, [1])

        test_cardinality = cardinalities[5]
        self.assertTrue(isinstance(test_cardinality, ConditionSpecStates))
        self.assertEqual(test_cardinality.field, 'C')
        self.assertEqual(test_cardinality.states, [0, 1])


if __name__ == '__main__':
    test_main()
