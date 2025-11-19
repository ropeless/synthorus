from synthorus.model.defaults import DEFAULT_ENTITY_NAME, DEFAULT_ID_FIELD, DEFAULT_COUNT_FIELD, DEFAULT_NAME, \
    DEFAULT_AUTHOR, DEFAULT_COMMENT, DEFAULT_RNG_N
from synthorus.model.model_spec import ModelSpec, ModelEntitySpec
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


if __name__ == '__main__':
    test_main()
