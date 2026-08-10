from typing import List, Dict

import pandas as pd
from ck.pgm import State

from synthorus.dataset import Dataset
from synthorus.error import SpecFileError
from synthorus.model.defaults import DEFAULT_ENTITY_NAME, DEFAULT_ID_FIELD, DEFAULT_COUNT_FIELD, DEFAULT_NAME, \
    DEFAULT_AUTHOR, DEFAULT_COMMENT, DEFAULT_RNG_N
from synthorus.model.make_model_index import make_model_index
from synthorus.model.model_index import ModelIndex, RVIndex
from synthorus.model.model_spec import ModelSpec, ModelEntitySpec, ModelFieldSpec, ModelFieldSpecSum, \
    ModelCrosstabSpec, ModelRVSpec
from synthorus.model.noiser_spec import NoiserSpec, NoiserSpecNaiveLaplace, NoiserSpecDecompositionLaplace
from synthorus.simulator.condition_spec import ConditionSpec, ConditionSpecFixedLimit, ConditionSpecVariableLimit, \
    ConditionSpecStates
from synthorus.spec_file import keys
from synthorus.spec_file.interpret_spec_file import interpret_spec_file, load_spec_file
from synthorus.utils.string_extras import unindent
from tests.helpers.tmp_dir import tmp_dir
from tests.helpers.unittest_fixture import Fixture, test_main


class InterpretSpecFileTest(Fixture):

    def test_minimal(self) -> None:
        spec = {}

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

    def test_no_defaults_ok(self) -> None:
        spec = {}
        model_spec: ModelSpec = interpret_spec_file(spec, defaults={})

        # Default metadata
        self.assertEqual(model_spec.name, DEFAULT_NAME)
        self.assertEqual(model_spec.author, DEFAULT_AUTHOR)
        self.assertEqual(model_spec.comment, DEFAULT_COMMENT)
        self.assertEqual(model_spec.roots, [])
        self.assertEqual(model_spec.rng_n, DEFAULT_RNG_N)

    def test_no_defaults_error(self) -> None:
        spec = {
            keys.rvs: {'bad rvs': 42},
        }

        with self.assertRaises(SpecFileError) as context:
            _: ModelSpec = interpret_spec_file(spec, defaults={})
        err: SpecFileError = context.exception
        self.assertEqual(err.section, f'{keys.rvs}')

    def test_default_overrides(self) -> None:
        spec = {
            keys.name: 'a name',
            keys.author: 'an author',
            keys.comment: 'a comment',
            keys.roots: 'one_root',
            keys.rng_n: 1234,
        }

        model_spec: ModelSpec = interpret_spec_file(spec)

        self.assertEqual(model_spec.name, 'a name')
        self.assertEqual(model_spec.author, 'an author')
        self.assertEqual(model_spec.comment, 'a comment')
        self.assertEqual(model_spec.roots, ['one_root'])
        self.assertEqual(model_spec.rng_n, 1234)

    def test_two_roots(self) -> None:
        spec = {
            keys.roots: ['root_1', 'root_2']
        }

        model_spec: ModelSpec = interpret_spec_file(spec)

        self.assertEqual(model_spec.roots, ['root_1', 'root_2'])

    def test_parameters(self) -> None:
        spec = {
            keys.parameters: {
                'A': 'a',
                'B': 2,
                'C': 3.4,
                'D': None,
                'E': True,
            }
        }
        expect: Dict[str, State] = {
            'A': 'a',
            'B': 2,
            'C': 3.4,
            'D': None,
            'E': True,
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        self.assertEqual(model_spec.parameters, expect)

    def test_parameters_bad_state(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.parameters: {
                'A': range(1, 10),
            }
        }

        with self.assertRaises(SpecFileError) as context:
            _: ModelSpec = interpret_spec_file(spec)
        err: SpecFileError = context.exception
        self.assertEqual(err.section, f'test_spec:{keys.parameters}')
        self.assertEqual(repr(err.details), repr(range(1, 10)))

    def test_entity_field_clash(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.id_field: 'my_field',
                    keys.count_field: 'my_field',
                },
            },
        }

        with self.assertRaises(SpecFileError) as context:
            _: ModelSpec = interpret_spec_file(spec)
        err: SpecFileError = context.exception
        self.assertEqual(err.section, f'test_spec:{keys.entities}:test_entity')
        self.assertEqual(err.details, 'my_field')

    def test_entity_foreign_keys_none(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.parent: None,
                },
            },
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        entity: ModelEntitySpec = model_spec.entities['test_entity']
        self.assertIsNone(entity.parent)
        self.assertIsNone(entity.foreign_field_name)

    def test_entity_foreign_keys_singleton(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.parent: 'other_entity',
                },
                'other_entity': {},
            },
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        entity: ModelEntitySpec = model_spec.entities['test_entity']
        self.assertEqual(entity.parent, 'other_entity')
        self.assertEqual(entity.foreign_field_name, '_other_entity__id_')

    def test_entity_foreign_keys_string(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.parent: 'other_entity',
                },
                'other_entity': {},
            },
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        entity: ModelEntitySpec = model_spec.entities['test_entity']
        self.assertEqual(entity.parent, 'other_entity')
        self.assertEqual(entity.foreign_field_name, '_other_entity__id_')

    def test_entity_foreign_keys_dict(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.parent: 'other_entity',
                    keys.foreign_field: 'other_entity_key'
                },
                'other_entity': {},
            },
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        entity: ModelEntitySpec = model_spec.entities['test_entity']
        self.assertEqual(entity.parent, 'other_entity')
        self.assertEqual(entity.foreign_field_name, 'other_entity_key')

    def test_entity_foreign_keys_dict_default_field(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.parent: 'other_entity',
                },
                'other_entity': {},
            },
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        entity: ModelEntitySpec = model_spec.entities['test_entity']
        self.assertEqual(entity.parent, 'other_entity')
        self.assertEqual(entity.foreign_field_name, '_other_entity__id_')

    def test_entity_foreign_keys_str_error(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.parent: 'missing_entity',
                },
                'other_entity': {},
            },
        }

        with self.assertRaises(SpecFileError) as context:
            _: ModelSpec = interpret_spec_file(spec)
        err: SpecFileError = context.exception
        self.assertEqual(err.section, f'test_spec:{keys.entities}:test_entity')
        self.assertEqual(err.details, 'missing_entity')

    def test_entity_foreign_keys_dict_error(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.parent: 'missing_entity',
                    keys.foreign_field: 'other_entity_key'
                },
                'other_entity': {},
            },
        }

        with self.assertRaises(SpecFileError) as context:
            _: ModelSpec = interpret_spec_file(spec)
        err: SpecFileError = context.exception
        self.assertEqual(err.section, f'test_spec:{keys.entities}:test_entity')
        self.assertEqual(err.details, 'missing_entity')

    def test_entity_foreign_key_field_clash_id(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.id_field: 'my_field',
                    keys.parent: 'other_entity',
                    keys.foreign_field: 'my_field'
                },
                'other_entity': {},
            },
        }

        with self.assertRaises(SpecFileError) as context:
            _: ModelSpec = interpret_spec_file(spec)
        err: SpecFileError = context.exception
        self.assertEqual(err.section, f'test_spec:{keys.entities}:test_entity')
        self.assertEqual(err.details, 'my_field')

    def test_entity_foreign_key_field_clash_count(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.count_field: 'my_field',
                    keys.parent: 'other_entity',
                    keys.foreign_field: 'my_field'
                },
                'other_entity': {},
            },
        }

        with self.assertRaises(SpecFileError) as context:
            _: ModelSpec = interpret_spec_file(spec)
        err: SpecFileError = context.exception
        self.assertEqual(err.section, f'test_spec:{keys.entities}:test_entity')
        self.assertEqual(err.details, 'my_field')

    def test_entity_fields(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.rvs: ['A', 'B'],
                    keys.fields: {
                        'one': {keys.value: 1},
                        'two': {keys.sum: ['one', 'B']},
                        'three': {keys.function: 'A * A', keys.input: 'A'},
                        'four': {keys.sample: 'C'},
                    }
                },
            },
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        entity: ModelEntitySpec = model_spec.entities['test_entity']
        self.assertEqual(set(entity.fields.keys()), {'A', 'B', 'one', 'two', 'three', 'four'})
        sampled_fields = {field_name for field_name, _ in entity.sampled_fields()}
        self.assertEqual(sampled_fields, {'A', 'B', 'four'})

    def test_entity_fields_bad(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.fields: {
                        'one': {},
                    }
                },
            },
        }

        with self.assertRaises(SpecFileError) as context:
            _: ModelSpec = interpret_spec_file(spec)
        err: SpecFileError = context.exception
        self.assertEqual(err.section, f'test_spec:{keys.entities}:test_entity:{keys.fields}:one')

    def test_entity_fields_sum_empty(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.fields: {
                        'one': {keys.sum: []},
                    }
                },
            },
        }
        model_spec: ModelSpec = interpret_spec_file(spec)
        entity: ModelEntitySpec = model_spec.entities['test_entity']
        field: ModelFieldSpec = entity.fields['one']

        self.assertEqual(field.type, 'sum')
        field: ModelFieldSpecSum
        self.assertEqual(field.initial_value, 0)
        self.assertEqual(field.sum, [])
        self.assertEqual(field.offset, 0)

    def test_entity_fields_sum_constant(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.fields: {
                        'one': {keys.sum: 5},
                    }
                },
            },
        }
        model_spec: ModelSpec = interpret_spec_file(spec)
        entity: ModelEntitySpec = model_spec.entities['test_entity']
        field: ModelFieldSpec = entity.fields['one']

        self.assertEqual(field.type, 'sum')
        field: ModelFieldSpecSum
        self.assertEqual(field.initial_value, 0)
        self.assertEqual(field.sum, [])
        self.assertEqual(field.offset, 5)

    def test_entity_fields_sum_field(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.fields: {
                        'one': {keys.sum: 'two'},
                    }
                },
            },
        }
        model_spec: ModelSpec = interpret_spec_file(spec)
        entity: ModelEntitySpec = model_spec.entities['test_entity']
        field: ModelFieldSpec = entity.fields['one']

        self.assertEqual(field.type, 'sum')
        field: ModelFieldSpecSum
        self.assertEqual(field.initial_value, 0)
        self.assertEqual(field.sum, ['two'])
        self.assertEqual(field.offset, 0)

    def test_entity_fields_sum_mixture(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.fields: {
                        'one': {keys.sum: ['two', 5, 'three'], keys.value: 3},
                    }
                },
            },
        }
        model_spec: ModelSpec = interpret_spec_file(spec)
        entity: ModelEntitySpec = model_spec.entities['test_entity']
        field: ModelFieldSpec = entity.fields['one']

        self.assertEqual(field.type, 'sum')
        field: ModelFieldSpecSum
        self.assertEqual(field.initial_value, 3)
        self.assertEqual(field.sum, ['two', 'three'])
        self.assertEqual(field.offset, 5)

    def test_entity_fields_sum_none(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.fields: {
                        'one': {keys.sum: [None]},
                    }
                },
            },
        }

        with self.assertRaises(SpecFileError) as context:
            _: ModelSpec = interpret_spec_file(spec)
        err: SpecFileError = context.exception
        self.assertEqual(err.section, f'test_spec:{keys.entities}:test_entity:{keys.fields}:one')

    def test_entity_duplicated_sample_rvs(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.rvs: ['A', 'A'],
                    keys.fields: {
                        'one': {keys.value: 1},
                        'two': {keys.value: 2},
                    }
                },
            },
        }

        with self.assertRaises(SpecFileError) as context:
            _: ModelSpec = interpret_spec_file(spec)
        err: SpecFileError = context.exception
        self.assertEqual(err.section, f'test_spec:{keys.entities}:test_entity')
        self.assertEqual(err.details, 'A')

    def test_entity_duplicated_fields(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.rvs: ['A', 'B'],
                    keys.fields: {
                        'one': {keys.value: 1},
                        'B': {keys.value: 2},
                    }
                },
            },
        }

        with self.assertRaises(SpecFileError) as context:
            _: ModelSpec = interpret_spec_file(spec)
        err: SpecFileError = context.exception
        self.assertEqual(err.section, f'test_spec:{keys.entities}:test_entity:{keys.fields}')
        self.assertEqual(err.details, 'B')

    def test_entity_duplicated_fields_clash_id(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.id_field: 'two',
                    keys.rvs: ['A', 'B'],
                    keys.fields: {
                        'one': {keys.value: 1},
                        'two': {keys.value: 2},
                    }
                },
            },
        }

        with self.assertRaises(SpecFileError) as context:
            _: ModelSpec = interpret_spec_file(spec)
        err: SpecFileError = context.exception
        self.assertEqual(err.section, f'test_spec:{keys.entities}:test_entity:{keys.fields}')
        self.assertEqual(err.details, 'two')

    def test_entity_duplicated_fields_clash_count(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.count_field: 'two',
                    keys.rvs: ['A', 'B'],
                    keys.fields: {
                        'one': {keys.value: 1},
                        'two': {keys.value: 2},
                    }
                },
            },
        }

        with self.assertRaises(SpecFileError) as context:
            _: ModelSpec = interpret_spec_file(spec)
        err: SpecFileError = context.exception
        self.assertEqual(err.section, f'test_spec:{keys.entities}:test_entity:{keys.fields}')
        self.assertEqual(err.details, 'two')

    def test_entity_duplicated_fields_clash_foreign(self) -> None:
        spec = {
            keys.name: 'test_spec',
            keys.entities: {
                'test_entity': {
                    keys.parent: 'other_entity',
                    keys.foreign_field: 'two',
                    keys.rvs: ['A', 'B'],
                    keys.fields: {
                        'one': {keys.value: 1},
                        'two': {keys.value: 2},
                    }
                },
                'other_entity': {},
            },
        }

        with self.assertRaises(SpecFileError) as context:
            _: ModelSpec = interpret_spec_file(spec)
        err: SpecFileError = context.exception
        self.assertEqual(err.section, f'test_spec:{keys.entities}:test_entity:fields')
        self.assertEqual(err.details, 'two')

    def test_entity_cardinalities(self) -> None:
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

        test_cardinality: ConditionSpec = cardinalities[0]
        self.assertTrue(isinstance(test_cardinality, ConditionSpecFixedLimit))
        test_cardinality: ConditionSpecFixedLimit
        self.assertEqual(test_cardinality.field, DEFAULT_COUNT_FIELD)
        self.assertEqual(test_cardinality.op, '>=')
        self.assertEqual(test_cardinality.limit, 10)

        test_cardinality: ConditionSpec = cardinalities[1]
        self.assertTrue(isinstance(test_cardinality, ConditionSpecVariableLimit))
        test_cardinality: ConditionSpecVariableLimit
        self.assertEqual(test_cardinality.field, DEFAULT_COUNT_FIELD)
        self.assertEqual(test_cardinality.op, '>=')
        self.assertEqual(test_cardinality.limit_field, 'A')

        test_cardinality: ConditionSpec = cardinalities[2]
        self.assertTrue(isinstance(test_cardinality, ConditionSpecFixedLimit))
        test_cardinality: ConditionSpecFixedLimit
        self.assertEqual(test_cardinality.field, 'B')
        self.assertEqual(test_cardinality.op, '>=')
        self.assertEqual(test_cardinality.limit, 3)

        test_cardinality: ConditionSpec = cardinalities[3]
        self.assertTrue(isinstance(test_cardinality, ConditionSpecVariableLimit))
        test_cardinality: ConditionSpecVariableLimit
        self.assertEqual(test_cardinality.field, 'B')
        self.assertEqual(test_cardinality.op, '>=')
        self.assertEqual(test_cardinality.limit_field, 'C')

        test_cardinality: ConditionSpec = cardinalities[4]
        self.assertTrue(isinstance(test_cardinality, ConditionSpecStates))
        test_cardinality: ConditionSpecStates
        self.assertEqual(test_cardinality.field, 'B')
        self.assertEqual(test_cardinality.states, [1])

        test_cardinality: ConditionSpec = cardinalities[5]
        self.assertTrue(isinstance(test_cardinality, ConditionSpecStates))
        test_cardinality: ConditionSpecStates
        self.assertEqual(test_cardinality.field, 'C')
        self.assertEqual(test_cardinality.states, [0, 1])

    def test_crosstabs_dict_dict(self):
        spec = {
            keys.name: 'test_spec',
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
            keys.crosstabs: {
                'test_crosstab': {
                    keys.rvs: ['A'],
                },
            },
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        crosstab: ModelCrosstabSpec = model_spec.crosstabs['test_crosstab']
        self.assertEqual(crosstab.rvs, ['A'])
        self.assertEqual(crosstab.datasource, 'my_datasource')

    def test_crosstabs_dict_dict_datasource(self):
        spec = {
            keys.name: 'test_spec',
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
            keys.crosstabs: {
                'test_crosstab': {
                    keys.rvs: ['B', 'C'],
                    keys.datasource: 'my_datasource',
                },
            },
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        crosstab: ModelCrosstabSpec = model_spec.crosstabs['test_crosstab']
        self.assertEqual(crosstab.rvs, ['B', 'C'])
        self.assertEqual(crosstab.datasource, 'my_datasource')

    def test_crosstabs_list_list(self):
        spec = {
            keys.name: 'test_spec',
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
            keys.crosstabs: [['A']],
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        crosstab: ModelCrosstabSpec = model_spec.crosstabs['A']
        self.assertEqual(crosstab.rvs, ['A'])
        self.assertEqual(crosstab.datasource, 'my_datasource')

    def test_crosstabs_list_dict_single_rv(self):
        spec = {
            keys.name: 'test_spec',
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
            keys.crosstabs: [
                {
                    keys.rvs: 'A',
                },
            ],
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        crosstab: ModelCrosstabSpec = model_spec.crosstabs['A']
        self.assertEqual(crosstab.rvs, ['A'])
        self.assertEqual(crosstab.datasource, 'my_datasource')

    def test_crosstabs_list_dict_multi_rv(self):
        spec = {
            keys.name: 'test_spec',
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
            keys.crosstabs: [
                {
                    keys.rvs: ['A', 'B'],
                },
            ],
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        crosstab: ModelCrosstabSpec = model_spec.crosstabs['A,B']
        self.assertEqual(crosstab.rvs, ['A', 'B'])
        self.assertEqual(crosstab.datasource, 'my_datasource')

    def test_crosstabs_dict_list(self):
        spec = {
            keys.name: 'test_spec',
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
            keys.crosstabs: {
                'test_crosstab_1': ['A'],
                'test_crosstab_2': ['B', 'C'],
            },
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        crosstab: ModelCrosstabSpec = model_spec.crosstabs['test_crosstab_1']
        self.assertEqual(crosstab.rvs, ['A'])
        crosstab: ModelCrosstabSpec = model_spec.crosstabs['test_crosstab_2']
        self.assertEqual(crosstab.rvs, ['B', 'C'])

    def test_crosstabs_list_string(self):
        spec = {
            keys.name: 'test_spec',
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
            keys.crosstabs: ['my_datasource']
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        crosstab: ModelCrosstabSpec = model_spec.crosstabs['my_datasource']
        self.assertEqual(crosstab.rvs, ['A', 'B', 'C'])
        self.assertEqual(crosstab.datasource, 'my_datasource')

    def test_crosstabs_naming(self):
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
            keys.crosstabs: {
                '_A_': ['B', 'C'],  # steal the name that will be used for the default cross-table for rv 'A'.
            },
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        crosstab: ModelCrosstabSpec = model_spec.crosstabs['_A_']
        self.assertEqual(crosstab.rvs, ['B', 'C'])
        crosstab: ModelCrosstabSpec = model_spec.crosstabs['_A_(1)']
        self.assertEqual(crosstab.rvs, ['A'])

    def test_crosstabs_noiser_naive_laplace(self):
        spec = {
            keys.name: 'test_spec',
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
            keys.crosstabs: [
                {
                    keys.rvs: ['A'],
                    keys.noise: keys.naive_laplace,
                    keys.max_add_rows: 20,
                },
            ],
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        crosstab: ModelCrosstabSpec = model_spec.crosstabs['A']
        self.assertEqual(crosstab.rvs, ['A'])
        noiser: NoiserSpec = crosstab.noiser
        self.assertEqual(noiser.type, 'naive_laplace')
        noiser: NoiserSpecNaiveLaplace
        self.assertEqual(noiser.max_add_rows, 20)

    def test_crosstabs_noiser_basic_laplace(self):
        spec = {
            keys.name: 'test_spec',
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
            keys.crosstabs: [
                {
                    keys.rvs: ['A'],
                    keys.noise: keys.basic_laplace,
                },
            ],
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        crosstab: ModelCrosstabSpec = model_spec.crosstabs['A']
        self.assertEqual(crosstab.rvs, ['A'])
        noiser: NoiserSpec = crosstab.noiser
        self.assertEqual(noiser.type, 'basic_laplace')

    def test_crosstabs_noiser_decomposition_laplace(self):
        spec = {
            keys.name: 'test_spec',
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
            keys.crosstabs: [
                {
                    keys.rvs: ['A'],
                    keys.noise: {
                        keys.noise: keys.decomposition_laplace,
                        keys.max_add_rows: 99,
                    },
                },
            ],
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        crosstab: ModelCrosstabSpec = model_spec.crosstabs['A']
        self.assertEqual(crosstab.rvs, ['A'])
        noiser: NoiserSpec = crosstab.noiser
        self.assertEqual(noiser.type, 'decomposition_laplace')
        noiser: NoiserSpecDecompositionLaplace
        self.assertEqual(noiser.max_add_rows, 99)

    def test_rvs_empty(self):
        spec = {
            keys.name: 'test_spec',
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
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        self.assertEqual(set(model_spec.rvs.keys()), {'A', 'B', 'C'})

    def test_rvs_states(self):
        spec = {
            keys.name: 'test_spec',
            keys.data_format: keys.csv,
            keys.sensitivity: 0,
            keys.min_cell_size: 0,
            keys.datasources: {
                'datasource_ABC': {
                    keys.inline: """
                    A, B, C
                    1, 2, 3
                    4, 5, 6
                    """
                },
                'datasource_EFG': {
                    keys.inline: """
                    E, F, G
                    7, 8, 9
                    4, 3, 2
                    """
                },
                'datasource_XYZ': {
                    keys.inline: """
                    X, Y, Z
                    3, 2, 1
                    7, 8, 9
                    """
                },
                'datasource_QRS': {
                    keys.inline: """
                    Q, R, S
                    3, 2, 1
                    4, 5, 6
                    """
                },
            },
            keys.rvs: {
                'A': {keys.states: keys.infer_distinct},
                'B': {keys.states: keys.infer_range},
                'C': {keys.states: keys.infer_max},
                'E': {keys.states: 10},
                'F': {keys.states: {keys.start: 2, keys.stop: 9, keys.step: 0.5}},
                'G': {keys.states: {keys.start: 2, keys.stop: 9}},
                'X': {keys.states: range(1, 10)},
                'Y': {keys.states: (2, 4, 6, 8)},
                'Z': {keys.states: {keys.start: 9, keys.stop: 1, keys.step: -1}},
                'Q': {keys.states: {keys.start: 4, keys.stop: 2, keys.step: -0.5}},
            },
            keys.crosstabs: ['datasource_ABC'],
        }

        model_spec: ModelSpec = interpret_spec_file(spec)
        self.assertEqual(set(model_spec.rvs.keys()), {'A', 'B', 'C', 'E', 'F', 'G', 'X', 'Y', 'Z', 'Q'})
        rv_spec: ModelRVSpec = model_spec.rvs['A']
        self.assertEqual(rv_spec.states, 'infer_distinct')
        rv_spec: ModelRVSpec = model_spec.rvs['B']
        self.assertEqual(rv_spec.states, 'infer_range')
        rv_spec: ModelRVSpec = model_spec.rvs['C']
        self.assertEqual(rv_spec.states, 'infer_max')
        rv_spec: ModelRVSpec = model_spec.rvs['E']
        self.assertEqual(rv_spec.states, 10)
        rv_spec: ModelRVSpec = model_spec.rvs['F']
        self.assertEqual(rv_spec.states, [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
        rv_spec: ModelRVSpec = model_spec.rvs['G']
        self.assertEqual(rv_spec.states, [2, 3, 4, 5, 6, 7, 8, 9])
        rv_spec: ModelRVSpec = model_spec.rvs['X']
        self.assertEqual(rv_spec.states, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        rv_spec: ModelRVSpec = model_spec.rvs['Y']
        self.assertEqual(rv_spec.states, [2, 4, 6, 8])
        rv_spec: ModelRVSpec = model_spec.rvs['Z']
        self.assertEqual(rv_spec.states, [9, 8, 7, 6, 5, 4, 3, 2, 1])
        rv_spec: ModelRVSpec = model_spec.rvs['Q']
        self.assertEqual(rv_spec.states, [4.0, 3.5, 3.0, 2.5, 2.0])

        # Check indexed states resolve
        model_index: ModelIndex = make_model_index(model_spec, dataset_cache=None)
        self.assertEqual(set(model_index.rvs.keys()), {'A', 'B', 'C', 'E', 'F', 'G', 'X', 'Y', 'Z', 'Q'})
        rv_index: RVIndex = model_index.rvs['A']
        self.assertEqual(rv_index.states, [1, 4])
        rv_index: RVIndex = model_index.rvs['B']
        self.assertEqual(rv_index.states, [2, 3, 4, 5])
        rv_index: RVIndex = model_index.rvs['C']
        self.assertEqual(rv_index.states, [0, 1, 2, 3, 4, 5, 6])
        rv_index: RVIndex = model_index.rvs['E']
        self.assertEqual(rv_index.states, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        rv_index: RVIndex = model_index.rvs['F']
        self.assertEqual(rv_index.states, [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
        rv_index: RVIndex = model_index.rvs['G']
        self.assertEqual(rv_index.states, [2, 3, 4, 5, 6, 7, 8, 9])
        rv_index: RVIndex = model_index.rvs['X']
        self.assertEqual(rv_index.states, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        rv_index: RVIndex = model_index.rvs['Y']
        self.assertEqual(rv_index.states, [2, 4, 6, 8])
        rv_index: RVIndex = model_index.rvs['Z']
        self.assertEqual(rv_index.states, [9, 8, 7, 6, 5, 4, 3, 2, 1])
        rv_index: RVIndex = model_index.rvs['Q']
        self.assertEqual(rv_index.states, [4.0, 3.5, 3.0, 2.5, 2.0])

    def test_find_datasource(self):
        spec = {
            keys.name: 'test_spec',
            keys.sensitivity: 0,
            keys.min_cell_size: 0,
            keys.states: keys.infer_distinct,
            keys.data_format: keys.csv,
            keys.datasources: {
                'datasource_1': {
                    keys.inline: """
                    A, B, C
                    1, 2, 3
                    4, 5, 6
                    """
                },
                'datasource_2': {
                    keys.inline: """
                    A, B
                    1, 2
                    4, 5
                    """
                },
            },
            keys.rvs: {'A': {}},
            keys.crosstabs: {'_A_': ['A']},
        }
        model_spec: ModelSpec = interpret_spec_file(spec)

        crosstab: ModelCrosstabSpec = model_spec.crosstabs['_A_']
        self.assertEqual(crosstab.rvs, ['A'])
        # The preferred datasource has fewer random variables
        self.assertEqual(crosstab.datasource, 'datasource_2')

    def test_datasources_inline(self):
        spec = {
            keys.name: 'test_spec',
            keys.sensitivity: 0,
            keys.min_cell_size: 0,
            keys.states: keys.infer_distinct,
            keys.datasources: {
                'datasource_ABC': {
                    keys.data_format: keys.csv,
                    keys.inline: """
                    A, B, C
                    1, 2, 3
                    4, 5, 6
                    """
                },
                'datasource_EFG': {
                    keys.data_format: keys.tsv,
                    keys.inline: """
                    E\tF\tG
                    7\t8\t9
                    4\t3\t2
                    3\t6\t9
                    """
                },
                'datasource_XYZ': {
                    keys.data_format: keys.csv,
                    keys.sep: '.',
                    keys.inline: """
                    X.Y.Z
                    3.2.1
                    7.8.9
                    2.4.8
                    3.1.4
                    """
                },
                'datasource_table_builder': {
                    keys.data_format: keys.table_builder,
                    keys.inline: """
                    Australian Bureau of Statistics
                    
                    "Census - employment, income and education"
                    
                    Filters: None                    
                    
                    "age","sex",
                    "0","Male",47069,
                    ,"Female",44443,
                    "1","Male",47785,
                    ,"Female",45457,
                    "2","Male",48413,
                    ,"Female",45172,
                    """
                },
            },
        }
        model_spec: ModelSpec = interpret_spec_file(spec)

        dataset: Dataset = model_spec.datasources['datasource_ABC'].dataset()
        self.assertEqual(list(dataset.rvs), ['A', 'B', 'C'])
        self.assertEqual(dataset.number_of_records(), 2)

        dataset: Dataset = model_spec.datasources['datasource_EFG'].dataset()
        self.assertEqual(list(dataset.rvs), ['E', 'F', 'G'])
        self.assertEqual(dataset.number_of_records(), 3)

        dataset: Dataset = model_spec.datasources['datasource_XYZ'].dataset()
        self.assertEqual(list(dataset.rvs), ['X', 'Y', 'Z'])
        self.assertEqual(dataset.number_of_records(), 4)

        dataset: Dataset = model_spec.datasources['datasource_table_builder'].dataset()
        self.assertEqual(list(dataset.rvs), ['age', 'sex'])
        self.assertEqual(dataset.number_of_records(), 6)

    def test_datasources_function(self):
        spec = {
            keys.states: keys.infer_distinct,
            keys.datasources: {
                'my_func': {
                    keys.sensitivity: 0,
                    keys.function: 'A * B + C',
                    keys.input: {
                        'A': 5,
                        'B': range(6),
                        'C': {keys.start: 1, keys.stop: 7}
                    },
                },
            },
        }
        model_spec: ModelSpec = interpret_spec_file(spec)
        dataset: Dataset = model_spec.datasources['my_func'].dataset()
        self.assertEqual(list(dataset.rvs), ['A', 'B', 'C', 'my_func'])
        self.assertEqual(dataset.number_of_records(), 5 * 6 * 7)

    def test_datasources_file(self):
        spec = {
            keys.states: keys.infer_distinct,
            keys.sensitivity: 0,
            keys.datasources: {
                'datasource_1': {keys.location: 'data.csv'},
                'datasource_2': {keys.location: 'data.tsv'},
                'datasource_3': {keys.location: 'data.parquet'},
                'datasource_4': {keys.location: 'data.feather'},
                'datasource_5': {keys.location: 'data.pkl'},
            },
        }
        with tmp_dir() as dir_path:
            data = pd.DataFrame({
                'A': [1, 2, 3],
                'B': [4, 5, 6],
            })
            data.to_csv(dir_path / 'data.csv', index=False)
            data.to_csv(dir_path / 'data.tsv', index=False, sep='\t')
            data.to_parquet(dir_path / 'data.parquet', index=False)
            data.to_feather(dir_path / 'data.feather')
            data.to_pickle(dir_path / 'data.pkl')
            model_spec: ModelSpec = interpret_spec_file(spec, cwd=dir_path)

            self.assertEqual(
                set(model_spec.datasources.keys()),
                {'datasource_1', 'datasource_2', 'datasource_3', 'datasource_4', 'datasource_5'}
            )
            for datasource in model_spec.datasources.values():
                dataset: Dataset = datasource.dataset([dir_path])
                self.assertEqual(list(dataset.rvs), ['A', 'B'])
                self.assertEqual(dataset.number_of_records(), 3)

    def test_datasources_define_function(self):
        spec = {
            keys.sensitivity: 0,
            keys.min_cell_size: 0,
            keys.states: keys.infer_distinct,
            keys.datasources: {
                'datasource_ABC': {
                    keys.data_format: keys.csv,
                    keys.inline: """
                    A, B, C
                    1, 2, 3
                    4, 5, 6
                    """,
                    keys.define: {
                        'fAB': {
                            keys.function: 'A + B',
                            keys.input: ['A', 'B'],
                            keys.delete_input: False,
                        }
                    }
                },
            },
        }
        model_spec: ModelSpec = interpret_spec_file(spec)

        dataset: Dataset = model_spec.datasources['datasource_ABC'].dataset()
        self.assertEqual(list(dataset.rvs), ['A', 'B', 'C', 'fAB'])
        self.assertEqual(dataset.number_of_records(), 2)

        crosstab: pd.DataFrame = dataset.crosstab(dataset.rvs)
        self.assertEqual(list(crosstab.columns), ['A', 'B', 'C', 'fAB', ''])
        self.assertEqual(crosstab.shape, (2, 5))
        self.assertEqual(list(crosstab.iloc[0, :]), [1, 2, 3, 3, 1])
        self.assertEqual(list(crosstab.iloc[1, :]), [4, 5, 6, 9, 1])

    def test_datasources_define_group(self):
        spec = {
            keys.sensitivity: 0,
            keys.min_cell_size: 0,
            keys.states: keys.infer_distinct,
            keys.datasources: {
                'datasource_1': {
                    keys.data_format: keys.csv,
                    keys.inline: """
                    A, B, C
                    1, 2, 3
                    2, 4, 6
                    3, 6, 6
                    4, 8, 3
                    """,
                    keys.define: {
                        'gA': {
                            keys.grouping: keys.group_qcut,
                            keys.input: 'A',
                            keys.size: 2,
                            keys.delete_input: False,
                        }
                    }
                },
            },
        }
        model_spec: ModelSpec = interpret_spec_file(spec)

        dataset: Dataset = model_spec.datasources['datasource_1'].dataset()
        self.assertEqual(list(dataset.rvs), ['A', 'B', 'C', 'gA'])
        self.assertEqual(dataset.number_of_records(), 4)

        crosstab: pd.DataFrame = dataset.crosstab(dataset.rvs)
        self.assertEqual(list(crosstab.columns), ['A', 'B', 'C', 'gA', ''])
        self.assertEqual(crosstab.shape, (4, 5))
        self.assertEqual(list(crosstab.iloc[0, :]), [1, 2, 3, 0, 1])
        self.assertEqual(list(crosstab.iloc[1, :]), [2, 4, 6, 0, 1])
        self.assertEqual(list(crosstab.iloc[2, :]), [3, 6, 6, 1, 1])
        self.assertEqual(list(crosstab.iloc[3, :]), [4, 8, 3, 1, 1])

    def test_datasources_rvmap_dict(self):
        spec = {
            keys.sensitivity: 0,
            keys.min_cell_size: 0,
            keys.states: keys.infer_distinct,
            keys.datasources: {
                'datasource_1': {
                    keys.data_format: keys.csv,
                    keys.inline: """
                    A, B, C, D
                    1, 2, 7, 9
                    4, 6, 3, 8
                    """,
                    keys.rvs: {
                        'D': 'B',
                        'E': -1,
                        'F': 0,
                    }
                },
            },
        }
        model_spec: ModelSpec = interpret_spec_file(spec)

        dataset: Dataset = model_spec.datasources['datasource_1'].dataset()
        self.assertEqual(set(dataset.rvs), {'D', 'E', 'F'})
        self.assertEqual(dataset.number_of_records(), 2)

        crosstab: pd.DataFrame = dataset.crosstab(['D', 'E', 'F'])
        self.assertEqual(list(crosstab.columns), ['D', 'E', 'F', ''])
        self.assertEqual(crosstab.shape, (2, 4))
        self.assertEqual(list(crosstab.iloc[0, :]), [2, 9, 1, 1])
        self.assertEqual(list(crosstab.iloc[1, :]), [6, 8, 4, 1])

    def test_load_spec_file(self):
        test_file_name: str = 'a_test_spec.py'
        test_file: str = \
            """
            spec = {
                'entities': {
                    'entity_1': {},
                    'entity_2': {},
                },
            }
            """

        with tmp_dir():
            with open(test_file_name, 'w') as f:
                f.write(unindent(test_file))
            model_spec: ModelSpec = load_spec_file(test_file_name)
        self.assertEqual(model_spec.name, 'a_test_spec')
        self.assertEqual(len(model_spec.entities), 2)


if __name__ == '__main__':
    test_main()
