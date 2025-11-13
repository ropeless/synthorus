from synthorus.model.dataset_cache import DatasetCache
from synthorus.model.make_model_index import make_model_index
from synthorus.model.model_index import ModelIndex
from synthorus.model.model_spec import ModelSpec, ModelRVSpec, ModelCrosstabSpec, ModelEntitySpec, ModelFieldSpecSample
from tests.helpers.make_model_spec import make_model_spec, make_dataset_csv_inline
from tests.helpers.unittest_fixture import Fixture, test_main


class DatasetCacheTest(Fixture):

    def test_empty(self) -> None:
        model_spec: ModelSpec = make_model_spec({})

        model_index: ModelIndex = make_model_index(model_spec, DatasetCache(model_spec, cwd=None))

        self.assertEqual(len(model_index.rvs), 0)
        self.assertEqual(len(model_index.crosstabs), 0)
        self.assertEqual(len(model_index.entities), 0)

    def test_simple(self) -> None:
        datasource = 'acx'
        model_spec: ModelSpec = make_model_spec({
            datasource: make_dataset_csv_inline()
        })

        model_spec.rvs['A'] = ModelRVSpec(states='infer_distinct')
        model_spec.rvs['C'] = ModelRVSpec(states='infer_distinct')
        model_spec.rvs['X'] = ModelRVSpec(states='infer_distinct')

        model_spec.crosstabs['crosstab_ac'] = ModelCrosstabSpec(rvs=['A', 'C'], datasource=datasource)
        model_spec.crosstabs['crosstab_cx'] = ModelCrosstabSpec(rvs=['C', 'X'], datasource=datasource)

        model_spec.entities['entity_a'] = ModelEntitySpec(
            fields={
                'field_a': ModelFieldSpecSample(rv_name='A')
            },
        )
        model_spec.entities['entity_x'] = ModelEntitySpec(
            fields={
                'field_x': ModelFieldSpecSample(rv_name='X')
            },
            parent='entity_a',
            foreign_field_name='foreign_a',
        )

        model_index: ModelIndex = make_model_index(model_spec, DatasetCache(model_spec, cwd=None))

        self.assertEqual(len(model_index.rvs), 3)
        self.assertEqual(len(model_index.crosstabs), 2)
        self.assertEqual(len(model_index.entities), 2)

        self.assertEqual(model_index.rvs['A'].name, 'A')
        self.assertEqual(set(model_index.rvs['A'].states), {'y', 'n'})
        self.assertEqual(model_index.rvs['A'].primary_datasource, datasource)
        self.assertEqual(model_index.rvs['A'].all_datasources, [datasource])
        self.assertEqual(model_index.rvs['A'].all_distribution_crosstabs, ['crosstab_ac'])
        self.assertEqual(model_index.rvs['A'].all_sampling_entities, ['entity_a'])

        self.assertEqual(model_index.rvs['C'].name, 'C')
        self.assertEqual(set(model_index.rvs['C'].states), {'y', 'n'})
        self.assertEqual(model_index.rvs['C'].primary_datasource, datasource)
        self.assertEqual(model_index.rvs['C'].all_datasources, [datasource])
        self.assertEqual(set(model_index.rvs['C'].all_distribution_crosstabs), {'crosstab_ac', 'crosstab_cx'})
        self.assertEqual(model_index.rvs['C'].all_sampling_entities, [])

        self.assertEqual(model_index.rvs['X'].name, 'X')
        self.assertEqual(set(model_index.rvs['X'].states), {'y', 'n'})
        self.assertEqual(model_index.rvs['X'].primary_datasource, datasource)
        self.assertEqual(model_index.rvs['X'].all_datasources, [datasource])
        self.assertEqual(model_index.rvs['X'].all_distribution_crosstabs, ['crosstab_cx'])
        self.assertEqual(model_index.rvs['X'].all_sampling_entities, ['entity_x'])

        self.assertEqual(model_index.crosstabs['crosstab_ac'].name, 'crosstab_ac')
        self.assertEqual(model_index.crosstabs['crosstab_ac'].rvs, ['A', 'C'])
        self.assertEqual(model_index.crosstabs['crosstab_ac'].non_distribution_rvs, [])
        self.assertEqual(model_index.crosstabs['crosstab_ac'].distribution_rvs, ['A', 'C'])
        self.assertEqual(model_index.crosstabs['crosstab_ac'].datasource, datasource)
        self.assertEqual(model_index.crosstabs['crosstab_ac'].number_of_states, 4)

        self.assertEqual(model_index.crosstabs['crosstab_cx'].name, 'crosstab_cx')
        self.assertEqual(model_index.crosstabs['crosstab_cx'].rvs, ['C', 'X'])
        self.assertEqual(model_index.crosstabs['crosstab_cx'].non_distribution_rvs, [])
        self.assertEqual(model_index.crosstabs['crosstab_cx'].distribution_rvs, ['C', 'X'])
        self.assertEqual(model_index.crosstabs['crosstab_cx'].datasource, datasource)
        self.assertEqual(model_index.crosstabs['crosstab_cx'].number_of_states, 4)

        self.assertEqual(model_index.entities['entity_a'].name, 'entity_a')
        self.assertIsNone(model_index.entities['entity_a'].parent)
        self.assertEqual(model_index.entities['entity_a'].sampled_fields, {'field_a': 'A'})
        self.assertEqual(len(model_index.entities['entity_a'].entity_crosstabs), 1)
        self.assertEqual(model_index.entities['entity_a'].entity_crosstabs[0].crosstab, 'crosstab_ac')
        self.assertEqual(model_index.entities['entity_a'].entity_crosstabs[0].sampled_rvs, ['A'])
        self.assertEqual(model_index.entities['entity_a'].entity_crosstabs[0].condition_rvs, ['C'])
        self.assertEqual(model_index.entities['entity_a'].entity_crosstabs[0].non_distribution_rvs, [])
        self.assertEqual(len(model_index.entities['entity_a'].ancestor_conditions), 0)

        self.assertEqual(model_index.entities['entity_x'].name, 'entity_x')
        self.assertEqual(model_index.entities['entity_x'].parent, 'entity_a')
        self.assertEqual(model_index.entities['entity_x'].sampled_fields, {'field_x': 'X'})
        self.assertEqual(len(model_index.entities['entity_x'].entity_crosstabs), 1)
        self.assertEqual(model_index.entities['entity_x'].entity_crosstabs[0].crosstab, 'crosstab_cx')
        self.assertEqual(model_index.entities['entity_x'].entity_crosstabs[0].sampled_rvs, ['X'])
        self.assertEqual(model_index.entities['entity_x'].entity_crosstabs[0].condition_rvs, ['C'])
        self.assertEqual(model_index.entities['entity_x'].entity_crosstabs[0].non_distribution_rvs, [])
        self.assertEqual(len(model_index.entities['entity_x'].ancestor_conditions), 0)

    def test_ancestor_conditions(self) -> None:
        datasource = 'acx'
        model_spec: ModelSpec = make_model_spec({
            datasource: make_dataset_csv_inline()
        })

        model_spec.rvs['A'] = ModelRVSpec(states='infer_distinct')
        model_spec.rvs['C'] = ModelRVSpec(states='infer_distinct')
        model_spec.rvs['X'] = ModelRVSpec(states='infer_distinct')

        model_spec.crosstabs['crosstab_acx'] = ModelCrosstabSpec(rvs=['A', 'C', 'X'], datasource=datasource)

        model_spec.entities['entity_a'] = ModelEntitySpec(
            fields={
                'field_a': ModelFieldSpecSample(rv_name='A')
            },
        )
        model_spec.entities['entity_x'] = ModelEntitySpec(
            fields={
                'field_x': ModelFieldSpecSample(rv_name='X')
            },
            parent='entity_a',
            foreign_field_name='foreign_a',
        )

        model_index: ModelIndex = make_model_index(model_spec, DatasetCache(model_spec, cwd=None))

        self.assertEqual(len(model_index.rvs), 3)
        self.assertEqual(len(model_index.crosstabs), 1)
        self.assertEqual(len(model_index.entities), 2)

        self.assertEqual(model_index.crosstabs['crosstab_acx'].name, 'crosstab_acx')
        self.assertEqual(model_index.crosstabs['crosstab_acx'].rvs, ['A', 'C', 'X'])
        self.assertEqual(model_index.crosstabs['crosstab_acx'].non_distribution_rvs, [])
        self.assertEqual(model_index.crosstabs['crosstab_acx'].distribution_rvs, ['A', 'C', 'X'])
        self.assertEqual(model_index.crosstabs['crosstab_acx'].datasource, datasource)
        self.assertEqual(model_index.crosstabs['crosstab_acx'].number_of_states, 8)

        self.assertEqual(model_index.entities['entity_a'].name, 'entity_a')
        self.assertIsNone(model_index.entities['entity_a'].parent)
        self.assertEqual(model_index.entities['entity_a'].sampled_fields, {'field_a': 'A'})
        self.assertEqual(len(model_index.entities['entity_a'].entity_crosstabs), 1)
        self.assertEqual(model_index.entities['entity_a'].entity_crosstabs[0].crosstab, 'crosstab_acx')
        self.assertEqual(model_index.entities['entity_a'].entity_crosstabs[0].sampled_rvs, ['A'])
        self.assertEqual(set(model_index.entities['entity_a'].entity_crosstabs[0].condition_rvs), {'C', 'X'})
        self.assertEqual(model_index.entities['entity_a'].entity_crosstabs[0].non_distribution_rvs, [])
        self.assertEqual(len(model_index.entities['entity_a'].ancestor_conditions), 0)

        self.assertEqual(model_index.entities['entity_x'].name, 'entity_x')
        self.assertEqual(model_index.entities['entity_x'].parent, 'entity_a')
        self.assertEqual(model_index.entities['entity_x'].sampled_fields, {'field_x': 'X'})
        self.assertEqual(len(model_index.entities['entity_x'].entity_crosstabs), 1)
        self.assertEqual(model_index.entities['entity_x'].entity_crosstabs[0].crosstab, 'crosstab_acx')
        self.assertEqual(model_index.entities['entity_x'].entity_crosstabs[0].sampled_rvs, ['X'])
        self.assertEqual(set(model_index.entities['entity_x'].entity_crosstabs[0].condition_rvs), {'A', 'C'})
        self.assertEqual(model_index.entities['entity_x'].entity_crosstabs[0].non_distribution_rvs, [])
        self.assertEqual(len(model_index.entities['entity_x'].ancestor_conditions), 1)
        self.assertEqual(model_index.entities['entity_x'].ancestor_conditions[0].entity, 'entity_a')
        self.assertEqual(model_index.entities['entity_x'].ancestor_conditions[0].field, 'field_a')
        self.assertEqual(model_index.entities['entity_x'].ancestor_conditions[0].rv, 'A')


if __name__ == '__main__':
    test_main()
