from typing import Dict

from synthorus.model.datasource_spec import DatasourceSpec
from synthorus.model.model_spec import ModelRVSpec, ModelEntitySpec, ModelSpec, ModelCrosstabSpec, \
    ModelFieldSpecSample
from synthorus_demos.dataset import example_datasource


def make_model_spec_one_entity() -> ModelSpec:
    """
    Make a model spec that has a single dataset, cross-table and entity;
    all use three random variables: A, C and X.
    """
    datasource_name: str = 'acx'
    datasource: DatasourceSpec = example_datasource.make_datasource_spec_acx()
    datasources: Dict[str, DatasourceSpec] = {datasource_name: datasource}

    # Define an RVSpec for each rv of the datasource
    rvs: Dict[str, ModelRVSpec] = make_rvs_for_datasources(datasources)

    # Define a cross-table over all datasource rvs
    crosstabs = {'my_crosstab': ModelCrosstabSpec(rvs=datasource.rvs, datasource=datasource_name)}

    # Define an entity, sampling all rvs
    entities = {'my_entity': ModelEntitySpec(fields=sample_rvs(*datasource.rvs))}

    # Put it all together to make a model spec
    model_spec = ModelSpec(
        name=f'{__name__}.make_model_spec_one_entity',
        datasources=datasources,
        rvs=rvs,
        crosstabs=crosstabs,
        entities=entities,
    )

    return model_spec


def make_model_spec_two_entities() -> ModelSpec:
    """
    Make a model spec that has a single dataset, cross-table with
    random variables A, C and X.

    There are two entities:
        'entity_1': a root entity sampling random variable A,
        'entity_2': a child entity sampling random variable C.
    """
    datasource_name: str = 'acx'
    datasource: DatasourceSpec = example_datasource.make_datasource_spec_acx()
    datasources: Dict[str, DatasourceSpec] = {datasource_name: datasource}

    # Define an RVSpec for each rv of the datasource
    rvs: Dict[str, ModelRVSpec] = make_rvs_for_datasources(datasources)

    # Define a cross-table over the whole datasource
    crosstabs = {'my_crosstab': ModelCrosstabSpec(rvs=datasource.rvs, datasource=datasource_name)}

    # Define entities
    entities = {
        'entity_1': ModelEntitySpec(fields=sample_rvs('A')),
        'entity_2': ModelEntitySpec(fields=sample_rvs('C'), parent='entity_1', foreign_field_name='_entity_1__id_'),
    }

    # Put it all together to make a model spec
    model_spec = ModelSpec(
        name=f'{__name__}.make_model_spec_two_entities',
        datasources=datasources,
        rvs=rvs,
        crosstabs=crosstabs,
        entities=entities,
    )

    return model_spec


def sample_rvs(*names: str) -> Dict[str, ModelFieldSpecSample]:
    """
    A convenience method to construct 'fields' for an `ModelEntitySpec` where
    all the fields are sampled random variables, and where each field
    has the same name as its random variable.
    """
    return {
        name: ModelFieldSpecSample(rv_name=name)
        for name in names
    }


def make_rvs_for_datasources(datasources: Dict[str, DatasourceSpec]) -> Dict[str, ModelRVSpec]:
    """
    Define an RVSpec for each rv of the given datasource.
    The states for each rv will be 'infer_distinct'.

    Assumes:
        each random variable is in just one datasource.
    """
    rvs: Dict[str, ModelRVSpec] = {
        rv_name: ModelRVSpec(states='infer_distinct')
        for datasource_name, datasource in datasources.items()
        for rv_name in datasource.rvs
    }
    return rvs
