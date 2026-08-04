from __future__ import annotations

import math
from functools import cmp_to_key
from typing import List, Set, Dict, Sequence, Collection, Tuple, Mapping, Optional

from ck.pgm import State
from ck.utils.map_list import MapList
from ck.utils.map_set import MapSet

from synthorus.dataset import Dataset
from synthorus.error import SynthorusError
from synthorus.model.dataset_cache import DatasetCache
from synthorus.model.datasource_spec import DatasourceSpec
from synthorus.model.model_index import ModelIndex, RVIndex, CrosstabIndex, EntityIndex, EntityCrosstabIndex, \
    AncestorConditionsIndex
from synthorus.model.model_spec import ModelRVSpec, ModelCrosstabSpec, ModelEntitySpec, ModelSpec, ModelFieldSpecSample, \
    ForeignKeyField
from synthorus.utils.clean_state import clean_state


def make_model_index(model_spec: ModelSpec, dataset_cache: Optional[DatasetCache]) -> ModelIndex:
    """
    Make a ModelIndex object for the given model spec.

    Args:
        model_spec: The model spec to index.
        dataset_cache: Access to datasources, if None, one will be created automatically.

    Returns:
        model_index.

    Raises:
        SynthorusError the given model spec is inconsistent and no index can be formed.
    """
    if dataset_cache is None:
        dataset_cache = DatasetCache(model_spec=model_spec, cwd=None)

    index = ModelIndex(parameters=model_spec.parameters)
    potential_dist_datasources: MapList[str, str] = MapList()

    _register_rvs(index, model_spec.rvs)

    _register_datasources(index, model_spec.datasources, potential_dist_datasources)
    _resolve_rvs_states(index, model_spec.rvs, dataset_cache)
    _resolve_primary_datasources(index, potential_dist_datasources)

    _register_crosstabs(index, model_spec)
    _register_entities(index, model_spec)

    return index


def _register_rvs(index: ModelIndex, rvs: Dict[str, ModelRVSpec]) -> None:
    """
    Register `RVIndex` objects with the given model index for each provided random variable.
    All `RVIndex.states` values will have a placeholder object that should be resolved.
    """
    dummy_states: List[State] = []
    dummy_datasource: str = ''

    for rv_name, rv_spec in rvs.items():
        index.rvs[rv_name] = RVIndex(states=dummy_states, primary_datasource=dummy_datasource)


def _register_datasources(
        index: ModelIndex,
        datasources: Dict[str, DatasourceSpec],
        potential_dist_datasources: MapList[str, str],
) -> None:
    """
    For each dataset in `datasources`:
        for each rv in dataset:
            register the dataset with rv.all_datasources.
            If rv is not a conditioned random variable:
                register the dataset with potential_dist_datasources[rv].

    Args:
        index: index being updated
        datasources: all available datasources.
        potential_dist_datasources: for each random variable, a list of datasources
            providing a distribution for the random variable.
    """
    datasource_name: str
    datasource: DatasourceSpec
    for datasource_name, datasource in datasources.items():
        # Index the datasource with its random variables
        non_distribution_rvs: Set[str] = set(datasource.non_distribution_rvs)
        for rv_name in datasource.rvs:
            rv_index: Optional[RVIndex] = index.rvs.get(rv_name)
            if rv_index is not None:
                rv_index.all_datasources.append(datasource_name)
                if rv_name not in non_distribution_rvs:
                    potential_dist_datasources.append(rv_name, datasource_name)


def _resolve_primary_datasources(
        index: ModelIndex,
        potential_dist_datasources: MapList[str, str],
) -> None:
    """
    For each random variable, rv, in the index:
        Set rv.primary_datasource to a datasource with the rv as a non-conditioned random variable.

    Args:
        index: The model index being checked.
        potential_dist_datasources: List of

    Raises:
        SynthorusError if a primary datasource cannot be found.
    """
    for rv_name, rv_index in index.rvs.items():
        dist_datasources: Sequence[str] = potential_dist_datasources.get(rv_name, ())
        if len(dist_datasources) == 0:
            raise SynthorusError(
                f'random variable {rv_name!r} has no distribution datasource'
            )
        rv_index.primary_datasource = dist_datasources[0]  # TODO should we do better?


def _register_crosstabs(index: ModelIndex, model_spec: ModelSpec) -> None:
    """
    Register cross-tables with the given model index.
    All `CrosstabIndex.dataset` values will have a placeholder object that should be resolved.
    Metadata about the cross-table will be filled in later.
    """
    crosstabs: Dict[str, ModelCrosstabSpec] = model_spec.crosstabs

    crosstab_name: str
    crosstab_spec: ModelCrosstabSpec
    for crosstab_name, crosstab_spec in crosstabs.items():
        datasource_name: str = crosstab_spec.datasource
        datasource: DatasourceSpec = model_spec.datasources[datasource_name]
        rvs: List[str] = crosstab_spec.rvs
        non_distribution_rvs: List[str] = datasource.non_distribution_rvs
        distribution_rvs: List[str] = [rv_name for rv_name in rvs if rv_name not in non_distribution_rvs]
        number_of_states: int = math.prod(len(index.rvs[rv].states) for rv in rvs)

        for non_distribution_rv in non_distribution_rvs:
            if non_distribution_rv not in rvs:
                raise SynthorusError(
                    f'cross-table {crosstab_name!r}'
                    f' must include non-distribution random variable: {non_distribution_rv!r}'
                )

        # Index the cross-table with its "distribution" random variables
        for rv_name in distribution_rvs:
            rv_index: RVIndex = index.rvs[rv_name]
            rv_index.all_distribution_crosstabs.append(crosstab_name)

        index.crosstabs[crosstab_name] = CrosstabIndex(
            rvs=rvs,
            number_of_states=number_of_states,
            non_distribution_rvs=non_distribution_rvs,
            distribution_rvs=distribution_rvs,
            datasource=datasource_name,
        )


def _register_entities(index: ModelIndex, model_spec: ModelSpec) -> None:
    """
    Assumes:
        rvs and cross-tables are already registered with the index.

    Args:
        index: The model index being updated.
        model_spec: The model spec holding the entities to be registered.
    Raises:
        SynthorusError if there is a missing or ambiguous conditioning random variable.
    """
    ancestors: Mapping[str, Set[str]] = _find_ancestors(model_spec.entities)
    children: Mapping[str, Set[str]] = _find_children(model_spec.entities)

    entity_name: str
    entity_spec: ModelEntitySpec
    for entity_name, entity_spec in model_spec.entities.items():
        field: ModelFieldSpecSample
        sampled_fields: Dict[str, str] = {
            field_name: field.rv_name
            for field_name, field in entity_spec.sampled_fields()
        }
        entity_crosstabs: List[EntityCrosstabIndex] = find_covering_crosstabs(sampled_fields.values(), index)
        condition_rvs: Set[str] = {
            rv_name
            for entity_crosstab in entity_crosstabs
            for rv_name in entity_crosstab.condition_rvs
        }

        ancestor_conditions: List[AncestorConditionsIndex] = _find_ancestor_conditions(
            entity_name,
            model_spec.entities,
            condition_rvs,
            ancestors[entity_name],
        )

        # Register the entity with its rvs
        for rv_name in sampled_fields.values():
            index.rvs[rv_name].all_sampling_entities.append(entity_name)

        index.entities[entity_name] = EntityIndex(
            model_entity=entity_spec,
            entity_crosstabs=entity_crosstabs,
            ancestors=list(ancestors[entity_name]),
            ancestor_conditions=ancestor_conditions,
            children=list(children.get(entity_name, ())),
        )


def _find_children(entities: Dict[str, ModelEntitySpec]) -> Mapping[str, Set[str]]:
    """
    Return a dictionary mapping entity name to set of ancestors.
    """
    children: MapSet[str, str] = MapSet()
    for entity_name, entity_spec in entities.items():
        for dependency in entity_spec.foreign_key_fields:
            children.add(dependency.foreign_entity, entity_name)
    return children


def _find_ancestors(entities: Dict[str, ModelEntitySpec]) -> Dict[str, Set[str]]:
    """
    Return a dictionary mapping entity name to set of ancestors.
    """
    ancestors: Dict[str, Set[str]] = {}
    for entity_name in entities.keys():
        _find_ancestors_r(entity_name, entities, ancestors)
    return ancestors


def _find_ancestors_r(
        entity_name: str,
        entities: Dict[str, ModelEntitySpec],
        ancestors: Dict[str, Set[str]],
) -> Set[str]:
    """
    Recursively find all ancestors of the named entity.
    Results are cached in ancestors.
    Args:
        entity_name: The name of the entity.
        entities: all entities in the model.
        ancestors: cache of ancestors indexed by entity name.
    """
    entity_ancestors = ancestors.get(entity_name)
    if entity_ancestors is not None:
        return entity_ancestors
    spec = entities[entity_name]
    dependency: ForeignKeyField
    entity_ancestors: Set[str] = set()
    for dependency in spec.foreign_key_fields:
        entity_ancestors.add(dependency.foreign_entity)
        entity_ancestors.update(_find_ancestors_r(dependency.foreign_entity, entities, ancestors))
    ancestors[entity_name] = entity_ancestors
    return entity_ancestors


def _find_ancestor_conditions(
        entity_name: str,
        model_entities: Dict[str, ModelEntitySpec],
        condition_rvs: Set[str],
        ancestors: Set[str],
) -> List[AncestorConditionsIndex]:
    """
    Find ancestors of the given entity_spec that cover the given conditioning random variables.

    Args:
        entity_name: name of the entity.
        model_entities: the model specification.
        condition_rvs: The set of conditioning random variables.
        ancestors: set of ancestors for the given entity.
    Returns:
        A list of AncestorConditionsIndex, one for each conditioning random variable.
    Raises:
        SynthorusError if an ambiguous conditioning random variable is found.
        SynthorusError if a conditioning random variable is not found.
    """
    result: List[AncestorConditionsIndex] = []
    found: Set[str] = set()
    for ancestor in ancestors:
        ancestor_spec: ModelEntitySpec = model_entities[ancestor]
        for field_name, field in ancestor_spec.sampled_fields():
            rv_name: str = field.rv_name
            if rv_name in condition_rvs:
                if rv_name in found:
                    # The random variable appears multiple times in the ancestors.
                    # I.e. two or more fields sampling the same rv.
                    raise SynthorusError(
                        f'ambiguous conditioning random variable: {rv_name!r}, for entity: {entity_name!r}')
                found.add(rv_name)
                result.append(
                    AncestorConditionsIndex(
                        entity=ancestor,
                        field=field_name,
                        rv=rv_name,
                    )
                )
    return result


def find_covering_crosstabs(
        rvs: Collection[str],
        index: ModelIndex,
) -> List[EntityCrosstabIndex]:
    """
    Find a small set of cross-tables with distribution random variables covering the given rvs.

    Args:
        rvs: random variables to cover.
        index: The model index, with rvs and cross-tables already registered.

    Assumes:
        rvs and cross-tables are already registered with the index.

    Returns:
        entity_crosstabs: A list of `EntityCrosstabIndex` objects covering the given entity.
        condition_rvs: A list of rvs in the cross-tables but not sampled by the entity.
    """
    if len(rvs) == 0:
        # Base case and trivial case
        return []

    # Find all potential cross-tables
    # These are cross-tables that collectively provide distibutions for the given rvs.
    covering_crosstables: Set[str] = set()
    for rv_name in rvs:
        rv_index: RVIndex = index.rvs[rv_name]
        covering_crosstables.update(rv_index.all_distribution_crosstabs)

    # Find all potential EntityCrosstabIndex objects
    rvs_set: Set[str] = set(rvs)
    potentials: List[EntityCrosstabIndex] = []
    for crosstab_name in covering_crosstables:
        crosstab_index: CrosstabIndex = index.crosstabs[crosstab_name]
        crosstab_rvs = crosstab_index.rvs
        sampled_rvs: List[str] = list(rvs_set.intersection(crosstab_rvs))
        condition_rvs: List[str] = list(set(crosstab_rvs).difference(rvs))
        non_distribution_rvs: List[str] = crosstab_index.non_distribution_rvs.copy()

        potentials.append(
            EntityCrosstabIndex(
                crosstab=crosstab_name,
                sampled_rvs=sampled_rvs,
                condition_rvs=condition_rvs,
                non_distribution_rvs=non_distribution_rvs,
            )
        )

    def sort_key(_eci: EntityCrosstabIndex) -> int:
        """
        Used to order `potentials` from smallest to largest overlap with rvs_set
        """
        nonlocal rvs_set
        return len(rvs_set.intersection(_eci.sampled_rvs))

    # Find a covering set of EntityCrosstabIndex
    entity_crosstabs: List[EntityCrosstabIndex] = []
    entity_condition_rvs: Set[str] = set()
    while len(rvs_set) > 0 and len(potentials) > 0:
        potentials.sort(key=sort_key)
        potential: EntityCrosstabIndex = potentials.pop()
        entity_crosstabs.append(potential)
        rvs_set.difference_update(potential.sampled_rvs)
        entity_condition_rvs.update(potential.condition_rvs)

    # Check to see if any remaining potential cross-table adds more conditioning rvs
    potentials.sort(key=sort_key)
    while len(potentials) > 0:
        potential: EntityCrosstabIndex = potentials.pop()
        if not entity_condition_rvs.issuperset(potential.condition_rvs):
            entity_crosstabs.append(potential)
            entity_condition_rvs.update(potential.condition_rvs)

    return entity_crosstabs


def _resolve_rvs_states(
        model_index: ModelIndex,
        rvs_specs: Dict[str, ModelRVSpec],
        dataset_cache: DatasetCache,
) -> None:
    for rv_name, rv_index in model_index.rvs.items():
        rv_index.states = _resolve_rv_states(
            rv_name,
            rvs_specs[rv_name],
            rv_index,
            dataset_cache
        )


def _resolve_rv_states(
        rv_name: str,
        rv_spec: ModelRVSpec,
        rv_index: RVIndex,
        data_source_cache: DatasetCache,
) -> List[State]:
    """
    Convert the specified states in the rv_spec to a list of states.

    Args:
        rv_spec: spec of the random variable.
        rv_index: index of the random variable.
        data_source_cache: source of data, if needed.

    Returns:
        a list of states for the random variable.
    """
    states_spec = rv_spec.states
    need_none: bool = rv_spec.ensure_none
    base_states: List[State]

    # RV states specified directly
    if isinstance(states_spec, int):
        base_states = list(range(states_spec))
    elif isinstance(states_spec, list):
        base_states = list(states_spec)
    else:
        # Remaining options require access to datasources
        datasets: List[Dataset] = [
            data_source_cache[datasource_name]
            for datasource_name in rv_index.all_datasources
        ]
        if not need_none:
            need_none = any(datasource.value_maybe_none(rv_name) for datasource in datasets)

        if states_spec == 'infer_distinct':
            distinct: Set[State] = set(
                clean_state(s)
                for dataset in datasets
                for s in dataset.value_set(rv_name)
            )
            if not need_none:
                need_none = None in distinct
            distinct.discard(None)
            base_states = sorted(distinct, key=cmp_to_key(_compare_states))
        elif states_spec == 'infer_range':
            min_val, max_val = _infer_range(rv_name, datasets)
            base_states = list(range(min_val, max_val + 1))
        elif states_spec == 'infer_max':
            _, max_val = _infer_range(rv_name, datasets)
            base_states = list(range(max_val + 1))
        else:
            raise SynthorusError(f'random variable ({rv_name!r}) state specification not understood: {states_spec!r}')

    if need_none:
        base_states.append(None)

    return base_states


def _infer_range(rv_name: str, datasets: List[Dataset]) -> Tuple[int, int]:
    """
    Infer the range of values for a random variable over one or more datasets.

    Raises:
        SynthorusError: If values cannot be interpreted as an integer.
    """
    if len(datasets) == 0:
        raise SynthorusError(f'cannot infer value range for random variable {rv_name!r}: no datasets')

    min_val: int
    max_val: int
    min_val, max_val = _min_max_ints(rv_name, datasets[0])

    for dataset in datasets[1:]:
        min_val_dataset, max_val_dataset = _min_max_ints(rv_name, dataset)
        min_val = min(min_val, min_val_dataset)
        max_val = max(max_val, max_val_dataset)

    return min_val, max_val


def _min_max_ints(rv_name: str, dataset: Dataset) -> tuple[int, int]:
    """
    Get the minimum and maximum values of a random variable for a dataset, ensuring that they are integers.
    """
    min_state: State = dataset.value_min(rv_name)
    max_state: State = dataset.value_max(rv_name)

    try:
        min_int: int = int(min_state)  # type: ignore
    except (ValueError, TypeError):
        raise SynthorusError(f'random variable ({rv_name!r}) state minimum value not an int: {min_state!r}')
    try:
        max_int: int = int(max_state)  # type: ignore
    except (ValueError, TypeError):
        raise SynthorusError(f'random variable ({rv_name!r}) state maximum value not an int: {max_state!r}')

    return min_int, max_int


def _compare_states(s1: State, s2: State) -> int:
    """
    Compare two states for sorting purposes.
    This function is used to sort states meaningfully, especially when states are not directly comparable.
    """
    # None is always larger than anything else
    if s1 is None:
        return 1  # s2 comes first
    if s2 is None:
        return -1  # s1 comes first

    # Normally comparable...
    if isinstance(s1, bool) and isinstance(s2, bool):
        return (s1 > s2) - (s1 < s2)
    if isinstance(s1, (int, float)) and isinstance(s2, (int, float)):
        return (s1 > s2) - (s1 < s2)
    if isinstance(s1, str) and isinstance(s2, str):
        return (s1 > s2) - (s1 < s2)

    # Put booleans first
    if isinstance(s1, bool):
        return -1
    if isinstance(s2, bool):
        return 1

    # Then numbers
    if isinstance(s1, (int, float)):
        return -1
    if isinstance(s2, (int, float)):
        return 1

    # This should never be reached as we should have handled all cases
    assert False, f'cannot compare {s1!r} and {s2!r}'
