from __future__ import annotations

from typing import List, Set, Dict, Sequence, Optional, Collection

from ck.pgm import State
from ck.utils.iter_extras import multiply
from ck.utils.map_list import MapList

from synthorus.dataset import Dataset
from synthorus.error import SynthorusError
from synthorus.model.dataset_cache import DatasetCache
from synthorus.model.datasource_spec import DatasourceSpec
from synthorus.model.model_index import ModelIndex, RVIndex, CrosstabIndex, EntityIndex, EntityCrosstabIndex, \
    AncestorConditionsIndex
from synthorus.model.model_spec import ModelRVSpec, ModelCrosstabSpec, ModelEntitySpec, ModelSpec, ModelFieldSpecSample
from synthorus.utils.clean_state import clean_state


def make_model_index(model_spec: ModelSpec, dataset_cache: DatasetCache) -> ModelIndex:
    """
    Make a ModelIndex object for the given model spec.

    Args:
        model_spec: The model spec to index.
        dataset_cache: Access to datasources.

    Returns:
        model_index.
    """
    index = ModelIndex()
    potential_dist_datasources = MapList[str, str]()

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
        index.rvs[rv_name] = RVIndex(name=rv_name, states=dummy_states, primary_datasource=dummy_datasource)


def _register_datasources(
        index: ModelIndex,
        datasources: Dict[str, DatasourceSpec],
        potential_dist_datasources: MapList[str, str],
) -> None:
    """
    For each dataset in `datasources`:
        for each rv in dataset:
            register the dataset with rv.all_datasources.
            if rv is not a conditioned random variable
                register the dataset with potential_dist_datasources[rv].

    Args:
        index: index being updated
        datasources: all available datasources.
        potential_dist_datasources: for each random variable, a list datasources
            providing a distribution for the random variable.
    """
    datasource_name: str
    datasource: DatasourceSpec
    for datasource_name, datasource in datasources.items():
        # Index the datasource with its random variables
        non_distribution_rvs: Set[str] = set(datasource.non_distribution_rvs)
        for rv_name in datasource.rvs:
            rv_index: RVIndex = index.rvs[rv_name]
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
    for rv_index in index.rvs.values():
        dist_datasources: Sequence[str] = potential_dist_datasources.get(rv_index.name, ())
        if len(dist_datasources) == 0:
            raise SynthorusError(
                f'random variable {rv_index.name!r} has no distribution datasource'
            )
        rv_index.primary_datasource = dist_datasources[0]  # TODO should we do better?


def _register_crosstabs(index: ModelIndex, model_spec: ModelSpec) -> None:
    """
    Register cross-tables with the given model index.
    All `CrosstabIndex.dataset` values will have a placeholder object that should be resolved.
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

        for non_distribution_rv in non_distribution_rvs:
            if non_distribution_rv not in rvs:
                raise SynthorusError(
                    f'cross-table {crosstab_name!r}'
                    f' must include non-distribution random variable: {non_distribution_rv!r}'
                )

        # Index the cross-table with its distribution random variables
        for rv_name in distribution_rvs:
            rv_index: RVIndex = index.rvs[rv_name]
            rv_index.all_distribution_crosstabs.append(crosstab_name)

        # Calculate the total state space
        number_of_states: int = multiply(len(index.rvs[rv_name].states) for rv_name in rvs)

        index.crosstabs[crosstab_name] = CrosstabIndex(
            name=crosstab_name,
            rvs=rvs,
            non_distribution_rvs=non_distribution_rvs,
            distribution_rvs=distribution_rvs,
            datasource=datasource_name,
            number_of_states=number_of_states,
        )


def _register_entities(index: ModelIndex, model_spec: ModelSpec) -> None:
    """
    Assumes:
        rvs and cross-tables are already registered with the index.

    Args:
        index: The model index being updated.
        model_spec: The model spec holding the entities to be registered.
    """
    entity_name: str
    entity_spec: ModelEntitySpec
    for entity_name, entity_spec in model_spec.entities.items():
        field: ModelFieldSpecSample
        sampled_fields: Dict[str, str] = {
            field_name: field.rv_name
            for field_name, field in entity_spec.sampled_fields()
        }
        entity_crosstabs: List[EntityCrosstabIndex] = find_covering_crosstabs(sampled_fields.values(), index)
        ancestor_conditions: List[AncestorConditionsIndex] = []
        condition_rvs: Set[str] = {
            rv_name
            for entity_crosstab in entity_crosstabs
            for rv_name in entity_crosstab.condition_rvs
        }
        _find_ancestor_conditions(entity_name, model_spec, entity_spec, condition_rvs, set(), ancestor_conditions)

        # Register the entity with its rvs
        for rv_name in sampled_fields.values():
            index.rvs[rv_name].all_sampling_entities.append(entity_name)

        index.entities[entity_name] = EntityIndex(
            name=entity_name,
            parent=entity_spec.parent,
            sampled_fields=sampled_fields,
            entity_crosstabs=entity_crosstabs,
            ancestor_conditions=ancestor_conditions,
        )


def _find_ancestor_conditions(
        entity_name: str,
        model_spec: ModelSpec,
        entity_spec: ModelEntitySpec,
        condition_rvs: Set[str],
        found: Set[str],
        result: List[AncestorConditionsIndex],
) -> None:
    """
    Find ancestors of the given entity_spec that cover the given conditioning random variables.

    Args:
        model_spec: the model specification.
        entity_spec: The entity of interest.
        condition_rvs: The set of conditioning random variables.
        found: The names of random variables in `result`.
        result: Where to store the findings.
    """
    parent_name: Optional[str] = entity_spec.parent
    if parent_name is not None:
        parent_spec: ModelEntitySpec = model_spec.entities[parent_name]
        for field_name, field in parent_spec.sampled_fields():
            rv_name: str = field.rv_name
            if rv_name in condition_rvs:
                if rv_name in found:
                    # The random variable appears multiple times in the ancestors.
                    # I.e. two or more fields sampling the same rv.
                    raise SynthorusError(f'ambiguous conditioning random variable: {rv_name!r}, for entity: {entity_name!r}')
                found.add(rv_name)
                result.append(
                    AncestorConditionsIndex(
                        entity=parent_name,
                        field=field_name,
                        rv=rv_name,
                    )
                )
        # Continue searching up the entity hierarchy
        _find_ancestor_conditions(entity_name, model_spec, parent_spec, condition_rvs, found, result)


def find_covering_crosstabs(
        rvs: Collection[str],
        index: ModelIndex,
) -> List[EntityCrosstabIndex]:
    """
    Find a small set of cross-tables with distribution random variables covering the given rvs.

    Args:
        rvs: random variables to cover.
        index: The model index, with rvs and cross-tables are already registered.

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
    # These are crosstables that collectively provide distibutions for the given rvs.
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
    for rv_index in model_index.rvs.values():
        rv_index.states = _resolve_rv_states(
            rvs_specs[rv_index.name],
            rv_index,
            dataset_cache
        )


def _resolve_rv_states(
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
    rv_name: str = rv_index.name
    need_none: bool = rv_spec.ensure_none
    base_states: List[State]

    # RV states specified directly
    if isinstance(states_spec, int):
        base_states = _make_states_range(rv_name, 0, states_spec - 1)
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
            base_states = sorted(distinct)
        elif states_spec == 'infer_range':
            min_val = max(datasource.value_min(rv_name) for datasource in datasets)
            max_val = max(datasource.value_max(rv_name) for datasource in datasets)
            base_states = _make_states_range(rv_name, min_val, max_val)
        elif states_spec == 'infer_max':
            max_val = max(datasource.value_max(rv_name) for datasource in datasets)
            base_states = _make_states_range(rv_name, 0, max_val)
        else:
            raise SynthorusError(f'random variable ({rv_name!r}) state specification not understood: {states_spec!r}')

    if need_none:
        base_states.append(None)

    return base_states


def _make_states_range(rv_name: str, from_val: State, to_val: State) -> List[int]:
    """
    Construct a list of states (for the named random variable) `from_val` -- `to_val`, inclusive.
    Raises:
        SynthorusError: If `from_val` or `to_val` cannot be interpreted as an integer.
    """
    try:
        from_int: int = int(from_val)
    except ValueError:
        raise SynthorusError(f'random variable ({rv_name!r}) state minimum value not an int: {from_val!r}')
    try:
        to_int: int = int(to_val)
    except ValueError:
        raise SynthorusError(f'random variable ({rv_name!r}) state maximum value not an int: {from_val!r}')

    return list(range(from_int, to_int + 1))
