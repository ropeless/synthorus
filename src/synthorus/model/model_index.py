from __future__ import annotations

from typing import Dict, List, Optional, Set

from ck.pgm import State
from pydantic import BaseModel


class ModelIndex(BaseModel):
    """
    A model index keeps track of interrelationships between components
    of a model spec.

    A model index can be constructed for a `ModelSpec` object using
    `make_model_index` or as part of `make_model_definition_files`.
    """
    rvs: Dict[str, RVIndex] = {}
    crosstabs: Dict[str, CrosstabIndex] = {}
    entities: Dict[str, EntityIndex] = {}


class RVIndex(BaseModel):
    """
    A random variable in a ModelIndex.
    """
    name: str  # The name of the random variable

    states: List[State]  # The states of the random variable (in order).

    primary_datasource: str
    all_datasources: List[str] = []  # All datasources that mention this random variable.
    all_distribution_crosstabs: List[str] = []  # All cross-tables mentioning this random variable as a distribution rv.
    all_sampling_entities: List[str] = []  # All entities that sample this random variable.


class CrosstabIndex(BaseModel):
    """
    A cross-table in a ModelIndex.
    """
    name: str  # The name of the cross-table

    rvs: List[str]  # The random variables of the cross-table.
    non_distribution_rvs: List[str]  # The rvs that should _not_ be considered as providing a distribution
    distribution_rvs: List[str]  # The rvs that should be considered as providing a distribution
    datasource: str  # The covering dataset for the cross-table
    number_of_states: int  # The total number of possible states of the cross-table


class EntityIndex(BaseModel):
    """
    An entity in a ModelIndex.
    """
    name: str  # The name of the entity

    parent: Optional[str]  # The parent entity.
    sampled_fields: Dict[str, str]  # mapping field name -> random variable name
    entity_crosstabs: List[EntityCrosstabIndex]  # What cross-tables relate to this entity.
    ancestor_conditions: List[AncestorConditionsIndex]  # what ancestor entities cover `condition_rvs`

    def sample_rvs(self) -> Set[str]:
        return set(self.sampled_fields.values())

    def ancestor_rvs(self) -> Set[str]:
        return {i.rv for i in self.ancestor_conditions}

    def condition_rvs(self) -> Set[str]:
        result = set()
        for crosstab in self.entity_crosstabs:
            result.update(crosstab.condition_rvs)
        return result


class EntityCrosstabIndex(BaseModel):
    """
    The intersection between an entity and cross-tables.


    A set of random variables of an entity, that can be used to
    create and/or evaluate and entity's PGM.
    """
    crosstab: str  # The cross-table providing the random variables
    sampled_rvs: List[str]  # The intersection of the entity's rvs and the cross-table's rvs.
    condition_rvs: List[str]  # The cross-table rvs not sampled by the entity
    non_distribution_rvs: List[str]  # The rvs that should _not_ be considered as providing a distribution

    def distribution_rvs(self) -> Set[str]:
        """
        Cross-table rvs that are not `non_distribution_rvs`.
        """
        return set(self.sampled_rvs).union(self.condition_rvs).difference(self.non_distribution_rvs)


class AncestorConditionsIndex(BaseModel):
    """
    Keep track of where an ancestor field is for a random variable that
    is used to condition the random variable of PGM for a descendant entity.
    """
    entity: str  # an ancestor entity
    field: str  # the sampled field of `entity` that samples the rv.
    rv: str  # the random variable.
