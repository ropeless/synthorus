from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import List, TypeAlias, Dict, Set, Sequence, Collection

import pandas as pd
from ck.dataset.cross_table import CrossTable
from ck.in_out.pgm_python import write_python
from ck.learning.condition_cross_table import condition
from ck.learning.model_from_cross_tables import model_from_cross_tables
from ck.pgm import PGM, RandomVariable

from synthorus.error import SynthorusError
from synthorus.model.make_model_index import find_covering_crosstabs
from synthorus.model.model_index import ModelIndex, EntityIndex, EntityCrosstabIndex
from synthorus.utils.clean_state import clean_state
from synthorus.utils.print_function import PrintFunction
from synthorus.workflows.cross_table_loader import CrossTableLoader

_Factor: TypeAlias = List[str]  # A collection of random variable names.


def make_entity_pgms(
        model_index: ModelIndex,
        crosstab_loader: CrossTableLoader,
        save_directory_path: Path,
        *,
        log: PrintFunction = print,
) -> None:
    """
    Make a PGM model for each entity needing a sampler.
    Saves each PGM, in Python format, in `save_directory_path`.

    This method will read the cross-table files (and associated metadata)
    as stored in the given directory path. These would have been created
    by a previous call to 'make_cross_tables` (in module `model_crosstabs`)
    with the same directory path.

    Model files will be written to the given directory. The model files
    may be subsequently used by function `make_simulator`.

    Args:
        model_index: All the important characteristics of the model for PGMs.
        crosstab_loader: Access to crosstables.
        save_directory_path: Where to write the PGMs - an existing empty directory.
        log: A destination for log messages, a print function.

    Assumes:
        `save_directory_path` exists and is empty.
    """
    log('make_entity_pgms started')
    for entity_name in model_index.entities.keys():
        pgm: PGM = _make_entity_pgm(
            entity_name,
            model_index,
            crosstab_loader,
            log,
        )
        # Save the PGM for the entity
        with open(save_directory_path / f'{entity_name}.py', 'w') as file:
            write_python(pgm, file=file)

    log('make_entity_pgms completed')


def _make_entity_pgm(
        entity_name: str,
        model_index: ModelIndex,
        crosstab_loader: CrossTableLoader,
        log,
) -> PGM:
    """
    Make a PGM for the given entity, assuming that PGMs for ancestor
    entities are completed (and registered in `completed_entities`).

    The algorithm:
    1. find all cross-tables with at least one random variable in common
       with the entity's random variables (sampled fields).

    Args:
        entity_name:
        model_index:
        crosstab_loader:
        log:

    Returns:
        a PGM for the entity.
    """
    log(f'making PGM for entity: {entity_name}')
    pgm = PGM(entity_name)

    # Prepare to get all CK cross-tables that pertain to this entity
    crosstab_maker = EntityCrossTableMaker(
        crosstab_loader=crosstab_loader,
        model_index=model_index,
        entity_index=model_index.entities[entity_name],
        pgm=pgm,
        add_rvs=True,
    )

    log('sample_rvs:', ', '.join(crosstab_maker.sample_rv_names))

    if len(crosstab_maker.sample_rv_names) == 0:
        # Nothing to do
        return pgm

    # Get all the cross-tables for the entity
    cross_tables: List[CrossTable] = crosstab_maker.get_cross_tables()

    # Add factors to the PGM (using CK).
    model_from_cross_tables(pgm, cross_tables)

    # Check that no potential function is zero as that implies the PGM cannot be sampled.
    # If so, it's probably an empty cross-tables.
    if any(factor.is_zero for factor in pgm.factors):
        raise SynthorusError(f'could not make a valid PGM for entity {entity_name} (probably an empty cross-table)')

    log(f'finished making PGM for entity: {entity_name}')
    return pgm


class EntityCrossTableMaker:
    """
    Loads cross-tables, converts them to CK cross-tables, resolves
    non-distribution rvs, and projects to remove any unneeded random variables.
    """

    def __init__(
            self,
            crosstab_loader: CrossTableLoader,
            model_index: ModelIndex,
            entity_index: EntityIndex,
            pgm: PGM,
            add_rvs: bool,
    ):
        self.crosstab_loader = crosstab_loader
        self.model_index = model_index
        self.entity_index = entity_index

        self.sample_rv_names: Set[str] = entity_index.sample_rvs()
        self.ancestor_rv_names: Set[str] = entity_index.ancestor_rvs()
        self.condition_rv_names: Set[str] = entity_index.condition_rvs()

        if add_rvs:
            for rv_name in chain(self.sample_rv_names, self.condition_rv_names):
                pgm.new_rv(rv_name, model_index.rvs[rv_name].states)

        # Index PGM random variables by name
        self.rv_map: Dict[str, RandomVariable] = {
            rv.name: rv
            for rv in pgm.rvs
        }

    @property
    def needed_rv_names(self) -> Collection[str]:
        """
        What random variables are needed for the entity.
        This is sampled random variables plus ancestor conditioning random variables.

        Returns:
            A set like object with str elements.
        """
        return self.rv_map.keys()

    def get_cross_tables(self) -> List[CrossTable]:
        """
        Get all the cross-tables needed for the entity.
        """
        return [
            self._get_cross_table(entity_crosstab_index, allow_ancestor_conditioning=True)
            for entity_crosstab_index in self.entity_index.entity_crosstabs
        ]

    def _get_cross_table(
            self,
            entity_crosstab_index: EntityCrosstabIndex,
            allow_ancestor_conditioning: bool,
    ) -> CrossTable:
        """
        Construct a CrossTable object from the given cross-table dataframe.

        This is a recursive method that will call itself to resolve non-distribution rvs.

        Args:
            entity_crosstab_index: The cross-table to get for the entity.
            allow_ancestor_conditioning: Whether to allow ancestor conditioning of non-distribution rvs.

        Returns:
            a CK CrossTable object.
        """
        # Get the synthorus cross-table as a dataframe
        dataframe: pd.DataFrame = self.crosstab_loader[entity_crosstab_index.crosstab]

        # Get the CK random variables from dataframe column names
        rvs: List[RandomVariable] = [self.rv_map[rv_name] for rv_name in dataframe.columns[:-1]]

        # Convert to a CK cross-table
        ck_crosstab: CrossTable = CrossTable(
            rvs,
            update=(
                (tuple(rv.state_idx(clean_state(state)) for rv, state in zip(rvs, row[:-1])), row[-1])
                for row in dataframe.itertuples(index=False)
            )
        )

        # Get the non-distribution rvs for the current cross-table.
        # If there are any non-distribution rvs, then we need to recondition the cross-table.
        non_distribution_rvs: List[str] = entity_crosstab_index.non_distribution_rvs
        if allow_ancestor_conditioning:
            # Remove non-distribution rvs that an ancestor will condition.
            non_distribution_rvs = [rv for rv in non_distribution_rvs if rv not in self.ancestor_rv_names]
        else:
            # Not possible to correctly condition non-distribution rvs that an ancestor will condition
            for rv_name in non_distribution_rvs:
                if rv_name not in self.ancestor_rv_names:
                    raise SynthorusError(
                        f'random variable {rv_name!r} cannot be correctly conditioned'
                        f' for cross-table {entity_crosstab_index.crosstab!r}'
                    )

        # Apply cross-table conditioning for any non-distribution rvs
        for entity_crosstab_index in find_covering_crosstabs(non_distribution_rvs, self.model_index):
            conditioner = self._get_cross_table(entity_crosstab_index, allow_ancestor_conditioning=False)
            condition_rvs_set = set(ck_crosstab.rvs).intersection(conditioner.rvs)
            ck_crosstab = condition(ck_crosstab, conditioner, condition_rvs_set)

        # Project cross-table to remove unnecessary rvs.
        project_rvs: Sequence[RandomVariable] = tuple(rv for rv in ck_crosstab.rvs if rv.name in self.needed_rv_names)
        if project_rvs != ck_crosstab.rvs:
            ck_crosstab = ck_crosstab.project(project_rvs)

        return ck_crosstab
