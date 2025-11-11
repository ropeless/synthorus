from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, List, Optional, Sequence, Mapping

from synthorus.error import SynthorusError
from synthorus.model.defaults import DEFAULT_ID_FIELD, DEFAULT_COUNT_FIELD
from synthorus.simulator.no_sim_sampler import NO_SIM_SAMPLER
from synthorus.simulator.sim_entity import SimEntity, SimSampler
from synthorus.simulator.sim_field import SimField
from synthorus.simulator.sim_field_updaters import NO_UPDATE
from synthorus.simulator.sim_recorder import SimRecorder


class Simulator:

    def __init__(self):
        self._entities: Dict[str, SimEntity] = {}
        self._parameters: Dict[str, SimField] = {}

    def add_entity(
            self,
            name: str,
            *,
            id_field_name: str = DEFAULT_ID_FIELD,
            count_field_name: str = DEFAULT_COUNT_FIELD,
            foreign_field_name: Optional[str] = None,
            parent: Optional[SimEntity] = None,
            sampler: SimSampler = NO_SIM_SAMPLER
    ) -> SimEntity:
        """
        Add an entity to the simulator.

        Args:
            name: a name of the entity, which must be unique.
            id_field_name: the name given to the entity's ID field.
            count_field_name: the name given to the entity's count field.
            foreign_field_name: the name given to the entity's foreign ID field. Ignored if
                `parent` is None, required otherwise.
            parent: optional parent entity.
            sampler: what sampler to use for the entities sampled fields.

        Returns:
             the newly created entity.
        """
        if name in self._entities.keys():
            raise SynthorusError(f'entity name must be unique: {name!r}')

        if parent is None:
            foreign_field_name = None  # not needed - ignore it
        else:
            if self._entities[parent.name] is not parent:
                raise SynthorusError(f'parent entity must be in this simulator: {parent.name!r}')
            if foreign_field_name is None:
                raise SynthorusError(f'parent entity specified by no foreign field name: {parent.name!r}')

        entity = SimEntity(
            name=name,
            id_field_name=id_field_name,
            count_field_name=count_field_name,
            foreign_id_field_name=foreign_field_name,
            parent=parent,
            sampler=sampler,
        )
        self._entities[name] = entity
        return entity

    def entity(self, entity_name: str) -> SimEntity:
        """
        Get an entity by name.
        """
        return self._entities[entity_name]

    def add_parameter(self, parameter_name, value=None) -> SimField:
        sim_field = SimField(
            parameter_name,
            value,
            NO_UPDATE
        )
        self._parameters[parameter_name] = sim_field
        return sim_field

    @property
    def parameters(self) -> Mapping[str, SimField]:
        """
        Get all parameters.
        """
        return MappingProxyType(self._parameters)

    def run(self, recorder: SimRecorder, iterations: int = 1) -> None:
        """
        Run the simulation.

        If iterations is 1 (the default) then each root entity
        is run once, and the number of records for the entity is
        determined by the entity's cardinality conditions.

        If iterations > 1 then the root entities will be run
        that multiple of times.

        If iterations <= 0, then the entities will be initialised
        but not run, i.e., no records will be generated.

        Ensures:
            `recorder.finish()` is called, even if an exception is raised
            by the simulator. This ensures that a SimRecorder is given
            an opportunity to close resources that may be allocated
            (like opened files).

        Warning:
            If the simulation raises an exception then entity fields
            may be left in an arbitrary state. This may cause issues
            any field that is updated based on previous states of fields.
        """
        try:
            # Infer an entity tree
            roots: Sequence[_SimNode] = self._form_tree()

            # Initialise entities and recorder
            for entity in self._entities.values():
                field_names: Sequence[str] = tuple(iter(entity))
                start_id: int = recorder.start_entity(entity.name, field_names)
                entity.initialise(start_id - 1)

            # Run the simulation
            for _ in range(iterations):
                self._run_nodes(roots, recorder)

            # Reset the entities field values
            for entity in self._entities.values():
                entity.reset_fields()

        finally:
            recorder.finish()

    def _run_nodes(self, nodes: Sequence[_SimNode], recorder: SimRecorder):
        for node in nodes:
            self._run_node(node, recorder)

    def _run_node(self, node: _SimNode, recorder: SimRecorder):
        entity: SimEntity = node.entity
        entity.start()
        while not entity.stop():
            record = entity.update()
            recorder.write_record(entity.name, record)
            self._run_nodes(node.children, recorder)

    def _form_tree(self) -> Sequence[_SimNode]:
        nodes: Dict[str, _SimNode] = {}
        for entity in self._entities.values():
            self._form_tree_r(nodes, entity)
        roots = [node for node in nodes.values() if node.entity.parent is None]
        return roots

    def _form_tree_r(self, nodes: Dict[str, _SimNode], entity: SimEntity) -> _SimNode:
        node = nodes.get(entity.name)
        if node is None:
            node = _SimNode(entity)
            nodes[entity.name] = node
            parent = entity.parent
            if parent is not None:
                parent_node = self._form_tree_r(nodes, parent)
                parent_node.children.append(node)
        return node


@dataclass
class _SimNode:
    """
    A Node in an entity tree.
    """

    entity: SimEntity
    children: List[_SimNode] = field(default_factory=list)
