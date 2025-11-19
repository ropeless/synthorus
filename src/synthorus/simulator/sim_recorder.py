from abc import ABC, abstractmethod
from io import TextIOWrapper
from os import PathLike
from pathlib import Path
from typing import Sequence, Iterator, Dict, Iterable, Tuple, Mapping, List

from ck.pgm import State

from synthorus.error import SynthorusError
from synthorus.simulator.sim_record import SimRecord
from synthorus.utils.data_catcher import RamDataCatcher


class SimRecorder(ABC):
    """
    Abstract class for recording simulation results.
    """

    @abstractmethod
    def start_entity(self, entity_name: str, field_names: Sequence[str]) -> int:
        """
        Prepare to write records for the given named entity.
        Return the first id number that should be used for records of this entity,
        I.e., record ids passed to write_record will be >= the value returned
        by this method.

        This gets called by the simulator once per entity before records are generate
        to let the recorder know what kinds of records to expect.
        """

    @abstractmethod
    def write_record(self, entity_name: str, record: SimRecord) -> None:
        """
        Record one record for the given named entity.
        This will only be called between self.start_entity(entity_name, field_names)
        and self.finish(). The fields of the given record are co-indexed with
        the field_names passed to start_entity.
        """

    @abstractmethod
    def finish(self) -> None:
        """
        Called when the simulation finished, even if an exception is thrown.
        """


class CompositeRecorder(SimRecorder):
    """
    A SimRecorder that delegates to multiple other SimRecorder objects.
    """

    def __init__(self, *recorders: SimRecorder):
        """
        Create a CompositeRecorder that delegates to the given SimRecorder objects.
        There should be no duplicates in the objects passed (including the delegates
        of nested CompositeRecorder objects).
        If no recorders are passed to the constructor, then this SimRecorder
        does nothing.
        """
        self.recorders = recorders

        # Ensure there are no repeated recorders as
        # that can mess up the state of the recorders due to multiple
        # start and finish calls made to the same SimRecorder object.
        all_recorders = list(self._all_recorder_r())
        if len({id(r) for r in all_recorders}) != len(all_recorders):
            raise SynthorusError(f'duplicate SimRecorder objects not allowed')

    def _all_recorder_r(self) -> Iterator[SimRecorder]:
        """
        Recursively yield all recorders, including the delegates
        of nested CompositeRecorder objects, but not the
        CompositeRecorder objects themselves.
        """
        for recorder in self.recorders:
            if isinstance(recorder, CompositeRecorder):
                for sub_recorder in recorder._all_recorder_r():
                    yield sub_recorder
            else:
                yield recorder

    def start_entity(self, entity_name: str, field_names: Sequence[str]) -> int:
        """
        Passes 'start_entity' to all delegates.
        The result (first id number) is the maximum over results from delegates.
        """
        result = 0
        for recorder in self.recorders:
            start_id = recorder.start_entity(entity_name, field_names)
            result = max(result, start_id)
        return result

    def finish(self) -> None:
        """
        Passes 'finish' to all delegates.
        """
        for recorder in self.recorders:
            recorder.finish()

    def write_record(self, entity_name: str, record: SimRecord) -> None:
        """
        Passes 'write_record' to all delegates.
        """
        for recorder in self.recorders:
            recorder.write_record(entity_name, record)


class CSVRecorder(SimRecorder):
    """
    A SimRecorder that writes records to a CSV file for each entity.
    There is one file per entity in a nominated directory.

    The recorder saves each record to file as the record is provided.
    Thus, if the simulator fails to complete for some reason, the
    records are still available.
    """

    def __init__(self, directory: PathLike, sep: str = ',', start_id: int = 1) -> None:
        self._directory: Path = Path(directory)
        self._files: Dict[str, TextIOWrapper] = {}
        self._field_names: Dict[str, List[str]] = {}
        self._sep: str = sep
        self._start_id: int = start_id

    def start_entity(self, entity_name: str, field_names: Sequence[str]) -> int:
        assert entity_name not in self._files.keys()

        self._directory.mkdir(parents=False, exist_ok=True)

        path = self._directory / (entity_name + '.csv')
        file = open(path, 'w')
        self._files[entity_name] = file
        self._field_names[entity_name] = list(field_names)
        self._write(file, field_names)
        return self._start_id

    def finish(self) -> None:
        exception = None
        for file in self._files.values():
            try:
                file.close()
            except Exception as err:
                if exception is None:
                    exception = err
        self._files = {}
        if exception is not None:
            raise exception

    def write_record(self, entity_name: str, record: SimRecord) -> None:
        file = self._files[entity_name]
        fields = self._field_names[entity_name]
        self._write(file, [record.get(field) for field in fields])

    def _write(self, file, values: Sequence[State]) -> None:
        file.write(self._sep.join(self._clean_val(val) for val in values))
        file.write('\n')
        file.flush()

    @staticmethod
    def _clean_val(val: State) -> str:
        """
        This is just a hack to fix some issues with CSV field quoting.
        It can be improved.
        """
        if isinstance(val, (int, float, type(None))):
            return str(val)
        if not isinstance(val, str):
            val = repr(val)
        if ',' in val or ' ' in val:
            return '"' + val.replace('"', "'") + '"'
        return val

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        self.finish()
        return t is None


class MemoryRecorder(SimRecorder):
    """
    A SimRecorder that records to memory.

    Access the records using the `records` property.
    The records of each entity is stored as a `RamDataCatcher`.
    """

    def __init__(self):
        self._records: Dict[str, RamDataCatcher] = {}

    @property
    def records(self) -> Mapping[str, RamDataCatcher]:
        """
        Get the records as a mapping from entity name to `RamDataCatcher`.

        Examples usage:
        ```
        simulator: Simulator = ...

        recoder = MemoryRecorder()
        simulator.run(recorder)

        my_entity_records: RamDataCatcher = recorder.records['my_entity']
        for record in my_entity_records:
            print(record)
        ```

        Returns:
            A mapping from entity name to `RamDataCatcher`.
        """
        return self._records

    def start_entity(self, entity_name: str, field_names: Iterable[str]) -> int:
        records = self._records.get(entity_name)
        if records is None:
            self._records[entity_name] = records = RamDataCatcher()
        return len(records) + 1

    def finish(self) -> None:
        pass

    def write_record(self, entity_name: str, record: SimRecord) -> None:
        mem_record = self._records[entity_name].append()
        for col, val in record.items():
            mem_record[col] = val


class DebugRecorder(SimRecorder):
    """
    A SimRecorder that prints records (for debugging and demonstrations).
    """

    def __init__(
            self,
            file=None,
            blank_line_between_entities: bool = False,
            entity_start_ids: Mapping[str, int] | Iterable[Tuple[str, int]] = (),
    ):
        """
        Create a DebugRecorder.

        Args:
            file: optional file to write records to.
            entity_start_ids: optional mapping from entity names to start id (default start id is 1).
            blank_line_between_entities: flag to insert a blank line between blocks of entity records.
        """
        self._file = file
        self._blank_line_between_entities = blank_line_between_entities
        self._entity_start_ids: Dict[str, int] = dict(entity_start_ids)

        self._writing: bool = False
        self._last_entity: str = ''

    def start_entity(self, entity_name: str, field_names: Iterable[str]) -> int:
        """
        Print the entity name and its fields.
        """
        print(f'Entity: {entity_name} {list(field_names)}', file=self._file)

        self._writing = False
        self._last_entity = ''

        # Return the starting id number
        return self._entity_start_ids.get(entity_name, 1)

    def write_record(self, entity_name: str, record: SimRecord) -> None:
        """
        Print the record.
        """
        # This puts a blank line between any `start_entity` lines and the record lines.
        if not self._writing:
            print(file=self._file)
            self._writing = True

        if self._blank_line_between_entities and entity_name != self._last_entity:
            if self._last_entity != '':
                print(file=self._file)
            self._last_entity = entity_name

        # Print the record
        values = ', '.join(repr(val) for val in record.items())
        print(f'{entity_name} [{values}]', file=self._file)

    def finish(self) -> None:
        """
        Print a blank line then "Finished".
        """
        self._writing = False
        self._last_entity = ''

        print(file=self._file)
        print('Finished', file=self._file)
