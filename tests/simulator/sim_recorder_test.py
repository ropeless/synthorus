from io import StringIO
from typing import List

from synthorus.simulator.sim_recorder import MemoryRecorder, SimRecorder, CSVRecorder, DebugRecorder, CompositeRecorder
from synthorus.utils.data_catcher import RamDataCatcher
from synthorus.utils.string_extras import unindent
from tests.helpers.tmp_dir import tmp_dir
from tests.helpers.unittest_fixture import Fixture, test_main


class SimRecorderTest(Fixture):

    def test_memory_recorder(self) -> None:
        recorder = MemoryRecorder()
        simulate(recorder)

        self.assertEqual(set(recorder.records.keys()), {'entity_1', 'entity_2'})

        entity_1: RamDataCatcher = recorder.records['entity_1']
        entity_2: RamDataCatcher = recorder.records['entity_2']

        self.assertEqual(len(entity_1), 3)
        self.assertEqual(entity_1[0], {'A': 1, 'B': 2, 'C': 3})
        self.assertEqual(entity_1[1], {'A': 4, 'B': 5, 'C': 6})
        self.assertEqual(entity_1[2], {'A': 7, 'B': 8, 'C': 9})

        self.assertEqual(len(entity_2), 4)
        self.assertEqual(entity_2[0], {'D': 'a', 'E': 'b'})
        self.assertEqual(entity_2[1], {'D': 'c', 'E': 'd'})
        self.assertEqual(entity_2[2], {'D': 'e', 'E': 'f'})
        self.assertEqual(entity_2[3], {'D': 'g', 'E': 'h'})

    def test_debug_recorder(self) -> None:
        file = StringIO()
        recorder = DebugRecorder(file)
        simulate(recorder)

        expect: str = unindent('''
            Entity: entity_1 ['A', 'B', 'C']
            Entity: entity_2 ['D', 'E']
            
            entity_1 [('A', 1), ('B', 2), ('C', 3)]
            entity_1 [('A', 4), ('B', 5), ('C', 6)]
            entity_2 [('D', 'a'), ('E', 'b')]
            entity_1 [('A', 7), ('B', 8), ('C', 9)]
            entity_2 [('D', 'c'), ('E', 'd')]
            entity_2 [('D', 'e'), ('E', 'f')]
            entity_2 [('D', 'g'), ('E', 'h')]
            
            Finished
        ''')

        self.assertEqual(file.getvalue(), expect)

    def test_csv_recorder(self) -> None:
        with tmp_dir() as tmp:
            recorder = CSVRecorder(directory=tmp)
            simulate(recorder)

            with open('entity_1.csv') as file:
                entity_1: List[str] = file.readlines()

            with open('entity_2.csv') as file:
                entity_2: List[str] = file.readlines()

        self.assertEqual(len(entity_1), 4)
        self.assertEqual(entity_1[0], 'A,B,C\n')
        self.assertEqual(entity_1[1], '1,2,3\n')
        self.assertEqual(entity_1[2], '4,5,6\n')
        self.assertEqual(entity_1[3], '7,8,9\n')

        self.assertEqual(len(entity_2), 5)
        self.assertEqual(entity_2[0], 'D,E\n')
        self.assertEqual(entity_2[1], 'a,b\n')
        self.assertEqual(entity_2[2], 'c,d\n')
        self.assertEqual(entity_2[3], 'e,f\n')
        self.assertEqual(entity_2[4], 'g,h\n')

    def test_composite_recorder(self) -> None:
        debug_file = StringIO()
        memory_recorder = MemoryRecorder()
        recorder = CompositeRecorder(DebugRecorder(debug_file), memory_recorder)
        simulate(recorder)

        expect_file: str = unindent('''
            Entity: entity_1 ['A', 'B', 'C']
            Entity: entity_2 ['D', 'E']

            entity_1 [('A', 1), ('B', 2), ('C', 3)]
            entity_1 [('A', 4), ('B', 5), ('C', 6)]
            entity_2 [('D', 'a'), ('E', 'b')]
            entity_1 [('A', 7), ('B', 8), ('C', 9)]
            entity_2 [('D', 'c'), ('E', 'd')]
            entity_2 [('D', 'e'), ('E', 'f')]
            entity_2 [('D', 'g'), ('E', 'h')]

            Finished
        ''')
        self.assertEqual(debug_file.getvalue(), expect_file)

        entity_1: RamDataCatcher = memory_recorder.records['entity_1']
        entity_2: RamDataCatcher = memory_recorder.records['entity_2']

        self.assertEqual(len(entity_1), 3)
        self.assertEqual(entity_1[0], {'A': 1, 'B': 2, 'C': 3})
        self.assertEqual(entity_1[1], {'A': 4, 'B': 5, 'C': 6})
        self.assertEqual(entity_1[2], {'A': 7, 'B': 8, 'C': 9})

        self.assertEqual(len(entity_2), 4)
        self.assertEqual(entity_2[0], {'D': 'a', 'E': 'b'})
        self.assertEqual(entity_2[1], {'D': 'c', 'E': 'd'})
        self.assertEqual(entity_2[2], {'D': 'e', 'E': 'f'})
        self.assertEqual(entity_2[3], {'D': 'g', 'E': 'h'})


def simulate(recorder: SimRecorder) -> None:
    recorder.start_entity('entity_1', ['A', 'B', 'C'])
    recorder.start_entity('entity_2', ['D', 'E'])

    recorder.write_record('entity_1', {'A': 1, 'B': 2, 'C': 3})
    recorder.write_record('entity_1', {'A': 4, 'B': 5, 'C': 6})

    recorder.write_record('entity_2', {'D': 'a', 'E': 'b'})

    recorder.write_record('entity_1', {'A': 7, 'B': 8, 'C': 9})

    recorder.write_record('entity_2', {'D': 'c', 'E': 'd'})
    recorder.write_record('entity_2', {'D': 'e', 'E': 'f'})
    recorder.write_record('entity_2', {'D': 'g', 'E': 'h'})

    recorder.finish()


if __name__ == '__main__':
    test_main()
