from synthorus.noise.row_sampler import SmartRowSampler, calc_state_space, SamplerRVIndex
from tests.helpers.unittest_fixture import Fixture, test_main


class SamplerRVIndexTest(Fixture):

    def test_rv_index_draw_front(self):
        states = [
            (1, 2, 3, 4),
            (1, 2),
            (1, 2, 3),
        ]
        chunk_sizes = calc_state_space(states)[1:]

        index = SamplerRVIndex()
        self.assertEqual(index.num_used, 0)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 1, 1))
        self.assertEqual(index.num_used, 1)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 1, 2))
        self.assertEqual(index.num_used, 2)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 1, 3))
        self.assertEqual(index.num_used, 3)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 2, 1))
        self.assertEqual(index.num_used, 4)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 2, 2))
        self.assertEqual(index.num_used, 5)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 2, 3))
        self.assertEqual(index.num_used, 6)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (2, 1, 1))
        self.assertEqual(index.num_used, 7)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (2, 1, 2))
        self.assertEqual(index.num_used, 8)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (2, 1, 3))
        self.assertEqual(index.num_used, 9)

        # Try to draw beyond what is available
        with self.assertRaises(IndexError):
            _ = index.draw(15, states, chunk_sizes)

    def test_rv_index_draw_back(self):
        states = [
            (1, 2, 3, 4),
            (1, 2),
            (1, 2, 3),
        ]
        chunk_sizes = calc_state_space(states)[1:]

        index = SamplerRVIndex()
        self.assertEqual(index.num_used, 0)

        row = index.draw(23, states, chunk_sizes)
        self.assertEqual(row, (4, 2, 3))
        self.assertEqual(index.num_used, 1)

        row = index.draw(22, states, chunk_sizes)
        self.assertEqual(row, (4, 2, 2))
        self.assertEqual(index.num_used, 2)

        row = index.draw(21, states, chunk_sizes)
        self.assertEqual(row, (4, 2, 1))
        self.assertEqual(index.num_used, 3)

        row = index.draw(20, states, chunk_sizes)
        self.assertEqual(row, (4, 1, 3))
        self.assertEqual(index.num_used, 4)

        row = index.draw(19, states, chunk_sizes)
        self.assertEqual(row, (4, 1, 2))
        self.assertEqual(index.num_used, 5)

        row = index.draw(18, states, chunk_sizes)
        self.assertEqual(row, (4, 1, 1))
        self.assertEqual(index.num_used, 6)

        row = index.draw(17, states, chunk_sizes)
        self.assertEqual(row, (3, 2, 3))
        self.assertEqual(index.num_used, 7)

        row = index.draw(16, states, chunk_sizes)
        self.assertEqual(row, (3, 2, 2))
        self.assertEqual(index.num_used, 8)

        row = index.draw(15, states, chunk_sizes)
        self.assertEqual(row, (3, 2, 1))
        self.assertEqual(index.num_used, 9)

        row = index.draw(14, states, chunk_sizes)
        self.assertEqual(row, (3, 1, 3))
        self.assertEqual(index.num_used, 10)

        row = index.draw(13, states, chunk_sizes)
        self.assertEqual(row, (3, 1, 2))
        self.assertEqual(index.num_used, 11)

        # Try to draw beyond what is available
        with self.assertRaises(IndexError):
            _ = index.draw(13, states, chunk_sizes)

    def test_rv_draw_mid(self):
        states = [
            (1, 2, 3, 4),
            (1, 2),
            (1, 2, 3),
        ]
        chunk_sizes = calc_state_space(states)[1:]

        index = SamplerRVIndex()
        self.assertEqual(index.num_used, 0)

        expect_row = (1, 2, 3)  # row at index 5
        row = index.draw(5, states, chunk_sizes)
        self.assertEqual(row, expect_row)
        self.assertEqual(index.num_used, 1)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 1, 1))
        self.assertEqual(index.num_used, 2)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 1, 2))
        self.assertEqual(index.num_used, 3)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 1, 3))
        self.assertEqual(index.num_used, 4)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 2, 1))
        self.assertEqual(index.num_used, 5)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 2, 2))
        self.assertEqual(index.num_used, 6)

        # (1, 2, 3) not available

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (2, 1, 1))
        self.assertEqual(index.num_used, 7)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (2, 1, 2))
        self.assertEqual(index.num_used, 8)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (2, 1, 3))
        self.assertEqual(index.num_used, 9)

    def test_rv_index_add(self):
        states = [
            (1, 2, 3, 4),
            (1, 2),
            (1, 2, 3),
        ]
        chunk_sizes = calc_state_space(states)[1:]

        to_remove = (1, 2, 3)

        index = SamplerRVIndex()
        self.assertEqual(index.num_used, 0)

        index.add(to_remove, 0)
        self.assertEqual(index.num_used, 1)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 1, 1))
        self.assertEqual(index.num_used, 2)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 1, 2))
        self.assertEqual(index.num_used, 3)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 1, 3))
        self.assertEqual(index.num_used, 4)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 2, 1))
        self.assertEqual(index.num_used, 5)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (1, 2, 2))
        self.assertEqual(index.num_used, 6)

        # (1, 2, 3) not available

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (2, 1, 1))
        self.assertEqual(index.num_used, 7)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (2, 1, 2))
        self.assertEqual(index.num_used, 8)

        row = index.draw(0, states, chunk_sizes)
        self.assertEqual(row, (2, 1, 3))
        self.assertEqual(index.num_used, 9)

    def test_rv_index_add_duplicate(self):
        to_add = (1, 2, 3)

        index = SamplerRVIndex()
        self.assertEqual(index.num_used, 0)

        index.add(to_add, 0)
        self.assertEqual(index.num_used, 1)

        index.add(to_add, 0)
        self.assertEqual(index.num_used, 1)


class SmartRowSamplerTest(Fixture):

    def test_draw_rows(self):
        states = [
            (1, 2, 3, 4),
            (1, 2),
            (1, 2, 3),
        ]

        sampler = SmartRowSampler(states)
        self.assertEqual(sampler.available_rows, 24)

        # should be able to draw 24 rows.
        rows = sampler.draw_rows(24)
        self.assertEqual(sampler.available_rows, 0)

        self.assertEqual(len(rows), 24)
        self.assertEqual(len(set(rows)), 24, 'expect no duplicates')
        rows = sorted(rows)
        self.assertEqual(rows[0], (1, 1, 1))
        self.assertEqual(rows[-1], (4, 2, 3))

        # Try to draw more is an error
        with self.assertRaises(ValueError):
            sampler.draw_rows(1)

    def test_remove_rows(self):
        states = [
            (1, 2, 3, 4),
            (1, 2),
            (1, 2, 3),
        ]

        to_remove = (1, 2, 3)

        sampler = SmartRowSampler(states)
        self.assertEqual(sampler.available_rows, 24)

        sampler.remove_rows([to_remove])
        self.assertEqual(sampler.available_rows, 23)

        # should be able to draw 23 rows.
        rows = sampler.draw_rows(23)
        row_set = set(rows)
        self.assertEqual(sampler.available_rows, 0)
        self.assertEqual(len(rows), 23)
        self.assertEqual(len(row_set), 23, 'expect no duplicates')

        self.assertNotIn(to_remove, row_set)
        self.assertIn((1, 1, 1), row_set)
        self.assertIn((4, 2, 3), row_set)


if __name__ == '__main__':
    test_main()
