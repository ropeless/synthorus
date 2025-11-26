from __future__ import annotations

import random
from typing import Tuple, List, Dict, Iterable, Any

from synthorus.utils.const import Const


class SmartRowSampler:
    """
    A sampler to draw rows from a state space, without replacement.

    This sampler uses inverse transform sampling by carefully indexing the
    set of rows that are used.
    """

    def __init__(self, states: List[Tuple]):
        self._index = SamplerRVIndex()
        state_space = calc_state_space(states)
        self._states = states
        self._space_size = state_space.pop(0)
        self._available = self._space_size
        self._chunk_sizes = state_space

    def remove_rows(self, rows: Iterable[Tuple]) -> None:
        """
        Remove the given rows from the possible rows to draw.
        """
        for row in rows:
            self._index.add(row, 0)
        self._available = self._space_size - self._index.num_used

    @property
    def available_rows(self) -> int:
        """
        How many rows are available to be drawn.
        """
        return self._available

    def draw_rows(self, k: int) -> List[Tuple]:
        """
        Draw `k` new rows from the state space, without replacement.
        """
        if self._available < k:
            raise ValueError('insufficient unused rows')
        return [self._draw_row() for _ in range(k)]

    def _draw_row(self) -> Tuple:
        available: int = self._available
        self._available = available - 1
        idx: int = random.randrange(available)
        return self._index.draw(idx, self._states, self._chunk_sizes)


_END = Const('_END')  # A sentinel object


class SamplerRVIndex:
    """
    Support for SmartRowSampler.

    An SamplerRVIndex is essentially a mapping from a random variable state to
    how much of the state space is use with the random variable in that state
    (and assuming all previous random variables in particular states).
    """

    def __init__(self, num_used: int = 0):
        self.used: Dict[Any, SamplerRVIndex | Const] = {}  # map state of RV to next link or sentinel _END
        self.num_used: int = num_used  # how much of the state space is used

    def add(self, row: Tuple, i: int) -> int:
        """
        Record the fact that the given row is used and should not be drawn.
        Assumes this is an RV index for the ith random variable, indexed from zero.

        Returns:
            1 if added, 0 if already in the index.
        """
        state = row[i]
        state_index = self.used.get(state)
        if state_index is None:
            state_index = _END
            for j in range(len(row) - 1, i, -1):
                next_index = SamplerRVIndex(1)
                next_index.used[row[j]] = state_index
                state_index = next_index
            self.used[state] = state_index
            self.num_used += 1
            return 1
        else:
            i += 1
            if i == len(row):
                # The row is already in the index.
                return 0
            else:
                added = state_index.add(row, i)
                self.num_used += added
                return added

    def draw(self, idx: int, states: List[Tuple], chunk_sizes: List[int]) -> Tuple:
        """
        Draw a row from the unused state space where the row is at position `idx`.

        The drawn row is returned and the row is added to `used`, so it cannot
        be drawn again. The number of used rows is incremented by 1.

        Assumes 0 <= idx < number of unused rows, where the number of unused rows
        is the size of the state space minus the number of rows used.

        This method uses `states` and `chunk_sizes` as cached by the caller.

        Args:
            idx: is the index into the unused rows, starting from zero.
            states: states[i] are the possible states for the ith random variable.
            chunk_sizes: chunk_sizes[i] is the state space chunk size for the ith random variable.

        Returns:
            the drawn row, and marks the drawn row as used.
        """
        if idx < 0:
            raise IndexError('index out of range')
        result = []
        self._draw_r(idx, states, chunk_sizes, result)
        return tuple(result)

    def _draw_r(self, idx: int, states: List[Tuple], chunk_sizes: List[int], result: List) -> None:
        """
        Recursive implementation of `draw`.
        Accumulates the drawn row in `result`.
        """
        rv_idx: int = len(result)
        cur_states = states[rv_idx]
        chunk_size = chunk_sizes[rv_idx]

        for state in cur_states:
            state_index = self.used.get(state)
            if state_index is _END:
                # This is a leaf index - cannot draw it
                continue
            elif state_index is None:
                # Unused state
                next_idx = idx - chunk_size
                if next_idx < 0:
                    # choose this state
                    result.append(state)

                    # append remaining states
                    used = []
                    for ss in states[rv_idx + 1:]:
                        ss_size = len(ss)
                        ss_state = ss[idx % ss_size]
                        idx //= ss_size
                        result.append(ss_state)
                        used.append(ss_state)

                    # record the used states
                    state_index = _END
                    for next_state in reversed(used):
                        next_index = SamplerRVIndex(1)
                        next_index.used[next_state] = state_index
                        state_index = next_index
                    self.used[state] = state_index
                    self.num_used += 1

                    if idx != 0:
                        raise IndexError('index out of range')
                    return
            else:
                available = chunk_size - state_index.num_used
                next_idx = idx - available
                if next_idx < 0:
                    # choose this state
                    result.append(state)
                    state_index._draw_r(idx, states, chunk_sizes, result)
                    self.num_used += 1
                    return
            idx = next_idx
        raise IndexError('index out of range')


def calc_state_space(states: List[Tuple]) -> List[int]:
    """
    Return the size of the state spaces co-indexed with the given random variable states.
    The result will have one extra element appended, which has the value 1.
    E.g., if states is [('a', 'b', 'c'), (1, 2, 3, 4)] then the result
    will be [12, 4, 1].
    """
    state_space = [1]
    for ss in reversed(states):
        state_space.append(len(ss) * state_space[-1])
    state_space.reverse()
    return state_space

# Deprecated
#
# class RejectionRowSampler:
#     """
#     A sampler to draw rows from a state space, without replacement.
#
#     This sampler uses rejection sampling. I.e., rows that have already
#     been seen are rejected. This is a kind of sampling may be inefficient
#     when the proportion of used rows is large.
#     """
#
#     def __init__(self, states: List[Tuple]):
#         self._row_set: Set[Tuple] = set()
#         self._states = states
#
#     def remove_rows(self, rows: Iterable[Tuple]) -> None:
#         """
#         Remove the given rows from the possible rows to draw.
#         """
#         self._row_set.update(rows)
#
#     @property
#     def available_rows(self) -> int:
#         """
#         How many rows are available to be drawn.
#         """
#         return math.prod(len(ss) for ss in self._states) - len(self._row_set)
#
#     def draw_rows(self, k: int) -> List[Tuple]:
#         """
#         Draw `k` new rows from the state space, without replacement.
#         """
#         row_set = self._row_set
#         states = self._states
#
#         # Protect against infinite loop when looking for new rows.
#         # This is the maximum number of random rows to try.
#         max_tries = k * 20
#
#         new_rows = []
#         while len(new_rows) < k and max_tries > 0:
#             # Draw a random row, uniformly from all possible rows.
#             # The possible random variable states for the ith random variable
#             # is given by states[i].
#             row = tuple(
#                 random.choice(ss)
#                 for ss in states
#             )
#             # Accept the row if it is not already in the set
#             if row not in row_set:
#                 row_set.add(row)
#                 new_rows.append(row)
#             max_tries -= 1
#         if len(new_rows) < k:
#             raise RuntimeError('cannot find enough new rows: max tries exceeded')
#
#         return new_rows
