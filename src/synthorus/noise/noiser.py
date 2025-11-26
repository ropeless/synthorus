"""
This module supports adding noise to cross-tables.

This module defines a type 'Noiser'. A Noiser is a function from a source
cross-table and DP parameters to a noisy-cross-table.
It is permitted for the function to have side effects (in place changes to the source
cross-table), but it is generally not recommended.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Iterable, Optional, Dict, Sequence

import numpy as np
import pandas as pd
from ck.pgm import State
from ck.utils.iter_extras import combos, multiply

from synthorus.error import SynthorusError
from synthorus.utils.print_function import PrintFunction
from .row_sampler import SmartRowSampler
from .safe_random import SafeRandom


@dataclass
class NoiserResult:
    """
    The result of adding noise to a cross-table.
    """
    cross_table: pd.DataFrame
    rows_original: int
    rows_added: int
    rows_lost: int

    @property
    def rows_final(self) -> int:
        return self.cross_table.shape[0]


class Noiser(ABC):
    """
    A function to add noise to a cross-table.
    """

    def __init__(self, rvs: Dict[str, Sequence[State]]):
        """
        Construct a Noiser that can manage cross-tables including the given random variables.

        Args:
            rvs: all the random variables that this noiser can manage.
        """
        self.rvs: Dict[str, Sequence[State]] = rvs

    @abstractmethod
    def __call__(
            self,
            cross_table: pd.DataFrame,
            sensitivity: float,
            epsilon: float,
            min_cell_size: float
    ) -> NoiserResult:
        """
        Return a cross-table that adds noise to the given cross-table.

        The added noise is potentially parameterised by the three Differential
        Privacy (DP) parameters. However, each implementation will make its
        own guarantees and claims.

        Args:
            cross_table: The source cross-table (to have noise added).
            sensitivity: The DP parameter.
            epsilon: The DP parameter.
            min_cell_size: The DP parameter.
        
        Returns:
            a NoiserResult that includes the noised cross-table and other statistics.
        """
        ...

    def get_states(self, rv_names: Iterable[str]) -> List[Sequence[State]]:
        """
        Get the possible states for each random variable in rv_names.

        Returns:
            a list co-indexed with the given rv names.
        """
        return [
            self.rvs[rv_name]
            for rv_name in rv_names
        ]

    def state_space_size(self, rv_names: Iterable[str]) -> int:
        """
        What is the state space size for the given named random variables.
        Assumes the named random variables are known to this Noiser.
        """
        states = self.get_states(rv_names)
        return multiply(len(ss) for ss in states)


class LaplaceNoise(Noiser):
    """
    This noiser adds Laplace noise to cross-table weights where noise
    is drawn from safe_random.laplace(0, b), where
    b = sensitivity / epsilon. Then rows with weight < min_cell_size
    are removed from the cross-table.

    This uses the "Decomposition of additive Laplacian noise" method.
    Rows not in a cross-table are taken as having weight zero, and may
    end up with entry in the resulting table.
    """

    def __init__(
            self,
            safe_random: SafeRandom,
            rvs: Dict[str, Sequence[State]],
            max_add_rows: int,
            log: PrintFunction,
    ):
        """
        Construct a LaplaceNoise noiser.

        Args:
            safe_random: an instance of SafeRandom, to generate random numbers.
            rvs: all the random variables that this noiser can manage.
            max_add_rows: a limit on the number of rows to add to a cross-table.
            log: optional print function for logging progress messages.
        """
        super().__init__(rvs)
        self._safe_random = safe_random
        self._basic = BasicLaplaceNoise(safe_random, rvs)
        self._naive = NaiveLaplaceNoise(safe_random, rvs, max_add_rows)
        self._max_add_rows = max_add_rows
        self._log = log

    def __call__(
            self,
            cross_table: pd.DataFrame,
            sensitivity: float,
            epsilon: float,
            min_cell_size: float
    ) -> NoiserResult:
        """
        Add Laplace noise to cross-table weights where noise
        is drawn from safe_random.laplace(0, b), where
        b = sensitivity / epsilon, then enforce min_cell_size.

        Assumes the given cross-table has no duplicated entries (i.e. same
        row, excluding the last, weight column).

        Noise may be added to implied zero rows, therefore new rows may be added
        to the result. See class comment.
        """
        log = self._log

        if sensitivity <= 0:
            # Not adding Laplace noise.
            if min_cell_size <= 0:
                # and not enforcing min cell size
                log(f'no sensitivity or min cell size: applying no noise')
                return NoiserResult(cross_table, cross_table.shape[0], 0, 0)
            else:
                # but still enforcing min cell size
                log(f'no sensitivity: using {self._basic.__class__.__name__}')
                return self._basic(cross_table, sensitivity, epsilon, min_cell_size)

        num_states = self.state_space_size(cross_table.columns[:-1])
        num_rows = cross_table.shape[0]
        num_suppressed = num_states - num_rows
        alpha = self.alpha(sensitivity, epsilon, min_cell_size)
        log(f'alpha: {alpha}')
        log(f'Expected-rows: {alpha * num_suppressed}')

        # Heuristically choose which method to apply.
        if num_suppressed == 0:
            # There are no suppressed rows - just use the basic method.
            log(f'no suppressed rows: using {self._basic.__class__.__name__}')
            return self._basic(cross_table, sensitivity, epsilon, min_cell_size)

        elif num_suppressed <= self._max_add_rows and alpha > 0.5:
            # The number of suppressed rows is low enough to just add them,
            # and alpha is high so no value expected from the decomposition method.
            log(f'low suppressed rows: using {self._naive.__class__.__name__}')
            return self._naive(cross_table, sensitivity, epsilon, min_cell_size)

        elif min_cell_size <= 0:
            # The decomposition method only works when min_cell_size > 0.
            log(f'no min cell size: using {self._naive.__class__.__name__}')
            return self._naive(cross_table, sensitivity, epsilon, min_cell_size)

        else:
            # Use the decomposition method.
            log('using decomposition_method')
            return self.decomposition_method(cross_table, sensitivity, epsilon, min_cell_size)

    @property
    def max_add_rows(self) -> int:
        return self._max_add_rows

    def decomposition_method(
            self,
            cross_table: pd.DataFrame,
            sensitivity: float,
            epsilon: float,
            min_cell_size: float
    ) -> NoiserResult:
        """
        Make a noisy cross-table using the decomposition method.

        Args:
            cross_table: clean cross-table.
            sensitivity: DP parameter.
            epsilon: DP parameter.
            min_cell_size: DP parameter.
        """
        weight_col = cross_table.columns[-1]
        rvs = list(cross_table.columns[:-1])
        states = self.get_states(rvs)
        num_states = multiply(len(ss) for ss in states)
        rows_original = cross_table.shape[0]
        num_suppressed = num_states - rows_original
        max_weight = cross_table[weight_col].max()

        # Apply the basic method to original rows (i.e., rows already in cross_table).
        basic_result = self._basic(cross_table, sensitivity, epsilon, min_cell_size)
        cross_table_1 = basic_result.cross_table
        rows_lost = basic_result.rows_lost

        # Create new rows, that are not in cross_table.
        cross_table_2 = self._make_zn_rows(
            cross_table,
            rvs,
            states,
            num_suppressed,
            sensitivity,
            epsilon,
            min_cell_size,
            weight_col,
            max_weight
        )
        rows_added = cross_table_2.shape[0]

        cross_table = pd.concat([cross_table_1, cross_table_2], ignore_index=True)
        return NoiserResult(cross_table, rows_original, rows_added, rows_lost)

    def _make_zn_rows(
            self,
            cross_table,
            rvs,
            states,
            num_suppressed,
            sensitivity,
            epsilon,
            min_cell_size,
            weight_col,
            max_weight
    ) -> pd.DataFrame:
        """
        Create new rows with random weights >= min_cell_size, that
        are not in the given cross_table.

        The number of rows returned is a random number, k, drawn from
        k ~ binomial(num_suppressed, alpha)
        where alpha = 0.5 * exp(-min_cell_size * sensitivity / epsilon).

        Will raise a SynthorusError if `k > self.max_add_rows`.
        """

        safe_random = self._safe_random
        alpha = self.alpha(sensitivity, epsilon, min_cell_size)
        k = safe_random.binomial(num_suppressed, alpha)

        # Check that not adding too many rows
        if k > self._max_add_rows:
            suggested_min_cell_size = self.recommended_min_cell_size(
                epsilon,
                sensitivity,
                num_suppressed,
                target_rows=self._max_add_rows
            )
            message = f'too many rows to add: {k:,}, maximum {self._max_add_rows:,}'
            if suggested_min_cell_size > max_weight:
                message += f', no feasible min cell size, consider reducing epsilon or refactoring the cross-table'
            else:
                message += f', suggested min cell size = {suggested_min_cell_size}'
            raise SynthorusError(message)

        row_sampler = SmartRowSampler(states)
        row_sampler.remove_rows(row[:-1] for row in cross_table.itertuples(index=False))
        new_rows = row_sampler.draw_rows(k)

        # Make the cross-table, as a Dataframe
        cross_table = pd.DataFrame(new_rows, columns=rvs)
        b = sensitivity / epsilon
        laplace_noise = np.fromiter(
            count=k,
            dtype=np.double,
            iter=(
                min_cell_size + abs(safe_random.laplace(0, b))
                for _ in range(k)
            ),
        )
        cross_table[weight_col] = laplace_noise
        return cross_table

    @staticmethod
    def alpha(sensitivity: float, epsilon: float, min_cell_size: float) -> float:
        """
        Return the probability of a suppressed row being in a noisy cross-table
        after adding Laplace noise and enforcing min cell size.
        This assumes sensitivity > 0 and epsilon > 0.
        """
        # b = sensitivity / epsilon
        # return 0.5 * math.exp(-min_cell_size / b)
        return 0.5 * math.exp(-min_cell_size * epsilon / sensitivity)

    @staticmethod
    def recommended_min_cell_size(epsilon, sensitivity, num_suppressed, target_rows) -> float:
        """
        What min-cell-size leads to the expected given number of target rows (or less)
        for a cross-table with the given sensitivity and number of suppressed rows.

        Returns min-cell-size >= 0.
        If no such min-cell-size is sufficient, then float('inf') is returned.
        """
        if target_rows >= num_suppressed:
            # There is no min-cell-size required
            return 0

        target_probability = target_rows / num_suppressed

        if target_probability <= 0:
            # There is no min-cell-size high enough!
            return float('inf')

        recommendation = (
                - sensitivity / epsilon * math.log(2 * target_probability)
        )

        # It is possible that the recommendation goes
        # negative if num_suppressed is low
        recommendation = max(recommendation, 0)

        return recommendation


class NaiveLaplaceNoise(Noiser):
    """
    This noiser adds Laplace noise to cross-table weights where noise
    is drawn from safe_random.laplace(0, b), where
    b = sensitivity / epsilon. Then rows with weight < min_cell_size
    are removed from the cross-table.

    This does not use "Decomposition of additive Laplacian noise" method.
    Rows not in a cross-table are taken as having weight zero, and will
    end up with entry in the resulting table, even if left as zero.

    This is a baseline algorithm for implementing Differential Privacy.
    """

    def __init__(
            self, safe_random: SafeRandom,
            rvs: Dict[str, Sequence[State]],
            max_add_rows: Optional[int] = None,
    ):
        """
        Construct a LaplaceNoise noiser.

        Args:
            safe_random: an instance of SafeRandom, to generate random numbers.
            rvs: all the random variables that this noiser can manage.
            max_add_rows: an optional limit on the number of rows to add to a cross-table.
                If provided then an exception is raised if this limit is exceeded.
        """
        super().__init__(rvs)
        self._safe_random = safe_random
        self._basic = BasicLaplaceNoise(safe_random, rvs)
        self._max_add_rows = max_add_rows

    def __call__(
            self,
            cross_table: pd.DataFrame,
            sensitivity: float,
            epsilon: float,
            min_cell_size: float
    ) -> NoiserResult:
        """
        Make a noisy cross-table by constructing a data frame with
        all possible rows, then apply the basic method.

        Args:
            cross_table: clean cross-table.
            sensitivity: DP parameter.
            epsilon: DP parameter.
            min_cell_size: DP parameter.
        """
        weight_col = cross_table.columns[-1]
        rvs = list(cross_table.columns[:-1])
        states = self.get_states(rvs)

        rows_original = cross_table.shape[0]
        num_complete_rows = multiply(len(ss) for ss in states)
        num_rows_to_add = num_complete_rows - rows_original
        if self._max_add_rows is not None:
            if num_rows_to_add > self._max_add_rows:
                raise SynthorusError(f'too many rows to add: {num_rows_to_add:,}, maximum {self._max_add_rows:,}')

        # Dictionary 'weight_dict' maps an original row to its weight in 'cross_table'
        weight_dict: Dict[Tuple, float] = {
            row[:-1]: row[-1]
            for row in cross_table.itertuples(index=False)
            if row[-1] != 0
        }

        # Create a row for every possible combinations for states.
        # Excludes the weight column.
        cross_table = pd.DataFrame(combos(states), columns=rvs)

        # A function to map a row in `cross_table` to its original weight
        def get_weight(_row) -> float:
            return weight_dict.get(tuple(_row), 0)

        # Add the weight column to `cross_table` with the original weight values
        cross_table[weight_col] = cross_table.apply(get_weight, axis=1)

        # Use the `basic` method to add noise to the weights,
        # then enforce min_cell_size, which may end up deleting rows.
        cross_table = self._basic(cross_table, sensitivity, epsilon, min_cell_size).cross_table

        # A function to check if a row from `cross_table` was an existing row
        def is_old_row(_row) -> bool:
            return tuple(_row[:-1]) in weight_dict

        # Work out how many new and lost rows there were
        num_old_rows = cross_table.apply(is_old_row, axis=1).sum()  # original rows that survived to the new table
        rows_added = cross_table.shape[0] - num_old_rows
        rows_lost = rows_original - num_old_rows

        return NoiserResult(cross_table, rows_original, rows_added, rows_lost)

    @property
    def max_add_rows(self) -> Optional[int]:
        return self._max_add_rows


class BasicLaplaceNoise(Noiser):
    """
    This noiser adds Laplace noise to cross-table weights where noise
    is drawn from safe_random.laplace(0, b), where
    b = sensitivity / epsilon. Then rows with weight < min_cell_size
    are removed from the cross-table.

    Laplace noise is only added to rows that exist in the source cross
    table, and only added if sensitivity is > 0.

    Generally epsilon should be strictly positive (non-zero and finite),
    however this only needs to be ensured when used with sensitivity > 0.

    THIS DOES NOT SATISFY DIFFERENTIAL PRIVACY REQUIREMENTS.
    This noiser will never add new rows.
    """

    def __init__(self, safe_random: SafeRandom, rvs: Dict[str, Sequence[State]]):
        """
        Construct a BasicLaplaceNoise noiser.

        Args:
            safe_random: an instance of SafeRandom, to generate random numbers.
            rvs: all the random variables that this noiser can manage.
        """
        super().__init__(rvs)
        self._safe_random = safe_random

    def __call__(
            self,
            cross_table: pd.DataFrame,
            sensitivity: float,
            epsilon: float,
            min_cell_size: float
    ) -> NoiserResult:
        """
        Add Laplace noise to cross-table weights where noise
        is drawn from safe_random.laplace(0, b), where
        b = sensitivity / epsilon, then enforce min_cell_size.
        """
        initial_size = cross_table.shape[0]
        if sensitivity > 0:
            b = sensitivity / epsilon
            cross_table = self.add_laplace_noise(cross_table, b)
        cross_table = self.enforce_min_cell_size(cross_table, min_cell_size)
        rows_lost = initial_size - cross_table.shape[0]
        return NoiserResult(cross_table, initial_size, 0, rows_lost)

    def add_laplace_noise(self, cross_table: pd.DataFrame, b: float) -> pd.DataFrame:
        size = cross_table.shape[0]
        safe_random = self._safe_random
        laplace_noise = np.fromiter(
            (
                safe_random.laplace(0, b)
                for _ in range(size)
            ),
            count=size,
            dtype=np.double
        )
        weight_series = cross_table.iloc[:, -1] + laplace_noise
        cross_table[cross_table.columns[-1]] = weight_series
        return cross_table

    @staticmethod
    def enforce_min_cell_size(cross_table: pd.DataFrame, min_cell_size: float) -> pd.DataFrame:
        """
        Only keep rows where the weight (last column) is >= min_cell_size.
        """
        weight_series = cross_table[cross_table.columns[-1]]
        keep_rows = (weight_series >= min_cell_size)
        cross_table = cross_table.loc[keep_rows]
        return cross_table
