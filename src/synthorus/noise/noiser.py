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

    @abstractmethod
    def add_noise(
            self,
            cross_table: pd.DataFrame,
            rvs: Dict[str, Sequence[State]],
            safe_random: SafeRandom,
            sensitivity: float,
            epsilon: float,
            min_cell_size: float,
            log: PrintFunction,
    ) -> NoiserResult:
        """
        Return a cross-table that adds noise to the given cross-table.

        The added noise is potentially parameterised by the three Differential
        Privacy (DP) parameters. However, each implementation will make its
        own guarantees and claims.

        Args:
            cross_table: The source cross-table (to have noise added).
            rvs: Known random variables and their states to help implementers do their job.
            safe_random: The random number generator to use.
            sensitivity: The DP parameter.
            epsilon: The DP parameter.
            min_cell_size: The DP parameter.
            log: optional print function for logging progress messages.

        Returns:
            a NoiserResult that includes the noised cross-table and other statistics.
        """
        ...


class LaplaceNoise(Noiser):
    """
    This noiser adds Laplace noise to cross-table weights where noise
    is drawn from safe_random.laplace(0, b), where
    b = sensitivity / epsilon. Then rows with weight < min_cell_size
    are removed from the cross-table.

    Rows not in a cross-table are taken as having weight zero, and may
    end up with entry in the resulting table.

    This noiser chooses between `BasicLaplaceNoise`, `NaiveLaplaceNoise`, and `DecompositionLaplaceNoise`
    heuristically to get the most efficient computation.
    """

    def __init__(
            self,
            max_add_rows: int
    ):
        """
        Construct a LaplaceNoise noiser.

        Args:
            max_add_rows: a limit on the number of rows to add to a cross-table.
        """
        self._basic: Noiser = BasicLaplaceNoise()
        self._naive: Noiser = NaiveLaplaceNoise(max_add_rows)
        self._decomp: Noiser = DecompositionLaplaceNoise(max_add_rows)
        self.max_add_rows: int = max_add_rows

    def add_noise(
            self,
            cross_table: pd.DataFrame,
            rvs: Dict[str, Sequence[State]],
            safe_random: SafeRandom,
            sensitivity: float,
            epsilon: float,
            min_cell_size: float,
            log: PrintFunction,
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
        if sensitivity <= 0:
            # Not adding Laplace noise.
            if min_cell_size <= 0:
                # and not enforcing min cell size
                log(f'no sensitivity or min cell size: applying no noise')
                return NoiserResult(cross_table, cross_table.shape[0], 0, 0)
            else:
                # but still enforcing min cell size
                log(f'no sensitivity: using {self._basic.__class__.__name__}')
                return self._basic.add_noise(cross_table, rvs, safe_random, sensitivity, epsilon, min_cell_size, log)

        num_states = state_space_size(rvs, cross_table.columns[:-1])
        num_rows = cross_table.shape[0]
        num_suppressed = num_states - num_rows
        alpha = compute_alpha(sensitivity, epsilon, min_cell_size)
        log(f'alpha: {alpha}')
        log(f'Expected-rows: {alpha * num_suppressed}')

        # Heuristically choose which method to apply.
        if num_suppressed == 0:
            # There are no suppressed rows - just use the basic method.
            log(f'no suppressed rows: using {self._basic.__class__.__name__}')
            return self._basic.add_noise(cross_table, rvs, safe_random, sensitivity, epsilon, min_cell_size, log)

        elif num_suppressed <= self.max_add_rows and alpha > 0.5:
            # The number of suppressed rows is low enough to just add them,
            # and alpha is high so no value expected from the decomposition method.
            log(f'low suppressed rows: using {self._naive.__class__.__name__}')
            return self._naive.add_noise(cross_table, rvs, safe_random, sensitivity, epsilon, min_cell_size, log)

        elif min_cell_size <= 0:
            # The decomposition method only works when min_cell_size > 0.
            log(f'no min cell size: using {self._naive.__class__.__name__}')
            return self._naive.add_noise(cross_table, rvs, safe_random, sensitivity, epsilon, min_cell_size, log)

        else:
            # Use the decomposition method.
            log('using decomposition_method')
            return self._decomp.add_noise(cross_table, rvs, safe_random, sensitivity, epsilon, min_cell_size, log)


class DecompositionLaplaceNoise(Noiser):
    """
    This noiser adds Laplace noise to cross-table weights where noise
    is drawn from safe_random.laplace(0, b), where
    b = sensitivity / epsilon. Then rows with weight < min_cell_size
    are removed from the cross-table.

    Rows not in a cross-table are taken as having weight zero, and may
    end up with entry in the resulting table.

    This always uses the "Decomposition of additive Laplacian noise" method.

    WARNING: This requires positive sensitivity, epsilon and min_cell_size.
    """

    def __init__(
            self,
            max_add_rows: int
    ):
        """
        Construct a LaplaceNoise noiser.

        Args:
            max_add_rows: a limit on the number of rows to add to a cross-table.
        """
        self._basic: Noiser = BasicLaplaceNoise()
        self.max_add_rows: int = max_add_rows

    def add_noise(
            self,
            cross_table: pd.DataFrame,
            rvs: Dict[str, Sequence[State]],
            safe_random: SafeRandom,
            sensitivity: float,
            epsilon: float,
            min_cell_size: float,
            log: PrintFunction,
    ) -> NoiserResult:
        """
        Make a noisy cross-table using the decomposition method.
        """
        weight_col = cross_table.columns[-1]
        rvs_names = list(cross_table.columns[:-1])
        states = get_states(rvs, rvs_names)
        num_states = multiply(len(ss) for ss in states)
        rows_original = cross_table.shape[0]
        num_suppressed = num_states - rows_original
        max_weight = cross_table[weight_col].max()

        # Apply the basic method to original rows (i.e., rows already in cross_table).
        basic_result = self._basic.add_noise(cross_table, rvs, safe_random, sensitivity, epsilon, min_cell_size, log)
        if num_suppressed == 0:
            # If there were no suppressed rows, then it is all done
            return basic_result
        cross_table_1 = basic_result.cross_table
        rows_lost = basic_result.rows_lost

        # Create new rows, that are not in the original cross-table, `cross_table`.
        cross_table_2 = self._make_zn_rows(
            cross_table,
            rvs_names,
            safe_random,
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
            safe_random: SafeRandom,
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

        alpha = compute_alpha(sensitivity, epsilon, min_cell_size)
        k = safe_random.binomial(num_suppressed, alpha)

        # Check that not adding too many rows
        if k > self.max_add_rows:
            suggested_min_cell_size = recommended_min_cell_size(
                epsilon,
                sensitivity,
                num_suppressed,
                target_rows=self.max_add_rows
            )
            message = f'too many rows to add: {k:,}, maximum {self.max_add_rows:,}'
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


class NaiveLaplaceNoise(Noiser):
    """
    This noiser adds Laplace noise to cross-table weights where noise
    is drawn from safe_random.laplace(0, b), where
    b = sensitivity / epsilon. Then rows with weight < min_cell_size
    are removed from the cross-table.

    Rows not in a cross-table are taken as having weight zero, and will
    end up with entry in the resulting table, even if left as zero.

    This is a baseline algorithm for implementing Differential Privacy.
    """

    def __init__(
            self,
            max_add_rows: Optional[int] = None,
    ):
        """
        Construct a LaplaceNoise noiser.

        Args:
            max_add_rows: an optional limit on the number of rows to add to a cross-table.
                If provided then an exception is raised if this limit is exceeded.
        """
        self._basic: Noiser = BasicLaplaceNoise()
        self.max_add_rows: Optional[int] = max_add_rows

    def add_noise(
            self,
            cross_table: pd.DataFrame,
            rvs: Dict[str, Sequence[State]],
            safe_random: SafeRandom,
            sensitivity: float,
            epsilon: float,
            min_cell_size: float,
            log: PrintFunction,
    ) -> NoiserResult:
        """
        Make a noisy cross-table by constructing a data frame with
        all possible rows, then apply the basic method.
        """
        weight_col = cross_table.columns[-1]
        rvs_names = list(cross_table.columns[:-1])
        states = get_states(rvs, rvs_names)

        rows_original = cross_table.shape[0]
        num_complete_rows = multiply(len(ss) for ss in states)
        num_rows_to_add = num_complete_rows - rows_original
        if self.max_add_rows is not None:
            if num_rows_to_add > self.max_add_rows:
                raise SynthorusError(f'too many rows to add: {num_rows_to_add:,}, maximum {self.max_add_rows:,}')

        # Dictionary 'weight_dict' maps an original row to its weight in 'cross_table'
        weight_dict: Dict[Tuple, float] = {
            row[:-1]: row[-1]
            for row in cross_table.itertuples(index=False)
            if row[-1] != 0
        }

        # Create a row for every possible combinations for states.
        # Excludes the weight column.
        cross_table = pd.DataFrame(combos(states), columns=rvs_names)

        # A function to map a row in `cross_table` to its original weight
        def get_weight(_row) -> float:
            return weight_dict.get(tuple(_row), 0)

        # Add the weight column to `cross_table` with the original weight values
        cross_table[weight_col] = cross_table.apply(get_weight, axis=1)

        # Use the `basic` method to add noise to the weights,
        # then enforce min_cell_size, which may end up deleting rows.
        cross_table = self._basic.add_noise(cross_table, rvs, safe_random, sensitivity, epsilon, min_cell_size, log).cross_table

        # A function to check if a row from `cross_table` was an existing row
        def is_old_row(_row) -> bool:
            return tuple(_row[:-1]) in weight_dict

        # Work out how many new and lost rows there were
        num_old_rows = cross_table.apply(is_old_row, axis=1).sum()  # original rows that survived to the new table
        rows_added = cross_table.shape[0] - num_old_rows
        rows_lost = rows_original - num_old_rows

        return NoiserResult(cross_table, rows_original, rows_added, rows_lost)


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

    def add_noise(
            self,
            cross_table: pd.DataFrame,
            rvs: Dict[str, Sequence[State]],
            safe_random: SafeRandom,
            sensitivity: float,
            epsilon: float,
            min_cell_size: float,
            log: PrintFunction,
    ) -> NoiserResult:
        """
        Add Laplace noise to cross-table weights where noise
        is drawn from safe_random.laplace(0, b), where
        b = sensitivity / epsilon, then enforce min_cell_size.
        """
        initial_size = cross_table.shape[0]
        if sensitivity > 0:
            b = sensitivity / epsilon
            cross_table = add_laplace_noise(cross_table, safe_random, b)
        cross_table = self.enforce_min_cell_size(cross_table, min_cell_size)
        rows_lost = initial_size - cross_table.shape[0]
        return NoiserResult(cross_table, initial_size, 0, rows_lost)

    @staticmethod
    def enforce_min_cell_size(cross_table: pd.DataFrame, min_cell_size: float) -> pd.DataFrame:
        """
        Only keep rows where the weight (last column) is >= min_cell_size.
        """
        weight_series = cross_table[cross_table.columns[-1]]
        keep_rows = (weight_series >= min_cell_size)
        cross_table = cross_table.loc[keep_rows]
        return cross_table


def get_states(rvs: Dict[str, Sequence[State]], rv_names: Iterable[str]) -> List[Sequence[State]]:
    """
    Get the possible states for each random variable in rv_names.

    Support for Noiser implementations.

    Assumes:
        named random variables are in `rvs`.

    Returns:
        a list co-indexed with the given rv names.
    """
    return [
        rvs[rv_name]
        for rv_name in rv_names
    ]


def state_space_size(rvs: Dict[str, Sequence[State]], rv_names: Iterable[str]) -> int:
    """
    What is the state space size for the given named random variables.
    Support for Noiser implementations.

    Assumes:
        named random variables are in `rvs`.

    """
    states = get_states(rvs, rv_names)
    return multiply(len(ss) for ss in states)


def add_laplace_noise(cross_table: pd.DataFrame, safe_random: SafeRandom, b: float) -> pd.DataFrame:
    """
    Add Laplace noise to the given dataframe.
    Support for Noiser implementations.
    """
    size = cross_table.shape[0]
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


def compute_alpha(sensitivity: float, epsilon: float, min_cell_size: float) -> float:
    """
    Return the probability of a suppressed row being in a noisy cross-table
    after adding Laplace noise and enforcing min cell size.
    This assumes sensitivity > 0 and epsilon > 0.
    """
    # b = sensitivity / epsilon
    # return 0.5 * math.exp(-min_cell_size / b)
    return 0.5 * math.exp(-min_cell_size * epsilon / sensitivity)


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
