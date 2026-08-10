"""
A module with extra iteration functions.
"""
from collections import Counter
from itertools import combinations, chain
from typing import Iterable, TypeVar, Sequence, Optional

_T = TypeVar('_T')


def flatten(lists: Iterable[Iterable[_T]]) -> Iterable[_T]:
    """
    Return a generator over the elements of a list of lists.
    """
    return (elem for the_list in lists for elem in the_list)


def deep_flatten(lists: Iterable) -> Iterable:
    """
    Generator over the flattening of nested collections (Iterable objects).
    Strings are not treated as an Iterable.
    """
    for el in lists:
        if isinstance(el, Iterable) and not isinstance(el, str):
            for sub in deep_flatten(el):
                yield sub
        else:
            yield el


def combos(list_of_lists: Sequence[Sequence[_T]], flip=False) -> Iterable[tuple[_T, ...]]:
    """
    Iterate over all combinations of taking one element from each of the lists.
    The order of results has the first element changing most rapidly.
    For example, given [[1,2,3],[4,5],[6,7]], `combos` yields the following:
        (1,4,6), (2,4,6), (3,4,6), (1,5,6), (2,5,6), (3,5,6),
        (1,4,7), (2,4,7), (3,4,7), (1,5,7), (2,5,7), (3,5,7).
    If flip, then the last changes most rapidly.
    """
    num = len(list_of_lists)
    if num == 0:
        yield ()
        return
    rng = range(num)
    indexes = [0] * num
    if flip:
        start = num - 1
        inc = -1
        end = -1
    else:
        start = 0
        inc = 1
        end = num
    while True:
        yield tuple(list_of_lists[i][indexes[i]] for i in rng)
        i = start
        while True:
            indexes[i] += 1
            if indexes[i] < len(list_of_lists[i]):
                break
            indexes[i] = 0
            i += inc
            if i == end:
                return


def combos_ranges(list_of_lens: Sequence[int], flip=False) -> Iterable[tuple[int, ...]]:
    """
    Equivalent to combos([range(l) for l in list_of_lens], flip)
    The order of results has the first element changing most rapidly.
    If flip, then the last changes most rapidly.
    """
    num = len(list_of_lens)
    if num == 0:
        yield ()
        return
    indexes = [0] * num
    if flip:
        start = num - 1
        inc = -1
        end = -1
    else:
        start = 0
        inc = 1
        end = num
    while True:
        yield tuple(indexes)
        i = start
        while True:
            indexes[i] += 1
            if indexes[i] < list_of_lens[i]:
                break
            indexes[i] = 0
            i += inc
            if i == end:
                return


def pairs(elements: Sequence[_T]) -> Iterable[tuple[_T, _T]]:
    """
    Iterate over all possible pairs in the given list of elements.
    """
    return combinations(elements, 2)


def sequential_pairs(elements: Sequence[_T]) -> Iterable[tuple[_T, _T]]:
    """
    Iterate over sequential pairs in the given list of elements.
    """
    for i in range(len(elements) - 1):
        yield elements[i], elements[i + 1]


def powerset(iterable: Iterable[_T], min_size: int = 0, max_size: Optional[int] = None) -> Iterable[tuple[_T, ...]]:
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    if not isinstance(iterable, (list, tuple)):
        iterable = list(iterable)
    if min_size is None:
        min_size = 0
    if max_size is None:
        max_size = len(iterable)
    return chain.from_iterable(
        combinations(iterable, size)
        for size in range(min_size, max_size + 1)
    )


def unzip(xs):
    """
    Inverse function of zip.
    a, b, c = unzip(zip(a, b, c))
    """
    return zip(*xs)


def ith(items: Iterable[_T], i: int) -> _T:
    """
    Return the ith element of the given iterable.
    """
    it = iter(items)
    x = next(it)
    for _ in range(i):
        x = next(it)
    return x


def first(items: Iterable[_T]) -> _T:
    """
    Return the first element of the given iterable.
    """
    return next(iter(items))


def second(items: Iterable[_T]) -> _T:
    """
    Return the second element of the given iterable.
    """
    return ith(items, 1)


def third(items: Iterable[_T]) -> _T:
    """
    Return the third element of the given iterable.
    """
    return ith(items, 2)


def last(items: Iterable[_T]) -> _T:
    """
    Return the last item of the given iterable.
    """
    it = iter(items)
    x = next(it)
    for y in it:
        x = y
    return x


def duplicates(iterable: Iterable[_T]) -> Iterable[_T]:
    """
    Return only elements that appear more than once in the given iterable.
    """
    return map(
        first,
        filter(
            lambda x: x[1] > 1,
            Counter(iterable).items()
        )
    )
