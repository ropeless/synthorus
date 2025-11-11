from typing import Set, List, Tuple

from synthorus.error import SynthorusError
from synthorus.model.model_spec import ModelSpec


def _find_crosstab_datasource(crosstab_name: str, rvs: Set[str], model_spec: ModelSpec) -> str:
    """
    find a suitable datasource to cover the given random variables.

    This may also update the cross-table rvs and non_distribution_rvs
    if the datasource has non-distribution rvs.

    Args:
        crosstab_name: the name of the cross-table.
        rvs: the cross-table random variables.
        model_spec: the specified model.

    Returns:
        a datasource name
    """

    # Find all datasources that cover the cross-table's random variables
    candidate_datasources: List[Tuple[int, str]] = [
        (len(datasource.rvs), datasource_name)
        for datasource_name, datasource in model_spec.datasources.items()
        if rvs.issubset(datasource.rvs)
    ]

    if len(candidate_datasources) == 0:
        raise SynthorusError(
            f'cannot find a datasource to cover all random variables of cross-table: {crosstab_name!r}'
        )

    # Use the candidate with the fewest random variables.
    return min(candidate_datasources)[1]
