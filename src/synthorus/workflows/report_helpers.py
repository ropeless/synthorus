from typing import Dict, Any, List

from dominate.tags import table, tbody, tr, td, p, pre, code, style, html_tag

from synthorus.model.model_index import ModelIndex, CrosstabIndex
from synthorus.model.model_spec import ModelSpec, ModelCrosstabSpec
from synthorus.utils.string_extras import strip_lines, unindent


def add_head_styles() -> None:
    style("""
        .tree-menu details {
          margin-left: 20px;
          margin-top: 8px;
        }
    """)
    style("""
        code {
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
        }
    """)


def dict_table(data: Dict[Any, Any]) -> None:
    """
    Use `dominate` to render a dictionary as a two-column table.
    """
    cell_style: str = 'padding: 8px;'
    with table(border='1', style='width: 1%; white-space: nowrap; border-collapse: collapse;'):
        with tbody():
            for key, value in data.items():
                with tr():
                    td(str(key), style='color: gray; ' + cell_style)
                    if value is None:
                        td('', style=cell_style)
                    elif value == '':
                        td('""', style=cell_style)
                    elif isinstance(value, int):
                        td(f'{value:,}', style=cell_style)
                    elif isinstance(value, html_tag):
                        td(value, style=cell_style)
                    else:
                        td(str(value), style=cell_style)


def render_comment(comment: str) -> None:
    """
    Use `dominate` to render a comment.
    """
    # Break up a comment into paragraphs
    paras: List[str] = ['']
    for comment_line in comment.strip().split('\n'):
        comment_line = comment_line.strip()
        if len(comment_line) == 0:
            # Blank line
            if len(paras[-1]) > 0:
                paras.append('')
        else:
            paras[-1] += ' '
            paras[-1] += comment_line

    # Render paragraphs.
    for para in paras:
        if len(para) > 0:
            p(para)


def render_code(code_string: str) -> None:
    """
    Use `dominate` to render a block of code.
    """
    pre(code('\n' + unindent(code_string)))


def render_inline_data(data: str) -> None:
    """
    Use `dominate` to render a block of inline data.
    """
    pre(code('\n' + strip_lines(data)))


def rng_n_str(rng_n: int) -> str:
    """
    Interpret the random number generator security level as a
    human-readable string.

    See class SafeRandom in package modelling.noise.

    For details on the interpretation, see:
    Holohan, N., & Braghin, S. (2021, October). Secure random sampling in differential privacy.
    In European Symposium on Research in Computer Security (pp. 523-542). Springer, Cham.
    """
    if rng_n < 4:
        return 'lower than AES128 - unverified security'
    if rng_n == 4:
        return 'equivalent to AES128 - adequate security'
    if rng_n == 5:
        return 'equivalent to AES192 - good security'
    if rng_n == 6:
        return 'equivalent to AES256 - very good security'
    if rng_n > 6:
        return 'better than AES256 - excellent security'
    return 'no interpretation'


def budget_str(privacy_budget: float) -> str:
    """
    Interpret the privacy_budget as a human-readable string.
    """
    if privacy_budget <= 0:
        return 'no privacy risk'
    if privacy_budget < 0.01:
        return 'very good privacy protection'
    if privacy_budget < 1.0:
        return 'good privacy protection'
    if privacy_budget < 10.0:
        return 'low privacy protection'
    return 'effectively no privacy protection'


def calculate_privacy_budget(model_spec: ModelSpec, model_index: ModelIndex) -> float:
    """
    Calculate the total privacy budget for a model. This is
    the sum of cross-table epsilon values, for cross-tables of a
    datasource with sensitivity > 0.

    Args:
        model_spec: synthetic data model specification.
        model_index: cached relationships between model components.

    Returns:
        sum of cross-table epsilon using sources with sensitivity > 0.
    """
    total: float = 0
    crosstab_name: str
    crosstab_spec: ModelCrosstabSpec
    for crosstab_name, crosstab_spec in model_spec.crosstabs.items():
        crosstab_index: CrosstabIndex = model_index.crosstabs[crosstab_name]
        datasource_name: str = crosstab_index.datasource
        sensitivity: float = model_spec.datasources[datasource_name].sensitivity
        if sensitivity > 0:
            total += crosstab_spec.epsilon
    return total
