from pathlib import Path

from synthorus.utils.file_extras import DataPathLike, open_text


def head(file_path: DataPathLike, number_of_lines: int = 5) -> None:
    """
    Print the first few lines of a text file.

    Args:
        file_path: path to the file.
        number_of_lines: how many lines to print. Default is 5.
    """
    with open_text(file_path) as file:
        for _ in range(number_of_lines):
            line: str = file.readline()
            if line == '':
                break
            print(line.rstrip())


def cat(file_path: DataPathLike) -> None:
    """
    Print all lines of a text file.

    Args:
        file_path: path to the file.
    """
    with open_text(file_path) as file:
        while line := file.readline():
            print(line.rstrip())


def print_file_tree(start: DataPathLike, indent: str = '  ', prefix: str = '') -> None:
    if isinstance(start, str):
        start = Path(start)
    if start.is_dir():
        print(f'{prefix}{start.name}/')
        next_prefix: str = prefix + indent
        for file in start.iterdir():
            print_file_tree(file, indent, next_prefix)
    elif start.is_file():
        print(f'{prefix}{start.name}')
