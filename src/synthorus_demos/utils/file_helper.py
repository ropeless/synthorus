from importlib.abc import Traversable
from os import PathLike
from pathlib import Path


def head(file: Path | Traversable, number_of_lines: int = 5) -> None:
    """
    Print the first few lines of a text file.

    Args:
        file: path to the file.
        number_of_lines: how many lines to print. Default is 5.
    """
    with open(file, 'r') as file:
        for _ in range(number_of_lines):
            line: str = file.readline()
            if line == '':
                break
            print(line.rstrip())


def cat(file: PathLike | Traversable) -> None:
    """
    Print all lines of a text file.

    Args:
        file: path to the file.
    """
    with open(file, 'r') as file:
        while line := file.readline():
            print(line.rstrip())


def print_file_tree(start: PathLike | Traversable, indent: str = '  ', prefix: str = '') -> None:
    start = Path(start)
    if start.exists():
        if start.is_dir():
            print(f'{prefix}{start.name}/')
            next_prefix: str = prefix + indent
            for file in start.iterdir():
                print_file_tree(file, indent, next_prefix)
        else:
            print(f'{prefix}{start.name}')
