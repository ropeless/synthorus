import math
import pickle
from pathlib import Path
from typing import Dict, Sequence, Mapping, Optional

from jupyter_lsp.specs import json

from synthorus.error import SynthorusError
from synthorus.utils import py_loader


def save_dict(the_dict: Dict, filepath: Path, assign_var='the_dict'):
    if '.' not in filepath.name:
        raise SynthorusError('could not infer file type: no file name extension')

    ext = filepath.suffix.lower()
    if ext == '.py':
        with open(filepath, 'w') as file:
            pretty_print_dict(
                the_dict=the_dict,
                assign_var=assign_var,
                indent=4,
                file=file
            )
    elif ext == '.json':
        with open(filepath, 'w') as file:
            json.dump(the_dict, file, indent=4)
    elif ext == '.pk':
        with open(filepath, 'wb') as file:
            pickle.dump(the_dict, file)
    else:
        raise SynthorusError(f'could not infer file type from file name extension: {ext!r}')


def load_dict(filepath: Path, assign_var=None) -> Dict:
    if '.' not in filepath.name:
        raise SynthorusError('could not infer file type: no file name extension')

    ext = filepath.suffix.lower()
    if ext == '.py':
        return py_loader.load_object(
            filepath,
            variable=assign_var,
            object_type=dict
        )
    elif ext == '.json':
        with open(filepath, 'w') as file:
            return json.load(file)
    elif ext == '.pk':
        with open(filepath, 'wb') as file:
            return pickle.load(file)
    else:
        raise SynthorusError(f'could not infer file type from file name extension: {ext!r}')


def pretty_print_dict(
        the_dict: Dict,
        assign_var: Optional[str] = None,
        indent: int = 4,
        keys: Optional[Mapping] = None,
        keys_module_name: Optional[str] = None,
        keys_import_name: str = '*',
        print_import: bool = False,
        file=None
):
    """
    Render the given dict as pretty-printed Python code.

    Args:
        the_dict: Python dict to render.
        assign_var: if not None, then assign the dict to a variable in the render.
        indent: number of spaces for indentation.
        keys: is vars(keys_module) where 'keys_module' is an imported module.
        keys_module_name: is the name to import the 'keys' module.
        keys_import_name: a name to import keys as, or '*'.
        print_import: if true, and import_keys_as is not None, then render the import line.
        file: the output file to write to or None for std out (passed to print).
    """
    assert (keys is None) == (keys_module_name is None)

    indent = ' ' * indent  # convert indent to a string
    if keys is not None:
        if keys_import_name == '*':
            import_str = f'from {keys_module_name} import *'
            key_map = {
                value: name
                for name, value in keys.items()
                if isinstance(value, str) and not name.startswith('_')
            }
        else:
            import_str = f'import {keys_module_name} as {keys_import_name}'
            key_map = {
                value: f'{keys_import_name}.{name}'
                for name, value in keys.items()
                if isinstance(value, str) and not name.startswith('_')
            }
        if print_import:
            print(import_str, file=file)
            print(file=file)
    else:
        key_map = {}
    if assign_var is not None:
        print(f'{assign_var} = ', end='', file=file)
    _write_python_dict_r(the_dict, file, key_map, prefix='', indent=indent, postfix='')


def _write_python_r(value, file, key_map, prefix, indent, postfix):
    if isinstance(value, Mapping):
        _write_python_dict_r(value, file, key_map, prefix, indent, postfix)
    elif not isinstance(value, str) and isinstance(value, tuple):
        _write_python_list_r(value, file, key_map, prefix, indent, postfix, '(', ')')
    elif not isinstance(value, str) and isinstance(value, Sequence):
        _write_python_list_r(value, file, key_map, prefix, indent, postfix, '[', ']')
    elif isinstance(value, float) and math.isnan(value):
        # It seems 'nan' needs to be treated specially.
        # See https://bugs.python.org/issue1732212
        print("float('nan')" + postfix, file=file)
    else:
        print(repr(value) + postfix, file=file)


def _write_python_dict_r(the_dict: Mapping, file, key_map, prefix, indent, postfix):
    keys = list(the_dict.keys())
    if len(keys) == 0:
        print('{}' + postfix, file=file)
    elif len(keys) == 1 and isinstance(the_dict[keys[0]], (int, float, str, type(None))):
        _key = keys[0]
        value = the_dict[_key]
        key_str = key_map.get(_key, repr(_key))
        print('{' + key_str + ': ' + repr(value) + '}' + postfix, file=file)
    else:
        next_prefix = prefix + indent
        print('{', file=file)
        for _key in keys[:-1]:
            value = the_dict[_key]
            key_str = key_map.get(_key, repr(_key))
            print(f'{next_prefix}{key_str}: ', file=file, end='')
            _write_python_r(value, file, key_map, next_prefix, indent, ',')

        _key = keys[-1]
        _value = the_dict[_key]
        key_str = key_map.get(_key, repr(_key))
        print(f'{next_prefix}{key_str}: ', file=file, end='')
        _write_python_r(_value, file, key_map, next_prefix, indent, '')

        print(prefix + '}' + postfix, file=file)


def _write_python_list_r(the_list: Sequence, file, key_map, prefix, indent, postfix, open_str, close_str):
    if len(the_list) == 0:
        print(open_str + close_str + postfix, file=file)
    elif len(the_list) == 1 and isinstance(the_list[0], (int, float, str, type(None))):
        if close_str == ')':
            # tuple special case
            print(f'{open_str}{the_list[0]!r},{close_str}{postfix}', file=file)
        else:
            print(f'{open_str}{the_list[0]!r}{close_str}{postfix}', file=file)
    else:
        next_prefix = prefix + indent
        print(open_str, file=file)
        for value in the_list[:-1]:
            print(next_prefix, file=file, end='')
            _write_python_r(value, file, key_map, next_prefix, indent, ',')

        print(next_prefix, file=file, end='')
        _write_python_r(the_list[-1], file, key_map, next_prefix, indent, '')

        print(prefix + close_str + postfix, file=file)
