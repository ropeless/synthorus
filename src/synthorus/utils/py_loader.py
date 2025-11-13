"""
Module for explicitly loading Python modules (source files).
"""

import importlib.machinery as _machinery
import os as _os
from types import ModuleType
from typing import Optional


def load(filepath, *, module_name: Optional[str] = None) -> ModuleType:
    """
    Load the given Python module (source file), named filepath, as a Python module.
    The module can be given an explicit name, module_name, or
    a default name will be constructed from filepath.
    The loaded module is returned.

    Args:
        filepath: is a string or path to the Python source.
        module_name: is an optional name for the module (used by SourceFileLoader).
    """
    if module_name is None:
        module_name = _os.path.basename(filepath)
        module_name = _os.path.splitext(module_name)[0]
    loader = _machinery.SourceFileLoader(str(module_name), str(filepath))
    module = ModuleType(loader.name)
    loader.exec_module(module)
    return module


def load_object(filepath, *, object_type=object, variable: Optional[str] = None, module_name: Optional[str] = None):
    """
    Load the given Python module (source file), named filepath and return
    the identified object in the module. The object is identified by type
    and/or variable name.

    Will throw RuntimeError if a unique object cannot be found in the loaded module.

    Args:
        filepath: is a string or path to the Python source.
        object_type: a restriction on the expected type of the object in the loaded module.
        variable: a restriction on the expected type of the object in the loaded module.
            If None, the all variables not starting with underscore (_) are checked.
        module_name: is an optional name for the module (used by SourceFileLoader).
    
    Returns:
        an object of type 'object_type'.
    """
    module = load(filepath, module_name=module_name)
    return get_object(module, object_type=object_type, variable=variable)


def get_object(module: ModuleType, *, object_type=object, variable: Optional[str] = None):
    """
    Given the loaded Python module, return the identified object
    in the module. The object is identified by type and/or variable name.

    Will throw RuntimeError if a unique object cannot be found in the loaded module.

    Args:
        module: is module loaded using py_loader.load(...).
        object_type: a restriction on the expected type of the object in the loaded module.
        variable: a restriction on the expected type of the object in the loaded module.
            If None, the all variables not starting with underscore (_) are checked.
    
    Returns:
        an object of type 'object_type'.
    """
    module_vars = vars(module)

    if variable is None:
        potentials = [
            (var, value)
            for var, value in module_vars.items()
            if not var.startswith('_') and isinstance(value, object_type)
        ]
        if len(potentials) == 0:
            raise RuntimeError(f'no objects of type {object_type} defined in the file')
        if len(potentials) > 1:
            matches = ', '.join(var for var, value in potentials)
            raise RuntimeError(f'multiple objects of type {object_type} defined in the file: {matches}')
        obj = potentials[0][1]
    else:
        if variable not in module_vars.keys():
            raise RuntimeError(f'object {variable!r} is not defined in the file')
        obj = module_vars[variable]
        if not isinstance(obj, object_type):
            raise RuntimeError(f'object {variable} is not of type {object_type}')
    return obj


def load_dict(filepath, *, module_name: Optional[str] = None) -> dict:
    """
    Load the given Python source file, named filepath,
    and return the loaded objects as a dictionary.

    Args:
        filepath: is a string or path to the Python source.
        module_name: is an optional name for the module (used by SourceFileLoader).
    """
    return vars(load(filepath, module_name=module_name))
