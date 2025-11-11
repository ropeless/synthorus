from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Iterable, List, Union, TypeAlias, Sequence, Set, MutableMapping, Dict, Any

from ck.pgm import State

from synthorus.error import SpecFileError
from synthorus.utils.const import Const

_VALID_ID_CHARS: Set[str] = set(
    '0123456789'
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '_-.,'
)

SpecValue: TypeAlias = Union[State, Sequence[State], Set[State], 'SpecDict']

_None = Const('_None')


class Key:
    """
    Common keys for any SpecDict
    """
    location = 'location'
    inline = 'inline'


class SpecDict(MutableMapping[str, SpecValue]):
    """
    This is a dict that allows us to get values in a controlled and defaulting way.

    A SpecDict has local (key, value) items as per a normal dict. The
    find(key) method will check local items and also items defined in parent
    Mapping objects.
    """

    def __init__(
            self,
            section: str,
            dict_id: str,
            *defaults: Mapping[str, SpecValue],
            update: Optional[Mapping[str, SpecValue]] = None,
            dont_inherit: Iterable[str] = ()
    ):
        """
        Construct a SpecDict object.

        Args:
            section: section name for this SpecDict.
            dict_id: the parent-given name for this SpecDict.
            defaults: zero or more Mappings of default values.
            the_key: key in this model dict that identifies the sub-dict.
            update: is an optional set of key-value pair to update in created mapping.
            dont_inherit: a collection of keys to not inherit from parents.
        """
        self._section = section
        self._dict_id = dict_id
        self._defaults = defaults
        self._dict = {} if update is None else dict(update)
        self._dont_inherit = set() if dont_inherit is None else set(dont_inherit)
        self._warnings = []

    @property
    def section(self) -> str:
        """
        Section name for this SpecDict.
        This is normally set to '' for a root SpecDict.
        """
        return self._section

    @property
    def dict_id(self) -> str:
        """
        The parent-given name for this SpecDict.
        This is normally set to '' for a root SpecDict.
        """
        return self._dict_id

    def copy(self, update: Optional[Mapping] = None) -> SpecDict:
        """
        Shallow copy - only copies the local entries, not parents.
        """
        result = SpecDict(
            self._section,
            self._dict_id,
            *self._defaults,
            update=self._dict,
            dont_inherit=self._dont_inherit
        )
        if update is not None:
            result._dict.update(update)
        return result

    def sub_section(self, the_key) -> str:
        """
        What would be the section name, for a sub-dict
        identified by the given key?
        """
        if self.section == '':
            return str(the_key)
        else:
            return f'{self.section}:{the_key}'

    def sub_dict(
            self,
            the_key,
            update: Optional[Mapping] = None,
            dont_inherit: Optional[Iterable] = None
    ) -> SpecDict:
        """
        Get the sub-dict, identified by the given key.

        Args:
            the_key: key in this model dict that identifies the sub-dict.
            update: is an optional set of key-value pair to update the returned mapping.
            dont_inherit: a list of keys to not inherit from parents.
        """
        result = SpecDict(
            self.sub_section(the_key),
            the_key,
            self,
            update=update,
            dont_inherit=dont_inherit
        )
        return result

    def __str__(self):
        return self.section

    def __iter__(self):
        """
        Iterator over keys of local entries.
        """
        return iter(self._dict)

    def __len__(self) -> int:
        """
        Number of local entries.
        """
        return len(self._dict)

    def keys(self):
        """
        Return the set of locally defined keys (not inherited keys).
        """
        return self._dict.keys()

    def items(self):
        """
        Return the set of locally defined (key, value) pairs (not inherited keys).
        """
        return self._dict.items()

    def __getitem__(self, item_key):
        """
        Get the value of the given key, local entries only.
        """
        return self._dict[item_key]

    def __setitem__(self, item_key, value):
        """
        Add or update a local entry for this SpecDict.
        """
        self._dict[item_key] = value

    def __delitem__(self, key, /):
        del self._dict[key]

    def update(self, to_copy, **kwargs):
        self._dict.update(to_copy, **kwargs)

    def get(self, item_key, default=None):
        """
        Get the value for the given key from
        the local entries only.
        """
        return self._dict.get(item_key, default)

    def find(self, item_key, default=None):
        """
        Get the value for the given key. First checks local
        entries, then checks parents. And if not found,
        then the provided default value is returned.
        """
        val = self._dict.get(item_key, _None)
        if val is not _None:
            return val
        if item_key not in self._dont_inherit:
            for d in self._defaults:
                if isinstance(d, SpecDict):
                    val = d.find(item_key, _None)
                else:
                    val = d.get(item_key, _None)
                if val is not _None:
                    return val
        return default

    def get_string(self, the_key, default: Optional[str] = None) -> str:
        the_val = self.check_exists(the_key, default=default)
        if not isinstance(the_val, str):
            raise self.error('invalid value', f'{the_key} must be a string')
        return the_val

    def get_string_optional(self, the_key) -> Optional[str]:
        the_val = self.find(the_key)
        if not isinstance(the_val, (type(None), str)):
            raise self.error('invalid value', f'{the_key} must be a string')
        return the_val

    def get_string_list(self, the_key, default: Optional[List[str]] = None) -> List[str]:
        the_val = self.check_exists(the_key, default)
        if isinstance(the_val, str):
            return [the_val]
        elif isinstance(the_val, Mapping):
            raise self.error('invalid value', f'{the_key} must be a string or list of strings')
        elif isinstance(the_val, Iterable):
            the_val = list(the_val)
        else:
            raise self.error('invalid value', f'{the_key} must be a string or list of strings')
        for elem in the_val:
            if not isinstance(elem, str):
                raise self.error('invalid value', f'{the_key} must be a string or list of strings')
        return the_val

    def get_state_optional(self, the_key: str) -> State:
        the_val = self.find(the_key)
        if not isinstance(the_val, (str, int, float, bool, type(None))):
            raise self.error('invalid value', f'{the_key} must be a random variable state value')
        return the_val

    def get_bool(self, the_key, default: Optional[bool] = None) -> bool:
        the_val = self.check_exists(the_key, default=default)
        return self.check_is_bool(the_val)

    def get_bool_optional(self, the_key) -> Optional[bool]:
        the_val = self.find(the_key)
        if the_val is None:
            return None
        else:
            return self.check_is_bool(the_val)

    def get_positive(self, the_key, default: Optional[float] = None) -> float:
        return self.get_numeric(
            the_key,
            lambda x: x > 0,
            'must be a positive number',
            default=default
        )

    def get_non_neg(self, the_key, default: Optional[float] = None) -> float:
        return self.get_numeric(
            the_key,
            lambda x: x >= 0,
            'must be a non-negative number',
            default=default
        )

    def get_positive_int(self, the_key, default: Optional[int] = None) -> int:
        return self.get_int(
            the_key,
            lambda x: x > 0,
            'must be a positive integer',
            default=default
        )

    def get_non_neg_int(self, the_key, default: Optional[int] = None) -> int:
        return self.get_int(
            the_key,
            lambda x: x >= 0,
            'must be a non-negative integer',
            default=default
        )

    def get_numeric(self, the_key, predicate=None, error='must be numeric', default: Optional[float] = None) -> float:
        the_val = self.check_exists(the_key, default=default)
        if not isinstance(the_val, (int, float)):
            raise self.error('invalid value', f'{the_key} {error}')
        if predicate is not None:
            if not predicate(the_val):
                raise self.error('invalid value', f'{the_key} {error}')
        return float(the_val)

    def get_numeric_optional(self, the_key, predicate=None, error='must be numeric') -> Optional[float]:
        the_val = self.find(the_key)
        if the_val is None:
            return None
        if not isinstance(the_val, (int, float)):
            raise self.error('invalid value', f'{the_key} {error}')
        if predicate is not None:
            if not predicate(the_val):
                raise self.error('invalid value', f'{the_key} {error}')
        return float(the_val)

    def get_int(self, the_key, predicate=None, error='must be an integer', default: Optional[int] = None) -> int:
        the_val = self.check_exists(the_key, default=default)
        if not isinstance(the_val, int):
            raise self.error('invalid value', f'{the_key} {error}')
        if predicate is not None:
            if not predicate(the_val):
                raise self.error('invalid value', f'{the_key} {error}')
        return the_val

    def get_int_optional(self, the_key, predicate=None, error='must be an integer') -> Optional[int]:
        the_val = self.find(the_key)
        if the_val is None:
            return None

        if not isinstance(the_val, int):
            raise self.error('invalid value', f'{the_key} {error}')
        if predicate is not None:
            if not predicate(the_val):
                raise self.error('invalid value', f'{the_key} {error}')
        return the_val

    def get_dict(
            self,
            the_key,
            *,
            dont_inherit: Optional[Iterable] = None,
            default: Optional[Dict[str, Any]] = None,
    ) -> SpecDict:
        """
        Args:
            the_key: key in this model dict that identifies the sub-dict.
            dont_inherit: a list of keys to not inherit from parents.
            default: optional default value.
        """
        the_val = self.check_exists(the_key, default=default)
        if not isinstance(the_val, Mapping):
            raise self.error('invalid value', f'{the_key} must be a dictionary')
        return self.sub_dict(the_key, update=the_val, dont_inherit=dont_inherit)

    def get_dict_optional(self, the_key, *, dont_inherit: Optional[Iterable] = None) -> Optional[SpecDict]:
        """
        Args:
            the_key: key in this model dict that identifies the sub-dict.
            dont_inherit: a list of keys to not inherit from parents.
        """
        the_val = self.find(the_key)
        if the_val is None:
            return None
        if not isinstance(the_val, Mapping):
            raise self.error('invalid value', f'{the_key} must be a dictionary')
        return self.sub_dict(the_key, update=the_val, dont_inherit=dont_inherit)

    def check_exists(self, the_key, default=None) -> Any:
        """
        Confirm the key is defined, either in our local entries
        or inherited entries.
        """
        the_val = self.find(the_key, default)
        if the_val is None:
            raise self.error('missing key', the_key)
        return the_val

    def check_not_exists(self, *the_keys, message='unexpected key'):
        """
        Confirm the keys are not defined in our local keys.
        I.e., does not check inherited entries.
        """
        for the_key in the_keys:
            if the_key in self.keys():
                raise self.error(message, the_key)

    def check_mutually_exclusive(self, *the_keys, message='incompatible keys'):
        """
        Confirm that if one of the given key is defined in our local keys,
        then no other is.
        I.e., does not check inherited entries.
        """
        found = set(self.keys())
        found.intersection_update(the_keys)
        if len(found) > 1:
            raise self.error(message, found)

    def check_restricted(self, allowed_keys, *, message='unexpected key'):
        """
        Confirm only the allowed keys are defined in our local keys.
        I.e., does not check inherited entries.
        """
        if not isinstance(allowed_keys, set):
            allowed_keys = set(allowed_keys)
        for the_key in self.keys():
            if the_key not in allowed_keys:
                raise self.error(message, the_key)

    def get_path(self, roots, default_location) -> Path:
        """
        Just like get_pandas_source, but only key 'location' is checked.
        Key 'inline' is explicitly not allowed.
        """
        self.check_not_exists(Key.inline)
        location = self.get_string_optional(Key.location)
        location = default_location if location is None else location
        return self._make_path(location, roots)

    def check_is_id(self, the_val, error='invalid id') -> str:
        if not isinstance(the_val, str):
            raise self.error(error, repr(the_val))
        if the_val == '':
            raise self.error(error, 'empty string is not permitted')
        if not set(the_val) <= _VALID_ID_CHARS:
            raise self.error(error, repr(the_val))
        return the_val

    def check_is_state(self, the_val) -> State:
        if not isinstance(the_val, (str, int, float, type(None))):
            raise self.error('invalid state', repr(the_val))
        return the_val

    def check_is_string(self, the_val) -> str:
        if not isinstance(the_val, str):
            raise self.error('invalid string', repr(the_val))
        return the_val

    def check_is_bool(self, the_val) -> bool:
        if isinstance(the_val, str):
            check_val = the_val.lower()
        else:
            check_val = the_val
        if check_val in (True, 1, '1', 'yes', 'true'):
            return True
        if check_val in (False, 0, '0', 'no', 'false'):
            return False
        raise self.error('invalid Boolean', repr(the_val))

    def check_is_positive(self, the_val) -> int | float:
        return self.check_is_numeric(
            the_val,
            predicate=(lambda x: x > 0),
            error='not a positive number'
        )

    def check_is_positive_int(self, the_val) -> int:
        return self.check_is_int(
            the_val,
            predicate=(lambda x: x > 0),
            error='not a positive integer'
        )

    def check_is_non_neg(self, the_val) -> int | float:
        return self.check_is_numeric(
            the_val,
            predicate=(lambda x: x >= 0),
            error='not a non-negative number'
        )

    def check_is_numeric(self, the_val, *, predicate=None, error='not a number') -> int | float:
        if not isinstance(the_val, (int, float)):
            raise self.error(error, repr(the_val))
        if predicate is not None:
            if not predicate(the_val):
                raise self.error(error, repr(the_val))
        return the_val

    def check_is_int(self, the_val, *, predicate=None, error='not an integer') -> int:
        if not isinstance(the_val, int):
            raise self.error(error, repr(the_val))
        if predicate is not None:
            if not predicate(the_val):
                raise self.error(error, repr(the_val))
        return the_val

    def error(self, error: str, details=None) -> SpecFileError:
        """
        Return a SpecError, with the 'section' as per this SpecDict.

        Usage:
            raise my_model_dict.error(...)
        """
        return SpecFileError(error, self.section, details)

    def warn(self, error: str, details=None) -> None:
        msg = str(self.error(error, details))
        self._record_warning(msg)

    def _record_warning(self, msg: str) -> None:
        self._warnings.append(msg)
        for default in self._defaults:
            if isinstance(default, SpecDict):
                default._record_warning(msg)

    @property
    def warnings(self) -> List[str]:
        return self._warnings

    def _make_path(self, location, roots) -> Path:
        """
        Support for get_source and get_path.

        If location is not an absolute path, then looks in all roots for 'location'.
        Raises an error if not exactly one path found to exist.
        """
        location_as_path = Path(location)
        if location_as_path.is_absolute():
            if not location_as_path.exists():
                raise self.error(f'absolute file location but file not found', location)
            return location_as_path
        found = None
        for root in roots:
            location_as_path = root / location
            if location_as_path.exists():
                if found is not None:
                    raise self.error(f'multiple source files found', location)
                found = location_as_path
        if found is None:
            raise self.error(f'could not resolve file location', location)
        return found
