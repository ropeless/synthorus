from typing import Dict, Mapping, Any, Optional

# noinspection PyProtectedMember
from synthorus.dataset._dataset_impl._connection_params import connection_str, resolve_connection
from synthorus.error import SynthorusError
from tests.helpers.unittest_fixture import Fixture, test_main


class MathDatasetTest(Fixture):

    def test_resolve_connection_param(self):
        env: Mapping[str, Any] = {
            'DB_A': 'a',
            'DB_B': 2,
            'DB_C': 3.5,
            'DB_D': None,
            'DB_E': None,
        }
        connection_params: Optional[Dict[str, Optional[str | int]]] = {
            'A': 'A',
            'B': None,
            'E': 'E',
        }
        expect: Dict[str, str] = {
            'A': 'A',
            'B': '2',
            'E': 'E',
        }

        result: Dict[str, str] = resolve_connection(connection_params, env=env)
        self.assertEqual(result, expect)

    def test_resolve_connection_param_missing(self):
        env: Mapping[str, Any] = {
            'DB_A': 'a',
            'DB_B': 2,
            'DB_C': 3.5,
            'DB_D': None,
            'DB_E': None,
        }
        connection_params: Optional[Dict[str, Optional[str | int]]] = {
            'A': 'A',
            'B': None,
            'E': None,
        }

        with self.assertRaises(SynthorusError) as context:
            _: Dict[str, str] = resolve_connection(connection_params, env=env)
        self.assertEqual(str(context.exception), "cannot resolve connection parameter: 'E'")

    def test_resolve_connection_str(self):
        env: Mapping[str, Any] = {
            'DB_A': 'a',
            'DB_B': 2,
            'DB_C': 3.5,
            'DB_D': None,
            'DB_E': None,
            'DB_CONNECTION': {
                'A': 'A',
                'B': None,
                'E': 'E',
            },
        }
        expect: Dict[str, str] = {
            'A': 'A',
            'B': '2',
            'E': 'E',
        }

        result: Dict[str, str] = resolve_connection(None, env=env)
        self.assertEqual(result, expect)

    def test_resolve_connection_str_missing(self):
        env: Mapping[str, Any] = {
            'DB_A': 'a',
            'DB_B': 2,
            'DB_C': 3.5,
            'DB_D': None,
            'DB_E': None,
        }

        with self.assertRaises(SynthorusError) as context:
            _: Dict[str, str] = resolve_connection(None, env=env)
        self.assertEqual(str(context.exception), "cannot resolve connection parameter: 'DB_CONNECTION'")

    def test_connection_str(self):
        connection_params: Dict[str, str] = {
            'A': '1',
            'B': '2',
            'C': '3',
            'D': '4',
        }
        result = connection_str(connection_params)
        self.assertEqual(result, 'A=1;B=2;C=3;D=4')

    def test_connection_str_sorted(self):
        connection_params: Dict[str, str] = {
            'D': '4',
            'A': '1',
            'C': '3',
            'B': '2',
        }
        result = connection_str(connection_params)
        self.assertEqual(result, 'A=1;B=2;C=3;D=4')


if __name__ == '__main__':
    test_main()
