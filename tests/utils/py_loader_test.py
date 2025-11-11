from synthorus.utils import py_loader
from tests.helpers.tmp_dir import tmp_dir
from tests.helpers.unittest_fixture import Fixture, test_main

TEST_CONTENTS = """
a = 'a string'
b = 1234
c = '-' * 10
d = None

def double(x):
    return 2 * x
"""


def make_test_file(contents: str, file_name: str) -> None:
    with open(file_name, 'w') as file:
        print(contents, file=file)


class TestPyLoader(Fixture):

    def test_load_with_module_name(self):
        file_name = 'test_file.py'
        module_name = 'my_test_module'
        with tmp_dir():
            make_test_file(TEST_CONTENTS, file_name)
            module = py_loader.load(file_name, module_name=module_name)

        self.assertEqual(module.__name__, module_name)
        self.assertEqual(module.a, 'a string')
        self.assertEqual(module.b, 1234)
        self.assertEqual(module.c, '----------')
        self.assertIsNone(module.d)
        self.assertEqual(module.double(7), 14)

    def test_load_without_module_name(self):
        file_name = 'test_file.py'
        with tmp_dir():
            make_test_file(TEST_CONTENTS, file_name)
            module = py_loader.load(file_name)

        self.assertEqual(module.__name__, 'test_file')
        self.assertEqual(module.a, 'a string')
        self.assertEqual(module.b, 1234)
        self.assertEqual(module.c, '----------')
        self.assertIsNone(module.d)
        self.assertEqual(module.double(7), 14)

    def test_load_dict(self):
        file_name = 'test_file.py'
        with tmp_dir():
            make_test_file(TEST_CONTENTS, file_name)
            contents = py_loader.load_dict(file_name)

        self.assertEqual(contents['a'], 'a string')
        self.assertEqual(contents['b'], 1234)
        self.assertEqual(contents['c'], '----------')
        self.assertIsNone(contents['d'])
        self.assertEqual(contents['double'](7), 14)

    def test_get_object_by_type(self):
        file_name = 'test_file.py'
        with tmp_dir():
            make_test_file(TEST_CONTENTS, file_name)
            module = py_loader.load(file_name)
            obj = py_loader.get_object(module, object_type=int)
            self.assertEqual(obj, 1234)

    def test_get_object_by_name(self):
        file_name = 'test_file.py'
        with tmp_dir():
            make_test_file(TEST_CONTENTS, file_name)
            module = py_loader.load(file_name)
            obj = py_loader.get_object(module, variable='a')
            self.assertEqual(obj, 'a string')

    def test_load_object_by_type(self):
        file_name = 'test_file.py'
        with tmp_dir():
            make_test_file(TEST_CONTENTS, file_name)
            obj = py_loader.load_object(file_name, object_type=int)
            self.assertEqual(obj, 1234)

    def test_load_object_by_name(self):
        file_name = 'test_file.py'
        with tmp_dir():
            make_test_file(TEST_CONTENTS, file_name)
            obj = py_loader.load_object(file_name, variable='a')
            self.assertEqual(obj, 'a string')

    def test_no_load_object_by_type(self):
        file_name = 'test_file.py'
        with tmp_dir():
            make_test_file(TEST_CONTENTS, file_name)
            with self.assertRaises(Exception):
                py_loader.load_object(file_name, object_type=dict)

    def test_multi_load_object_by_type(self):
        file_name = 'test_file.py'
        with tmp_dir():
            make_test_file(TEST_CONTENTS, file_name)
            with self.assertRaises(Exception):
                py_loader.load_object(file_name, object_type=str)

    def test_fail_load_object_by_name(self):
        file_name = 'test_file.py'
        with tmp_dir():
            make_test_file(TEST_CONTENTS, file_name)
            with self.assertRaises(Exception):
                py_loader.load_object(file_name, variable='x')

    def test_fail_load_object_by_name_and_type(self):
        file_name = 'test_file.py'
        with tmp_dir():
            make_test_file(TEST_CONTENTS, file_name)
            with self.assertRaises(Exception):
                py_loader.load_object(file_name, variable='a', object_type=int)


if __name__ == '__main__':
    test_main()
