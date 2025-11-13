from synthorus.error import SynthorusError
from synthorus.utils.parse_formula import parse_formula
from tests.helpers.unittest_fixture import Fixture, test_main


class ParseFormulaTest(Fixture):

    def test_no_inputs(self):
        expression = '2 + 3 * 4'
        inputs = []

        function = parse_formula(expression, inputs)

        self.assertEqual(function(), 2 + 3 * 4)

    def test_simple_parse(self):
        expression = 'a + b * c'
        inputs = ['a', 'b', 'c']

        function = parse_formula(expression, inputs)

        self.assertEqual(function(2, 3, 4), 2 + 3 * 4)

    def test_complex_parse(self):
        # This expression includes:
        #     accepting complex arguments,
        #     the use of builtins,
        #     multiple lines,
        #     returns a tuple.
        expression = '''
        (
            (x[2] + y) ** 2 + min(z),
            y * max(z)
        )
        '''
        inputs = ['x', 'y', 'z']

        function = parse_formula(expression, inputs)

        self.assertEqual(function([3, 2, 1, 0], 3, [16, 17]), (32, 51))
        self.assertEqual(function([3, 2, 1, 0], 3, {16, 11}), (27, 48))

    def test_invalid_identifiers(self):
        expression = 'a + b * c'
        inputs = ['a, b, c']

        with self.assertRaises(SynthorusError) as context:
            _ = parse_formula(expression, inputs)
        self.assertIn('invalid argument identifier', str(context.exception))

    def test_unsafe(self):
        expression = 'a + b; a * b'
        inputs = ['a', 'b']

        with self.assertRaises(SynthorusError) as context:
            _ = parse_formula(expression, inputs)
        self.assertIn('could not parse', str(context.exception))

    def test_not_parseable(self):
        expression = 'a +++ b *** c'
        inputs = ['a', 'b', 'c']

        with self.assertRaises(SynthorusError) as context:
            _ = parse_formula(expression, inputs)
        self.assertIn('could not parse', str(context.exception))

    def test_invalid_token_exit(self):
        expression = 'exit(0)'
        inputs = []

        with self.assertRaises(SynthorusError) as context:
            _ = parse_formula(expression, inputs)
        self.assertIn('exit', str(context.exception))
        self.assertIn('invalid character sequence', str(context.exception))

    def test_invalid_token_import(self):
        expression = 'import os'
        inputs = []

        with self.assertRaises(SynthorusError) as context:
            _ = parse_formula(expression, inputs)
        self.assertIn('import', str(context.exception))
        self.assertIn('invalid character sequence', str(context.exception))

    def test_invalid_token_eval(self):
        expression = 'eval("1 + 2")'
        inputs = []

        with self.assertRaises(SynthorusError) as context:
            _ = parse_formula(expression, inputs)
        self.assertIn('eval', str(context.exception))
        self.assertIn('invalid character sequence', str(context.exception))

    def test_invalid_token_print(self):
        expression = 'print("1 + 2")'
        inputs = []

        with self.assertRaises(SynthorusError) as context:
            _ = parse_formula(expression, inputs)
        self.assertIn('print', str(context.exception))
        self.assertIn('invalid character sequence', str(context.exception))

    def test_invalid_token_input(self):
        expression = 'input()'
        inputs = []

        with self.assertRaises(SynthorusError) as context:
            _ = parse_formula(expression, inputs)
        self.assertIn('input', str(context.exception))
        self.assertIn('invalid character sequence', str(context.exception))

    def test_invalid_token_open(self):
        expression = 'open("my_file")'
        inputs = []

        with self.assertRaises(SynthorusError) as context:
            _ = parse_formula(expression, inputs)
        self.assertIn('open', str(context.exception))
        self.assertIn('invalid character sequence', str(context.exception))


if __name__ == '__main__':
    test_main()
