from synthorus.utils.validate_inputs import validate_inputs
from tests.helpers.unittest_fixture import Fixture, test_main


class ValidateInputsTest(Fixture):

    def test_validate_empty(self):
        # Should not throw
        validate_inputs([])

    def test_validate_okay(self):
        # Should not throw
        validate_inputs(['a', 'b', '_c'])

    def test_validate_duplicated(self):
        with self.assertRaises(ValueError) as context:
            validate_inputs(['a', 'b', 'a'])
        self.assertIn('duplicated', str(context.exception))

    def test_validate_numeric(self):
        with self.assertRaises(ValueError) as context:
            validate_inputs(['a', '123', '_c'])
        self.assertIn('invalid', str(context.exception))

    def test_validate_spaces(self):
        with self.assertRaises(ValueError) as context:
            validate_inputs(['a', 'b ', '_c'])
        self.assertIn('invalid', str(context.exception))

    def test_validate_non_identifier(self):
        with self.assertRaises(ValueError) as context:
            validate_inputs(['a', 'b.c', '_c'])
        self.assertIn('invalid', str(context.exception))


if __name__ == '__main__':
    test_main()
