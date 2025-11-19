from synthorus.utils.string_extras import unindent, strip_lines
from tests.helpers.unittest_fixture import Fixture, test_main


class UnindentTest(Fixture):

    def test_unindent_empty(self):
        input_string = ''

        expect_string = '\n'

        clean_string = unindent(input_string)

        self.assertEqual(clean_string, expect_string)

    def test_unindent(self):
        input_string = """
            This is the first line
              This line starts with two spaces
            This is the last line
            """

        expect_string = (
            "This is the first line\n"
            "  This line starts with two spaces\n"
            "This is the last line\n"
        )

        clean_string = unindent(input_string)

        self.assertEqual(clean_string, expect_string)

    def test_unindent_trailing_spaces(self):
        input_string = (
            "This is the first line            \n"
            "  This line starts with two spaces          \n"
            "This is the last line  \n"
        )

        expect_string = (
            "This is the first line\n"
            "  This line starts with two spaces\n"
            "This is the last line\n"
        )

        clean_string = unindent(input_string)

        self.assertEqual(clean_string, expect_string)

    def test_unindent_no_last_return(self):
        input_string = (
            "This is the first line\n"
            "  This line starts with two spaces\n"
            "This is the last line"
        )

        expect_string = (
            "This is the first line\n"
            "  This line starts with two spaces\n"
            "This is the last line\n"
        )

        clean_string = unindent(input_string)

        self.assertEqual(clean_string, expect_string)

    def test_unindent_already_clean(self):
        input_string = (
            "This is the first line\n"
            "  This line starts with two spaces\n"
            "This is the last line\n"
        )

        expect_string = (
            "This is the first line\n"
            "  This line starts with two spaces\n"
            "This is the last line\n"
        )

        clean_string = unindent(input_string)

        self.assertEqual(clean_string, expect_string)


class StripLinesTest(Fixture):

    def test_strip_lines_empty(self):
        input_string = ''

        expect_string = ''

        clean_string = strip_lines(input_string)

        self.assertEqual(clean_string, expect_string)

    def test_strip_lines(self):
        input_string = """
            This is the first line
              This line starts with two spaces
            This is the last line
            """

        expect_string = (
            "This is the first line\n"
            "This line starts with two spaces\n"
            "This is the last line\n"
        )

        clean_string = strip_lines(input_string)

        self.assertEqual(clean_string, expect_string)

    def test_strip_lines_trailing_spaces(self):
        input_string = (
            "This is the first line            \n"
            "  This line starts with two spaces          \n"
            "This is the last line  \n"
        )

        expect_string = (
            "This is the first line\n"
            "This line starts with two spaces\n"
            "This is the last line\n"
        )

        clean_string = strip_lines(input_string)

        self.assertEqual(clean_string, expect_string)

    def test_strip_lines_no_last_return(self):
        input_string = (
            "This is the first line\n"
            "  This line starts with two spaces\n"
            "This is the last line"
        )

        expect_string = (
            "This is the first line\n"
            "This line starts with two spaces\n"
            "This is the last line\n"
        )

        clean_string = strip_lines(input_string)

        self.assertEqual(clean_string, expect_string)

    def test_strip_blank_lines(self):
        input_string = (
            "This is the first line\n"
            "\n"
            "  This line starts with two spaces  \n"
            "\n"
            "\n"
            "This is the last line\n"
        )

        expect_string = (
            "This is the first line\n"
            "This line starts with two spaces\n"
            "This is the last line\n"
        )

        clean_string = strip_lines(input_string)

        self.assertEqual(clean_string, expect_string)

    def test_strip_lines_already_clean(self):
        input_string = (
            "This is the first line\n"
            "This line does not start with two spaces\n"
            "This is the last line\n"
        )

        expect_string = (
            "This is the first line\n"
            "This line does not start with two spaces\n"
            "This is the last line\n"
        )

        clean_string = strip_lines(input_string)

        self.assertEqual(clean_string, expect_string)


if __name__ == '__main__':
    test_main()
