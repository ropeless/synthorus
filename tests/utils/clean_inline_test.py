from synthorus.utils.clean_inline import clean_inline
from tests.helpers.unittest_fixture import Fixture, test_main


class CleanInlineTest(Fixture):

    def test_clean_inline_empty(self):
        input_string = ''

        expect_string = '\n'

        clean_string = clean_inline(input_string)

        self.assertNotEqual(input_string, expect_string)
        self.assertEqual(clean_string, expect_string)

    def test_clean_inline(self):
        input_string = """
            This is the first line
              This line start with two spaces
            This is the last line
            """

        expect_string = (
            "This is the first line\n"
            "  This line start with two spaces\n"
            "This is the last line\n"
        )

        clean_string = clean_inline(input_string)

        self.assertNotEqual(input_string, expect_string)
        self.assertEqual(clean_string, expect_string)

    def test_clean_inline_no_last_return(self):
        input_string = (
            "This is the first line\n"
            "  This line start with two spaces\n"
            "This is the last line"
        )

        expect_string = (
            "This is the first line\n"
            "  This line start with two spaces\n"
            "This is the last line\n"
        )

        clean_string = clean_inline(input_string)

        self.assertEqual(clean_string, expect_string)

    def test_clean_inline_already_clean(self):
        input_string = (
            "This is the first line\n"
            "  This line start with two spaces\n"
            "This is the last line\n"
        )

        expect_string = (
            "This is the first line\n"
            "  This line start with two spaces\n"
            "This is the last line\n"
        )

        clean_string = clean_inline(input_string)

        self.assertEqual(clean_string, expect_string)


if __name__ == '__main__':
    test_main()
