#!/usr/bin/env python

"""
Report on coverage of unit tests (in all_tests.py).
"""
import webbrowser
from pathlib import Path

from coverage.cmdline import main as coverage_main


def main():
    own_dir = Path(__file__).parent
    all_tests = own_dir / 'all_tests.py'
    html_dir = own_dir / 'all_coverage_html'
    html_index = html_dir / 'index.html'

    print(f'Running {all_tests}')
    coverage_main([
        'run',
        '--rcfile=all_coveragerc.txt',
        all_tests.as_posix()
    ])

    print(f'Creating report')
    coverage_main([
        'html',
        '--rcfile=all_coveragerc.txt',
        '-d', html_dir.as_posix()
    ])
    webbrowser.open(html_index.as_uri())


if __name__ == '__main__':
    main()
