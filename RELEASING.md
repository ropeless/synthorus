Release Processes
=================

Version Identification
----------------------

Synthorus version identification conforms to [PEP 440](https://peps.python.org/pep-0440/)
with MAJOR.MINOR.PATCH [semantic versioning](https://semver.org/). 

Here are some example version identifiers that conform:
* `4.5.6` a stable release,
* `4.5.6a78` a prerelease of version `4.5.6`.

Versions will be tagged in the repository using the version identifier as the label, prepended with "v".

Here are the repository tags corresponding to the version identifiers above:
* `v4.5.6`,
* `v4.5.6a78`.

Prepare for a Release
---------------------

1. Check out the "main" branch.
2. Merge any new features and fixes for the release.
3. Run `tests/all_tests.py` for unit testing, confirming all unit tests pass.
4. Run `synthorus_demos/all_demos.py` for smoke testing, confirming no errors are reported.
5. Edit `pyproject.toml` to update the project version number (must do before building documentation).
6. Run `python build_docs.py` and confirm the documentation builds okay.
7. View the documentation build to confirm it: [`docs/_build/html/index.html`](docs/_build/html/index.html).

Only proceed if the "main" branch is ready for release.

Perform a Release - GitHub Action
---------------------------------

1. Commit, tag and push the "main" branch.
   This will automatically release the documentation.
2. Go to the project [GitHub Actions, Upload Python Package](https://github.com/ropeless/synthorus/actions/workflows/python-publish.yml).
3. Select "Run workflow".

Perform a Release - Manual Action
---------------------------------

1. Commit, tag and push the "main" branch.
   This will automatically release the documentation.
2. Ensure you are on an up-to-date checkout of the "main" branch.
3. Delete any existing project `dist` and `build` directories.
4. Build a pure Python wheel distribution using: `python -m build --wheel`.
5. Upload the package to PyPI using: `python -m twine upload dist/*`.

Post-release Checks
-------------------

1. Check the online version of the documentation:  https://synthorus.readthedocs.io/.
2. Check the PyPI release history: https://pypi.org/project/synthorus/#history.
3. Open a Synthorus client test project. Update the dependencies (e.g., `poetry update`).
   Ensure Synthorus upgraded and that the test project works as expected.

Actions for a Broken Release
----------------------------

If the post-release checks fail:

1. Delete the broken PyPI release: https://pypi.org/project/synthorus/#history.
2. Create a "fix/..." branch for a fix.
3. When a fix is found, perform the release process above.
