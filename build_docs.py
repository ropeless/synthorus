import asyncio
import shutil
import subprocess
import sys
import warnings
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import nbformat
import toml
from nbconvert.preprocessors import ExecutePreprocessor

from synthorus.utils.config_help import config
from synthorus.utils.time_extras import timestamp
from synthorus_demos.utils import output_directory

FORCE_REBUILD: bool = config.get('BUILD_DOCS_FORCE_REBUILD', True)
BUILD_API_DOCS: bool = config.get('BUILD_DOCS_BUILD_API_DOCS', True)
INSTANTIATE_TEMPLATES: bool = config.get('BUILD_DOCS_INSTANTIATE_TEMPLATES', True)
EXECUTE_NOTEBOOKS: bool = config.get('BUILD_DOCS_EXECUTE_NOTEBOOKS', True)
OPEN_DOCUMENT_HTML: bool = config.get('BUILD_DOCS_OPEN_DOCUMENT_HTML', True)


def main() -> None:
    """
    Build the documentation.

    This script will explicitly update the document files by instantiating
    templates and executing notebooks in-place.

    The updated document files should be checked and committed to the repository as
    they will be used directly by Read The Docs when pushed to GitHub.
    """
    project_dir: Path = Path(__file__).parent
    src_dir: Path = project_dir / 'src'
    ck_package_dir: Path = src_dir / 'synthorus'
    docs_dir: Path = project_dir / 'docs'
    api_docs_dir: Path = docs_dir / 'api'
    doc_out_dir: Path = docs_dir / 'output_directory'
    build_dir: Path = docs_dir / '_build'
    html_index: Path = build_dir / 'html' / 'index.html'

    if FORCE_REBUILD:
        if build_dir.exists():
            shutil.rmtree(build_dir)

    if BUILD_API_DOCS:
        build_api_docs(ck_package_dir, api_docs_dir)

    if INSTANTIATE_TEMPLATES:
        instantiate_templates(project_dir, docs_dir)

    if EXECUTE_NOTEBOOKS:
        config_out_dir: Path = get_config_out_dir()
        start_timestamp: float = datetime.now().timestamp()
        for notebook_path in docs_dir.glob('*.ipynb'):
            execute_notebook(notebook_path)
        copy_output_directory(config_out_dir, doc_out_dir, start_timestamp)

    run_jupyter_book()

    uri: str = html_index.as_uri()
    print(f'Documentation available at: {uri}')
    if OPEN_DOCUMENT_HTML:
        webbrowser.open(uri)


def get_config_out_dir() -> Path:
    """
    The caller must have 'OUT_DIR' defined in their local config so that we can
    collect notebook output files for the document build.
    """
    config_out_dir = config.get(output_directory.DEMO_OUT_CONFIG)
    if config_out_dir is None:
        raise RuntimeError(
            f'Need to have a defined output directory using config: {output_directory.DEMO_OUT_CONFIG}'
        )
    return Path(config_out_dir)


def copy_output_directory(
        config_out_dir: Path,
        doc_out_dir: Path,
        start_timestamp: float,
) -> None:
    """
    Copy the output directory results to the document area, in a clean directory.

    Args:
        config_out_dir: The output directory to copy from.
        doc_out_dir: The output directory to copy to.
        start_timestamp: Only files modified after this timestamp are copied.
    """
    if doc_out_dir.exists():
        shutil.rmtree(doc_out_dir)

    # Deep copy text files.
    for source_path in config_out_dir.rglob('*'):
        if source_path.suffix.lower() not in {'.html', '.txt', '.json', '.csv'}:
            continue
        if source_path.stat().st_mtime <= start_timestamp:
            continue
        relative_path = source_path.relative_to(config_out_dir)
        target_path = doc_out_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def build_api_docs(api_package_dir: Path, api_docs_dir: Path) -> None:
    """
    Build the API documentation from docstring comments.

    Args:
        api_package_dir: where to find the API packages.
        api_docs_dir: where to put the API rst files.
c    """
    print('Generating API docs')
    api_docs_dir.mkdir(exist_ok=True)
    shutil.rmtree(api_docs_dir)
    cmd: List[str] = ['sphinx-apidoc', '-o', api_docs_dir.as_posix(), api_package_dir.as_posix()]
    subprocess.run(cmd, capture_output=False, check=True)


def run_jupyter_book() -> None:
    """
    Run the Jupyter Book command to build the documentation.
    """
    print('Running jupyter-book')
    cmd: List[str] = ['jupyter-book', 'build', '-qq', 'docs']
    subprocess.run(cmd, capture_output=False, check=True, stdout=subprocess.DEVNULL)
    print('Finished generating HTML')


def load_pyproject(file_path: Path) -> Dict[str, Any]:
    """
    Loads the pyproject.toml file as a nested dictionary.

    Args:
        file_path: Path to the pyproject.toml file.

    Returns:
        the toml dictionary
    """
    with open(file_path, 'r') as f:
        return toml.load(f)


def instantiate_templates(project_dir: Path, docs_dir: Path) -> None:
    """
    Instantiate Markdown documents from found templates.

    Args:
        project_dir: where to find the 'pyproject.toml' file.
        docs_dir: where to find the document files.
    """
    print('Instantiating templates')
    pyproject = load_pyproject(project_dir / 'pyproject.toml')

    # These values will be inserted into the template using `str.format`.
    fields: Dict[str, str] = {
        'version': pyproject['project']['version'],
        'version_note': pyproject.get('doc_extra', {}).get('version_note', ''),
        'date': timestamp(),
    }

    for template_file in docs_dir.glob('*_template.md'):
        dest_name = template_file.name[:-len('_template.md')] + '.md'
        dest_file: Path = docs_dir / dest_name

        with open(template_file, 'r') as f:
            lines = [
                line.format(**fields)
                for line in f.readlines()
            ]

        with open(dest_file, 'w') as f:
            f.writelines(lines)


def execute_notebook(notebook_path: Path) -> None:
    """
    Executes a Jupyter notebook and save the output inplace.

    Args:
        notebook_path: Path to the Jupyter notebook file.
    """
    print(f'Executing Jupyter notebook {notebook_path.name}')
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    asyncio.run(_execute_notebook_inplace(notebook_path), loop_factory=factory)


async def _execute_notebook_inplace(notebook_path: Path) -> None:
    """
    Executes a Jupyter notebook and save the output inplace.

    Args:
        notebook_path: Path to the Jupyter notebook file.
    """
    # 1. Read the existing notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 2. Configure the execution engine
    # timeout=600 gives cells up to 10 minutes to run; kernel_name targets the current environment
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.extra_arguments = ['Application.log_level=CRITICAL', '--IPKernelApp.log_level=CRITICAL']

    # 3. Execute the notebook cells, ignoring some warnings
    # noinspection PyTypeChecker
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=(
                '(.*Socket operation on non-socket.*)'
                '|(.*Proactor event loop does not implement.*)'
                '|(.*Connection reset by peer.*)'
            )
        )
        ep.preprocess(nb, {'metadata': {'path': './'}})

    # 4. Write the executed notebook back to the original path
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)


if __name__ == '__main__':
    main()
