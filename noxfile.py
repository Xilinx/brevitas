import os

import nox
from packaging import version
from platform import system
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.join('.', '.github', 'workflows')))
from gen_github_actions import *


PYTORCH_CPU_VIRTUAL_PKG = '1.2.0'
NOX_WIN_NUMPY_VERSION = '1.17.4'  # avoid errors from more recent Numpy called through Nox on Windows
CONDA_PYTHON_IDS = tuple([f'conda_python_{i}' for i in CONDA_PYTHON_VERSIONS])
PYTORCH_IDS = tuple([f'pytorch_{i}' for i in PYTORCH_VERSIONS])
JIT_IDS = tuple([f'{i}'.lower() for i in JIT_STATUSES])


def install_pytorch(pytorch, session):
    is_win = system() == 'Windows'
    is_cpu_virtual = version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG)
    if is_cpu_virtual and is_win:
        session.conda_install('-c', 'pytorch', f'pytorch=={pytorch}', f'numpy=={NOX_WIN_NUMPY_VERSION}', 'cpuonly')
    elif is_cpu_virtual and not is_win:
        session.conda_install('-c', 'pytorch', f'pytorch=={pytorch}', 'cpuonly')
    elif not is_cpu_virtual and is_win:
        session.conda_install('-c', 'pytorch', f'pytorch-cpu=={pytorch}', f'numpy=={NOX_WIN_NUMPY_VERSION}')
    else:
        session.conda_install('-c', 'pytorch', f'pytorch-cpu=={pytorch}')


def dry_run_install_pytorch_deps(python, pytorch, session, deps_only):
    deps = '--only-deps' if deps_only else '--no-deps'
    is_win = system() == 'Windows'
    is_cpu_virtual = version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG)
    if is_cpu_virtual and is_win:
        session.run('conda', 'create', '-n', 'dry_run', deps, '-d', '-c', 'pytorch', f'pytorch=={pytorch}',
                    f'numpy=={NOX_WIN_NUMPY_VERSION}', 'cpuonly', f'python={python}')
    elif is_cpu_virtual and not is_win:
        session.run('conda', 'create', '-n', 'dry_run', deps, '-d', '-c', 'pytorch', f'pytorch=={pytorch}',
                    'cpuonly', f'python={python}')
    elif not is_cpu_virtual and not is_win:
        session.run('conda', 'create', '-n', 'dry_run', deps, '-d', '-c', 'pytorch', f'pytorch-cpu=={pytorch}',
                    'cpuonly', f'python={python}')
    else:
        session.run('conda', 'create', '-n', 'dry_run', deps, '-d', '-c', 'pytorch', f'pytorch-cpu=={pytorch}',
                    f'numpy=={NOX_WIN_NUMPY_VERSION}', f'python={python}')


@nox.session(venv_backend="conda", python=CONDA_PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("jit_status", JIT_STATUSES, ids=JIT_IDS)
def tests_brevitas_cpu(session, pytorch, jit_status):
    session.env['PYTORCH_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    install_pytorch(pytorch, session)
    session.install('.[test]')
    session.run('pytest', 'test/brevitas', '-v')


@nox.session(venv_backend="conda", python=CONDA_PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_install_dev(session, pytorch):
    install_pytorch(pytorch, session)
    session.install('-e', '.[test]')
    session.run('pytest', '-v', 'test/brevitas/test_import.py')


@nox.session(venv_backend="conda", python=CONDA_PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_examples_install_dev(session, pytorch):
    install_pytorch(pytorch, session)
    session.conda_install('scipy')  # For Hadamard example
    session.install('-e', '.[test, tts, stt]')
    session.run('pytest', '-v', 'test/brevitas_examples/test_import.py')


@nox.session(python=False)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("python", CONDA_PYTHON_VERSIONS, ids=CONDA_PYTHON_IDS)
def dry_run_pytorch_only_deps(session, pytorch, python):
    dry_run_install_pytorch_deps(python, pytorch, session, deps_only=True)


@nox.session(python=False)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("python", CONDA_PYTHON_VERSIONS, ids=CONDA_PYTHON_IDS)
def dry_run_pytorch_no_deps(session, pytorch, python):
    dry_run_install_pytorch_deps(python, pytorch, session, deps_only=False)
