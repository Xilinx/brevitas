import os

import nox
from packaging import version
from platform import system
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.join('.', '.github', 'workflows')))
from gen_github_actions import *

PYTORCH_CPU_VIRTUAL_PKG = '1.2.0'
PYTORCH_PILLOW_FIXED = '1.4.0'
NOX_WIN_NUMPY_VERSION = '1.17.4'  # avoid errors from more recent Numpy called through Nox on Windows
CONDA_PYTHON_IDS = tuple([f'conda_python_{i}' for i in CONDA_PYTHON_VERSIONS])
PYTORCH_IDS = tuple([f'pytorch_{i}' for i in PYTORCH_VERSIONS])
JIT_IDS = tuple([f'{i}'.lower() for i in JIT_STATUSES])

def install_pytorch(pytorch, session):
    is_win = system() == 'Windows'
    is_cpu_virtual = version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG)
    if is_cpu_virtual:
        cmd = ['-c', 'pytorch', f'pytorch=={pytorch}', 'cpuonly']
    else:
        cmd = ['-c', 'pytorch', f'pytorch-cpu=={pytorch}']
    if is_win:
        cmd += [f'numpy=={NOX_WIN_NUMPY_VERSION}']
    session.conda_install(*cmd)


def install_torchvision(pytorch, session):
    is_cpu_virtual = version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG)
    fixed_pillow = version.parse(pytorch) >= version.parse(PYTORCH_PILLOW_FIXED)
    if is_cpu_virtual:
        cmd = ['-c', 'pytorch', 'torchvision', 'cpuonly']
    else:
        cmd = ['-c', 'pytorch', 'torchvision-cpu']
    if not fixed_pillow:
        cmd += ['Pillow==6.2.0']
    session.conda_install(*cmd)


def dry_run_install_pytorch_deps(python, pytorch, session, deps_only):
    deps = '--only-deps' if deps_only else '--no-deps'
    is_win = system() == 'Windows'
    is_cpu_virtual = version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG)
    if is_cpu_virtual:
        cmd = ['conda', 'create', '-n', 'dry_run', deps, '-d', '-c', 'pytorch', f'pytorch=={pytorch}',
               f'numpy=={NOX_WIN_NUMPY_VERSION}', 'cpuonly', f'python={python}']
    else:
        cmd = ['conda', 'create', '-n', 'dry_run', deps, '-d', '-c', 'pytorch', f'pytorch-cpu=={pytorch}',
               'cpuonly', f'python={python}']
    if is_win:
        cmd += [f'numpy=={NOX_WIN_NUMPY_VERSION}']
    session.run(*cmd)


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
@nox.parametrize("jit_status", JIT_STATUSES, ids=JIT_IDS)
def tests_brevitas_examples_cpu(session, pytorch, jit_status):
    session.env['PYTORCH_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)  # For CV eval scripts
    session.conda_install('scipy')  # For Hadamard example
    session.install('.[test, tts, stt, vision]')
    session.run('pytest', 'test/brevitas_examples', '-v')


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
