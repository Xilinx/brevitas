import os

import nox
from packaging import version
from platform import system
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.join('.', '.github', 'workflows')))
from gen_github_actions import *


PYTORCH_CPU_VIRTUAL_PKG = '1.2.0'
CONDA_PYTHON_IDS = tuple([f'conda_python_{i}' for i in CONDA_PYTHON_VERSIONS])
PYTORCH_IDS = tuple([f'pytorch_{i}' for i in PYTORCH_VERSIONS])
JIT_IDS = tuple([f'{i}'.lower() for i in JIT_STATUSES])


@nox.session(venv_backend="conda", python=CONDA_PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("jit_status", JIT_STATUSES, ids=JIT_IDS)
def tests_brevitas_cpu(session, pytorch, jit_status):
    session.env['PYTORCH_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    if version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG):
        session.conda_install('-c', 'pytorch', f'pytorch=={pytorch}', 'cpuonly')
    else:
        session.conda_install('-c', 'pytorch', f'pytorch-cpu=={pytorch}')
    session.install('.[test]')
    session.run('pytest', 'test/brevitas', '-v')


@nox.session(venv_backend="conda", python=CONDA_PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_install_dev(session, pytorch):
    if version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG):
        session.conda_install('-c', 'pytorch', f'pytorch=={pytorch}', 'cpuonly')
    else:
        session.conda_install('-c', 'pytorch', f'pytorch-cpu=={pytorch}')
    session.install('-e', '.[test]')
    session.run('pytest', '-v', 'test/brevitas/test_import.py')


@nox.session(venv_backend="conda", python=CONDA_PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_examples_install_dev(session, pytorch):
    if version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG):
        session.conda_install('-c', 'pytorch', f'pytorch=={pytorch}', 'cpuonly')
    else:
        session.conda_install('-c', 'pytorch', f'pytorch-cpu=={pytorch}')
    session.install('-e', '.[test, tts, stt]')
    session.run('pytest', '-v', 'test/brevitas_examples/test_import.py')


@nox.session(python=False)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("python", CONDA_PYTHON_VERSIONS, ids=CONDA_PYTHON_IDS)
def dry_run_pytorch_only_deps(session, pytorch, python):
    if version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG):
        session.run('conda', 'create', '-n', 'dry_run', '--only-deps', '-d', '-c', 'pytorch', f'pytorch=={pytorch}',
                    'cpuonly', f'python={python}')
    else:
        session.run('conda', 'create', '-n', 'dry_run', '--only-deps', '-d', '-c', 'pytorch', f'pytorch-cpu=={pytorch}',
                    f'python={python}')


@nox.session(python=False)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("python", CONDA_PYTHON_VERSIONS, ids=CONDA_PYTHON_IDS)
def dry_run_pytorch_no_deps(session, pytorch, python):
    if version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG):
        session.run('conda', 'create', '-n', 'dry_run', '--no-deps', '-d', '-c', 'pytorch', f'pytorch=={pytorch}',
                    'cpuonly', f'python={python}')
    else:
        session.run('conda', 'create', '-n', 'dry_run', '--no-deps', '-d', '-c', 'pytorch', f'pytorch-cpu=={pytorch}',
                    f'python={python}')
