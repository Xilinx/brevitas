import os

import yaml
import nox
from packaging import version


PYTORCH_CPU_VIRTUAL_PKG = '1.2.0'

GITHUB_WORKFLOWS_PATH = os.path.join('.', '.github', 'workflows')
REQUIREMENTS_PATH = os.path.join('.', 'requirements')

TESTS_YML = yaml.load(open(os.path.join(GITHUB_WORKFLOWS_PATH, 'tests.yml')), Loader=yaml.FullLoader)
TESTS_MATRIX = TESTS_YML['jobs']['build']['strategy']['matrix']

CONDA_PYTHON_VERSIONS = tuple(TESTS_MATRIX['conda_python_version'])
PYTORCH_VERSIONS = tuple(TESTS_MATRIX['pytorch_version'])
JIT_ENABLED = tuple(TESTS_MATRIX['jit_enabled'])

CONDA_PYTHON_IDS = tuple([f'conda_python_{i}' for i in CONDA_PYTHON_VERSIONS])
PYTORCH_IDS = tuple([f'pytorch_{i}' for i in PYTORCH_VERSIONS])
JIT_IDS = tuple([f'jit_{i}'.lower() for i in JIT_ENABLED])


@nox.session(venv_backend="conda", python=CONDA_PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("jit_enabled", JIT_ENABLED, ids=JIT_IDS)
def tests_cpu(session, pytorch, jit_enabled):
    session.env['PYTORCH_JIT'] = '{}'.format(int(jit_enabled))
    if version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG):
        session.conda_install('-c', 'pytorch', f'pytorch=={pytorch}', 'cpuonly')
    else:
        session.conda_install('-c', 'pytorch', f'pytorch-cpu=={pytorch}')
    session.install('.[test]')
    session.run('pytest', '-v')


@nox.session(python=False)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("python", CONDA_PYTHON_VERSIONS, ids=CONDA_PYTHON_IDS)
def dry_run_pytorch_only_deps(session, pytorch, python):
    if version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG):
        session.run('conda', 'create', '-n', 'dry_run', '--only-deps', '-d', '-c', 'pytorch', f'pytorch=={pytorch}', 'cpuonly', f'python={python}')
    else:
        session.run('conda', 'create', '-n', 'dry_run', '--only-deps', '-d', '-c', 'pytorch', f'pytorch-cpu=={pytorch}', f'python={python}')


@nox.session(python=False)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("python", CONDA_PYTHON_VERSIONS, ids=CONDA_PYTHON_IDS)
def dry_run_pytorch_no_deps(session, pytorch, python):
    if version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG):
        session.run('conda', 'create', '-n', 'dry_run', '--no-deps', '-d', '-c', 'pytorch', f'pytorch=={pytorch}', 'cpuonly', f'python={python}')
    else:
        session.run('conda', 'create', '-n', 'dry_run', '--no-deps', '-d', '-c', 'pytorch', f'pytorch-cpu=={pytorch}', f'python={python}')