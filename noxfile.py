import nox
from packaging import version
from platform import system
from gen_github_actions import *

PYTORCH_CPU_VIRTUAL_PKG = '1.2.0'
GITHUB_WORKFLOWS_PATH = os.path.join('.', '.github', 'workflows')
TESTS_YAML_TEMPLATE_PATH = os.path.join(GITHUB_WORKFLOWS_PATH, 'tests.yaml.template')

CONDA_PYTHON_IDS = tuple([f'conda_python_{i}' for i in CONDA_PYTHON_VERSIONS])
PYTORCH_IDS = tuple([f'pytorch_{i}' for i in PYTORCH_VERSIONS])
JIT_IDS = tuple([f'{i}'.lower() for i in JIT_STATUSES])


@nox.session(venv_backend="conda", python=CONDA_PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("jit_status", JIT_STATUSES, ids=JIT_IDS)
def tests_cpu(session, pytorch, jit_status):
    session.env['PYTORCH_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    if version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG):
        session.conda_install('-c', 'pytorch', f'pytorch=={pytorch}', 'cpuonly')
    else:
        session.conda_install('-c', 'pytorch', f'pytorch-cpu=={pytorch}')
    session.install('.[test]')
    session.run('pytest', '-v')


@nox.session(venv_backend="conda", python=CONDA_PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_install_develop(session, pytorch):
    if version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG):
        session.conda_install('-c', 'pytorch', f'pytorch=={pytorch}', 'cpuonly')
    else:
        session.conda_install('-c', 'pytorch', f'pytorch-cpu=={pytorch}')
    session.install('-e', '.')
    if system() == 'Windows':
        env = session.env
        nox.command.run(['python', '-c', 'import brevitas'], env=env, path=os.path.dirname(session.bin))
        nox.command.run(['python', '-c', 'import brevitas.function.ops_ste'], env=env,
                        path=os.path.dirname(session.bin))
    else:
        session.run('python', '-c', 'import brevitas')
        session.run('python', '-c', 'import brevitas.function.ops_ste')


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
