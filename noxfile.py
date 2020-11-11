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
FINN_NUMPY_VERSION = '1.18.1' # FINN has a strict req on 1.18.0+
CONDA_PYTHON_IDS = tuple([f'conda_python_{i}' for i in CONDA_PYTHON_VERSIONS])
PYTORCH_IDS = tuple([f'pytorch_{i}' for i in PYTORCH_VERSIONS])
JIT_IDS = tuple([f'{i}'.lower() for i in JIT_STATUSES])


# This combination has proven to be problematic and without access to
# a physical system it's tricky to debug
def is_torchvision_broken(python_version, pytorch_version):
    return (system() == 'Darwin' and
            version.parse(python_version) == version.parse('3.6') and
            version.parse(pytorch_version) == version.parse('1.5'))


def install_pytorch(pytorch, session, numpy=None):
    is_win = system() == 'Windows'
    is_cpu_virtual = version.parse(pytorch) >= version.parse(PYTORCH_CPU_VIRTUAL_PKG)
    if is_cpu_virtual:
        cmd = ['-c', 'pytorch', f'pytorch=={pytorch}', 'cpuonly']
    else:
        cmd = ['-c', 'pytorch', f'pytorch-cpu=={pytorch}']
    if numpy is not None:
        cmd += [f'numpy=={numpy}']
    elif numpy is None and is_win:
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
    session.env['BREVITAS_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install( '--upgrade', '.[test]')
    # run non graph tests
    session.run('pytest', 'test/brevitas', '-v', '--ignore', 'test/brevitas/graph')
    # run graph tests
    if not is_torchvision_broken(session.python, pytorch):
        session.run('pytest', 'test/brevitas/graph/test_generator.py', '-v')
        session.run('pytest', 'test/brevitas/graph/test_tracer.py', '-v')
        session.run('pytest', 'test/brevitas/graph/test_rewriter.py', '-v')

@nox.session(venv_backend="conda", python=CONDA_PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("jit_status", JIT_STATUSES, ids=JIT_IDS)
def tests_brevitas_examples_cpu(session, pytorch, jit_status):
    session.env['PYTORCH_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)  # For CV eval scripts
    session.install('--upgrade', '.[test, tts, stt, vision]')
    session.run('pytest', 'test/brevitas_examples')


@nox.session(venv_backend="conda", python=CONDA_PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_install_dev(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test]')
    session.env['BREVITAS_VERBOSE'] = '1'
    session.run('pytest', '-v', 'test/brevitas/test_brevitas_import.py')


@nox.session(venv_backend="conda", python=CONDA_PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_examples_install_dev(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test, tts, stt]')
    session.run('pytest', '-v', 'test/brevitas_examples/test_examples_import.py')


@nox.session(venv_backend="conda", python=CONDA_PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_finn_integration(session, pytorch):
    install_pytorch(pytorch, session, FINN_NUMPY_VERSION)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test, stt, finn_integration]')
    env = {'FINN_INST_NAME': 'finn'}
    session.run('pytest', '-v', 'test/brevitas_finn_integration', env=env)


@nox.session(venv_backend="conda", python=CONDA_PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_pyxir_integration(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test, pyxir_integration]')
    session.run('pytest', '-v', 'test/brevitas_pyxir_integration')


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
