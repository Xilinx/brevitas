import os

import nox
from packaging import version
from platform import system
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.join('.', '.github', 'workflows')))
from gen_github_actions import PYTORCH_VERSIONS, PYTHON_VERSIONS, JIT_STATUSES

IS_OSX = system() == 'Darwin'
PYTORCH_STABLE_WHEEL_SRC = 'https://download.pytorch.org/whl/torch_stable.html'
PYTORCH_110_CPU_WHEEL_SRC = 'https://download.pytorch.org/whl/cpu/torch_stable.html'
PYTORCH_PILLOW_FIXED = '1.4.0'
OLDER_PILLOW_VERSION = '6.2.0'
PYTORCH_IDS = tuple([f'pytorch_{i}' for i in PYTORCH_VERSIONS])
JIT_IDS = tuple([f'{i}'.lower() for i in JIT_STATUSES])

TORCHVISION_VERSION_DICT = {
    '1.1.0': '0.3.0',
    '1.2.0': '0.4.0',
    '1.3.0': '0.4.1',
    '1.3.1': '0.4.2',
    '1.4.0': '0.5.0',
    '1.5.0': '0.6.0',
    '1.5.1': '0.6.1',
    '1.6.0': '0.7.0',
    '1.7.0': '0.8.0',
    '1.7.1': '0.8.1'
}

PARSED_TORCHVISION_VERSION_DICT = {
    version.parse(k): v for k, v in TORCHVISION_VERSION_DICT.items()}


# This combination has proven to be problematic and without access to
# a physical system it's tricky to debug
def is_torchvision_broken(python_version, pytorch_version):
    return (IS_OSX and
            version.parse(python_version) == version.parse('3.6') and
            version.parse(pytorch_version) == version.parse('1.5'))


def install_pytorch(pytorch, session):
    if not IS_OSX and version.parse(pytorch) > version.parse('1.1.0'):
        cmd = [f'torch=={pytorch}+cpu', '-f', PYTORCH_STABLE_WHEEL_SRC]
    elif not IS_OSX and version.parse(pytorch) <= version.parse('1.1.0'):
        cmd = [f'torch=={pytorch}', '-f', PYTORCH_110_CPU_WHEEL_SRC]
    else:
        cmd = [f'torch=={pytorch}']
    session.install(*cmd)


def install_torchvision(pytorch, session):
    torchvision = PARSED_TORCHVISION_VERSION_DICT[version.parse(pytorch)]
    if not IS_OSX and version.parse(pytorch) > version.parse('1.1.0'):
        cmd = [f'torchvision=={torchvision}+cpu', '-f', PYTORCH_STABLE_WHEEL_SRC]
    elif not IS_OSX and version.parse(pytorch) <= version.parse('1.1.0'):
        cmd = [f'torchvision=={torchvision}', '-f', PYTORCH_110_CPU_WHEEL_SRC]
    else:
        cmd = [f'torchvision=={torchvision}']
    if version.parse(pytorch) < version.parse(PYTORCH_PILLOW_FIXED):
        cmd += [f'Pillow=={OLDER_PILLOW_VERSION}']
    session.install(*cmd)


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize('pytorch', PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize('jit_status', JIT_STATUSES, ids=JIT_IDS)
def tests_brevitas_cpu(session, pytorch, jit_status):
    session.env['PYTORCH_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    session.env['BREVITAS_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install( '--upgrade', '.[test]')
    # run non graph tests
    session.run('pytest', 'tests/brevitas', '-v', '--ignore', 'tests/brevitas/graph')
    # run graph tests
    if not is_torchvision_broken(session.python, pytorch):
        session.run('pytest', 'tests/brevitas/graph', '-v')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("jit_status", JIT_STATUSES, ids=JIT_IDS)
def tests_brevitas_examples_cpu(session, pytorch, jit_status):
    session.env['PYTORCH_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)  # For CV eval scripts
    session.install('--upgrade', '.[test, tts, stt, vision]')
    session.run('pytest', 'tests/brevitas_examples')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_install_dev(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test]')
    session.env['BREVITAS_VERBOSE'] = '1'
    session.run('pytest', '-v', 'tests/brevitas/test_brevitas_import.py')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_examples_install_dev(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test, tts, stt]')
    session.run('pytest', '-v', 'tests/brevitas_examples/test_examples_import.py')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_finn_integration(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test, stt, finn_integration]')
    env = {'FINN_INST_NAME': 'finn'}
    session.run('pytest', '-v', 'tests/brevitas_finn', env=env)


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_ort_integration(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test, ort_integration]')
    session.run('pytest', '-v', 'tests/brevitas_ort')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize('pytorch', PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_pyxir_integration(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test, pyxir_integration]')
    session.run('pytest', '-v', 'tests/brevitas_pyxir')
