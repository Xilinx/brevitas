# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import os
from platform import system
import sys

import nox
from packaging import version

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.join('.', '.github', 'workflows')))
from gen_github_actions import JIT_STATUSES
from gen_github_actions import PYTHON_VERSIONS
from gen_github_actions import PYTORCH_VERSIONS

IS_OSX = system() == 'Darwin'
PYTORCH_STABLE_WHEEL_SRC = 'https://download.pytorch.org/whl/torch_stable.html'
PYTORCH_IDS = tuple([f'pytorch_{i}' for i in PYTORCH_VERSIONS])
JIT_IDS = tuple([f'{i}'.lower() for i in JIT_STATUSES])
LSTM_EXPORT_MIN_PYTORCH = '1.10.1'

TORCHVISION_VERSION_DICT = {
    '1.5.1': '0.6.1',
    '1.6.0': '0.7.0',
    '1.7.0': '0.8.1',
    '1.7.1': '0.8.2',
    '1.8.0': '0.9.0',
    '1.8.1': '0.9.1',
    '1.9.1': '0.10.1',
    '1.10.1':'0.11.2',
    '1.11.0': '0.12.0',
    '1.12.1': '0.13.1',
    '1.13.0': '0.14.0'
}

PARSED_TORCHVISION_VERSION_DICT = {
    version.parse(k): v for k, v in TORCHVISION_VERSION_DICT.items()}


def install_pytorch(pytorch, session):
    if not IS_OSX:
        cmd = [f'torch=={pytorch}+cpu', '-f', PYTORCH_STABLE_WHEEL_SRC]
    else:
        cmd = [f'torch=={pytorch}']
    session.install(*cmd)


def install_torchvision(pytorch, session):
    torchvision = PARSED_TORCHVISION_VERSION_DICT[version.parse(pytorch)]
    if not IS_OSX:
        cmd = [
            f'torch=={pytorch}+cpu',  # make sure correct pytorch version is kept
            f'torchvision=={torchvision}+cpu',
            '-f', PYTORCH_STABLE_WHEEL_SRC]
    else:
        cmd = [f'torch=={pytorch}', f'torchvision=={torchvision}']
    session.install(*cmd)


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize('pytorch', PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize('jit_status', JIT_STATUSES, ids=JIT_IDS)
def tests_brevitas_cpu(session, pytorch, jit_status):
    session.env['BREVITAS_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install( '--upgrade', '.[test, export]')
    # run non graph tests
    session.run('pytest', 'tests/brevitas', '-n', 'auto', '-v', '--ignore', 'tests/brevitas/graph')
    # run graph tests separately
    session.run('pytest', 'tests/brevitas/graph','-n', 'auto', '-v')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("jit_status", JIT_STATUSES, ids=JIT_IDS)
def tests_brevitas_examples_cpu(session, pytorch, jit_status):
    session.env['PYTORCH_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)  # For CV eval scripts
    session.install('--upgrade', '.[test, tts, stt, vision]')
    session.run('pytest', '-n', 'auto', 'tests/brevitas_examples')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_install_dev(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test]')
    session.env['BREVITAS_VERBOSE'] = '1'
    session.run('pytest', '-n', 'auto', '-v', 'tests/brevitas/test_brevitas_import.py')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_examples_install_dev(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test, tts, stt]')
    session.run('pytest', '-n', 'auto', '-v', 'tests/brevitas_examples/test_examples_import.py')


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
    session.run('pytest', '-n', 'auto', '-v', 'tests/brevitas_ort')

@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_notebook(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test, ort_integration, notebook]')
    if version.parse(pytorch) >= version.parse(LSTM_EXPORT_MIN_PYTORCH):
        session.run('pytest', '-n', 'auto', '-v','--nbmake', '--nbmake-kernel=python3', 'notebooks')
    else:
        session.run('pytest', '-n', 'auto', '-v','--nbmake', '--nbmake-kernel=python3', 'notebooks', '--ignore', 'notebooks/quantized_recurrent.ipynb')

@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_end_to_end(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test, ort_integration]')
    session.run('pytest', '-v', 'tests/brevitas_end_to_end')
