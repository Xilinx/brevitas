# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
from platform import system
import sys

import nox
from packaging import version
from packaging.version import parse

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.join('.', '.github', 'workflows')))
from gen_github_actions import EXAMPLES_LLM_PYTEST_PYTORCH_VERSIONS
from gen_github_actions import JIT_STATUSES
from gen_github_actions import PYTHON_VERSIONS
from gen_github_actions import PYTORCH_VERSIONS

IS_OSX = system() == 'Darwin'
PYTORCH_STABLE_WHEEL_SRC = 'https://download.pytorch.org/whl/cpu'
PYTORCH_STABLE_WHEEL_SRC_LEGACY = 'https://download.pytorch.org/whl/torch_stable.html'
PYTORCH_IDS = tuple([f'pytorch_{i}' for i in PYTORCH_VERSIONS])
EXAMPLES_LLM_PYTEST_PYTORCH_IDS = tuple([
    f'pytorch_{i}' for i in EXAMPLES_LLM_PYTEST_PYTORCH_VERSIONS])
JIT_IDS = tuple([f'{i}'.lower() for i in JIT_STATUSES])
LSTM_EXPORT_MIN_PYTORCH = '1.10.1'

TORCHVISION_VERSION_DICT = {
    '1.9.1': '0.10.1',
    '1.10.1': '0.11.2',
    '1.11.0': '0.12.0',
    '1.12.1': '0.13.1',
    '1.13.0': '0.14.0',
    '2.0.1': '0.15.2',
    '2.1.0': '0.16.0',
    '2.2.2': '0.17.2',
    '2.3.1': '0.18.1',
    '2.4.0': '0.19.0'}

PARSED_TORCHVISION_VERSION_DICT = {version.parse(k): v for k, v in TORCHVISION_VERSION_DICT.items()}


def install_pytorch(pytorch, session):
    if not IS_OSX:
        if parse(pytorch) < parse('2.4.0'):
            cmd = [f'torch=={pytorch}+cpu', '-f', PYTORCH_STABLE_WHEEL_SRC_LEGACY]
        else:
            cmd = [f'torch=={pytorch}', '--index-url', PYTORCH_STABLE_WHEEL_SRC]

    else:
        cmd = [f'torch=={pytorch}']
    session.install(*cmd)


def install_torchvision(pytorch, session):
    torchvision = PARSED_TORCHVISION_VERSION_DICT[version.parse(pytorch)]
    if not IS_OSX:
        if parse(pytorch) < parse('2.4.0'):
            cmd = [
                f'torch=={pytorch}+cpu',  # make sure correct pytorch version is kept
                f'torchvision=={torchvision}+cpu',
                '-f',
                PYTORCH_STABLE_WHEEL_SRC_LEGACY]
        else:
            cmd = [
                f'torch=={pytorch}',
                f'torchvision=={torchvision}',
                '--index-url',
                PYTORCH_STABLE_WHEEL_SRC]
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
    session.install('--upgrade', '.[test, export]')
    if jit_status == 'jit_enabled':
        session.run('pytest', '-k', 'not _full', 'tests/brevitas/nn/test_nn_quantizers.py', '-v')
        session.run(
            'pytest',
            'tests/brevitas',
            '-n',
            'logical',
            '-v',
            '--ignore',
            'tests/brevitas/graph',
            '--ignore',
            'tests/brevitas/nn/test_nn_quantizers.py')
        session.run('pytest', 'tests/brevitas/graph', '-n', 'logical', '-v')
    else:
        session.run(
            'pytest',
            '-n',
            'logical',
            '-k',
            '_full or wbiol',
            'tests/brevitas/nn/test_nn_quantizers.py',
            '-v')
        # run non graph tests
        session.run(
            'pytest',
            'tests/brevitas',
            '-n',
            'logical',
            '-v',
            '--ignore',
            'tests/brevitas/graph',
            '--ignore',
            'tests/brevitas/nn/test_nn_quantizers.py')
        # run graph tests separately
        session.run('pytest', 'tests/brevitas/graph', '-n', 'logical', '-v')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("jit_status", JIT_STATUSES, ids=JIT_IDS)
def tests_brevitas_examples_cpu(session, pytorch, jit_status):
    session.env['BREVITAS_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)  # For CV eval scripts
    session.install('--upgrade', '.[test, tts, stt, vision]')
    session.run(
        'pytest',
        '-n',
        'logical',
        '--ignore-glob',
        'tests/brevitas_examples/*llm*',
        'tests/brevitas_examples')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize(
    "pytorch", EXAMPLES_LLM_PYTEST_PYTORCH_VERSIONS, ids=EXAMPLES_LLM_PYTEST_PYTORCH_IDS)
@nox.parametrize("jit_status", JIT_STATUSES, ids=JIT_IDS)
def tests_brevitas_examples_llm(session, pytorch, jit_status):
    session.env['BREVITAS_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)  # Optimum seems to require torchvision
    session.install('-e', '.[test, llm, export]')
    session.install(
        'optimum-amd[brevitas] @ git+https://github.com/huggingface/optimum-amd.git@main')
    session.run('pytest', '-n', 'logical', '-k', 'llm', 'tests/brevitas_examples/test_llm.py')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_install_dev(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test]')
    session.env['BREVITAS_VERBOSE'] = '1'
    session.run('pytest', '-n', 'logical', '-v', 'tests/brevitas/test_brevitas_import.py')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_examples_install_dev(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test, tts, stt]')
    session.run('pytest', '-n', 'logical', '-v', 'tests/brevitas_examples/test_examples_import.py')


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
    session.run('pytest', '-n', 'logical', '-v', 'tests/brevitas_ort')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_notebook(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test, ort_integration, notebook]')
    if version.parse(pytorch) >= version.parse(LSTM_EXPORT_MIN_PYTORCH):
        session.run(
            'pytest', '-n', 'logical', '-v', '--nbmake', '--nbmake-kernel=python3', 'notebooks')
    else:
        session.run(
            'pytest',
            '-n',
            'logical',
            '-v',
            '--nbmake',
            '--nbmake-kernel=python3',
            'notebooks',
            '--ignore',
            'notebooks/quantized_recurrent.ipynb')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_end_to_end(session, pytorch):
    install_pytorch(pytorch, session)
    install_torchvision(pytorch, session)
    session.install('--upgrade', '-e', '.[test, ort_integration]')
    session.run('pytest', '-n', 'logical', '-v', 'tests/brevitas_end_to_end')
