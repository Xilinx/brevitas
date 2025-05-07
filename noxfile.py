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
from gen_github_actions import TORCHVISION_VERSION_DICT

IS_OSX = system() == 'Darwin'
PYTORCH_STABLE_WHEEL_SRC = 'https://download.pytorch.org/whl/cpu'
PYTORCH_STABLE_WHEEL_SRC_LEGACY = 'https://download.pytorch.org/whl/torch_stable.html'
PIP_URL = 'https://pypi.org/simple'
PYTORCH_IDS = tuple([f'pytorch_{i}' for i in PYTORCH_VERSIONS])
EXAMPLES_LLM_PYTEST_PYTORCH_IDS = tuple([
    f'pytorch_{i}' for i in EXAMPLES_LLM_PYTEST_PYTORCH_VERSIONS])
JIT_IDS = tuple([f'{i}'.lower() for i in JIT_STATUSES])

PARSED_TORCHVISION_VERSION_DICT = {version.parse(k): v for k, v in TORCHVISION_VERSION_DICT.items()}


def install_pytorch_cmd(pytorch):
    if not IS_OSX:
        if parse(pytorch) < parse('2.4.0'):
            cmd = [f'torch=={pytorch}+cpu', '-f', PYTORCH_STABLE_WHEEL_SRC_LEGACY]
        else:
            cmd = [
                f'torch=={pytorch}',
                '--index-url',
                PYTORCH_STABLE_WHEEL_SRC,
                '--extra-index-url',
                PIP_URL]

    else:
        cmd = [f'torch=={pytorch}']
    return cmd


def install_torchvision_cmd(pytorch):
    torchvision = PARSED_TORCHVISION_VERSION_DICT[version.parse(pytorch)]
    if not IS_OSX:
        if parse(pytorch) < parse('2.4.0'):
            cmd = [f'torchvision=={torchvision}+cpu', '-f', PYTORCH_STABLE_WHEEL_SRC_LEGACY]
        else:
            cmd = [
                f'torchvision=={torchvision}',
                '--index-url',
                PYTORCH_STABLE_WHEEL_SRC,
                '--extra-index-url',
                PIP_URL]
    else:
        cmd = [f'torchvision=={torchvision}']
    return cmd


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize('pytorch', PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize('jit_status', JIT_STATUSES, ids=JIT_IDS)
def tests_brevitas_cpu(session, pytorch, jit_status):
    session.env['BREVITAS_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    cmd = []
    cmd += install_pytorch_cmd(pytorch)
    cmd += install_torchvision_cmd(pytorch)
    session.install('-e', '.[test, export]', *cmd)
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
        session.env['BREVITAS_NATIVE_STE_BACKEND'] = '1'
        session.run('pytest', '-n', 'logical', 'tests/brevitas/function/test_ops_ste.py', '-v')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
@nox.parametrize("jit_status", JIT_STATUSES, ids=JIT_IDS)
def tests_brevitas_examples_cpu(session, pytorch, jit_status):
    session.env['BREVITAS_JIT'] = '{}'.format(int(jit_status == 'jit_enabled'))
    cmd = []
    cmd += install_pytorch_cmd(pytorch)
    cmd += install_torchvision_cmd(pytorch)  # For CV eval scripts
    session.install('-e', '.[test, tts, stt, vision]', *cmd)
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
    cmd = []
    cmd += install_pytorch_cmd(pytorch)
    cmd += install_torchvision_cmd(pytorch)  # Optimum seems to require torchvision
    session.install('-e', '.[test, llm, export]', *cmd)
    session.run('pytest', '-n', 'logical', '-k', 'llm', 'tests/brevitas_examples/test_llm.py')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_install_dev(session, pytorch):
    cmd = []
    cmd += install_pytorch_cmd(pytorch)
    cmd += install_torchvision_cmd(pytorch)
    session.install('-e', '.[test]', *cmd)
    session.env['BREVITAS_VERBOSE'] = '1'
    session.run('pytest', '-n', 'logical', '-v', 'tests/brevitas/test_brevitas_import.py')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_examples_install_dev(session, pytorch):
    cmd = []
    cmd += install_pytorch_cmd(pytorch)
    cmd += install_torchvision_cmd(pytorch)
    session.install('-e', '.[test, tts, stt]', *cmd)
    session.run('pytest', '-n', 'logical', '-v', 'tests/brevitas_examples/test_examples_import.py')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_finn_integration(session, pytorch):
    cmd = []
    cmd += install_pytorch_cmd(pytorch)
    cmd += install_torchvision_cmd(pytorch)
    session.install('-e', '.[test, stt, finn_integration]', *cmd)
    env = {'FINN_INST_NAME': 'finn'}
    session.run('pytest', '-v', 'tests/brevitas_finn', env=env)


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_ort_integration(session, pytorch):
    cmd = []
    cmd += install_pytorch_cmd(pytorch)
    cmd += install_torchvision_cmd(pytorch)
    session.install('-e', '.[test, ort_integration]', *cmd)
    session.run('pytest', '-n', 'logical', '-v', 'tests/brevitas_ort')


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("pytorch", PYTORCH_VERSIONS, ids=PYTORCH_IDS)
def tests_brevitas_notebook(session, pytorch):
    cmd = []
    cmd += install_pytorch_cmd(pytorch)
    cmd += install_torchvision_cmd(pytorch)
    session.install('-e', '.[test, ort_integration, notebook]', *cmd)
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
    cmd = []
    cmd += install_pytorch_cmd(pytorch)
    cmd += install_torchvision_cmd(pytorch)
    session.install('-e', '.[test, ort_integration]', *cmd)
    session.run('pytest', '-n', 'logical', '-v', 'tests/brevitas_end_to_end')
