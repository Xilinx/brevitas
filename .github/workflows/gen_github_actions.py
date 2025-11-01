from collections import OrderedDict as od

from utils import Action
from utils import combine_od_list
from utils import generate_exclusion_list

TORCHVISION_VERSION_DICT = {
    '1.9.1': '0.10.1',
    '1.10.1': '0.11.2',
    '1.11.0': '0.12.0',
    '1.12.1': '0.13.1',
    '1.13.1': '0.14.1',
    '2.0.1': '0.15.2',
    '2.1.1': '0.16.1',
    '2.2.2': '0.17.2',
    '2.3.1': '0.18.1',
    '2.4.1': '0.19.1',
    '2.5.1': '0.20.1',
    '2.6.0': '0.21.0',
    '2.7.1': '0.22.1',
    '2.8.0': '0.23.0',
    '2.9.0': '0.24.0'}

BASE_YML_TEMPLATE = 'base.yml.template'
BASE_YML_REDUCED_TEMPLATE = 'base_reduced.yml.template'
BASE_YML_PERIODIC_TEMPLATE = 'base_periodic.yml.template'
PYTEST_YML = 'pytest.yml'
EXAMPLES_PYTEST_YML = 'examples_pytest.yml'
EXAMPLES_LLM_PYTEST_YML = 'examples_llm_pytest.yml'
EXAMPLES_DIFFUSION_PYTEST_YML = 'examples_diffusion_pytest.yml'
EXAMPLES_VISION_PYTEST_YML = 'examples_vision_pytest.yml'
DEVELOP_INSTALL_YML = 'develop_install.yml'
FINN_INTEGRATION_YML = 'finn_integration.yml'
ORT_INTEGRATION_YML = 'ort_integration.yml'
NOTEBOOK_YML = 'notebook.yml'
ENDTOEND_YML = 'end_to_end.yml'

# Reduced Test for PRs, except when a review is requested
PYTHON_VERSIONS_REDUCED = ('3.10',)

PYTORCH_LIST_REDUCED = ['2.2.2', '2.4.1', '2.7.1']

PLATFORM_LIST_REDUCED = ['ubuntu-latest']

MATRIX_REDUCED = od([('python_version', list(PYTHON_VERSIONS_REDUCED)),
                     ('pytorch_version', list(PYTORCH_LIST_REDUCED)),
                     ('platform', PLATFORM_LIST_REDUCED)])

EXAMPLES_LLM_PYTEST_MATRIX_REDUCED = od([('python_version', list(PYTHON_VERSIONS_REDUCED)),
                                         ('pytorch_version', list(
                                             ('2.4.1',))), ('platform', PLATFORM_LIST_REDUCED)])

EXAMPLES_DIFFUSION_PYTEST_MATRIX_REDUCED = od([
    ('python_version', list(PYTHON_VERSIONS_REDUCED)), ('pytorch_version', list(
        ('2.4.1',))), ('platform', PLATFORM_LIST_REDUCED)])

EXAMPLES_VISION_PYTEST_MATRIX_REDUCED = od([('python_version', list(PYTHON_VERSIONS_REDUCED)),
                                            ('pytorch_version', list(
                                                ('2.4.1',))), ('platform', PLATFORM_LIST_REDUCED)])

FINN_MATRIX_REDUCED = od([('python_version', list(PYTHON_VERSIONS_REDUCED)),
                          ('pytorch_version', list(PYTORCH_LIST_REDUCED)),
                          ('platform', PLATFORM_LIST_REDUCED)])

PYTEST_MATRIX_EXTRA_REDUCED = od([('jit_status', [
    'jit_disabled',])])

# Data shared betwen Nox sessions and Github Actions, formatted as tuples
ALL_SUPPORTED_PYTHON_VERSIONS = ('3.9', '3.10', '3.11', '3.12')

ALL_SUPPORTED_PYTORCH_VERSIONS = (
    '1.12.1',
    '1.13.1',
    '2.0.1',
    '2.1.1',
    '2.2.2',
    '2.3.1',
    '2.4.1',
    '2.5.1',
    '2.6.0',
    '2.7.1',
    '2.8.0',
    '2.9.0')
ALL_SUPPORTED_EXCLUSION_LIST = generate_exclusion_list([
    [['python_version', [
        '3.9',]], ['pytorch_version', [
            '2.9.0',]]],
    [['python_version', [
        '3.11',]], ['pytorch_version', ['1.12.1', '1.13.1']]],
    [['python_version', [
        '3.12',]], ['pytorch_version', ['1.12.1', '1.13.1', '2.0.1', '2.1.1']]],])

PYTHON_VERSIONS = ('3.10', '3.11')

PYTORCH_VERSIONS = ('2.2.2', '2.3.1', '2.4.1', '2.5.1', '2.6.0', '2.7.1')
JIT_STATUSES = ('jit_disabled', 'jit_enabled')

# Data used only by Github Actions, formatted as lists or lists of ordered dicts
PLATFORM_LIST = ['windows-latest', 'ubuntu-latest', 'macos-latest']
FINN_PLATFORM_LIST = ['windows-latest', 'ubuntu-latest']

STRATEGY = od([('fail-fast', 'false')])

EXCLUDE_LIST = []

JIT_EXCLUDE_LIST = []

NOTEBOOK_EXCLUDE_LIST = generate_exclusion_list([[['platform', [
    'macos-latest',]]]])

END_TO_END_EXCLUDE_LIST = generate_exclusion_list([[['platform', [
    'windows-latest',]]]])

MATRIX = od([('python_version', list(PYTHON_VERSIONS)), ('pytorch_version', list(PYTORCH_VERSIONS)),
             ('platform', PLATFORM_LIST)])

MATRIX_ALL_SUPPORTED = od([('python_version', list(ALL_SUPPORTED_PYTHON_VERSIONS)),
                           ('pytorch_version', list(ALL_SUPPORTED_PYTORCH_VERSIONS)),
                           ('platform', PLATFORM_LIST)])

EXAMPLES_LLM_PYTEST_PYTORCH_VERSIONS = ('2.4.1', '2.5.1', '2.6.0', '2.7.1')
EXAMPLES_LLM_PYTEST_MATRIX = od([('python_version', list(PYTHON_VERSIONS)),
                                 ('pytorch_version', list(EXAMPLES_LLM_PYTEST_PYTORCH_VERSIONS)),
                                 ('platform', PLATFORM_LIST)])

EXAMPLES_DIFFUSION_PYTEST_PYTORCH_VERSIONS = ('2.2.2', '2.3.1', '2.4.1', '2.5.1', '2.6.0', '2.7.1')
EXAMPLES_DIFFUSION_PYTEST_MATRIX = od([
    ('python_version', list(PYTHON_VERSIONS)),
    ('pytorch_version', list(EXAMPLES_DIFFUSION_PYTEST_PYTORCH_VERSIONS)),
    ('platform', PLATFORM_LIST)])

EXAMPLES_VISION_PYTEST_PYTORCH_VERSIONS = ('2.2.2', '2.3.1', '2.4.1', '2.5.1', '2.6.0', '2.7.1')
EXAMPLES_VISION_PYTEST_MATRIX = od([
    ('python_version', list(PYTHON_VERSIONS)),
    ('pytorch_version', list(EXAMPLES_VISION_PYTEST_PYTORCH_VERSIONS)),
    ('platform', PLATFORM_LIST)])

FINN_MATRIX = od([('python_version', list(PYTHON_VERSIONS)),
                  ('pytorch_version', list(PYTORCH_VERSIONS)), ('platform', FINN_PLATFORM_LIST)])

PYTEST_MATRIX_EXTRA = od([('jit_status', list(JIT_STATUSES))])

PYTEST_STEP_LIST = [
    od([('name', 'Run Nox session for brevitas pytest'), ('shell', 'bash'),
        (
            'run',
            'nox -v -s tests_brevitas_cpu-${{ matrix.python_version }}\(${{ matrix.jit_status }}\,\ pytorch_${{ matrix.pytorch_version }}\)'
        )]),]

EXAMPLES_PYTEST_STEP_LIST = [
    od([('name', 'Run Nox session for brevitas_examples pytest'), ('shell', 'bash'),
        (
            'run',
            'nox -v -s tests_brevitas_examples_cpu-${{ matrix.python_version }}\(${{ matrix.jit_status }}\,\ pytorch_${{ matrix.pytorch_version }}\)'
        )]),]

EXAMPLES_LLM_PYTEST_STEP_LIST = [
    od([('name', 'Run Nox session for brevitas_examples pytest'), ('shell', 'bash'),
        (
            'run',
            'nox -v -s tests_brevitas_examples_llm-${{ matrix.python_version }}\(${{ matrix.jit_status }}\,\ pytorch_${{ matrix.pytorch_version }}\)'
        )]),]

EXAMPLES_DIFFUSION_PYTEST_STEP_LIST = [
    od([('name', 'Run Nox session for brevitas_examples pytest'), ('shell', 'bash'),
        (
            'run',
            'nox -v -s tests_brevitas_examples_diffusion-${{ matrix.python_version }}\(${{ matrix.jit_status }}\,\ pytorch_${{ matrix.pytorch_version }}\)'
        )]),]

EXAMPLES_VISION_PYTEST_STEP_LIST = [
    od([('name', 'Run Nox session for brevitas_examples pytest'), ('shell', 'bash'),
        (
            'run',
            'nox -v -s tests_brevitas_examples_vision-${{ matrix.python_version }}\(${{ matrix.jit_status }}\,\ pytorch_${{ matrix.pytorch_version }}\)'
        )]),]

FINN_INTEGRATION_STEP_LIST = [
    od([('name', 'Install protobuf on Ubuntu'), ('shell', 'bash'),
        ('run', 'sudo apt-get install protobuf-compiler libprotoc-dev'),
        ('if', "startsWith(runner.os, 'Linux') == true")]),
    od([('name', 'Run Nox session for Brevitas-FINN integration'), ('shell', 'bash'),
        (
            'run',
            'nox -v -s tests_brevitas_finn_integration-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)'
        )])]

PYXIR_INTEGRATION_STEP_LIST = [
    od([('name', 'Run Nox session for Brevitas-PyXIR integration'), ('shell', 'bash'),
        (
            'run',
            'nox -v -s tests_brevitas_pyxir_integration-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)'
        )])]

ORT_INTEGRATION_STEP_LIST = [
    od([('name', 'Run Nox session for Brevitas-ORT integration'), ('shell', 'bash'),
        (
            'run',
            'nox -v -s tests_brevitas_ort_integration-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)'
        )])]

TEST_INSTALL_DEV_STEP_LIST = [
    od([('name', 'Run Nox session for testing brevitas develop install and imports'),
        ('shell', 'bash'),
        (
            'run',
            'nox -v -s tests_brevitas_install_dev-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)'
        )]),
    od([('name', 'Run Nox session for testing brevitas_examples develop install and imports'),
        ('shell', 'bash'),
        (
            'run',
            'nox -v -s tests_brevitas_examples_install_dev-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)'
        )])]

NOTEBOOK_STEP_LIST = [
    od([('name', 'Run Nox session for Notebook execution'), ('shell', 'bash'),
        (
            'run',
            'nox -v -s tests_brevitas_notebook-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)'
        )])]

ENDTOEND_STEP_LIST = [
    od([('name', 'Run Nox session for end-to-end flows'), ('shell', 'bash'),
        (
            'run',
            'nox -v -s tests_brevitas_end_to_end-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)'
        )])]


def gen_pytest_yml():
    pytest = Action(
        'Pytest',
        EXCLUDE_LIST + JIT_EXCLUDE_LIST,
        combine_od_list([MATRIX, PYTEST_MATRIX_EXTRA]),
        PYTEST_STEP_LIST,
        STRATEGY)
    pytest.gen_yaml(BASE_YML_TEMPLATE, PYTEST_YML)
    pytest = Action(
        'Pytest',
        EXCLUDE_LIST,
        combine_od_list([MATRIX_REDUCED, PYTEST_MATRIX_EXTRA_REDUCED]),
        PYTEST_STEP_LIST,
        STRATEGY)
    pytest.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + PYTEST_YML)
    pytest = Action(
        'Pytest',
        EXCLUDE_LIST + ALL_SUPPORTED_EXCLUSION_LIST,
        combine_od_list([MATRIX_ALL_SUPPORTED, PYTEST_MATRIX_EXTRA_REDUCED]),
        PYTEST_STEP_LIST,
        STRATEGY)
    pytest.gen_yaml(BASE_YML_PERIODIC_TEMPLATE, 'periodic_' + PYTEST_YML)


def gen_examples_pytest_yml():
    pytest = Action(
        'Examples Pytest',
        EXCLUDE_LIST + JIT_EXCLUDE_LIST,
        combine_od_list([MATRIX, PYTEST_MATRIX_EXTRA]),
        EXAMPLES_PYTEST_STEP_LIST,
        STRATEGY)
    pytest.gen_yaml(BASE_YML_TEMPLATE, EXAMPLES_PYTEST_YML)
    pytest = Action(
        'Examples Pytest',
        EXCLUDE_LIST,
        combine_od_list([MATRIX_REDUCED, PYTEST_MATRIX_EXTRA_REDUCED]),
        EXAMPLES_PYTEST_STEP_LIST,
        STRATEGY)
    pytest.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + EXAMPLES_PYTEST_YML)


def gen_examples_llm_pytest_yml():
    pytest = Action(
        'Examples LLM Pytest',
        EXCLUDE_LIST + JIT_EXCLUDE_LIST,
        combine_od_list([EXAMPLES_LLM_PYTEST_MATRIX, PYTEST_MATRIX_EXTRA]),
        EXAMPLES_LLM_PYTEST_STEP_LIST,
        STRATEGY)
    pytest.gen_yaml(BASE_YML_TEMPLATE, EXAMPLES_LLM_PYTEST_YML)
    pytest = Action(
        'Examples LLM Pytest',
        EXCLUDE_LIST,
        combine_od_list([EXAMPLES_LLM_PYTEST_MATRIX_REDUCED, PYTEST_MATRIX_EXTRA_REDUCED]),
        EXAMPLES_LLM_PYTEST_STEP_LIST,
        STRATEGY)
    pytest.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + EXAMPLES_LLM_PYTEST_YML)


def gen_examples_diffusion_pytest_yml():
    pytest = Action(
        'Examples Diffusion Pytest',
        EXCLUDE_LIST + JIT_EXCLUDE_LIST,
        combine_od_list([EXAMPLES_DIFFUSION_PYTEST_MATRIX, PYTEST_MATRIX_EXTRA]),
        EXAMPLES_DIFFUSION_PYTEST_STEP_LIST,
        STRATEGY)
    pytest.gen_yaml(BASE_YML_TEMPLATE, EXAMPLES_DIFFUSION_PYTEST_YML)
    pytest = Action(
        'Examples Diffusion Pytest',
        EXCLUDE_LIST,
        combine_od_list([EXAMPLES_DIFFUSION_PYTEST_MATRIX_REDUCED, PYTEST_MATRIX_EXTRA_REDUCED]),
        EXAMPLES_DIFFUSION_PYTEST_STEP_LIST,
        STRATEGY)
    pytest.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + EXAMPLES_DIFFUSION_PYTEST_YML)


def gen_examples_vision_pytest_yml():
    pytest = Action(
        'Examples Vision Pytest',
        EXCLUDE_LIST + JIT_EXCLUDE_LIST,
        combine_od_list([EXAMPLES_VISION_PYTEST_MATRIX, PYTEST_MATRIX_EXTRA]),
        EXAMPLES_VISION_PYTEST_STEP_LIST,
        STRATEGY)
    pytest.gen_yaml(BASE_YML_TEMPLATE, EXAMPLES_VISION_PYTEST_YML)
    pytest = Action(
        'Examples Vision Pytest',
        EXCLUDE_LIST,
        combine_od_list([EXAMPLES_VISION_PYTEST_MATRIX_REDUCED, PYTEST_MATRIX_EXTRA_REDUCED]),
        EXAMPLES_VISION_PYTEST_STEP_LIST,
        STRATEGY)
    pytest.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + EXAMPLES_VISION_PYTEST_YML)


def gen_test_develop_install_yml():
    test_develop_install = Action(
        'Test develop install', EXCLUDE_LIST, MATRIX, TEST_INSTALL_DEV_STEP_LIST, STRATEGY)
    test_develop_install.gen_yaml(BASE_YML_TEMPLATE, DEVELOP_INSTALL_YML)
    test_develop_install = Action(
        'Test develop install', EXCLUDE_LIST, MATRIX_REDUCED, TEST_INSTALL_DEV_STEP_LIST, STRATEGY)
    test_develop_install.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + DEVELOP_INSTALL_YML)
    test_develop_install = Action(
        'Test develop install',
        EXCLUDE_LIST + ALL_SUPPORTED_EXCLUSION_LIST,
        MATRIX_ALL_SUPPORTED,
        TEST_INSTALL_DEV_STEP_LIST,
        STRATEGY)
    test_develop_install.gen_yaml(BASE_YML_PERIODIC_TEMPLATE, 'periodic_' + DEVELOP_INSTALL_YML)


def gen_test_brevitas_finn_integration():
    test_finn_integration = Action(
        'Test Brevitas-FINN integration',
        EXCLUDE_LIST,
        FINN_MATRIX,
        FINN_INTEGRATION_STEP_LIST,
        STRATEGY)
    test_finn_integration.gen_yaml(BASE_YML_TEMPLATE, FINN_INTEGRATION_YML)
    test_finn_integration = Action(
        'Test Brevitas-FINN integration',
        EXCLUDE_LIST,
        FINN_MATRIX_REDUCED,
        FINN_INTEGRATION_STEP_LIST,
        STRATEGY)
    test_finn_integration.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + FINN_INTEGRATION_YML)


def gen_test_brevitas_ort_integration():
    test_ort_integration = Action(
        'Test Brevitas-ORT integration', EXCLUDE_LIST, MATRIX, ORT_INTEGRATION_STEP_LIST, STRATEGY)
    test_ort_integration.gen_yaml(BASE_YML_TEMPLATE, ORT_INTEGRATION_YML)
    test_ort_integration = Action(
        'Test Brevitas-ORT integration',
        EXCLUDE_LIST,
        MATRIX_REDUCED,
        ORT_INTEGRATION_STEP_LIST,
        STRATEGY)
    test_ort_integration.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + ORT_INTEGRATION_YML)
    test_ort_integration = Action(
        'Test Brevitas-ORT integration',
        EXCLUDE_LIST + ALL_SUPPORTED_EXCLUSION_LIST,
        MATRIX_ALL_SUPPORTED,
        ORT_INTEGRATION_STEP_LIST,
        STRATEGY)
    test_ort_integration.gen_yaml(BASE_YML_PERIODIC_TEMPLATE, 'periodic_' + ORT_INTEGRATION_YML)


def gen_test_brevitas_notebook():
    tests_brevitas_notebooks = Action(
        'Test Notebook execution',
        EXCLUDE_LIST + NOTEBOOK_EXCLUDE_LIST,
        MATRIX,
        NOTEBOOK_STEP_LIST,
        STRATEGY)
    tests_brevitas_notebooks.gen_yaml(BASE_YML_TEMPLATE, NOTEBOOK_YML)
    tests_brevitas_notebooks = Action(
        'Test Notebook execution',
        EXCLUDE_LIST + NOTEBOOK_EXCLUDE_LIST,
        MATRIX_REDUCED,
        NOTEBOOK_STEP_LIST,
        STRATEGY)
    tests_brevitas_notebooks.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + NOTEBOOK_YML)
    tests_brevitas_notebooks = Action(
        'Test Notebook execution',
        EXCLUDE_LIST + ALL_SUPPORTED_EXCLUSION_LIST + NOTEBOOK_EXCLUDE_LIST,
        MATRIX_ALL_SUPPORTED,
        NOTEBOOK_STEP_LIST,
        STRATEGY)
    tests_brevitas_notebooks.gen_yaml(BASE_YML_PERIODIC_TEMPLATE, 'periodic_' + NOTEBOOK_YML)


def gen_test_brevitas_end_to_end():
    tests_brevitas_end_to_end = Action(
        'Test End-to-end flows',
        EXCLUDE_LIST + END_TO_END_EXCLUDE_LIST,
        MATRIX,
        ENDTOEND_STEP_LIST,
        STRATEGY)
    tests_brevitas_end_to_end.gen_yaml(BASE_YML_TEMPLATE, ENDTOEND_YML)
    tests_brevitas_end_to_end = Action(
        'Test End-to-end flows',
        EXCLUDE_LIST + END_TO_END_EXCLUDE_LIST,
        MATRIX_REDUCED,
        ENDTOEND_STEP_LIST,
        STRATEGY)
    tests_brevitas_end_to_end.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + ENDTOEND_YML)
    tests_brevitas_end_to_end = Action(
        'Test End-to-end flows',
        EXCLUDE_LIST + ALL_SUPPORTED_EXCLUSION_LIST + END_TO_END_EXCLUDE_LIST,
        MATRIX_ALL_SUPPORTED,
        ENDTOEND_STEP_LIST,
        STRATEGY)
    tests_brevitas_end_to_end.gen_yaml(BASE_YML_PERIODIC_TEMPLATE, 'periodic_' + ENDTOEND_YML)


if __name__ == '__main__':
    gen_pytest_yml()
    gen_examples_pytest_yml()
    gen_examples_llm_pytest_yml()
    gen_examples_diffusion_pytest_yml()
    gen_examples_vision_pytest_yml()
    gen_test_develop_install_yml()
    gen_test_brevitas_finn_integration()
    gen_test_brevitas_ort_integration()
    gen_test_brevitas_notebook()
    gen_test_brevitas_end_to_end()
