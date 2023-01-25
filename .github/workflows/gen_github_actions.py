from collections import OrderedDict as od

from utils import Action
from utils import combine_od_list

BASE_YML_TEMPLATE = 'base.yml.template'
BASE_YML_REDUCED_TEMPLATE = 'base_reduced.yml.template'
PYTEST_YML = 'pytest.yml'
EXAMPLES_PYTEST_YML = 'examples_pytest.yml'
DEVELOP_INSTALL_YML = 'develop_install.yml'
FINN_INTEGRATION_YML = 'finn_integration.yml'
ORT_INTEGRATION_YML = 'ort_integration.yml'
NOTEBOOK_YML = 'notebook.yml'
ENDTOEND_YML = 'end_to_end.yml'

# Reduced Test for PRs, except when a review is requested
PYTHON_VERSIONS_REDUCED = ('3.8',)

PYTORCH_LIST_REDUCED = ["1.5.1", "1.10.1", "1.13.0"]

PLATFORM_LIST_REDUCED = ['ubuntu-latest']

MATRIX_REDUCED = od([('python_version', list(PYTHON_VERSIONS_REDUCED)), ('pytorch_version', list(PYTORCH_LIST_REDUCED)),
             ('platform', PLATFORM_LIST_REDUCED)])

FINN_MATRIX_REDUCED = od([('python_version', list(PYTHON_VERSIONS_REDUCED)),
                  ('pytorch_version', list(PYTORCH_LIST_REDUCED)), ('platform', PLATFORM_LIST_REDUCED)])

# Data shared betwen Nox sessions and Github Actions, formatted as tuples
PYTHON_VERSIONS = ('3.7', '3.8')

PYTORCH_VERSIONS = ('1.5.1', '1.6.0', '1.7.1', '1.8.1', '1.9.1', '1.10.1', '1.11.0', '1.12.1',
                    '1.13.0')
JIT_STATUSES = ('jit_disabled', 'jit_enabled')

# Data used only by Github Actions, formatted as lists or lists of ordered dicts
PLATFORM_LIST = ['windows-latest', 'ubuntu-latest', 'macos-latest']
FINN_PLATFORM_LIST = ['windows-latest', 'ubuntu-latest']

STRATEGY = od([('fail-fast', 'false')])

EXCLUDE_LIST = []
JIT_EXCLUDE_LIST = [
    od([('pytorch_version', ['1.5.1', ]),
         ('jit_status', ['jit_enabled',])]),
    od([('pytorch_version', ['1.6.0', ]),
         ('jit_status', ['jit_enabled',])]),
    od([('pytorch_version', ['1.7.1']),
        ('jit_status', ['jit_enabled',])]),
    od([('pytorch_version', ['1.8.1']),
        ('jit_status', ['jit_enabled',])]),
    od( [('pytorch_version', ['1.9.1']),
        ('jit_status', ['jit_enabled',])]),
    ]
NOTEBOOK_EXCLUDE_LIST = [od([('pytorch_version', ['1.5.1', '1.6.0', '1.7.1'])]),
                         od([('platform', ['macos-latest',])])]

END_TO_END_EXCLUDE_LIST = [od([('platform', ['windows-latest',])])]

MATRIX = od([('python_version', list(PYTHON_VERSIONS)),
             ('pytorch_version', list(PYTORCH_VERSIONS)),
             ('platform', PLATFORM_LIST)])

FINN_MATRIX = od([('python_version', list(PYTHON_VERSIONS)),
                  ('pytorch_version', list(PYTORCH_VERSIONS)),
                  ('platform', FINN_PLATFORM_LIST)])

PYTEST_MATRIX_EXTRA = od([('jit_status', list(JIT_STATUSES))])

PYTEST_STEP_LIST = [
    od([('name', 'Run Nox session for brevitas pytest'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_cpu-${{ matrix.python_version }}\(${{ matrix.jit_status }}\,\ pytorch_${{ matrix.pytorch_version }}\)')]),
]

EXAMPLES_PYTEST_STEP_LIST = [
    od([('name', 'Run Nox session for brevitas_examples pytest'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_examples_cpu-${{ matrix.python_version }}\(${{ matrix.jit_status }}\,\ pytorch_${{ matrix.pytorch_version }}\)')]),
]

FINN_INTEGRATION_STEP_LIST = [
    od([
        ('name', 'Install protobuf on Ubuntu'),
        ('shell', 'bash'),
        ('run',
         'sudo apt-get install protobuf-compiler libprotoc-dev'),
        ('if', "startsWith(runner.os, 'Linux') == true")
    ]),
    od([
        ('name', 'Run Nox session for Brevitas-FINN integration'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_finn_integration-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')
    ])]

PYXIR_INTEGRATION_STEP_LIST = [
    od([('name', 'Run Nox session for Brevitas-PyXIR integration'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_pyxir_integration-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')
    ])]

ORT_INTEGRATION_STEP_LIST = [
    od([('name', 'Run Nox session for Brevitas-ORT integration'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_ort_integration-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')
    ])]

TEST_INSTALL_DEV_STEP_LIST = [
    od([('name', 'Run Nox session for testing brevitas develop install and imports'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_install_dev-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)'
        )]),
    od([('name', 'Run Nox session for testing brevitas_examples develop install and imports'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_examples_install_dev-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')
    ])]

NOTEBOOK_STEP_LIST = [
    od([('name', 'Run Nox session for Notebook execution'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_notebook-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')
    ])]

ENDTOEND_STEP_LIST = [
    od([('name', 'Run Nox session for end-to-end flows'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_end_to_end-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')
    ])]

def gen_pytest_yml():
    pytest = Action(
        'Pytest',
        EXCLUDE_LIST + JIT_EXCLUDE_LIST,
        combine_od_list([MATRIX, PYTEST_MATRIX_EXTRA]),
        PYTEST_STEP_LIST,
        STRATEGY)
    pytest.gen_yaml(BASE_YML_TEMPLATE, PYTEST_YML)
    pytest = Action('Pytest', EXCLUDE_LIST, combine_od_list([MATRIX_REDUCED, PYTEST_MATRIX_EXTRA]),
                    PYTEST_STEP_LIST, STRATEGY)
    pytest.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + PYTEST_YML)

def gen_examples_pytest_yml():
    pytest = Action(
        'Examples Pytest',
        EXCLUDE_LIST + JIT_EXCLUDE_LIST,
        combine_od_list([MATRIX, PYTEST_MATRIX_EXTRA]),
        EXAMPLES_PYTEST_STEP_LIST,
        STRATEGY)
    pytest.gen_yaml(BASE_YML_TEMPLATE, EXAMPLES_PYTEST_YML)
    pytest = Action('Examples Pytest', EXCLUDE_LIST, combine_od_list([MATRIX_REDUCED, PYTEST_MATRIX_EXTRA]),
                    EXAMPLES_PYTEST_STEP_LIST, STRATEGY)
    pytest.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + EXAMPLES_PYTEST_YML)

def gen_test_develop_install_yml():
    test_develop_install = Action(
        'Test develop install',
        EXCLUDE_LIST,
        MATRIX,
        TEST_INSTALL_DEV_STEP_LIST,
        STRATEGY)
    test_develop_install.gen_yaml(BASE_YML_TEMPLATE, DEVELOP_INSTALL_YML)
    test_develop_install = Action('Test develop install', EXCLUDE_LIST, MATRIX_REDUCED,
                                  TEST_INSTALL_DEV_STEP_LIST, STRATEGY)
    test_develop_install.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + DEVELOP_INSTALL_YML)

def gen_test_brevitas_finn_integration():
    test_finn_integration = Action(
        'Test Brevitas-FINN integration',
        EXCLUDE_LIST,
        FINN_MATRIX,
        FINN_INTEGRATION_STEP_LIST,
        STRATEGY)
    test_finn_integration.gen_yaml(BASE_YML_TEMPLATE, FINN_INTEGRATION_YML)
    test_finn_integration = Action('Test Brevitas-FINN integration', EXCLUDE_LIST, FINN_MATRIX_REDUCED,
                                   FINN_INTEGRATION_STEP_LIST, STRATEGY)
    test_finn_integration.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + FINN_INTEGRATION_YML)

def gen_test_brevitas_ort_integration():
    test_ort_integration = Action(
        'Test Brevitas-ORT integration',
        EXCLUDE_LIST,
        MATRIX,
        ORT_INTEGRATION_STEP_LIST,
        STRATEGY)
    test_ort_integration.gen_yaml(BASE_YML_TEMPLATE, ORT_INTEGRATION_YML)
    test_ort_integration = Action('Test Brevitas-ORT integration', EXCLUDE_LIST, MATRIX_REDUCED,
                                  ORT_INTEGRATION_STEP_LIST, STRATEGY)
    test_ort_integration.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + ORT_INTEGRATION_YML)

def gen_test_brevitas_notebook():
    tests_brevitas_notebooks = Action(
        'Test Notebook execution',
        EXCLUDE_LIST + NOTEBOOK_EXCLUDE_LIST,
        MATRIX,
        NOTEBOOK_STEP_LIST,
        STRATEGY)
    tests_brevitas_notebooks.gen_yaml(BASE_YML_TEMPLATE, NOTEBOOK_YML)
    tests_brevitas_notebooks = Action('Test Notebook execution',
                                      EXCLUDE_LIST + NOTEBOOK_EXCLUDE_LIST, MATRIX_REDUCED,
                                      NOTEBOOK_STEP_LIST, STRATEGY)
    tests_brevitas_notebooks.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + NOTEBOOK_YML)

def gen_test_brevitas_end_to_end():
    tests_brevitas_end_to_end = Action(
        'Test End-to-end flows',
        EXCLUDE_LIST + END_TO_END_EXCLUDE_LIST,
        MATRIX,
        ENDTOEND_STEP_LIST,
        STRATEGY)
    tests_brevitas_end_to_end.gen_yaml(BASE_YML_TEMPLATE, ENDTOEND_YML)
    tests_brevitas_end_to_end = Action('Test End-to-end flows',
                                       EXCLUDE_LIST + END_TO_END_EXCLUDE_LIST, MATRIX_REDUCED,
                                       ENDTOEND_STEP_LIST, STRATEGY)
    tests_brevitas_end_to_end.gen_yaml(BASE_YML_REDUCED_TEMPLATE, 'reduced_' + ENDTOEND_YML)

if __name__ == '__main__':
    gen_pytest_yml()
    gen_examples_pytest_yml()
    gen_test_develop_install_yml()
    gen_test_brevitas_finn_integration()
    gen_test_brevitas_ort_integration()
    gen_test_brevitas_notebook()
    gen_test_brevitas_end_to_end()
