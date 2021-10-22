from collections import OrderedDict as od

from utils import Action, combine_od_list


BASE_YML_TEMPLATE = 'base.yml.template'
PYTEST_YML = 'pytest.yml'
EXAMPLES_PYTEST_YML = 'examples_pytest.yml'
DEVELOP_INSTALL_YML = 'develop_install.yml'
FINN_INTEGRATION_YML = 'finn_integration.yml'
PYXIR_INTEGRATION_YML = 'pyxir_integration.yml'
ORT_INTEGRATION_YML = 'ort_integration.yml'


# Data shared betwen Nox sessions and Github Actions, formatted as tuples
PYTHON_VERSIONS = ('3.6', '3.7', '3.8')
PYTORCH_VERSIONS = ('1.5.1', '1.6.0', '1.7.1', '1.8.1', '1.9.1', '1.10.0')
JIT_STATUSES = ('jit_disabled',)

# Data used only by Github Actions, formatted as lists or lists of ordered dicts
PLATFORM_LIST = ['windows-latest', 'ubuntu-latest', 'macos-latest']
FINN_PLATFORM_LIST = ['windows-latest', 'ubuntu-latest']

EXCLUDE_LIST = []

PYTEST_EXAMPLE_EXCLUDE_LIST_EXTRA = [od([('platform', 'macos-latest'),
                                         ('pytorch_version', '1.5.0'),
                                         ('python_version', '3.6')])]

PYXIR_INTEGRATION_EXCLUDE_LIST_EXTRA = [od([('platform', 'macos-latest'),
                                            ('pytorch_version', '1.5.0'),
                                            ('python_version', '3.6')])]

FINN_INTEGRATION_EXCLUDE_LIST_EXTRA = [od([('platform', 'windows-latest'),
                                           ('python_version', '3.6')])]

MATRIX = od([('python_version', list(PYTHON_VERSIONS)),
             ('pytorch_version', list(PYTORCH_VERSIONS)),
             ('platform', PLATFORM_LIST)])

FINN_MATRIX = od([('python_version', list(PYTHON_VERSIONS)),
                  ('pytorch_version', list(PYTORCH_VERSIONS)),
                  ('platform', FINN_PLATFORM_LIST)])

PYTEST_MATRIX_EXTRA = od([('jit_status', list(JIT_STATUSES))])

PYTEST_STEP_LIST = [
    od([
        ('name', 'Run Nox session for brevitas pytest'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_cpu-${{ matrix.python_version }}\(${{ matrix.jit_status }}\,\ pytorch_${{ matrix.pytorch_version }}\)')]),
]

EXAMPLES_PYTEST_STEP_LIST = [
    od([
        ('name', 'Run Nox session for brevitas_examples pytest'),
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
    od([
        ('name', 'Run Nox session for Brevitas-PyXIR integration'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_pyxir_integration-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')
    ])]

ORT_INTEGRATION_STEP_LIST = [
    od([
        ('name', 'Run Nox session for Brevitas-ORT integration'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_ort_integration-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')
    ])]

TEST_INSTALL_DEV_STEP_LIST = [
    od([
        ('name', 'Run Nox session for testing brevitas develop install and imports'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_install_dev-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')]),
    od([
        ('name', 'Run Nox session for testing brevitas_examples develop install and imports'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_examples_install_dev-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')
    ])]


def gen_pytest_yml():
    pytest = Action(
        'Pytest',
        EXCLUDE_LIST,
        combine_od_list([MATRIX, PYTEST_MATRIX_EXTRA]),
        PYTEST_STEP_LIST)
    pytest.gen_yaml(BASE_YML_TEMPLATE, PYTEST_YML)


def gen_examples_pytest_yml():
    pytest = Action(
        'Examples Pytest',
        EXCLUDE_LIST + PYTEST_EXAMPLE_EXCLUDE_LIST_EXTRA,
        combine_od_list([MATRIX, PYTEST_MATRIX_EXTRA]),
        EXAMPLES_PYTEST_STEP_LIST)
    pytest.gen_yaml(BASE_YML_TEMPLATE, EXAMPLES_PYTEST_YML)


def gen_test_develop_install_yml():
    test_develop_install = Action(
        'Test develop install',
        EXCLUDE_LIST,
        MATRIX,
        TEST_INSTALL_DEV_STEP_LIST)
    test_develop_install.gen_yaml(BASE_YML_TEMPLATE, DEVELOP_INSTALL_YML)


def gen_test_brevitas_finn_integration():
    test_finn_integration = Action(
        'Test Brevitas-FINN integration',
        EXCLUDE_LIST + FINN_INTEGRATION_EXCLUDE_LIST_EXTRA,
        FINN_MATRIX,
        FINN_INTEGRATION_STEP_LIST)
    test_finn_integration.gen_yaml(BASE_YML_TEMPLATE, FINN_INTEGRATION_YML)


def gen_test_brevitas_pyxir_integration():
    test_pyxir_integration = Action(
        'Test Brevitas-PyXIR integration',
        EXCLUDE_LIST + PYXIR_INTEGRATION_EXCLUDE_LIST_EXTRA,
        MATRIX,
        PYXIR_INTEGRATION_STEP_LIST)
    test_pyxir_integration.gen_yaml(BASE_YML_TEMPLATE, PYXIR_INTEGRATION_YML)


def gen_test_brevitas_ort_integration():
    test_ort_integration = Action(
        'Test Brevitas-ORT integration',
        EXCLUDE_LIST,
        MATRIX,
        ORT_INTEGRATION_STEP_LIST)
    test_ort_integration.gen_yaml(BASE_YML_TEMPLATE, ORT_INTEGRATION_YML)


if __name__ == '__main__':
    gen_pytest_yml()
    gen_examples_pytest_yml()
    gen_test_develop_install_yml()
    gen_test_brevitas_finn_integration()
    gen_test_brevitas_pyxir_integration()
    gen_test_brevitas_ort_integration()