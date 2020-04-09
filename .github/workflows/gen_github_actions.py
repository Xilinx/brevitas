from functools import reduce
from string import Template
from textwrap import indent
from collections import OrderedDict as od

import yaml

BASE_YML_TEMPLATE = 'base.yml.template'
PYTEST_YML = 'pytest.yml'
EXAMPLES_PYTEST_YML = 'examples_pytest.yml'
DEVELOP_INSTALL_YML = 'develop_install.yml'
FINN_INTEGRATION_YML = 'finn_integration.yml'

NIX_NEWLINE = '\n'

# Data shared betwen Nox sessions and Github Actions, formatted as tuples
CONDA_PYTHON_VERSIONS = ('3.6', '3.7', '3.8')
PYTORCH_VERSIONS = ('1.1.0', '1.2.0', '1.3.0', '1.3.1', '1.4.0', '1.5.0')
JIT_STATUSES = ('jit_enabled', 'jit_disabled')

# Data used only by Github Actions, formatted as lists or lists of oredered dicts
PLATFORM_LIST = ['windows-latest', 'ubuntu-latest', 'macos-latest']

EXCLUDE_LIST = [od([('platform', 'macos-latest'),
                    ('pytorch_version', '1.1.0')]),
                od([('pytorch_version', '1.1.0'),
                    ('conda_python_version', '3.8')]),
                od([('pytorch_version', '1.2.0'),
                    ('conda_python_version', '3.8')]),
                od([('pytorch_version', '1.3.0'),
                    ('conda_python_version', '3.8')]),
                od([('pytorch_version', '1.3.1'),
                    ('conda_python_version', '3.8')])]

PYTEST_EXCLUDE_LIST_EXTRA = [od([('pytorch_version', '1.1.0'),
                                 ('jit_status', 'jit_disabled')]),
                             od([('pytorch_version', '1.2.0'),
                                 ('jit_status', 'jit_disabled')])]

PYTEST_EXAMPLE_EXCLUDE_LIST_EXTRA = [od([('platform', 'macos-latest'),
                                         ('pytorch_version', '1.5.0'),
                                         ('conda_python_version', '3.6')])]

MATRIX = od([('conda_python_version', list(CONDA_PYTHON_VERSIONS)),
             ('pytorch_version', list(PYTORCH_VERSIONS)),
             ('platform', PLATFORM_LIST)])

PYTEST_MATRIX_EXTRA = od([('jit_status', list(JIT_STATUSES))])

PYTEST_STEP_LIST = [
    od([
        ('name', 'Run Nox session for brevitas pytest'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_cpu-${{ matrix.conda_python_version }}\(${{ matrix.jit_status }}\,\ pytorch_${{ matrix.pytorch_version }}\)')]),
]

EXAMPLES_PYTEST_STEP_LIST = [
    od([
        ('name', 'Run Nox session for brevitas_examples pytest'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_examples_cpu-${{ matrix.conda_python_version }}\(${{ matrix.jit_status }}\,\ pytorch_${{ matrix.pytorch_version }}\)')]),
]

FINN_INTEGRATION_STEP_LIST = [
    od([
        ('name', 'Run Nox session for Brevitas-FINN integration'),
        ('shell', 'bash'),
        ('run', 'nox -v -s tests_brevitas_finn_integration-${{ matrix.conda_python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')
    ])]

TEST_INSTALL_DEV_STEP_LIST = [
    od([
        ('name', 'Run Nox session for testing brevitas develop install and imports'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_install_dev-${{ matrix.conda_python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')]),
    od([
        ('name', 'Run Nox session for testing brevitas_examples develop install and imports'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_examples_install_dev-${{ matrix.conda_python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')
    ])]

# whitespaces to indent generated portions of output yaml
STEP_INDENT = 4
MATRIX_INDENT = 8
EXCLUDE_INDENT = 10
RELATIVE_INDENT = 2


class CustomTemplate(Template):
    delimiter = '&'


class Action:

    def __init__(self, name, exclude_list, matrix, step_list):
        self.name = name
        self.exclude_list = exclude_list
        self.matrix = matrix
        self.step_list = step_list

    @staticmethod
    def list_of_dicts_str(list_of_dicts, quote_val, indent_first):
        repr = ''
        for e in list_of_dicts:
            repr += Action.dict_str(e, quote_val, indent_first)
        return repr

    @staticmethod
    def dict_str(d, quote_val, indent_first):
        first_line_prefix = '- ' if indent_first else ''
        repr = first_line_prefix
        for name, val in d.items():
            if quote_val:
                repr += f"{name}: '{val}'" + NIX_NEWLINE
            else:
                repr += f"{name}: {val}" + NIX_NEWLINE
        if indent_first:
            repr = indent(repr, RELATIVE_INDENT * ' ', predicate=lambda line: not first_line_prefix in line)
        repr += NIX_NEWLINE
        return repr

    def gen_yaml(self, output_path):
        d = {'name': self.name,
             'exclude': indent(Action.list_of_dicts_str(self.exclude_list, True, True), EXCLUDE_INDENT * ' '),
             'matrix': indent(Action.dict_str(self.matrix, False, False), MATRIX_INDENT * ' '),
             'steps': indent(Action.list_of_dicts_str(self.step_list, False, True), STEP_INDENT * ' ')}
        template = CustomTemplate(open(BASE_YML_TEMPLATE).read())
        generated_file = template.substitute(d)
        yaml.safe_load(generated_file)  # validate the generated yaml
        with open(output_path, 'w', newline=NIX_NEWLINE) as f:
            f.write(generated_file)


def combine_od_list(od_list):
    return od(reduce(lambda l1, l2: l1 + l2, list(map(lambda d: list(d.items()), od_list))))


def gen_pytest_yml():
    pytest = Action(
        'Pytest',
        EXCLUDE_LIST + PYTEST_EXCLUDE_LIST_EXTRA,
        combine_od_list([MATRIX, PYTEST_MATRIX_EXTRA]),
        PYTEST_STEP_LIST)
    pytest.gen_yaml(PYTEST_YML)


def gen_examples_pytest_yml():
    pytest = Action(
        'Examples Pytest',
        EXCLUDE_LIST + PYTEST_EXCLUDE_LIST_EXTRA + PYTEST_EXAMPLE_EXCLUDE_LIST_EXTRA,
        combine_od_list([MATRIX, PYTEST_MATRIX_EXTRA]),
        EXAMPLES_PYTEST_STEP_LIST)
    pytest.gen_yaml(EXAMPLES_PYTEST_YML)


def gen_test_develop_install_yml():
    test_develop_install = Action(
        'Test develop install',
        EXCLUDE_LIST,
        MATRIX,
        TEST_INSTALL_DEV_STEP_LIST)
    test_develop_install.gen_yaml(DEVELOP_INSTALL_YML)


def gen_test_brevitas_finn_integration():
    test_finn_integration = Action(
        'Test Brevitas-FINN integration',
        EXCLUDE_LIST,
        MATRIX,
        FINN_INTEGRATION_STEP_LIST)
    test_finn_integration.gen_yaml(FINN_INTEGRATION_YML)



if __name__ == '__main__':
    gen_pytest_yml()
    gen_examples_pytest_yml()
    gen_test_develop_install_yml()
    gen_test_brevitas_finn_integration()