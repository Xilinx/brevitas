import os
from functools import reduce
from string import Template
from textwrap import indent
from collections import OrderedDict as od

import nox
import yaml
from packaging import version
from platform import system


PYTORCH_CPU_VIRTUAL_PKG = '1.2.0'
GITHUB_WORKFLOWS_PATH = os.path.join('.', '.github', 'workflows')
TESTS_YAML_TEMPLATE_PATH = os.path.join(GITHUB_WORKFLOWS_PATH, 'tests.yaml.template')


# Data shared betwen Nox sessions and Github Actions, formatted as tuples
CONDA_PYTHON_VERSIONS = ('3.6', '3.7', '3.8')
PYTORCH_VERSIONS = ('1.1.0', '1.2.0', '1.3.0', '1.3.1', '1.4.0')
JIT_STATUSES = ('jit_enabled', 'jit_disabled')


# Data used only by Github Actions, formatted as lists or lists of oredered dicts
PLATFORM_LIST = ['windows-latest', 'ubuntu-latest', 'macos-latest']

EXCLUDE_LIST = [od([('platform', 'windows-latest'),
                    ('conda_python_version', '3.6')]),
                od([('platform', 'macos-latest'),
                    ('pytorch_version', '1.1.0')]),
                od([('pytorch_version', '1.1.0'),
                    ('conda_python_version', '3.8')]),
                od([('pytorch_version', '1.2.0'),
                    ('conda_python_version', '3.8')]),
                od([('pytorch_version', '1.3.0'),
                    ('conda_python_version', '3.8')]),
                od([('pytorch_version', '1.3.1'),
                    ('conda_python_version', '3.8')])]

PYTEST_EXCLUDE_LIST_EXTRA = od([('pytorch_version', '1.1.0'),
                                ('jit_status', 'jit_disabled')])

MATRIX = od([('conda_python_version', list(CONDA_PYTHON_VERSIONS)),
             ('pytorch_version', list(PYTORCH_VERSIONS)),
             ('platform', PLATFORM_LIST)])

PYTEST_MATRIX_EXTRA = od([('jit_status', list(JIT_STATUSES))])

PYTEST_STEP_LIST = [od([
    ('name', 'Run Nox session for pytest'),
    ('shell', 'bash'),
    ('run', 'nox --verbose --session tests_cpu-${{ matrix.conda_python_version }}\(${{ matrix.jit_status }}\,\ pytorch_${{ matrix.pytorch_version }}\)')])]

TEST_INSTALL_DEVELOP_STEP_LIST = [od([
    ('name', 'Run Nox session for testing develop install and imports'),
    ('shell', 'bash'),
    ('run', 'nox --verbose --session tests_install_develop-${{ matrix.conda_python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')])]


# IDs used by Nox
CONDA_PYTHON_IDS = tuple([f'conda_python_{i}' for i in CONDA_PYTHON_VERSIONS])
PYTORCH_IDS = tuple([f'pytorch_{i}' for i in PYTORCH_VERSIONS])
JIT_IDS = tuple([f'{i}'.lower() for i in JIT_STATUSES])


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
                repr += f"{name}: '{val}'\n"
            else:
                repr += f"{name}: {val}\n"
        if indent_first:
            repr = indent(repr, RELATIVE_INDENT*' ', predicate=lambda line: not first_line_prefix in line)
        return repr

    def gen_yaml(self, output_path):
        d = {'name': self.name,
             'exclude': indent(Action.list_of_dicts_str(self.exclude_list, True, True), EXCLUDE_INDENT*' '),
             'matrix': indent(Action.dict_str(self.matrix, False, False), MATRIX_INDENT*' '),
             'steps': indent(Action.list_of_dicts_str(self.step_list, False, True), STEP_INDENT*' ')}
        template = CustomTemplate(open(TESTS_YAML_TEMPLATE_PATH).read())
        generated_file = template.substitute(d)
        yaml.safe_load(generated_file)  # validate the generated yaml
        with open(output_path, 'w') as f:
            f.write(generated_file)


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


def combine_od_list(od_list):
    return od(reduce(lambda l1, l2: l1 + l2, list(map(lambda d: list(d.items()), od_list))))


def gen_pytest_yaml():
    pytest = Action(
        'pytest',
        combine_od_list([EXCLUDE_LIST, PYTEST_EXCLUDE_LIST_EXTRA]),
        combine_od_list([MATRIX, PYTEST_MATRIX_EXTRA]),
        PYTEST_STEP_LIST)
    pytest.gen_yaml(os.path.join(GITHUB_WORKFLOWS_PATH, 'pytest.yaml'))


def gen_test_develop_install_yaml():
    test_develop_install = Action(
        'Test develop install',
        EXCLUDE_LIST,
        MATRIX,
        TEST_INSTALL_DEVELOP_STEP_LIST)
    test_develop_install.gen_yaml(os.path.join(GITHUB_WORKFLOWS_PATH, 'develop_install.yaml'))


if __name__ == '__main__':
    gen_pytest_yaml()
    gen_test_develop_install_yaml()

