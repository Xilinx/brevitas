from collections import OrderedDict as od

from utils import Action


VITIS_AI_BASE_YML_TEMPLATE = 'vitis_ai_base.yml.template'
XIR_INTEGRATION_YML = 'xir_integration.yml'


# Data shared betwen Nox sessions and Github Actions, formatted as tuples
PYTHON_VERSIONS = ('3.6', '3.7')
PYTORCH_VERSIONS = ('1.1.0', '1.2.0', '1.3.1', '1.4.0')

# Data used only by Github Actions, formatted as lists or lists of oredered dicts
PLATFORM_LIST = ['ubuntu-latest']


MATRIX = od([('python_version', list(PYTHON_VERSIONS)),
             ('pytorch_version', list(PYTORCH_VERSIONS)),
             ('platform', PLATFORM_LIST)])


XIR_INTEGRATION_STEP_LIST = [
    od([
        ('name', 'Run Nox session for Brevitas-XIR integration'),
        ('shell', 'bash'),
        ('run',
         'nox -v -s tests_brevitas_xir_integration-${{ matrix.python_version }}\(\pytorch_${{ matrix.pytorch_version }}\)')
    ])]


def gen_test_brevitas_xir_integration():
    test_ort_integration = Action(
        'Test Brevitas-XIR integration',
        [],
        MATRIX,
        XIR_INTEGRATION_STEP_LIST)
    test_ort_integration.gen_yaml(VITIS_AI_BASE_YML_TEMPLATE, XIR_INTEGRATION_YML)


if __name__ == '__main__':
    gen_test_brevitas_xir_integration()