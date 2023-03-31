## Test Organization
This folder contains all the tests for the Brevitas repository.

Everytime a new function is added or modified, it is highly recommeded to adapt the associated test cases if present, or to add new ones if tests are missing.

We are in the process of expanding and re-organising the test suite, thus information and descriptions contained here might change in the future.

Brevitas test suite is built around the following packages:
- [Pytest](https://pypi.org/project/pytest/)
- [Hypothesis](https://pypi.org/project/hypothesis/)
- [Pytest Cases](https://pypi.org/project/pytest-cases/)
- [Mock](https://pypi.org/project/mock/) ([Pytest Mock](https://pypi.org/project/pytest-mock/))

Please feel free to familiriasize with their basic usage before contributing to new tests on Brevitas.

### Folder Structure
The root tests folder structure is based on requirements:
- brevitas: tests for the main functionalities of the library. Only brevitas and the test packages should be needed to run these tests.
- brevitas_examples: tests for the quantization examples in _src\_brevitas\_examples_.
- brevitas_finn: tests for the integration with the FINN export-flow.
- brevitas_ort: tests for the different brevitas ONNX exports with standard operators, leveraging onnxruntime.

In particular, within the _brevitas_ folder, the subfolder structure matches the one in _src\_brevitas_.
A class in _src\_brevitas\_core_ will have its corresponding test in _tests\_brevitas\_core_.

### Adding New Tests
When adding a new test, make sure to respect the folder structure described above.
When adding or modifying a file in _src\_brevitas_,  check that the corresponding test is adapted (or added, if missing) in the respective test folder.


Generally speaking, there are two main types of tests used for Brevitas.
The first type relies heavily on Hypothesis to stress-test desired numerical properties. This is particularly useful for example when investigating the numerical propertities of scale factors or gradients.
The second group is more focused on other properties of modules and functions. For example, testing that the same quantized module can be in brevitas and in onnxruntime with similar outputs, or veryfing that the export flows work as intended.

#### Numerical-based tests (i.e., Hypothesis-based tests)
We define a set of pre-configured hypothesis-based strategies that can be used for tests, that can be found in [hyp_helper.py](https://github.com/Xilinx/brevitas/blob/dev/tests/brevitas/hyp_helper.py).

To be completed.
