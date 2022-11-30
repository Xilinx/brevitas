## Test Organization
This folder contains all the tests for the Brevitas repository.

Everytime a new function is added or modified, it is highly recommeded to adapt the associated test cases if present, or to add new ones if tests are missing.

We are in the process of expanding and re-organising the test suite, thus information and descriptions contained here might change in the future.

Brevitas test suite is built around the following packages:
- Pytest
- Hypothesis
- Pytest Cases
- Mock (Pytest Mock)

Please feel free to familiriasize with their basic usage before contributing to new tests on Brevitas.

### Folder Structure
The root tests folder structure is based on requirements:
- brevitas: tests the main functionalities of the library. Only brevitas and the test packages should be needed to run these tests.
- brevitas_examples: tests different aspects of the tests present in _src\_brevitas\_examples_.
- brevitas_finn: tests the integration with the finn export-flow.
- brevitas_ort: tests the brevitas onnx export by comparing results with onnxruntime.

In particular, within the _brevitas_ folder, the subfolder structure matches the one in _src\_brevitas_.

A class in _src\_brevitas\_core_ will have its corresponding test in _tests\_brevitas\_core_.

### Adding New Tests
When adding a new test, make sure to respect the folder structure described above.
When adding or modifying a file in _src\_brevitas_,  make sure that the corresponding test is adapted (or added, if missing) in the respective test folder. 


Generally speaking, there are two main types of tests used for Brevitas. 
The first type relies heavily on Hypothesis to stress-test desired numerical properties. This is particularly useful for example when investigating the numerical propertities of scale factors or gradients.
The second group uses more _fixtures_, _parametrize_, and _pytest\_cases_ to test other properties of modules and functions. For example, testing that the same quantized module executed in brevitas and in onnxruntime have similar outputs.

#### Numerical-based tests (i.e., Hypothesis-based tests)
We define a set of pre-configured hypothesis-based strategies that can be used for tests, that can be found in [hyp_helper.py](https://github.com/Xilinx/brevitas/blob/dev/tests/brevitas/hyp_helper.py).

To be completed.
