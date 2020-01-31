#include "ops_register.hpp"
// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension
#ifdef _WIN32
#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_C(void) {
  // No need to do anything.
  // _custom_ops.py will run on load
  return NULL;
}
#else
PyMODINIT_FUNC PyInit_C(void) {
  // No need to do anything.
  // _custom_ops.py will run on load
  return NULL;
}
#endif
#endif

int64_t _cuda_version() {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}


static auto registry =
    torch::RegisterOperators()
    .op("brevitas::round_ste", &round_ste)
    .op("brevitas::tensor_clamp_ste", &tensor_clamp_ste)
    .op("brevitas::scalar_clamp_ste", &scalar_clamp_ste)
    .op("brevitas::binary_sign_ste", &binary_sign_ste)
    .op("brevitas::ternary_sign_ste", &ternary_sign_ste)
    .op("brevitas::ceil_ste", &ceil_ste)
    .op("brevitas::floor_ste", &floor_ste);
