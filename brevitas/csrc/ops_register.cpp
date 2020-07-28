//Copyright (c) 2020-     Xilinx            (Giuseppe Franco)
//Copyright (c) 2016      Soumith Chintala
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met:
//
//* Redistributions of source code must retain the above copyright notice, this
//  list of conditions and the following disclaimer.
//
//* Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
//
//* Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include "ops_register.hpp"

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension
#ifdef _WIN32
#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_C(void) {
  return NULL;
}
#else
PyMODINIT_FUNC PyInit_C(void) {
  return NULL;
}
#endif
#endif


static auto registry =
    torch::RegisterOperators()
    .op("brevitas::round_ste", &round_ste)
    .op("brevitas::tensor_clamp_ste", &tensor_clamp_ste)
    .op("brevitas::scalar_clamp_ste", &scalar_clamp_ste)
    .op("brevitas::binary_sign_ste", &binary_sign_ste)
    .op("brevitas::ternary_sign_ste", &ternary_sign_ste)
    .op("brevitas::ceil_ste", &ceil_ste)
    .op("brevitas::floor_ste", &floor_ste)
    .op("brevitas::round_to_zero_ste", &round_to_zero_ste);
