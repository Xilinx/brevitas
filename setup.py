# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


from setuptools import setup, find_packages
import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_DIR = os.path.join(PROJECT_ROOT, 'requirements')


def read(*path):
    return open(os.path.join(*path), encoding='utf8').read()


def read_requirements(filename):
    return read(REQUIREMENTS_DIR, filename).splitlines()


setup(name="brevitas",
      use_scm_version=True,
      setup_requires=read_requirements('requirements-setup.txt'),
      description="Quantization-aware training in PyTorch",
      long_description=read(PROJECT_ROOT, 'README.md'),
      long_description_content_type="text/markdown",
      author="Alessandro Pappalardo",
      author_email="alessand@xilinx.com",
      url="https://github.com/Xilinx/brevitas",
      python_requires=">=3.6",
      install_requires=read_requirements('requirements.txt'),
      extras_require={
          "hadamard": read_requirements('requirements-hadamard.txt'),
          "test": read_requirements('requirements-test.txt'),
          "tts": read_requirements('requirements-tts.txt'),
          "stt": read_requirements('requirements-stt.txt'),
          "vision": read_requirements('requirements-vision.txt'),
          "finn_integration_lt_pt150":
              read_requirements('requirements-finn-integration-lt-pt150.txt'),
          "finn_integration_ge_pt150":
              read_requirements('requirements-finn-integration-ge-pt150.txt'),
          "pyxir_integration": read_requirements('requirements-pyxir-integration.txt'),
          "ort_integration": read_requirements('requirements-ort-integration.txt')
      },
      packages=find_packages('src'),
      package_dir={'': 'src'},
      zip_safe=False,
      include_package_data=True,
      entry_points={
          'console_scripts': [
              'brevitas_bnn_pynq_train = brevitas_examples.bnn_pynq.bnn_pynq_train:main',
              'brevitas_imagenet_val = brevitas_examples.imagenet_classification.imagenet_val:main',
              'brevitas_quartznet_val = brevitas_examples.speech_to_text.quartznet_val:main',
              'brevitas_melgan_val = brevitas_examples.text_to_speech.melgan_val:main',
              'brevitas_quartznet_preprocess = brevitas_examples.speech_to_text.get_librispeech_data:main',
              'brevitas_melgan_preprocess = brevitas_examples.text_to_speech.preprocess_dataset:main'
          ],
      })


