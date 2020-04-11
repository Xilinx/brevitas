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
from string import Template
import torch
from torch.utils.cpp_extension import BuildExtension, include_paths, library_paths
from pkg_resources import normalize_path
from distutils.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools import Extension
import glob
import sys

MIN_TORCH_JITTABLE_VERSION = "1.3.0"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_DIR = os.path.join(PROJECT_ROOT, 'requirements')


def apply_template(dest_build, version):
    template_path = os.path.join('brevitas', 'function', 'ops_ste.py.template')
    generated_path = os.path.join(dest_build, 'ops_ste.py')
    if version.parse(torch.__version__) >= version.parse(MIN_TORCH_JITTABLE_VERSION):
        d = dict(
            function_suffix='',
            function_prefix='torch.ops.brevitas.',
            torch_jit_template='@torch.jit.script')
    else:
        d = dict(
            function_suffix='_fn.apply',
            function_prefix='',
            torch_jit_template='@torch.jit.ignore')

    template_file = Template(read(template_path))
    generated_file = template_file.substitute(d)
    with open(generated_path, 'w') as f:
        f.write(generated_file)


def read(*path):
    return open(os.path.join(*path)).read()


def read_requirements(filename):
    return read(REQUIREMENTS_DIR, filename).splitlines()


class JittableExtension(Extension):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BuildJittableExtension(BuildExtension):

    def run(self):
        from packaging import version
        if version.parse(torch.__version__) < version.parse(MIN_TORCH_JITTABLE_VERSION):
            self.extensions = [e for e in self.extensions if not isinstance(e, JittableExtension)]
        super().run()


def get_jittable_extension():
    ext_modules = []
    extensions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brevitas', 'csrc')

    sources = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir] + include_paths()
    define_macros = []
    libraries = []
    library_dirs = []
    extra_compile_args = {}

    if sys.platform == 'win32':
        define_macros += [('brevitas_EXPORTS', None)]
        extra_compile_args.setdefault('cxx', [])
        extra_compile_args['cxx'].append('/MP')
        library_dirs += library_paths()
        libraries.append('c10')
        libraries.append('torch')
        libraries.append('torch_python')
        libraries.append('_C')


    jittable_ext = JittableExtension(
        'brevitas._C',
        language='c++',
        sources=sources,
        libraries=libraries,
        library_dirs=library_dirs,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)
    ext_modules.append(jittable_ext)
    return ext_modules


class BuildPy(build_py):

    def run(self):
        if not self.dry_run:
            from packaging import version
            target_dir = os.path.join(self.build_lib, 'brevitas', 'function')
            self.mkpath(target_dir)
            apply_template(target_dir, version)
        build_py.run(self)

class DevelopInstall(develop):
    def run(self):
        from packaging import version
        super().run()
        bext_cmd = self.get_finalized_command('build_ext')
        build_path = normalize_path(bext_cmd.build_lib)
        build_lib = os.path.dirname(os.path.dirname(build_path))
        target_dir = os.path.join(build_lib, 'brevitas', 'function')
        apply_template(target_dir, version)


setup(name="Brevitas",
      version="0.2.0-alpha",
      description="Quantization-aware training in PyTorch",
      long_description=read(PROJECT_ROOT, 'README.md'),
      long_description_content_type="text/markdown",
      author="Alessandro Pappalardo",
      python_requires=">=3.6",
      setup_requires=read_requirements('requirements-setup.txt'),
      install_requires=read_requirements('requirements.txt'),
      extras_require={
          "Hadamard": read_requirements('requirements-hadamard.txt'),
          "test": read_requirements('requirements-test.txt'),
          "tts": read_requirements('requirements-tts.txt'),
          "stt": read_requirements('requirements-stt.txt'),
          "vision": read_requirements('requirements-vision.txt')
      },
      packages=find_packages(),
      zip_safe=False,
      ext_modules=get_jittable_extension(),
      cmdclass={
          'build_py': BuildPy,
          'build_ext': BuildJittableExtension.with_options(no_python_abi_suffix=True),
          'develop': DevelopInstall,
      },
      package_data={
          'brevitas_examples': ['*.ini', '*.yaml'],
      },
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


