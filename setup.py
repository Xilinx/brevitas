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
import io
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension
from pkg_resources import get_distribution, DistributionNotFound
from setuptools.command.build_py import build_py as build_py_orig
import fnmatch
from packaging import version
import glob
import sys
import distutils.command.clean
import distutils.spawn
import shutil

torch_version = version.parse(torch.__version__)


class build_py(build_py_orig):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if torch_version >= version.parse("1.3.0"):
            file_to_rename = ['ops_ste_o']
        else:
            file_to_rename = ['ops_ste_n']

        pacchetti = [(pkg, mod, file,) for (pkg, mod, file,) in modules
                     if not any(fnmatch.fnmatchcase(mod, pat=pattern)
                                for pattern in file_to_rename)]
        return pacchetti


    def build_module(self, module, module_file, package):
        _, file_name = os.path.split(module_file)
        if file_name == 'ops_ste_n.py' or file_name == 'ops_ste_o.py':
            file_name = 'ops_ste.py'
        if isinstance(package, str):
            package = package.split('.')
        elif not isinstance(package, (list, tuple)):
            raise TypeError(
                "'package' must be a string (dot-separated), list, or tuple")

        # Now put the module source file into the "build" area -- this is
        # easy, we just copy it somewhere under self.build_lib (the build
        # directory for Python source).
        outfile = self.get_module_outfile(self.build_lib, package, module)

        head, _ = os.path.split(outfile)
        outfile = os.path.join(head, file_name)

        dir = os.path.dirname(outfile)
        self.mkpath(dir)
        return self.copy_file(module_file, outfile, preserve_mode=0)


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


def get_extensions():
    if torch_version >= version.parse("1.3.0"):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        extensions_dir = os.path.join(this_dir, 'brevitas', 'csrc')

        main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
        source_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', '*.cpp'))
        sources = main_file + source_cpu
        extension = CppExtension

        define_macros = []

        extra_compile_args = {}

        if sys.platform == 'win32':
            define_macros += [('brevitas_EXPORTS', None)]
            extra_compile_args.setdefault('cxx', [])
            extra_compile_args['cxx'].append('/MP')
        sources = [os.path.join(extensions_dir, s) for s in sources]
        include_dirs = [extensions_dir]
        ext_modules = [
            extension(
                'brevitas._C',
                sources,
                include_dirs=include_dirs,
                define_macros=define_macros,
                extra_compile_args=extra_compile_args,
            )
        ]
        return ext_modules
    else:
        return []


class clean(distutils.command.clean.clean):
    def run(self):
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split('\n')):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


INSTALL_REQUIRES = ["torch>=1.1.0", "docrep", "scipy", "packaging"]
TEST_REQUIRES = ["pytest", "hypothesis", "mock"]


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if torch_version >= version.parse("1.3.0"):
    cmdclass_dict = {
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True),
        'clean': clean,
        'build_py': build_py
    }
else:
    cmdclass_dict = {
        'build_py': build_py
    }

setup(name="Brevitas",
      version="0.2.0-alpha",
      description="Training-aware quantization in PyTorch",
      long_description=read('README.md'),
      long_description_content_type="text/markdown",
      author="Alessandro Pappalardo",
      python_requires=">=3.6",
      install_requires=INSTALL_REQUIRES,
      extras_require={
          "test": TEST_REQUIRES,
      },
      packages=find_packages(),

      zip_safe=False,
      ext_modules=get_extensions(),
      cmdclass=cmdclass_dict
      )
