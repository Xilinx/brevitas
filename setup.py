# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os

from setuptools import find_packages
from setuptools import setup

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_DIR = os.path.join(PROJECT_ROOT, 'requirements')


def read(*path):
    return open(os.path.join(*path), encoding='utf8').read()


def read_requirements(filename):
    return read(REQUIREMENTS_DIR, filename).splitlines()


setup(
    name="brevitas",
    use_scm_version=True,
    setup_requires=read_requirements('requirements-setup.txt'),
    description="Quantization-aware training in PyTorch",
    long_description=read(PROJECT_ROOT, 'README.md'),
    long_description_content_type="text/markdown",
    author="Alessandro Pappalardo",
    author_email="alessand@xilinx.com",
    url="https://github.com/Xilinx/brevitas",
    python_requires=">=3.7",
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        "notebook": read_requirements('requirements-notebook.txt'),
        "docs": read_requirements("requirements-docs.txt"),
        "export": read_requirements('requirements-export.txt'),
        "hadamard": read_requirements('requirements-hadamard.txt'),
        "test": read_requirements('requirements-test.txt'),
        "tts": read_requirements('requirements-tts.txt'),
        "stt": read_requirements('requirements-stt.txt'),
        "vision": read_requirements('requirements-vision.txt'),
        "finn_integration": read_requirements('requirements-finn-integration.txt'),
        "ort_integration": read_requirements('requirements-ort-integration.txt')},
    packages=find_packages('src'),
    package_dir={'': 'src'},
    zip_safe=False,
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'brevitas_bnn_pynq_train = brevitas_examples.bnn_pynq.bnn_pynq_train:main',
            'brevitas_qat_imagenet_val = brevitas_examples.imagenet_classification.qat.imagenet_val:main',
            'brevitas_quartznet_val = brevitas_examples.speech_to_text.quartznet_val:main',
            'brevitas_melgan_val = brevitas_examples.text_to_speech.melgan_val:main',
            'brevitas_quartznet_preprocess = brevitas_examples.speech_to_text.get_librispeech_data:main',
            'brevitas_melgan_preprocess = brevitas_examples.text_to_speech.preprocess_dataset:main',
            'brevitas_ptq_imagenet_benchmark = brevitas_examples.imagenet_classification.ptq.ptq_benchmark:main',
            'brevitas_ptq_imagenet_val = brevitas_examples.imagenet_classification.ptq.ptq_evaluate:main'
        ],})
