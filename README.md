# Pytorch Quantization

## Introduction

This repository implements a set of quantization strategies to be applied to supported type of layers.

The code originally started from the Pytorch and ATen implementation of a fused GRU/LSTM, extracted as a CFFI extension and expanded from there.

## Requirements
Building currently requires an appropriate CUDA environment, but execution is supported on CPU as well.

* Nvidia CUDA Toolkit (tested with CUDA 9.0)
* [Pytorch](https://pytorch.org) (tested with version 0.3.1)

## Installation

1. Run `python build.py`
2. Add current path to the python path: `EXPORT PYTHONPATH=/path/to/pytorch-quantization:PYTHONPATH`

## Usage

The following quantization modes are implemented for weights:

* FP: full-precision, no quantization performed.
* SIGNED_FIXED_UNIT: fixed point quantization between [-1,1).

The following quantization modes are implemented for activations:

* FP: full-precision, no quantization performed.
* SIGNED_FIXED_UNIT: fixed point quantization between [-1,1).

The following quantized layers are implemented:

* QuantizedLinear
* QuantizedLSTM 