[![DOI](https://zenodo.org/badge/140494324.svg)](https://zenodo.org/badge/latestdoi/140494324)

# Brevitas

Brevitas is a Pytorch library for training-aware quantization.

*Brevitas is currently in alpha stage and under active development. APIs might and probably will change. Documentation, examples, and pretrained models will be progressively released.*

## Requirements
* [Pytorch](https://pytorch.org) >= 1.1.0

## Introduction

Brevitas implements a set of building blocks to model a reduced precision hardware data-path at training time.
While partially biased towards modelling dataflow-style, very low-precision implementations, the building blocks can be parametrized and assembled together to target all sorts of reduced precision hardware.

The implementations tries to adhere to the following design principles:
- Idiomatic Pytorch, when possible.
- Modularity first, at the cost of some verbosity.
- Easily extendible.

## Target audience
Brevitas is mainly targeted at researchers and practicioners in the fields of training for reduced precision inference. 

The implementation is quite rich in options and allows for very fine grained control over the trained model. However, compared to other software solutions in this space, the burden of correctly modelling the target data-path is currently placed on the user. 

## Features

Soon.

## Installation

Soon.

## Usage
Soon.

## Author

Alessandro Pappalardo @ Xilinx Research Labs.

##
