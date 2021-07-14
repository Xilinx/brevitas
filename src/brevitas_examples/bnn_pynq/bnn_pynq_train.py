# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import os
import sys

import torch
from brevitas_examples.bnn_pynq.trainer import Trainer

# Pytorch precision
torch.set_printoptions(precision=10)


# Util method to add mutually exclusive boolean
def add_bool_arg(parser, name, default):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, action="store_true")
    group.add_argument("--no_" + name, dest=name, action="store_false")
    parser.set_defaults(**{name: default})


# Util method to pass None as a string and be recognized as None value
def none_or_str(value):
    if value == "None":
        return None
    return value


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def parse_args(args):
    parser = argparse.ArgumentParser(description="PyTorch MNIST/CIFAR10 Training")
    # I/O
    parser.add_argument("--datadir", default="./data/", help="Dataset location")
    parser.add_argument("--experiments", default="./experiments", help="Path to experiments folder")
    parser.add_argument("--dry_run", action="store_true", help="Disable output files generation")
    parser.add_argument("--log_freq", type=int, default=10)
    # Execution modes
    parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
    parser.add_argument("--resume", dest="resume", type=none_or_str,
                        help="Resume from checkpoint. Overrides --pretrained flag.")
    add_bool_arg(parser, "detect_nan", default=False)
    # Compute resources
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers")
    parser.add_argument("--gpus", type=none_or_str, default="0", help="Comma separated GPUs")
    # Optimizer hyperparams
    parser.add_argument("--batch_size", default=100, type=int, help="batch size")
    parser.add_argument("--lr", default=0.02, type=float, help="Learning rate")
    parser.add_argument("--optim", type=none_or_str, default="ADAM", help="Optimizer to use")
    parser.add_argument("--loss", type=none_or_str, default="SqrHinge", help="Loss function to use")
    parser.add_argument("--scheduler", default="FIXED", type=none_or_str, help="LR Scheduler")
    parser.add_argument("--milestones", type=none_or_str, default='100,150,200,250', help="Scheduler milestones")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs")
    parser.add_argument("--random_seed", default=1, type=int, help="Random seed")
    # Neural network Architecture
    parser.add_argument("--network", default="LFC_1W1A", type=str, help="neural network")
    parser.add_argument("--pretrained", action='store_true', help="Load pretrained model")
    parser.add_argument("--strict", action='store_true', help="Strict state dictionary loading")
    return parser.parse_args(args)


class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def launch(cmd_args):
    args = parse_args(cmd_args)

    # Set relative paths relative to current workdir
    path_args = ["datadir", "experiments", "resume"]
    for path_arg in path_args:
        path = getattr(args, path_arg)
        if path is not None and not os.path.isabs(path):
            abs_path = os.path.abspath(os.path.join(os.getcwd(), path))
            setattr(args, path_arg, abs_path)

    # Access config as an object
    args = objdict(args.__dict__)

    # Avoid creating new folders etc.
    if args.evaluate:
        args.dry_run = True

    # Init trainer
    trainer = Trainer(args)

    # Execute
    if args.evaluate:
        with torch.no_grad():
            trainer.eval_model()
    else:
        trainer.train_model()


def main():
    launch(sys.argv[1:])


if __name__ == "__main__":
    main()
