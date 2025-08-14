# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from datetime import datetime
from hashlib import sha256
import os
import random
import time

from packaging.version import parse
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST

import brevitas.config as config
from brevitas.export import export_onnx_qcdq
from brevitas.export import export_qonnx

from .logger import EvalEpochMeters
from .logger import Logger
from .logger import TrainingEpochMeters
from .models import model_with_cfg
from .models.losses import SqrHingeLoss


class MirrorMNIST(MNIST):

    if parse(torchvision.__version__) < parse('0.9.1'):

        resources = [(
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
                     (
                         "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
                         "d53e105ee54ea40749a09fcbcd1e9432"),
                     (
                         "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
                         "9fb629c4189551a2d022fa330f9573f3"),
                     (
                         "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
                         "ec29112dd5afa0611ce80d1b7f02629c")]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Trainer(object):

    def __init__(self, args):

        model, cfg = model_with_cfg(args.network, args.pretrained)

        # Catch invalid settings
        self.validate(args)

        # Init arguments
        self.args = args
        prec_name = "_{}W{}A".format(
            cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH'), cfg.getint('QUANT', 'ACT_BIT_WIDTH'))
        experiment_name = '{}{}_{}'.format(
            args.network, prec_name, datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.output_dir_path = os.path.join(args.experiments, experiment_name)

        if self.args.resume:
            self.output_dir_path, _ = os.path.split(args.resume)
            self.output_dir_path, _ = os.path.split(self.output_dir_path)

        if not args.dry_run:
            self.checkpoints_dir_path = os.path.join(self.output_dir_path, 'checkpoints')
            if not args.resume:
                os.mkdir(self.output_dir_path)
                os.mkdir(self.checkpoints_dir_path)
        # If we want to export a ONNX model, we still need to make output dirs
        if args.export_qonnx or args.export_qcdq_onnx:
            self.output_onnx_path = os.path.join(self.output_dir_path, "onnx")
            os.makedirs(self.output_onnx_path, exist_ok=True)

        self.logger = Logger(self.output_dir_path, args.dry_run)

        # Randomness
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

        # Datasets
        transform_to_tensor = transforms.Compose([transforms.ToTensor()])

        dataset = cfg.get('MODEL', 'DATASET')
        self.num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
        if dataset == 'CIFAR10':
            train_transforms_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]
            transform_train = transforms.Compose(train_transforms_list)
            builder = CIFAR10

        elif dataset == 'MNIST':
            transform_train = transform_to_tensor
            builder = MirrorMNIST
        else:
            raise Exception("Dataset not supported: {}".format(args.dataset))

        train_set = builder(root=args.datadir, train=True, download=True, transform=transform_train)
        test_set = builder(
            root=args.datadir, train=False, download=True, transform=transform_to_tensor)
        self.train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        self.test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # Init starting values
        self.starting_epoch = 1
        self.best_val_acc = 0

        # Setup device
        if args.gpus is not None:
            args.gpus = [int(i) for i in args.gpus.split(',')]
            self.device = 'cuda:' + str(args.gpus[0])
            torch.backends.cudnn.benchmark = True
        else:
            self.device = 'cpu'
        self.device = torch.device(self.device)

        # Resume checkpoint, if any
        if args.resume:
            model = self.load_checkpoint(model, args.resume, args.strict)

        if args.state_dict_to_pth:
            state_dict = model.state_dict()
            name = args.network.lower()
            path = os.path.join(self.checkpoints_dir_path, name)
            torch.save(state_dict, path)
            with open(path, "rb") as f:
                bytes = f.read()
                readable_hash = sha256(bytes).hexdigest()[:8]
            new_path = path + '-' + readable_hash + '.pth'
            os.rename(path, new_path)
            self.logger.info("Saving checkpoint model to {}".format(new_path))
            exit(0)

        if args.gpus is not None:
            model = model.to(device=self.device)
        if args.gpus is not None and len(args.gpus) > 1:
            model = nn.DataParallel(model, args.gpus)
        self.model = model

        # Loss function
        if args.loss == 'SqrHinge':
            self.criterion = SqrHingeLoss()
        elif args.loss == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"{args.loss} not supported.")
        self.criterion = self.criterion.to(device=self.device)

        # Init optimizer
        if args.optim == 'ADAM':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay)

        # Resume optimizer, if any
        if args.resume and not args.evaluate:
            self.logger.log.info("Loading optimizer checkpoint")
            if 'optim_dict' in package.keys():
                self.optimizer.load_state_dict(package['optim_dict'])
            if 'epoch' in package.keys():
                self.starting_epoch = package['epoch']
            if 'best_val_acc' in package.keys():
                self.best_val_acc = package['best_val_acc']

        # LR scheduler
        if args.scheduler == 'STEP':
            milestones = [int(i) for i in args.milestones.split(',')]
            self.scheduler = MultiStepLR(optimizer=self.optimizer, milestones=milestones, gamma=0.1)
        elif args.scheduler == 'FIXED':
            self.scheduler = None
        else:
            raise Exception("Unrecognized scheduler {}".format(self.args.scheduler))

        # Resume scheduler, if any
        if args.resume and not args.evaluate and self.scheduler is not None:
            self.scheduler.last_epoch = package['epoch']

    def validate(self, args):
        if args.export_qonnx or args.export_qcdq_onnx:
            assert not config.JIT_ENABLED, "JIT must be disabled for ONNX export, please run with BREVITAS_JIT=0"

    # Move model to CPU for reloading state_dicts or export - handles if model is data parallel
    def to_cpu(self, model):
        if isinstance(model, nn.DataParallel):
            self.logger.log.info("Converting Model from `nn.DataParallel` to `nn.Module`")
            model = model.module
        model = model.to(device="cpu")
        return model

    # Load checkpoint onto CPU
    def load_checkpoint(self, model, checkpoint_path, strict):
        def maybe_remove_prefix(state_dict, prefix='module'):
            new_state_dict = dict()
            flag = False
            prefix_dot=f"{prefix}."
            for k,v in state_dict.items():
                if k.startswith(prefix_dot):
                    flag = True
                    new_key = "".join(k.split(prefix_dot)[1:])
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            if flag:
                self.logger.info(f"Renaming state_dict keys to remove {prefix}")
            return new_state_dict

        model = self.to_cpu(model)
        print('Loading model checkpoint at: {}'.format(checkpoint_path))
        package = torch.load(checkpoint_path, map_location='cpu')
        model_state_dict = maybe_remove_prefix(package['state_dict'])
        model.load_state_dict(model_state_dict, strict=strict)
        model.to(device="cpu")
        return model

    def checkpoint_best(self, epoch, name):
        best_path = os.path.join(self.checkpoints_dir_path, name)
        self.logger.info("Saving checkpoint model to {}".format(best_path))
        torch.save({
            'state_dict': self.model.state_dict(),
            'optim_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_val_acc': self.best_val_acc,},
                   best_path)

    def train_model(self):

        # training starts
        if self.args.detect_nan:
            torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.starting_epoch, self.args.epochs+1):

            # Set to training mode
            self.model.train()
            self.criterion.train()

            # Init metrics
            epoch_meters = TrainingEpochMeters()
            start_data_loading = time.time()

            for i, data in enumerate(self.train_loader):
                (input, target) = data
                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # for hingeloss only
                if isinstance(self.criterion, SqrHingeLoss):
                    target = target.unsqueeze(1)
                    target_onehot = torch.Tensor(target.size(0), self.num_classes).to(
                        self.device, non_blocking=True)
                    target_onehot.fill_(-1)
                    target_onehot.scatter_(1, target, 1)
                    target = target.squeeze()
                    target_var = target_onehot
                else:
                    target_var = target

                # measure data loading time
                epoch_meters.data_time.update(time.time() - start_data_loading)

                # Training batch starts
                start_batch = time.time()
                output = self.model(input)
                loss = self.criterion(output, target_var)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if hasattr(self.model, 'clip_weights'):
                    self.model.clip_weights(-1, 1)

                # measure elapsed time
                epoch_meters.batch_time.update(time.time() - start_batch)

                if i % int(self.args.log_freq) == 0 or i == len(self.train_loader) - 1:
                    prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
                    epoch_meters.losses.update(loss.item(), input.size(0))
                    epoch_meters.top1.update(prec1.item(), input.size(0))
                    epoch_meters.top5.update(prec5.item(), input.size(0))
                    self.logger.training_batch_cli_log(
                        epoch_meters, epoch, i, len(self.train_loader))

                # training batch ends
                start_data_loading = time.time()

            # Set the learning rate
            if self.scheduler is not None:
                self.scheduler.step(epoch)
            else:
                # Set the learning rate
                if epoch % 40 == 0:
                    self.optimizer.param_groups[0]['lr'] *= 0.5

            # Perform eval
            with torch.no_grad():
                top1avg = self.eval_model(epoch)

            # checkpoint
            if top1avg >= self.best_val_acc and not self.args.dry_run:
                self.best_val_acc = top1avg
                self.checkpoint_best(epoch, "best.tar")
            elif not self.args.dry_run:
                self.checkpoint_best(epoch, "checkpoint.tar")

        # training ends
        if not self.args.dry_run:
            best_path = os.path.join(self.checkpoints_dir_path, "best.tar")
            if self.args.export_qonnx or self.args.export_qonnx:
                self.model = self.load_checkpoint(self.model, best_path, strict=True)
            if self.args.export_qonnx:
                self.export_qonnx()
            if self.args.export_qcdq_onnx:
                self.export_qcdq_onnx()
            return best_path

    def eval_model(self, epoch=None):
        eval_meters = EvalEpochMeters()

        # switch to evaluate mode
        self.model.eval()
        self.criterion.eval()

        for i, data in enumerate(self.test_loader):

            end = time.time()
            (input, target) = data

            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # for hingeloss only
            if isinstance(self.criterion, SqrHingeLoss):
                target = target.unsqueeze(1)
                target_onehot = torch.Tensor(target.size(0), self.num_classes).to(
                    self.device, non_blocking=True)
                target_onehot.fill_(-1)
                target_onehot.scatter_(1, target, 1)
                target = target.squeeze()
                target_var = target_onehot
            else:
                target_var = target

            # compute output
            output = self.model(input)

            # measure model elapsed time
            eval_meters.model_time.update(time.time() - end)
            end = time.time()

            # compute loss
            loss = self.criterion(output, target_var)
            eval_meters.loss_time.update(time.time() - end)

            pred = output.data.argmax(1, keepdim=True)
            correct = pred.eq(target.data.view_as(pred)).sum()
            prec1 = 100. * correct.float() / input.size(0)

            _, prec5 = accuracy(output, target, topk=(1, 5))
            eval_meters.losses.update(loss.item(), input.size(0))
            eval_meters.top1.update(prec1.item(), input.size(0))
            eval_meters.top5.update(prec5.item(), input.size(0))

            # Eval batch ends
            self.logger.eval_batch_cli_log(eval_meters, i, len(self.test_loader))

        return eval_meters.top1.avg

    def export_onnx(self, onnx_type):
        name = self.args.network.lower()
        path = os.path.join(self.output_onnx_path, name)
        model = self.to_cpu(self.model) # Switch to CPU for ONNX export
        training_state = model.training
        model.eval()
        if onnx_type == "qonnx":
            logstr = "QONNX"
            export_qonnx(model, self.train_loader.dataset[0][0].unsqueeze(0), path)
        elif onnx_type == "qcdq":
            logstr = "QCDQ ONNX"
            export_qonnx(model, self.train_loader.dataset[0][0].unsqueeze(0), path)
            export_onnx_qcdq(model, self.train_loader.dataset[0][0].unsqueeze(0), path)
        else:
            self.logger.error(f"Unknown ONNX export format: {onnx_type}, expected qonnx or qcdq")
            exit(1)
        with open(path, "rb") as f:
            bytes = f.read()
            readable_hash = sha256(bytes).hexdigest()[:8]
        new_path = os.path.join(
            self.output_onnx_path, "{}-{}-{}.onnx".format(name, onnx_type, readable_hash))
        os.rename(path, new_path)
        self.logger.info("Exporting {} to {}".format(logstr, new_path))
        model.train(training_state)

    def export_qonnx(self):
        self.export_onnx(onnx_type="qonnx")

    def export_qcdq_onnx(self):
        self.export_onnx(onnx_type="qcdq")
