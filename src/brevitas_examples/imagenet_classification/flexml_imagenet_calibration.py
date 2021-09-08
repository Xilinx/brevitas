import argparse
import os
import random
from copy import deepcopy
from functools import partial

import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models

try:
    import timm
except ImportError:
    timm = None

from brevitas.graph.target.flexml import preprocess_flexml, quantize_flexml
from brevitas.graph.calibrate import finalize_collect_stats
from brevitas.graph.calibrate import BiasCorrection
from brevitas.graph.calibrate import DisableQuantInference
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from brevitas.graph.utils import get_module
from brevitas.fx import GraphModule

SEED = 123456
INPUT_CHANNELS = 3
INPUT_FEATURES = 224

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--imagenet-dir', help='path to folder containing Imagenet val folder')
parser.add_argument('--model', type=str, default='resnet18', help='Name of the torchvision model')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', default=250, type=int, help='Minibatch size')
parser.add_argument('--eq-iters', default=0, type=int, help='Number of equalization iterations')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--shuffle', action='store_true', help='Shuffle validation data.')
parser.add_argument('--source', type=str, default='torchvision', help='Source for the model')


def calibrate_model(calibration_loader, model, args):
    with torch.no_grad():
        model.train()
        for i, (images, _) in enumerate(calibration_loader):
            print(f'Calibration iteration {i}')
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                DisableQuantInference().apply(model, images)
        model.eval()
        bc = BiasCorrection(iterations=len(calibration_loader))
        for i, (images, _) in enumerate(calibration_loader):
            print(f'Bias correction iteration {i}')
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                model = bc.apply(model, images)
    model.apply(finalize_collect_stats)
    return model


def extract_layers(graph_model: GraphModule):
    layers = []
    for node in graph_model.graph.nodes:
        if node.op == 'call_module':
            module = get_module(graph_model, node.target)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers.append((node.target, module))
    return layers


def validate(val_loader, model, args):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    def print_accuracy(top1, top5, prefix=''):
        print('{}Avg acc@1 {top1.avg:.3f} Avg acc@5 {top5.avg:.3f}'
              .format(prefix, top1=top1, top5=top5))

    model.eval()
    with torch.no_grad():
        num_batches = len(val_loader)
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            output = model(images)
            # measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            print_accuracy(top1, top5, '{}/{}: '.format(i, num_batches))
        print_accuracy(top1, top5, 'Total:')
    return top1, top5


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul(100.0 / batch_size))
        return res


def main():
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    if args.source == 'torchvision':
        model = getattr(models, args.model)(pretrained=True)
    elif args.source == 'timm':
        if timm is None:
            raise RuntimeError("timm is not installed, run pip install timm")
        model = timm.create_model(args.model, pretrained=True)
    else:
        raise RuntimeError(f"{args.source} not recognized as source.")

    # preprocess the floating point network 
    rand_inp = torch.randn(2, INPUT_CHANNELS, INPUT_FEATURES, INPUT_FEATURES)
    flexml_model = preprocess_flexml(deepcopy(model), equalization_iters=args.eq_iters, input=rand_inp)

    # graph quantize the network
    flexml_model = quantize_flexml(flexml_model)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        flexml_model = flexml_model.cuda(args.gpu)

    valdir = os.path.join(args.imagenet_dir, 'val')
    calibrationdir = os.path.join(args.imagenet_dir, 'calibration')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(INPUT_FEATURES),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.workers, pin_memory=True)
    
    calibration_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(calibrationdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(INPUT_FEATURES),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.workers, pin_memory=True)

    # Perform calibration on the calibration set
    flexml_model = calibrate_model(calibration_loader, flexml_model, args)

    # Compute accuracy on the validation set
    top1, _ = validate(val_loader, flexml_model, args)

    # Export the model to BrevitasONNX
    BrevitasONNXManager.export(
        model.cpu(), 
        input_t=torch.randn(2, INPUT_CHANNELS, INPUT_FEATURES, INPUT_FEATURES), 
        export_path=f'{args.model}_flexml_{top1.avg:.2f}.onnx')


if __name__ == '__main__':
    main()
