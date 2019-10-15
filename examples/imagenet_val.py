import argparse
import os
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models import quant_mobilenet_v1

SEED = 123456

models = {'quant_mobilenet_v1': quant_mobilenet_v1}


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--imagenet-dir', help='path to folder containing Imagenet val folder')
parser.add_argument('--resume', type=str, help='Path to pretrained model')
parser.add_argument('--arch', choices=models.keys(), default='quant_mobilenet_v1',
                    help='model architecture: ' + ' | '.join(models.keys()))
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', default=256, type=int, help='Minibatch size')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--bit-width', default=4, type=int, help='Model bit-width')


def main():
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    model = models[args.arch](bit_width=args.bit_width)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        cudnn.benchmark = True

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'], strict=False)

    valdir = os.path.join(args.imagenet_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    validate(val_loader, model, args)
    return


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
    return


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
