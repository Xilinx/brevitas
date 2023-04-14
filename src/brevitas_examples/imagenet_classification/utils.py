import csv

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

SEED = 123456

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


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


def accuracy(output, target, topk=(1,), stable=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if stable and (max(topk) > 1 or len(topk) > 1):
        raise RuntimeError("Stable implementation supports only topk = (1,)")

    with torch.no_grad():
        batch_size = target.size(0)
        if stable:
            import numpy as np
            pred = np.argmax(output.cpu().numpy(), axis=1)
            pred = torch.tensor(pred, device=target.device, dtype=target.dtype).unsqueeze(0)
        else:
            maxk = max(topk)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul(100.0 / batch_size))
        return res


def validate(val_loader, model):
    """
    Run validation on the desired dataset
    """
    top1 = AverageMeter('Acc@1', ':6.2f')

    def print_accuracy(top1, prefix=''):
        print('{}Avg acc@1 {top1.avg:2.3f}'.format(prefix, top1=top1))

    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            target = target.to(device)
            target = target.to(dtype)
            images = images.to(device)
            images = images.to(dtype)

            output = model(images)
            # measure accuracy
            acc1, = accuracy(output, target, stable=True)
            top1.update(acc1[0], images.size(0))

        print_accuracy(top1, 'Total:')
    return top1.avg.cpu().numpy()


def generate_dataset(dir, resize_shape=256, center_crop_shape=224, inception_preprocessing=False):
    if inception_preprocessing:
        normalize = transforms.Normalize(mean=0.5, std=0.5)
    else:
        normalize = transforms.Normalize(mean=MEAN, std=STD)

    dataset = datasets.ImageFolder(
        dir,
        transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.CenterCrop(center_crop_shape),
            transforms.ToTensor(),
            normalize]))
    return dataset


def generate_dataloader(
        dir,
        batch_size,
        num_workers,
        resize_shape,
        center_crop_shape,
        subset_size=None,
        inception_preprocessing=False):
    dataset = generate_dataset(
        dir,
        resize_shape=resize_shape,
        center_crop_shape=center_crop_shape,
        inception_preprocessing=inception_preprocessing)
    if subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return loader
