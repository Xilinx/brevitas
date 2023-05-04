from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import math

import torch
import torch.nn as nn

from brevitas.graph.calibrate import DisableEnableQuantization
from brevitas.inject.enum import ScalingImplType
from brevitas.quant_tensor import QuantTensor


class StopFwdException(Exception):
    pass


@dataclass
class LayerHandler:
    layer_name: str = None


class gptq_mode():

    def __init__(self, model, inplace=True, use_quant_activations=False) -> None:
        if not inplace:
            model = deepcopy(model)
        self.model = model
        self.use_quant_activations = use_quant_activations
        self.hook_dict = dict()
        self.gptq_layers = dict()
        # reference for each layer to update
        self.current_layer = LayerHandler()
        self.num_layers = 0
        self.disable_quant_inference = DisableEnableQuantization()
        self.orig_forward = self.model.forward
        self.model.forward = self.catch_stopfwd

    def _is_module_supported(self, module):
        if isinstance(module, nn.Conv2d) and (module.groups == 1 or
                                              (module.groups == module.out_channels)):
            return True
        elif isinstance(module, nn.Linear):
            return True
        else:
            return False

    def __enter__(self):
        if self.use_quant_activations:
            self.disable_quant_inference.disable_act_quantization(
                self.model, is_training=self.model.training)
            self.disable_quant_inference.disable_bias_quantization(
                self.model, is_training=self.model.training)
        for name, module in self.model.named_modules():
            if self._is_module_supported(module):
                gptq = GPTQ(module)
                hook_fn = partial(gptq.update_batch, name=name, current_layer=self.current_layer)
                self.hook_dict[name] = module.register_forward_hook(hook_fn)
                self.gptq_layers[name] = gptq
        self.num_layers = len(self.gptq_layers)
        return self

    def __exit__(self, type, value, traceback):
        self.model.forward = self.orig_forward
        if self.use_quant_activations:
            self.disable_quant_inference.enable_act_quantization(
                self.model, is_training=self.model.training)
            self.disable_quant_inference.enable_bias_quantization(
                self.model, is_training=self.model.training)

    def update(self):
        self.gptq_layers[self.current_layer.layer_name].fasterquant()
        self.gptq_layers[self.current_layer.layer_name].free()
        self.hook_dict[self.current_layer.layer_name].remove()

    def catch_stopfwd(self, *args, **kwargs):
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass


class GPTQ():

    def __init__(self, layer) -> None:
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data
        self.groups = 1
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
            self.groups = self.layer.groups
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.groups, self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def update_batch(self, module, input, out, name, current_layer):
        inp = input[0]
        if isinstance(inp, QuantTensor):
            inp = inp.value
        current_layer.layer_name = name
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inps = inp.t().unsqueeze(0)
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride)
            inps = list(torch.chunk(inp, self.groups, 1))
            for i, inp in enumerate(inps):
                inp = unfold(inp)
                inp = inp.permute([1, 0, 2])
                inp = inp.flatten(1)
                inps[i] = inp
            inps = torch.stack(inps)
        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inps.bmm(inps.transpose(2, 1))
        raise StopFwdException

    def fasterquant(self, percdamp=.01):
        W = self.layer.weight.data
        blocksize = math.ceil(W.shape[1] / 4)

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)

        # Switch quantizer to const
        self.layer.weight_quant.quant_injector = self.layer.weight_quant.quant_injector.let(
            scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS)
        self.layer.weight_quant.init_tensor_quant()
        self.layer.to(self.layer.weight.device)

        H = self.H
        del self.H
        for i in range(self.groups):
            dead = torch.diag(H[i, :, :]) == 0
            H[i, dead, dead] = 1
            W[i, dead] = 0
            damp = percdamp * torch.mean(torch.diag(H[i, :, :]))
            diag = torch.arange(self.columns, device=self.dev)
            H[i, diag, diag] += damp
            H[i, :, :] = torch.linalg.cholesky(H[i, :, :])
            H[i, :, :] = torch.cholesky_inverse(H[i, :, :])
            H[i, :, :] = torch.linalg.cholesky(H[i, :, :], upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2]
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[:, i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[:, i, i]
                q = self.get_quant_weights(i, i1, i2)

                err1 = (w - q) / d
                if self.groups > 1:
                    # In case of depthwise convs, each weight matrix interacts with only
                    # part of the input values, thus with only one of the hessian matrix
                    for ii in range(self.groups):
                        W1[ii, i:] -= err1[ii] * Hinv1[ii, i, i:]
                else:
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[0, i, i:].unsqueeze(0))
                Err1[:, i] = err1

            if self.groups > 1:
                # In case of depthwise convs, each weight matrix interacts with only
                # part of the input values, thus with only one of the hessian matrix
                for ii in range(self.groups):
                    W[ii:ii + 1, i2:] -= Err1[ii:ii + 1, :].matmul(Hinv[ii, i1:i2, i2:])
            else:
                W[:, i2:] -= Err1.matmul(Hinv[0, i1:i2, i2:])

    def get_quant_weights(self, i, i1, i2):
        Q = self.layer.quant_weight()
        if isinstance(self.layer, nn.Conv2d):
            Q = Q.flatten(1)
        Q1 = Q.value[:, i1:i2]
        q = Q1[:, i]
        return q

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None


def main():
    import warnings

    import torchvision
    from torchvision import datasets
    from torchvision import transforms

    from brevitas.graph.calibrate import bias_correction_mode
    from brevitas.graph.calibrate import calibration_mode
    from brevitas.graph.quantize import preprocess_for_quantize
    from brevitas.graph.quantize import quantize
    from brevitas_examples.imagenet_classification.utils import validate

    warnings.filterwarnings("ignore")
    from tqdm import tqdm
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    model = preprocess_for_quantize(model, equalize_iters=20)
    model = quantize(model)
    model_no_gptq = deepcopy(model)

    def generate_dataset(dir, resize_shape=256, center_crop_shape=224):
        normalize = transforms.Normalize(mean=MEAN, std=STD)

        dataset = datasets.ImageFolder(
            dir,
            transforms.Compose([
                transforms.Resize(resize_shape),
                transforms.CenterCrop(center_crop_shape),
                transforms.ToTensor(),
                normalize]))
        return dataset

    valid_dataset = generate_dataset('/scratch/datasets/imagenet_symlink/val')
    valid_loader = tqdm(
        torch.utils.data.DataLoader(
            valid_dataset, batch_size=256, num_workers=10, pin_memory=True, shuffle=False))

    calib_dataset = generate_dataset('/scratch/datasets/imagenet_symlink/calibration')
    calib_dataset = torch.utils.data.Subset(calib_dataset, list(range(1000)))
    calib_loader = torch.utils.data.DataLoader(
        calib_dataset, batch_size=8, num_workers=10, pin_memory=True, shuffle=False)
    model_no_gptq.cuda()
    model_no_gptq.eval()
    model.cuda()
    model.eval()

    with torch.no_grad():
        with calibration_mode(model_no_gptq):
            for img, t in calib_loader:
                img = img.cuda()
                model_no_gptq(img)
        with bias_correction_mode(model_no_gptq):
            for img, t in calib_loader:
                img = img.cuda()
                model_no_gptq(img)
    print("Evaluation of PTQ model without GPTQ")
    validate(valid_loader, model_no_gptq)

    with torch.no_grad():
        with gptq_mode(model) as gptq:
            for i in tqdm(range(gptq.num_layers)):
                for img, t in calib_loader:
                    img = img.cuda()
                    model(img)
                gptq.update()
        with calibration_mode(model):
            for img, t in calib_loader:
                img = img.cuda()
                model(img)
        with bias_correction_mode(model):
            for img, t in calib_loader:
                img = img.cuda()
                model(img)
    print("Evaluation of PTQ model with GPTQ")
    validate(valid_loader, model)


if __name__ == '__main__':
    main()
