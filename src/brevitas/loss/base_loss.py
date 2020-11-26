from abc import ABCMeta, abstractmethod

from torch import nn


class SimpleBaseLoss(object):
    __metaclass__ = ABCMeta

    def __init__(self, model, reg_coeff):
        self.model: nn.Module = model
        self.reg_coeff = reg_coeff
        self.tot_loss: float = 0.0
        self.register_hooks()

    @abstractmethod
    def register_hooks(self):
        pass

    def zero_accumulated_values(self):
        del self.tot_loss
        self.tot_loss = 0.0

    def retrieve(self):
        return self.tot_loss * self.reg_coeff

    def log(self):
        return self.tot_loss.detach().clone() * self.reg_coeff