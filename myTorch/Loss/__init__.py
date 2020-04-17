import warnings
import torch


def _relative_difference(input, target):
    numerator = 2.0 * torch.abs(input - target)
    denominator = input + target
    return numerator/denominator


def _relative_percentage_difference(input, target):
    return _relative_difference(input, target) * 100


def _accuracy(input, target, eps=1e-7):
    error = torch.clamp(
        torch.abs(target-input)/torch.abs(target+eps),
        0,1
    )
    return (1-error) * 100

import numpy as np

def _incompressibleConstantContinuity3d(pred, truth, domain):
    # for the internal field calcualte the distances
    dx = domain[:,0,1:-1,2:,1:-1] - domain[:,0,1:-1,1:-1,1:-1] + 1e-7
    dy = domain[:,1,2:,1:-1,1:-1] - domain[:,1,1:-1,1:-1,1:-1] + 1e-7
    dz = domain[:,2,1:-1,1:-1,2:] - domain[:,2,1:-1,1:-1,1:-1] + 1e-7
    # now solve the derivative
    dudx = (pred[:,0,1:-1,2:,1:-1] - pred[:,0,1:-1,1:-1,1:-1])/dx
    dudy = (pred[:,1,2:,1:-1,1:-1] - pred[:,1,1:-1,1:-1,1:-1])/dy
    dudz = (pred[:,2,1:-1,1:-1,2:] - pred[:,2,1:-1,1:-1,1:-1])/dz
    return torch.abs(dudx + dudy + dudz + 1e-5)

def _noSlip3d(pred, truth, domain):

    lr = torch.abs(pred[:,:,0,:,:]) + torch.abs(pred[:,:,-1,:,:])
    tb = torch.abs(pred[:,:,:,0,:]) + torch.abs(pred[:,:,:,-1,:])

    return lr.mean() + tb.mean()


class _loss_template:

    def __init__(self, function):
        self.function = function

    def __call__(self, input, target, reduction='sum'):
        if not(target.size() == input.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the input size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(
                    target.size(), input.size()),
                stacklevel=2)

        ret = self.function(input, target)
        if reduction is not None:
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)

        return ret

class _loss_templateCFD:

    def __init__(self, function):
        self.function = function

    def __call__(self, input, target, domain, reduction='sum'):
        if not(target.size() == input.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the input size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(
                    target.size(), input.size()),
                stacklevel=2)

        ret = self.function(input, target, domain)
        if reduction is not None:
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)

        return ret



relative_difference = _loss_template(_relative_difference)
relative_percentage_difference = _loss_template(
    _relative_percentage_difference)
accuracy = _loss_template(_accuracy)
incompressibleConstantContinuity3d = _loss_templateCFD(_incompressibleConstantContinuity3d)
noSlip3d = _loss_templateCFD(_noSlip3d)

class _Loss(torch.nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(_Loss, self).__init__()
        self.reduction = reduction


class RelativeDifference(_Loss):

    def forward(self, input, target):
        return relative_difference(input, target, reduction=self.reduction)


class RelativePercentageDifference(RelativeDifference):

    def forward(self, input, target):
        return super(RelativePercentageDifference, self).forward(
            input, target
        )


class Accuracy(_Loss):
    def forward(self, input, target):
        return accuracy(input, target, reduction=self.reduction)


class ContinuityNoSlip(_Loss):
    def forward(self, input, target, domain):
        l1 = incompressibleConstantContinuity3d(input, target, domain, reduction='mean')
        l2 = noSlip3d(input, target, domain, reduction='mean')
        l3 = torch.nn.functional.l1_loss(
            input, target, reduction=self.reduction
        )
        return  l1 + l2 + l3


__all__ = [
    'relative_difference', 'relative_percentage_difference', 'accuracy',
    'RelativeDifference', 'RelativePercentageDifference', 'Accuracy'
]