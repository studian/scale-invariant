import torch
import math
from torch.optim.optimizer import Optimizer, required


class InvariantSGD(Optimizer):

    def __init__(self, params, lr=required):
        defaults = dict(lr=lr)
        super(InvariantSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(InvariantSGD, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)

            # Normalize the linear classifier
            p = group['params'][-1]
            p.data.div_(torch.norm(p.data)).mul_(math.sqrt(p.data.shape[0]))
        return loss
