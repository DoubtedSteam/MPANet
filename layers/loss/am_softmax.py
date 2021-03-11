import torch

import torch.nn as nn
import torch.nn.functional as F


class AMSoftmaxLoss(nn.Module):
    def __init__(self, scale, margin, weight=None, ignore_index=-100, reduction='mean'):
        super(AMSoftmaxLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.scale = scale
        self.margin = margin

    def forward(self, x, y):
        y_onehot = torch.zeros_like(x, device=x.device)
        y_onehot.scatter_(1, y.data.view(-1, 1), self.margin)

        out = self.scale * (x - y_onehot)
        loss = F.cross_entropy(out, y, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss
