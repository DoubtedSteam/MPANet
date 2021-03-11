import torch.nn as nn
# import torch.softmax as softmax
from torch.nn import functional as F

class js_div:
    def __init__(self):
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')

    def __call__(self, p_output, q_output, get_softmax=True):
        """
        Function that measures JS divergence between target and output logits:
        """
        if get_softmax:
            p_output = F.softmax(p_output, 1)
            q_output = F.softmax(q_output, 1)
        log_mean_output = ((p_output + q_output) / 2).log()
        return (self.KLDivLoss(log_mean_output, p_output) + self.KLDivLoss(log_mean_output, q_output))/2