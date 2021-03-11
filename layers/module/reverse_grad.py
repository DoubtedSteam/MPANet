from torch import nn
from torch.autograd import Function


class ReverseGradFunction(Function):

    @staticmethod
    def forward(ctx, data, alpha=1.0):
        ctx.alpha = alpha
        return data

    @staticmethod
    def backward(ctx, grad_outputs):
        grad = None

        if ctx.needs_input_grad[0]:
            grad = -ctx.alpha * grad_outputs

        return grad, None


class ReverseGrad(nn.Module):
    def __init__(self):
        super(ReverseGrad, self).__init__()

    def forward(self, x, alpha=1.0):
        return ReverseGradFunction.apply(x, alpha)
