import torch as T
import torch.nn as nn
import torch.nn.functional as F

class CustomBatchNorm2D(nn.Module):
    def __init__(self, dim, lr=0.99):
        super().__init__()
        self.beta = nn.Parameter(T.ones((1, dim, 1, 1)), requires_grad=True)
        self.alpha = nn.Parameter(T.zeros((1, dim, 1, 1)), requires_grad=True)

        self.mean_history = nn.Parameter(T.zeros(dim), requires_grad=False)
        self.std_history = nn.Parameter(T.ones(dim), requires_grad=False)

        self.lr = lr

    def forward(self, x):
        if self.training:
            mean = x.sum((0, 2, 3), keepdim=True)/(x.shape[0]*x.shape[2]*x.shape[3])
            x_centered = x - mean
            std = T.sqrt((x_centered**2).sum((0, 2, 3), keepdim=True)/(x.shape[0]*x.shape[2]*x.shape[3]))
            self.mean_history.data = self.mean_history.data*self.lr + mean.detach()*(1 - self.lr)
            self.std_history.data = self.std_history.data*self.lr + std.detach()*(1 - self.lr)
        else:
            mean = self.mean_history
            std = self.std_history
            x_centered = x - mean

        normed_x = (x_centered)/(std + 0.000001)
        scaled_x = normed_x * self.beta + self.alpha

        return scaled_x

