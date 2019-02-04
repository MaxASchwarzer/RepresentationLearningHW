import torch as T
import torch.nn as nn
import torch.nn.functional as F

class CustomBatchNorm2D(nn.Module):
    def __init__(self, dim, lr=0.99):
        super().__init__()
        self.beta = nn.Parameter(T.ones(dim), requires_grad=True)
        self.alpha = nn.Parameter(T.zeros(dim), requires_grad=True)

        self.mean_history = nn.Parameter(T.zeros(dim), requires_grad=False)
        self.std_history = nn.Parameter(T.ones(dim), requires_grad=False)

        self.lr = lr

    def forward(self, x):
        flat_x = x.permute(0, 2, 3, 1).contiguous().view(-1, x.shape[1])
        if self.training:
            mean = flat_x.mean(0)
            std = flat_x.std(0)
            self.mean_history.data = self.mean_history.data*self.lr + mean.detach()*(1 - self.lr)
            self.std_history.data = self.std_history.data*self.lr + std.detach()*(1 - self.lr)
        else:
            mean = self.mean_history
            std = self.std_history

        normed_x = (flat_x - mean)/(std + 0.000001)
        scaled_x = normed_x * self.beta + self.alpha

        reshaped_x = scaled_x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[1]).contiguous().permute(0, 3, 1, 2)

        return reshaped_x

