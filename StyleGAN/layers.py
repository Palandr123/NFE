import torch.nn as nn
from utils import scale_weights


class ScaledLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = scale_weights(linear)

    def forward(self, x):
        return self.linear(x)
        

class ScaledConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = scale_weights(conv)

    def forward(self, input):
        return self.conv(input)