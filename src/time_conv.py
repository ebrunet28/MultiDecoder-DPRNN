## for generating speaker embeddings

import torch
class Conv1DBlock(torch.nn.Module):
    def __init__(self, in_channels, N, kernel_size, padding = None, max_pool = 1):
        super().__init__()
        assert kernel_size%2 == 1 , "cannot have even filter length"
        if padding == None:
            padding = kernel_size//2
        self.bn = torch.nn.BatchNorm1d(in_channels)
        self.conv = torch.nn.Conv1d(in_channels, N, kernel_size, padding = padding)
        self.act = torch.nn.ReLU()
        self.mp = torch.nn.MaxPool1d(max_pool)
    def forward(self, X):
        return self.mp(self.act(self.conv(self.bn(X))))

class TimeConv(torch.nn.Module):
    def __init__(self):
        self.block1 = Conv1DBlock(1, 60, 30, )