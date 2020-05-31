# -------------------------------------------------------------------------------------------
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        input_size = 28*28*1
        hidden_sizes = [128, 64]
        output_size = 10
        self.layer = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.layer(x)
        return x