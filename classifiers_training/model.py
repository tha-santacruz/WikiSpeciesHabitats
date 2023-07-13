import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """One hidden layer MLP with ReLU Activation"""
    def __init__(self, size_in, size_out, hidden_size=4096):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(size_in, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/4))
        self.fc3 = nn.Linear(int(hidden_size/4), int(hidden_size/16))
        self.fc4 = nn.Linear(int(hidden_size/16), size_out)
        self.activation = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(num_features=hidden_size)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.batchnorm(y)
        y = self.fc2(y)
        y = self.activation(y)
        y = self.fc3(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.fc4(y)
        return y
