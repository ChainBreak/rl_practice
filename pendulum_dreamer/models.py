#!/usr/bin/env python2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(3,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(3,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,3)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

class Loss(nn.Module):
    def __init__(self,input_size=None):
        super(Loss, self).__init__()
        self.target = torch.tensor([1.0, 0.0, 0.0]).float()

    def forward(self,x):
        error = torch.sum((self.target - x)**2)

        return error





if __name__ == "__main__":
    print("Hello There")
