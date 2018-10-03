#!/usr/bin/env python2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self,input_size=None):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(3,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)

    def forward(self,x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x



class Critic(nn.Module):
    def __init__(self,input_size=None):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(4,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)

    def forward(self,x_state,x_action):
        x = torch.cat((x_state,x_action),dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





if __name__ == "__main__":
    print("Hello There")
