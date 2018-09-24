#!/usr/bin/env python2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Actor(nn.Module):
    def __init__(self,input_size=None):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,1,bias=False)

        self.new_noise(1.0)

    def forward(self,x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

    def new_noise(self,std):
        m = torch.distributions.Normal(torch.Tensor([0.0]), torch.Tensor([std]))
        self.rand_action = Variable(m.sample())
        

    def get_action(self,state):
        x = torch.from_numpy(state).unsqueeze(dim=0).float()
        x = Variable(x)
        x = self(x)
        # print(x)
        x += self.rand_action
        # print("after rand",x)
        action = x.data.numpy()[0]
        return action


class Critic(nn.Module):
    def __init__(self,input_size=None):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,1)

    def forward(self,x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x





if __name__ == "__main__":
    print("Hello There")
