#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Takes in a state and predicts a action"""
    def __init__(self,state_size,action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,action_size)


    def forward(self,x,noise_amplitude=0):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        noise = torch.zeros_like(x)
        noise = torch.normal(noise)

        x = x + noise * noise_amplitude
        return x


class Critic(nn.Module):
    """Takes in a state and action and predicts Q value"""
    def __init__(self,state_size,action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,1)

    def forward(self,state,action):
        x = torch.cat((state,action),dim=1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


class StateDreamer(nn.Module):
    """Takes in a state and action and predicts the next state"""
    def __init__(self,state_size,action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,state_size)

    def forward(self,state,action):
        x = torch.cat((state,action),dim=1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x + state
        return x

class RewardDreamer(nn.Module):
    """Takes in a state and predicts the reward for being in that state"""
    def __init__(self,state_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x






if __name__ == "__main__":
    print("Hello There")
