#!/usr/bin/env python2

import sys
import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv

import torch
import torch.nn as nn
import torch.optim as optim

from models import Actor, Model, Loss

class Dreamer():

    def __init__(self):
        self.replay_buffer = []

        self.env = PendulumEnv()

        self.actor = Actor()
        self.model = Model()
        self.loss = Loss()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def env_reset(self):
        self.env.reset()

    def env_step(self):
        return self.env.step()

    def condition_state(self,state):
        with torch.no_grad():
            state = torch.tensor(state).float()
            state /= torch.tensor([1.,1.,self.env.max_speed])
        return state

    def train_loop(self):

        self.env_reset()
        while True:
            state,_,_,_ = self.env.step(np.array([1.0]))
            # print(state)
            state = self.condition_state(state)
            print(self.loss(state))
            self.env.render()



if __name__ == "__main__":
    d = Dreamer()
    d.train_loop()
