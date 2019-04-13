#!/usr/bin/env python3

import sys
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain

import models
import replaybuffer

class Dreamer():

    def __init__(self):
        self.replay_buffer = replaybuffer.ReplayBuffer()

        self.env = gym.make('Pendulum-v0')

        state_size = 3
        action_size = 1

        self.device = torch.device("cuda")
        
        #instanciate models
        self.state_dreamer = models.StateDreamer(state_size,action_size)
        self.reward_dreamer = models.RewardDreamer(state_size)
        self.actor = models.Actor(state_size,action_size)
        self.critic = models.Critic(state_size,action_size)

        #put models on device
        self.state_dreamer.to(self.device)
        self.reward_dreamer.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)

        #create optimiser for each model
        self.state_dreamer_optimizer = optim.Adam(self.state_dreamer.parameters(), lr=0.001)
        self.reward_dreamer_optimizer = optim.Adam(self.reward_dreamer.parameters(), lr=0.001)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)


    def observation_to_tensor(self,observation):
        """Takes a numpy observation from the environment and scales and converts it to a tensor"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(observation).float()
            state_tensor /= torch.tensor([1.,1.,self.env.max_speed])
        return state_tensor

    def env_reset(self):
        """Resets the environment and converts to observation to a tensor"""
        observation = self.env.reset()
        state_tensor = self.observation_to_tensor(observation)
        return state_tensor

    def env_step(self,action_tensor):
        """Takes a action tensor passes it to the environment"""

        #convert the action tensor to numpy array
        action_np = action_tensor.cpu().detach().numpy()

        #take an environment step and record output
        observation, reward, done, info_dict = self.env.step(action_np)

        #convert next observation and reward to tensors
        state_tensor = self.observation_to_tensor(observation)
        reward_tensor = torch.tensor(reward).float()

        return state_tensor, reward_tensor

        

    def train_loop(self):

        state_tensor = self.env_reset()
        while True:
 
            #record some interations with the environment
            for i in range(100):
                action_tensor = self.actor(state_tensor.unsqueeze(0).to(device=self.device)).squeeze(0)
                last_state_tensor = state_tensor
                state_tensor,reward_tensor = self.env_step(action_tensor)
                transition_tuple = (last_state_tensor, action_tensor, state_tensor, reward_tensor)
                self.replay_buffer.append(transition_tuple)
                self.env.render()

            print(len(self.replay_buffer))
       



if __name__ == "__main__":
    d = Dreamer()
    d.train_loop()
