#!/usr/bin/env python3

import sys
import numpy as np
import gym
from gym.envs.classic_control.pendulum import PendulumEnv
import time

import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain
from torch.utils.data import Dataset, DataLoader

import models
import replaybuffer

import logging

logging.basicConfig(level=logging.DEBUG)

class Dreamer():

    def __init__(self):
        self.replay_buffer = replaybuffer.ReplayBuffer(5000)

        self.env = PendulumEnv()
   

        observation = self.env.reset()

        self.device = torch.device("cuda")
        
        #INSTANCIATE MODELS
        state_size = 3
        action_size = 1
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
        self.state_dreamer_optimizer =  optim.SGD(self.state_dreamer.parameters(),  lr=0.01     , momentum=0.9)
        self.reward_dreamer_optimizer = optim.SGD(self.reward_dreamer.parameters(), lr=0.01     , momentum=0.9)
        self.actor_optimizer =          optim.SGD(self.actor.parameters(),          lr=0.0001   , momentum=0.9)
        self.critic_optimizer =         optim.SGD(self.critic.parameters(),         lr=0.001    , momentum=0.9)


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
        # self.env.state = np.array([np.pi,0.])
        # state_tensor, _ = self.env_step(torch.tensor([0.]).to(self.device))
        return state_tensor

    def env_step(self,action_tensor):
        """Takes a action tensor passes it to the environment"""

        #convert the action tensor to numpy array
        action_np = action_tensor.cpu().detach().numpy()

        #take an environment step and record output
        observation, reward, done, info_dict = self.env.step(action_np)

        #convert next observation and reward to tensors
        state_tensor = self.observation_to_tensor(observation)

        #shape reward so that it's between [0,1]
        min_reward = np.pi**2 + .1*self.env.max_speed**2 + .001*(self.env.max_torque**2)
        reward = reward/min_reward + 1
        reward_tensor = torch.tensor([reward]).float()

        return state_tensor, reward_tensor


    def interact_with_environment(self,n_steps):
        state_tensor = self.env_reset()
        

        #record some interations with the environment
        for i in range(n_steps):
            with torch.no_grad():
                action_tensor = self.actor(state_tensor.unsqueeze(0).to(self.device),0.1).squeeze(0).cpu()
            next_state_tensor,reward_tensor = self.env_step(action_tensor)
            transition_tuple = (state_tensor, action_tensor, next_state_tensor, reward_tensor)
            state_tensor = next_state_tensor
            self.replay_buffer.append(transition_tuple)
            self.env.render()

        
    def train_dreamers(self,n_epochs):
        train_loader = DataLoader(self.replay_buffer,batch_size=500,shuffle=True)#,num_workers=4)
        criterion = nn.MSELoss()
        

        for i_epoch in range(n_epochs):
            count = 0
            avg_state_loss = 0.0
            avg_reward_loss = 0.0
            for i_batch, data in enumerate(train_loader):
                #get batch data
                state_tensor = data["state"].to(self.device)
                next_state_tensor = data["next_state"].to(self.device)
                action_tensor = data["action"].to(self.device)
                next_reward_tensor = data["reward"].to(self.device)

                self.state_dreamer_optimizer.zero_grad()
                predicted_state_tensor = self.state_dreamer(state_tensor,action_tensor)
                state_loss = criterion(predicted_state_tensor,next_state_tensor)
                state_loss.backward(retain_graph = True)
                self.state_dreamer_optimizer.step()

                self.reward_dreamer_optimizer.zero_grad()
                predicted_reward_tensor = self.reward_dreamer(next_state_tensor)
                reward_loss = criterion(predicted_reward_tensor,next_reward_tensor)
                reward_loss.backward(retain_graph = True)
                self.reward_dreamer_optimizer.step()

                avg_state_loss += float(state_loss)
                avg_reward_loss += float(reward_loss)
                count += 1

        avg_state_loss /= count
        avg_reward_loss /= count

        return avg_state_loss, avg_reward_loss


    def train_critic(self,n_epochs):
        train_loader = DataLoader(self.replay_buffer,batch_size=500,shuffle=True)#,num_workers=4)
        criterion = nn.MSELoss()
        discount=0.99

        for i_epoch in range(n_epochs):
            count = 0
            avg_loss = 0.0
            for i_batch, data in enumerate(train_loader):
                #get batch data
                with torch.no_grad():
                    state_tensor = data["state"].to(self.device)
                    action_tensor = self.actor(state_tensor,0.1)

                    next_state_tensor = self.state_dreamer(state_tensor,action_tensor)
                    next_reward_tensor = self.reward_dreamer(next_state_tensor)
                    next_action_tensor = self.actor(state_tensor)
                    next_Q_tensor = self.critic(next_state_tensor,next_action_tensor)
                    target_Q_tensor = next_reward_tensor + discount * next_Q_tensor

                

                self.critic_optimizer.zero_grad()
                Q_tensor = self.critic(state_tensor,action_tensor)
                loss = criterion(Q_tensor,target_Q_tensor)
                loss.backward(retain_graph = True)
                self.critic_optimizer.step()

                avg_loss += float(loss)
                count += 1

                # logging.info("Critic Training: Critic Loss: %8.4f",float(loss))
        avg_loss /= count
        return avg_loss

    def train_actor(self,n_epochs):
        train_loader = DataLoader(self.replay_buffer,batch_size=500,shuffle=True)#,num_workers=4)
     
        discount=0.9

        for i_epoch in range(n_epochs):
            count = 0
            avg_loss = 0.0

            for i_batch, data in enumerate(train_loader):
                #get batch data

                state_tensor = data["state"].to(self.device)

                self.actor_optimizer.zero_grad()

                action_tensor = self.actor(state_tensor)
                Q_tensor = self.critic(state_tensor,action_tensor)

                loss = -torch.mean(Q_tensor)
                loss.backward(retain_graph = True)
                self.actor_optimizer.step()

                avg_loss += float(loss)
                count += 1

                # logging.info("Actor Training: Actor Loss: %8.4f",float(loss))
        avg_loss /= count
        return avg_loss

    def train_loop(self):
        while True:
            #Interact with environment and record to replay buffer
            self.interact_with_environment(n_steps = 500)

            #Train Dreamers
            avg_state_loss, avg_reward_loss = self.train_dreamers(n_epochs=3)

            #Train Critic
            avg_critic_loss = self.train_critic(n_epochs=3)

            #Train Actor
            avg_actor_loss = self.train_actor(n_epochs=3)

            logging.info("State Loss: %8.4f, Reward Loss: %8.4f, Critic Loss: %8.4f, Actor Loss: %8.4f,",avg_state_loss,avg_reward_loss,avg_critic_loss,avg_actor_loss)


if __name__ == "__main__":
    d = Dreamer()
    d.train_loop()
