#!/usr/bin/env python2

import sys
import select
import numpy as np
import time
import random
from gym.envs.classic_control.pendulum import PendulumEnv
from models import Actor, Critic
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from smooth_noise import SmoothNoise

class RL_Trainer():

    def __init__(self):

        self.num_steps = 1000000
        self.reset_interval = 500
        self.update_interval = 1
        self.batch_size = 32
        self.replay_buffer_length = 2000
        self.horizon_buffer_length = 100

        #Calculate discount factor so that the horizon ends within the horizon_buffer
        self.discount_factor = 0.99# np.power(0.01,1.0/float(self.horizon_buffer_length))

        print(self.discount_factor)

        self.horizon_buffer = []
        self.replay_buffer = []
        self.env = PendulumEnv()
        self.noise = SmoothNoise((1,))
        self.draw_env = False

        self.actor = Actor()
        self.critic = Critic()

        self.actor_optimizer = optim.Adam(self.actor.parameters(),   lr=0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        self.actor_criterion = nn.MSELoss()
        self.critic_criterion = nn.MSELoss()


        self.env_reset()

    def env_reset(self):
        state = self.env.reset()
        #scale the states between [0,1]
        state /= np.array([1,1,self.env.max_speed])
        self.state_tensor = torch.tensor(state).float()
        del self.horizon_buffer[:]

    def env_step(self):
        with torch.no_grad():
            #get action from actor for the current state
            action_tensor = self.actor(self.state_tensor)

            #add noise to action tensor. Reduce the noise to zero at 100000 steps
            noise = max((1.0 - float(self.i_step)/50000.0),0) * self.noise.sample()

            action_tensor += noise

            #interact with the environment to get next state and reward
            next_state_np,reward_np,done,_ = self.env.step(action_tensor.cpu().numpy())
            next_state_np /= np.array([1,1,self.env.max_speed])

            #draw environment if enabled
            if self.draw_env:
                self.env.render()

            #shape reward so that it's between [0,1]
            min_reward = np.pi**2 + .1*self.env.max_speed**2 + .001*(self.env.max_torque**2)
            reward_np = reward_np/min_reward + 1

            if self.draw_env:
                print("state",self.state_tensor)
                print("noise", noise)
                print("action",action_tensor)
                print("reward",reward_np)


            #convert states and rewards to tensors
            next_state_tensor = torch.tensor(next_state_np).float()
            reward_tensor = torch.tensor([reward_np]).float()

            #append state action reward tuple to horizon_buffer
            self.add_to_horizon_buffer((self.state_tensor, action_tensor, reward_tensor))

            #update state
            self.state_tensor = next_state_tensor

    def update_models(self):
        if len(self.replay_buffer) > self.batch_size:

            states, actions, rewards = zip(*self.replay_buffer)

            reward_weight_np = np.array(rewards)
            reward_weight_np -= np.min(reward_weight_np)
            reward_weight_np /= np.sum(reward_weight_np)
            # print(reward_weight_np)
            sample_indexes = np.random.choice(len(reward_weight_np),self.batch_size,replace=True, p = reward_weight_np)



            state_batch = [states[i] for i in sample_indexes]
            action_batch = [actions[i] for i in sample_indexes]

            #stack the batch into tensor
            state_batch_tensor = torch.stack(state_batch)
            action_batch_tensor = torch.stack(action_batch)

            #Update Actor
            self.actor_optimizer.zero_grad()

            prediction_action_batch_tensor = self.actor(state_batch_tensor)
            actor_loss = self.actor_criterion(prediction_action_batch_tensor,action_batch_tensor)
            actor_loss.backward()

            # for p in self.actor.parameters():
            #     print(p.grad)
            self.actor_optimizer.step()


    def add_to_horizon_buffer(self,experience_tuple):
        self.horizon_buffer.insert(0,experience_tuple)

        if len(self.horizon_buffer) > self.horizon_buffer_length:
            discounted_reward_sum = 0.0

            for state,action,reward in self.horizon_buffer:
                discounted_reward_sum = reward + self.discount_factor * discounted_reward_sum
            print(discounted_reward_sum)

            del self.horizon_buffer[-1]

            self.replay_buffer.insert(0,(state,action,discounted_reward_sum))

        if len(self.replay_buffer) > self.replay_buffer_length:
            del self.replay_buffer[-1]

        print(len(self.horizon_buffer),len(self.replay_buffer))





    def enter_pressed(self):
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            return True
        return False

    def train_loop(self):
        self.i_step = 0
        while True:
            self.i_step += 1

            #press enter to toggle environment drawing
            if self.enter_pressed():
                self.draw_env = not self.draw_env
                if self.draw_env:
                    self.env_reset()
                print("Draw Enabled: %i" % self.draw_env)

            if self.draw_env:
                print(self.i_step)

            #reset enviroment every reset log_interval
            if self.i_step % self.reset_interval == 0 and not self.draw_env:
                self.env_reset()

            #take a step in the enviroment and append to replay buffer
            self.env_step()

            #update the actor critic models every update interval
            # # if self.i_step % self.update_interval == 0:
            # for i in range(3):
            self.update_models()


if __name__ == "__main__":
    print("Hello There")
    rlt = RL_Trainer()
    rlt.train_loop()
