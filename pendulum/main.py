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
from smooth_noise import SmoothNoise

class RL_Trainer():

    def __init__(self):
        self.discount_factor = 0.99
        self.num_steps = 1000000
        self.reset_interval = 100
        self.update_interval = 1
        self.batch_size = 100
        self.replay_buffer_length = 1000

        self.replay_buffer = []
        self.env = PendulumEnv()
        self.noise = SmoothNoise((1,))
        self.draw_env = False

        self.actor = Actor()
        self.critic = Critic()

        self.actor_target = Actor()
        self.critic_target = Critic()

        self.actor_optimizer = optim.Adam(self.actor.parameters(),   lr=0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        self.actor_criterion = nn.MSELoss()
        self.critic_criterion = nn.MSELoss()

        self.update_target(self.actor_target , self.actor, 1.0)
        self.update_target(self.critic_target , self.critic, 1.0)

        self.env_reset()

    def update_target(self,target_model,new_model,alpha):
        with torch.no_grad():
            for target_param, new_param in zip(target_model.parameters(), new_model.parameters()):
                target_param += alpha * (new_param - target_param)

    def env_reset(self):
        state = self.env.reset()
        #scale the states between [0,1]
        state /= np.array([1,1,self.env.max_speed])
        self.state_tensor = torch.tensor(state).float()

    def env_step(self):
        with torch.no_grad():
            #get action from actor for the current state
            action_tensor = self.actor(self.state_tensor)

            #add noise to action tensor. Reduce the noise to zero at 100000 steps
            noise = max((1.0 - float(self.i_step)/100000.0),0) * self.noise.sample()

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

            #append state action reward tuple to replay_buffer
            self.replay_buffer.append((self.state_tensor, action_tensor, reward_tensor, next_state_tensor))

            #update state
            self.state_tensor = next_state_tensor

    def update_models(self):
        if len(self.replay_buffer) > self.batch_size:
            #limit the size of the replay buffer
            over_size = len(self.replay_buffer) - self.replay_buffer_length
            if over_size > 0:
                del self.replay_buffer[:over_size]

            #take a batch sample from the replay buffer
            batch_list = random.sample(self.replay_buffer, self.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch = zip(*batch_list)

            #stack the batch into tensors
            state_batch_tensor = torch.stack(state_batch)
            action_batch_tensor = torch.stack(action_batch)
            reward_batch_tensor = torch.stack(reward_batch)
            next_state_batch_tensor = torch.stack(next_state_batch)

            #Calculate target Q for updating critic
            with torch.no_grad():
                #get the predicted action for the next state
                next_action_batch_tensor = self.actor_target( next_state_batch_tensor )

                #get the predicted q value for the next state
                next_q_batch_tensor = self.critic_target( next_state_batch_tensor , next_action_batch_tensor)

                #get the target q for this state
                target_q_batch_tensor = reward_batch_tensor + self.discount_factor * next_q_batch_tensor
                # print("target_q_batch_tensor",target_q_batch_tensor)

            #Update critic
            self.critic_optimizer.zero_grad()
            q_batch_tensor = self.critic(state_batch_tensor,action_batch_tensor)
            critic_loss = self.critic_criterion(q_batch_tensor,target_q_batch_tensor)
            critic_loss.backward()
            self.critic_optimizer.step()

            #Update Actor
            self.actor_optimizer.zero_grad()
            action_batch_tensor = self.actor(state_batch_tensor)
            reward_prediction = self.critic(state_batch_tensor,action_batch_tensor)
            reward_prediction_mean = -torch.mean(reward_prediction)
            if self.draw_env:
                print(-reward_prediction_mean)
            reward_prediction_mean.backward()
            self.actor_optimizer.step()

            #slowly update the target networks
            self.update_target(self.actor_target , self.actor, 0.001)
            self.update_target(self.critic_target , self.critic, 0.001)

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
