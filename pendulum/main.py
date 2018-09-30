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
from tensor_env import TensorEnv





class RL_Trainer():

    def __init__(self):
        self.discount_factor = 0.99
        self.num_steps = 1000000
        self.reset_interval = 100
        self.update_interval = 1
        self.batch_size = 32
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
        # self.env.state = np.array([0, 0])
        # self.state_tensor = torch.tensor(self.env._get_obs()).float()
        state = self.env.reset()
        state /= np.array([1,1,self.env.max_speed])
        self.state_tensor = torch.tensor(state).float()

    def env_step(self):
        with torch.no_grad():
            #get action from actor for the current state
            action_tensor = self.actor(self.state_tensor)

            #add noise to action tensor
            noise = (1.0 - float(self.i_step)/1000000.0) * self.noise.sample()

            # if not self.draw_env:
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
            #limit the size of the replay replay
            over_size = len(self.replay_buffer) - self.replay_buffer_length
            if over_size > 0:
                del self.replay_buffer[:over_size]


            #take a batch sample from the replay buffer
            batch_list = random.sample(self.replay_buffer, self.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch = zip(*batch_list)
            # print("\n\n")
            # print("State",state_batch)
            # print("Action",action_batch)
            # print("Reward",reward_batch)
            # print("Next State",next_state_batch)

            #stack the batch into tensors
            state_batch_tensor = torch.stack(state_batch)
            action_batch_tensor = torch.stack(action_batch)
            reward_batch_tensor = torch.stack(reward_batch)
            next_state_batch_tensor = torch.stack(next_state_batch)

            # print("State",state_batch_tensor)
            # print("Action",action_batch_tensor)
            # print("Reward",reward_batch_tensor)
            # print("Next State",next_state_batch_tensor)

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
            # print(float(critic_loss.data))
            critic_loss.backward()
            self.critic_optimizer.step()

            # if self.draw_env:
            #     print("Critic ############################################################")
            #     for p in self.critic.parameters():
            #         print(p.grad)


            #Update Actor
            self.actor_optimizer.zero_grad()
            action_batch_tensor = self.actor(state_batch_tensor)
            reward_prediction = self.critic(state_batch_tensor,action_batch_tensor)
            reward_prediction_mean = -torch.mean(reward_prediction)
            if self.draw_env:
                print(-reward_prediction_mean)
            reward_prediction_mean.backward()
            self.actor_optimizer.step()

            # if self.draw_env:
            #     print("Actor ############################################################")
            #     for p in self.actor.parameters():
            #         print(p.grad)


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
            if self.i_step % self.update_interval == 0:
                self.update_models()


if __name__ == "__main__":
    print("Hello There")
    rlt = RL_Trainer()
    rlt.train_loop()
    # for i_step in range(num_steps):
    #
    #
    #     avg_reward = 0
    #
    #     # for i_step in range(num_steps):
    #
    #         # if i_episode > (num_episodes - 5):
    #         #     env.render()
    #     avg_reward /= i_step
    #     avg_reward_list.append(avg_reward)
    #     print(avg_reward)
    #
    #     #propegate reward backward
    #     plt.plot(reward_list)
    #     future_reward = reward_list[-1]
    #     for i in range(len(reward_list)-2,-1,-1):
    #         reward_list[i] += discount_factor * (future_reward - reward_list[i])
    #         future_reward = reward_list[i]
    #     plt.plot(reward_list)
    #     plt.show()
    #
    #     # for a in reward_list:
    #     #     print(type(a))
    #     #     print(a.dtype)
    #     #     print(a.shape)
    #
    #
    #     #turn lists into tensor batches
    #     state_tensor = torch.from_numpy(np.stack(state_list)).float()
    #     action_tensor = torch.from_numpy(np.stack(action_list)).float()
    #     state_action_tensor = torch.cat((state_tensor,action_tensor),dim=1).float()
    #     reward_tensor = torch.from_numpy(np.stack(reward_list)).float()
    #
    #     # state_tensor = torch.autograd.Variable(state_tensor)
    #     # action_tensor = torch.autograd.Variable(action_tensor)
    #     # state_action_tensor = torch.autograd.Variable(state_action_tensor)
    #     # reward_tensor = torch.autograd.Variable(reward_tensor)
    #
    #
    #
    #     #update Critic
    #     critic_optimizer.zero_grad()
    #     reward_prediction = critic(state_action_tensor)
    #     critic_loss = critic_criterion(reward_prediction,reward_tensor)
    #     loss_list.append(float(critic_loss.data))
    #     # print(float(critic_loss.data))
    #     critic_loss.backward()
    #     critic_optimizer.step()
    #
    #
    #     #update actor
    #     if i_episode > 500:
    #         actor_optimizer.zero_grad()
    #         action_tensor = actor(state_tensor)
    #         state_action_tensor = torch.cat((state_tensor,action_tensor),dim=1).float()
    #         reward_prediction = -torch.mean(critic(state_action_tensor))
    #         reward_prediction.backward()
    #         actor_optimizer.step()
    #
    #
    # plt.plot(loss_list)
    # plt.plot(avg_reward_list)
    # plt.show()
