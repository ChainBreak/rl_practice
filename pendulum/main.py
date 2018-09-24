#!/usr/bin/env python2

import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv
from models import Actor, Critic
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    discount_factor = 0.99
    num_episodes = 1000
    num_steps = 50

    env = PendulumEnv()
    actor = Actor(input_size=3)
    critic = Critic(input_size=4)

    actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.01)

    actor_criterion = nn.MSELoss()
    critic_criterion = nn.MSELoss()

    loss_list = []
    avg_reward_list = []
    for i_episode in range(num_episodes):
        #initialise lists for recording
        action_list = []
        state_list = []
        reward_list = []

        for i_batch in range(50):
            state = env.reset()
            avg_reward = 0

            for i_step in range(num_steps):

                #interact with environment
                actor.new_noise((num_episodes - i_episode)/float(num_episodes))
                action = actor.get_action(state)
                new_state,reward,done,_ = env.step(action)
                avg_reward += reward
                reward = np.array([reward])

                #record what happend
                state_list.append(state)
                action_list.append(action)
                # reward_list.append(reward)

                #update state
                state = new_state
                if i_episode > (num_episodes - 2):
                    env.render()
            avg_reward /= i_step

            for i in range(num_steps):
                reward_list.append(avg_reward)

        avg_reward_list.append(avg_reward)
        print(avg_reward)

        #propegate reward backward
        # plt.plot(reward_list)
        # future_reward = reward_list[-1]
        # for i in range(len(reward_list)-2,-1,-1):
        #     reward_list[i] += discount_factor * (future_reward - reward_list[i])
        #     future_reward = reward_list[i]

        # for i in range(len(reward_list)-1,-1,-1):
        #     reward_list[i] = np.array([avg_reward])
        # plt.plot(reward_list)
        # plt.show()

        # for a in reward_list:
        #     print(type(a))
        #     print(a.dtype)
        #     print(a.shape)


        #turn lists into tensor batches
        state_tensor = torch.from_numpy(np.stack(state_list)).float()
        action_tensor = torch.from_numpy(np.stack(action_list)).float()
        state_action_tensor = torch.cat((state_tensor,action_tensor),dim=1).float()
        reward_tensor = torch.from_numpy(np.stack(reward_list)).float()

        state_tensor = torch.autograd.Variable(state_tensor)
        action_tensor = torch.autograd.Variable(action_tensor)
        state_action_tensor = torch.autograd.Variable(state_action_tensor)
        reward_tensor = torch.autograd.Variable(reward_tensor)



        #update Critic
        critic_optimizer.zero_grad()
        reward_prediction = critic(state_action_tensor)
        critic_loss = critic_criterion(reward_prediction,reward_tensor)
        loss_list.append(float(critic_loss.data))
        # print(float(critic_loss.data))
        critic_loss.backward()
        critic_optimizer.step()


        #update actor
        if i_episode > 10:
            actor_optimizer.zero_grad()
            action_tensor = actor(state_tensor)
            state_action_tensor = torch.cat((state_tensor,action_tensor),dim=1).float()
            reward_prediction = -torch.mean(critic(state_action_tensor))
            reward_prediction.backward()
            actor_optimizer.step()


    plt.plot(loss_list)
    plt.plot(avg_reward_list)
    plt.show()
