
#https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/1_simple_pg.py
#https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf

import logging
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gym
import numpy as np
from gym.envs.classic_control.cartpole import CartPoleEnv

logging.basicConfig(level=logging.INFO)

class Actor(nn.Module):

        def __init__(self, state_size, action_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size,64),
                nn.Tanh(),
                nn.Linear(64,action_size),
                nn.Softmax()
            )

        def forward(self,state):
            action_probs = self.net(state)
            return action_probs

class PolicyGradientAgent():
    
    def __init__(self, state_size, action_size):
        self.state_size =  state_size
        self.action_size = action_size


        self.actor = Actor(state_size, action_size)

        self.update_batch_size = 1000
        self.batch_experience_buffer = []

        self.episode_experience_buffer = []
        self.episode_reward_sum = 0

        self.optimizer = optim.Adam(self.actor.parameters(),lr=0.02)

    def get_action(self,state_np):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state_np).unsqueeze(0).float()
            prob_tensor = self.actor(state_tensor).squeeze(0)
            m = Categorical(prob_tensor)
            action = int(m.sample())

            logging.debug("action probs: %s, action: %s",prob_tensor,action)
        return action

    def record_experience(self,state_np,action,reward,done):
        self.episode_experience_buffer.append( (state_np,action))
        self.episode_reward_sum += reward

        if done:
            for experience_tuple in self.episode_experience_buffer:
                experience_tuple += (self.episode_reward_sum,)
                self.batch_experience_buffer.append(experience_tuple)

            self.episode_experience_buffer = []
            self.episode_reward_sum = 0

            if len(self.batch_experience_buffer) > self.update_batch_size:
                self.update_policy()

            
    
    def update_policy(self):

        #zip the list of experience tuples into a tuple of lists
        states_np, actions, rewards = zip(*self.batch_experience_buffer)
     
        #create state batch tensor
        state_tensor = torch.from_numpy(np.stack(states_np)).float()

        #create a action tensor of the integer actions chosen during the episode
        action_tensor = torch.LongTensor(actions)
        
        #create rewards batch tensor
        rewards_tensor = torch.FloatTensor(rewards)

        #Scale the rewards so that it has a mean of zero and a std of one.
        #It means that bad action become less likely as oppised to the good action just becomming more likely
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 10e-6)

        #Use the actor to get probibilites for each action. 
        #These will be that same as it predicted during the episode
        prob_tensor = self.actor(state_tensor)

        #create categorical distribution thing
        m = Categorical(prob_tensor)

        #Use the action tensor to get the log probablity of the action chosen during the episode
        log_prob_tensor = m.log_prob(action_tensor)

        #This is the pocicy gradient equation.
        #
        #The negative sign makes it gradient ascent.
        loss = -torch.mean(rewards_tensor* log_prob_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logging.debug("state: %s\n%s",     state_tensor.shape, state_tensor)
        logging.debug("probs: %s\n%s",     prob_tensor.shape, prob_tensor)
        logging.debug("actions: %s\n%s",   action_tensor.shape, action_tensor)
        logging.debug("log_prob: %s\n%s",  log_prob_tensor.shape, log_prob_tensor)
        logging.debug("log_prob test: %s\n%s",  prob_tensor.shape, torch.log(prob_tensor))
        logging.debug("rewards: %s\n%s",   rewards_tensor.shape, rewards_tensor)
        logging.debug("loss: %s\n%s",      loss.shape, loss)
        self.batch_experience_buffer = []
        

if __name__ == "__main__":
    # env = gym.make("CartPole-v0")
    env = CartPoleEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    logging.info("State Size: %i, Action Size: %i" ,state_size, action_size)

    agent = PolicyGradientAgent(state_size, action_size)

    done = True

    last_render_time = time.time()
    while True:
        
        if done:
            state_np = env.reset()
            render = (time.time() - last_render_time) > 5
            

        action = agent.get_action(state_np)

        new_state_np, reward, done, info_dict = env.step(action)

        agent.record_experience(state_np,action,reward,done)

        state_np = new_state_np

        if render:
            env.render()
            last_render_time = time.time()
            print(done)




    