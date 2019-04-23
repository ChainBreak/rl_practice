
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
    """This is our policy model. 
    It takes a state vector and returns probabilities of discrete actions"""
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
        #choose a batch size. Number of time steps used for calculating policy gradient
        self.update_batch_size = 1000

        #take note of the size of state and action
        self.state_size =  state_size
        self.action_size = action_size

        #instaciate our actor network
        self.actor = Actor(state_size, action_size)
        
        #state,action,reward buffer used to make batch
        self.batch_experience_buffer = []

        #state,action buffer used to record an episode
        self.episode_experience_buffer = []

        #accumulates the reward for a episode
        self.episode_reward_sum = 0

        #Create optimiser used for update the model
        self.optimizer = optim.Adam(self.actor.parameters(),lr=0.02)


    def get_action(self,state_np):
        #no gradients allowed
        with torch.no_grad():
            #convert the state from numpy to tensor and unsqeeze a batch dimension
            state_tensor = torch.from_numpy(state_np).unsqueeze(0).float()

            #us our actor to prdict the probabilities of each eaction
            prob_tensor = self.actor(state_tensor).squeeze(0)

            #create a categorical thingy
            m = Categorical(prob_tensor)

            #sample an action. This is just the index from the prob tensor
            action = int(m.sample())

            logging.debug("action probs: %s, action: %s",prob_tensor,action)
        return action


    def record_experience(self,state_np,action,reward,done):
        #accumulate the state action pairs for this episode
        self.episode_experience_buffer.append( (state_np,action))

        #accumulate the reward over this episode
        self.episode_reward_sum += reward

        #if the episode is done
        if done:
            #Propegate the total sum of rewards for this episode to each state action pair
            for experience_tuple in self.episode_experience_buffer:
                
                #extend the tuple to include the sum of rewards for this episode (s,a) -> (s,a,r)
                experience_tuple += (self.episode_reward_sum,)

                #append this state action reward tuple to the larger batch of experience for this update
                self.batch_experience_buffer.append(experience_tuple)

            #reset the episode acculators
            self.episode_experience_buffer = []
            self.episode_reward_sum = 0

            #if the batch experience buffer is full then lets update our policy
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

        #Scale the reward tensor so that it has a mean of zero and a std of one.
        #It means that bad action become less likely as opposed to the good action just becomming more likely
        #This seemed to realy speed up training
        rewards_tensor = (rewards_tensor - rewards_tensor.mean())# / (rewards_tensor.std() + 10e-6)

        #Use the actor to get probibilites for each action. 
        #These will be that same as it predicted during the episode
        prob_tensor = self.actor(state_tensor)

        #create categorical distribution thingy
        m = Categorical(prob_tensor)

        #Use the action tensor to get the log probablity of the action chosen during the episode
        log_prob_tensor = m.log_prob(action_tensor)

        #This is the pocicy gradient equation.
        #Basically the reward from that episode multiplyed by the log prob of that action
        #The negative sign makes it gradient ascent.
        loss = -torch.mean(rewards_tensor* log_prob_tensor)

        #Backprop and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #empty the expecence buffer
        self.batch_experience_buffer = []

        logging.debug("state: %s\n%s",     state_tensor.shape, state_tensor)
        logging.debug("probs: %s\n%s",     prob_tensor.shape, prob_tensor)
        logging.debug("actions: %s\n%s",   action_tensor.shape, action_tensor)
        logging.debug("log_prob: %s\n%s",  log_prob_tensor.shape, log_prob_tensor)
        logging.debug("log_prob test: %s\n%s",  prob_tensor.shape, torch.log(prob_tensor))
        logging.debug("rewards: %s\n%s",   rewards_tensor.shape, rewards_tensor)
        logging.debug("loss: %s\n%s",      loss.shape, loss)
        
        
if __name__ == "__main__":

    #Create environment. Note that make actually wraps the actual environment.
    #The wrapper will end an episode based on a time limit
    # env = gym.make("CartPole-v0") #state = [x,x_dot,theta,theta_dot], actions = [left_or_right] 0 = left 1 = right

    env = CartPoleEnv() #instanciate the env directly, without the wrapper

    #get the state and action size from the environement
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    #instanciate agent
    agent = PolicyGradientAgent(state_size, action_size)
    
    done = True
    last_render_time = time.time()
    episode_count = -1
    #forever
    while True:
        
        #if episode is done
        if done:
            episode_count += 1
            logging.info("Episode: %i",episode_count)
            #reset env
            state_np = env.reset()
            #set render flag based on time
            render = (time.time() - last_render_time) > 5
            
        #Get an action from our agent
        action = agent.get_action(state_np)

        #take a step in the environment
        new_state_np, reward, done, info_dict = env.step(action)

        #let the agent record the experience
        agent.record_experience(state_np,action,reward,done)

        #update the state
        state_np = new_state_np

        #if render is set then render this episode
        if render:
            env.render()
            #update the time so that its not rendered for a while
            last_render_time = time.time()





    