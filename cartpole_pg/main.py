
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

logging.basicConfig(level=logging.DEBUG)

class Actor(nn.Module):

        def __init__(self, state_size, action_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size,32),
                nn.Tanh(),
                nn.Linear(32,action_size),
                nn.Sigmoid()
            )

        def forward(self,state):
            action_probs = self.net(state)
            return action_probs

class PolicyGradientAgent():
    
    def __init__(self, state_size, action_size):
        self.state_size =  state_size
        self.action_size = action_size


        self.actor = Actor(state_size, action_size)

        self.update_batch_size = 10
        self.batch_experience_buffer = []

        self.episode_experience_buffer = []
        self.episode_reward_sum = 0

        self.optimizer = optim.Adam(self.actor.parameters(),lr=0.02)

    def get_action(self,state_np):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state_np).unsqueeze(0).float()
            action_tensor = self.actor(state_tensor).squeeze(0)
            logging.debug("action_tensor: %s",action_tensor)
            action = int(torch.multinomial(action_tensor,num_samples=1))

            logging.debug("action probs: %s, action: %s",action_tensor,action)
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
        batch_size = len(self.batch_experience_buffer)
        #convert experience tuples into batch
        states_np, actions, rewards = zip(*self.batch_experience_buffer)
     
        #create state batch tensor
        state_tensor = torch.from_numpy(np.stack(states_np)).float()

        #create one-hot vectors from each action int
        action_indicies = torch.LongTensor(actions).unsqueeze(1)
        action_mask = torch.FloatTensor(batch_size,self.action_size).scatter_(1,action_indicies,1)

        #create rewards batch tensor
        rewards_tensor = torch.FloatTensor(rewards)

        action_batch = self.actor(state_tensor)

        log_prob_tensor = torch.sum(action_mask * F.log_softmax(action_batch), dim=1 )

        loss = - torch.mean(rewards_tensor* log_prob_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logging.debug("state: %s",state_tensor)
        logging.debug("actions: %s",action_mask)
        logging.debug("rewards: %s",rewards_tensor)
        logging.debug("log_prob: %s",log_prob_tensor)
        logging.debug("loss %s",loss)
        self.batch_experience_buffer = []
        

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    logging.info("State Size: %i, Action Size: %i" ,state_size, action_size)

    agent = PolicyGradientAgent(state_size, action_size)

    done = True
    episode_counter = 0
    while True:
        if done:
            state_np = env.reset()
            episode_counter += 1

        action = agent.get_action(state_np)

        state_np, reward, done, info_dict = env.step(action)

        agent.record_experience(state_np,action,reward,done)

        if episode_counter % 50 == 1:
            env.render()




    