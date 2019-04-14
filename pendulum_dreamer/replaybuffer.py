
import torch
from torch.utils.data import Dataset, DataLoader
import random

class ReplayBuffer(Dataset):

    def __init__(self,max_size):
        self.replay_buffer = []
        self.max_size = max_size

    def append(self,item):
        self.replay_buffer.append(item)
        if len(self.replay_buffer) > self.max_size:
            remove = int(self.max_size * 0.1)
            self.replay_buffer = self.replay_buffer[remove:]

    def __len__(self):
        return len(self.replay_buffer)
        
    def __getitem__(self,index):
        state_tensor, action_tensor, next_state_tensor, reward_tensor = self.replay_buffer[index]

        data = {
            "state":state_tensor,
            "next_state":next_state_tensor, 
            "action":action_tensor, 
            "reward":reward_tensor}

        return data


    def get_random_batch(self,batch_size):

        #take a batch sample from the replay buffer
        batch_list = random.sample(self.replay_buffer, batch_size)
        state_batch_tensor_list, action_batch_tensor_list, reward_batch_tensor_list, next_state_batch_tensor_list = zip(*batch_list)

        #stack the batch into tensors
        state_batch_tensor = torch.stack(state_batch_tensor_list)
        action_batch_tensor = torch.stack(action_batch_tensor_list)
        reward_batch_tensor = torch.stack(reward_batch_tensor_list)
        next_state_batch_tensor = torch.stack(next_state_batch_tensor_list)
