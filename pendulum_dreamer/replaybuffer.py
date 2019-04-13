
import torch
import torch.utils.data
import random

class ReplayBuffer(torch.utils.data.Dataset):

    def __init__(self):
        self.replay_buffer = []

    def append(self,item):
        self.replay_buffer.append(item)

    def __len__(self):
        return len(self.replay_buffer)
        
    def __getitem__(self,index):
        pass
    def get_random_batch(self,batch_size):

        #take a batch sample from the replay buffer
        batch_list = random.sample(self.replay_buffer, batch_size)
        state_batch_tensor_list, action_batch_tensor_list, reward_batch_tensor_list, next_state_batch_tensor_list = zip(*batch_list)

        #stack the batch into tensors
        state_batch_tensor = torch.stack(state_batch_tensor_list)
        action_batch_tensor = torch.stack(action_batch_tensor_list)
        reward_batch_tensor = torch.stack(reward_batch_tensor_list)
        next_state_batch_tensor = torch.stack(next_state_batch_tensor_list)
