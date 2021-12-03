import torch
from torch.utils.data import Dataset




class TransitionRecorder():

    def __init__(self):
        self.state_list        = []
        self.state_reward_list = []
        self.action_list       = []
        self.action_cost_list  = []
        self.next_state_list   = []

    def add_transition(self, state, state_reward, action, action_cost, next_state ):
        self.state_list.append(state)
        self.state_reward_list.append(state_reward)
        self.action_list.append(action)
        self.action_cost_list.append(action_cost)
        self.next_state_list.append(next_state)

    def get_transition_dataset(self):

        transition_dataset = TransitionDataset(
            state_list        = self.state_list,
            state_reward_list = self.state_reward_list,
            action_list       = self.action_list,
            action_cost_list  = self.action_cost_list,
            next_state_list   = self.next_state_list,
        )
        return transition_dataset

class TransitionDataset(Dataset):
    
    def __init__(self,state_list, state_reward_list, action_list, action_cost_list, next_state_list):

        self.state_tensor        = torch.tensor(state_list        , dtype=torch.float32)
        self.state_reward_tensor = torch.tensor(state_reward_list , dtype=torch.float32).unsqueeze(1)
        self.action_tensor       = torch.tensor(action_list       , dtype=torch.float32)
        self.action_cost_tensor  = torch.tensor(action_cost_list  , dtype=torch.float32).unsqueeze(1)
        self.next_state_tensor   = torch.tensor(next_state_list   , dtype=torch.float32)

    def __len__(self):
        return len(self.state_tensor)
    
    def __getitem__(self,index):

        return {
            "state_tensor"       :self.state_tensor[index],
            "state_reward_tensor":self.state_reward_tensor[index],
            "action_tensor"      :self.action_tensor[index],
            "action_cost_tensor" :self.action_cost_tensor[index],
            "next_state_tensor"  :self.next_state_tensor[index],
        }