
import torch
from torch.utils.data import Dataset

class TransitionDataset(Dataset):
    
    def __init__(self,run_transition_data):

        state_list      = []
        action_list     = []
        next_state_list = []
        reward_list     = []

        for run_data in run_transition_data:
            
            run_state_list, run_action_list, run_next_state_list, run_reward_list = zip(*run_data)
            
            state_list.extend( run_state_list )
            action_list.extend( run_action_list )
            next_state_list.extend( run_next_state_list )
            reward_list.extend( run_reward_list )

        self.state_tensor      = torch.tensor(state_list      , dtype=torch.float)
        self.action_tensor     = torch.tensor(action_list     , dtype=torch.float)
        self.next_state_tensor = torch.tensor(next_state_list , dtype=torch.float)
        self.reward_tensor     = torch.tensor(reward_list     , dtype=torch.float).unsqueeze(1)

    def __len__(self):
        return len(self.state_tensor)
    
    def __getitem__(self,index):

        return {
            "state"   : self.give_tensor[index],
            "action": self.action_tensor[index],
            "next_state" : self.next_state_tensor[index],
            "reward": self.reward_tensor[index],
        }
