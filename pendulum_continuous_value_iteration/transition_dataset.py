import torch
from torch.utils.data import Dataset
import typing
from typing import Dict



class TransitionRecorder():

    def __init__(self):
        
        self.data_dict = {
            "state"        : [],
            "state_reward" : [],
            "action"       : [],
            "action_cost"  : [],
            "next_state"   : [],
        }

    def add_transition(self, state, state_reward, action, action_cost, next_state ):
        self.data_dict["state"].append(state)
        self.data_dict["state_reward"].append(state_reward)
        self.data_dict["action"].append(action)
        self.data_dict["action_cost"].append(action_cost)
        self.data_dict["next_state"].append(next_state)

    def get_transition_dataset(self):

        transition_dataset = TransitionDataset( self.data_dict )
  
        return transition_dataset

    



class TransitionDataset(Dataset):
    
    def __init__(self, transition_data: dict):

        self.transition_tensors = self.create_dict_of_tensors_from_transition_data(transition_data)
        

    def create_dict_of_tensors_from_transition_data(self,transition_data: dict):
        transition_tensors = {}

        for name, data in transition_data.items():

            new_tensor = torch.tensor(data, dtype=torch.float32)

            if len(new_tensor.shape)<=1:
                new_tensor = new_tensor.unsqueeze(1)

            transition_tensors[name] = new_tensor

        return transition_tensors

    def get_transition_tensors_dict(self):
        return self.transition_tensors


    def __len__(self):
        first_key = self.transition_tensors.keys()[0]

        return len(self.transition_tensor[first_key])
    
    def __getitem__(self,index):
        return { name : tensor[index] for name, tensor in self.transition_tensors.items() }