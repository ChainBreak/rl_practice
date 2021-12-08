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


    def get_dict_of_tensors(self):

        transition_tensors = {}

        for name, data in self.data_dict.items():

            new_tensor = torch.tensor(data, dtype=torch.float32)

            if len(new_tensor.shape)<=1:
                new_tensor = new_tensor.unsqueeze(1)

            transition_tensors[name] = new_tensor

        return transition_tensors

    
