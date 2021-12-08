
from torch.utils.data import Dataset
from typing import List

class TransitionDataset(Dataset):
    
    def __init__(self, transition_data: dict, keys_to_use: List[str]):

        # dict of tensors e.g. {"state": torch.tensor([1,2,3])}
        self.transition_data = { key: transition_data[key] for key in keys_to_use}

        self.length = self.get_number_of_transions_from_tensors(self.transition_data)


    def get_number_of_transions_from_tensors(self,transition_data):
        first_key = list(self.transition_data.keys())[0]
        return len(self.transition_data[first_key])


    def __len__(self):
        return self.length
        
            
    def __getitem__(self,index):
        return { key : tensor[index] for key, tensor in self.transition_data.items() }