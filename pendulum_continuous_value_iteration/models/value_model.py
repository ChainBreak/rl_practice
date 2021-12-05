import torch
import torch.nn as nn

class ValueModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.state_size = 3

        self.seq = nn.Sequential(
            nn.Linear(self.state_size,32),
            nn.SiLU(),
            nn.Linear(32,32),
            nn.SiLU(),
            nn.Linear(32,1),
        )

    def forward(self,state):
        value = self.seq(state)

        return value