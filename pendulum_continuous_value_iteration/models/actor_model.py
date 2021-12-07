import torch
import torch.nn as nn

class ActorModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.state_size = 3
        self.acton_size=1

        self.seq = nn.Sequential(
            nn.Linear(self.state_size,32),
            nn.SiLU(),
            nn.Linear(32,32),
            nn.SiLU(),
            nn.Linear(32,self.acton_size),
        )

    def forward(self,state):
        action = self.seq(state)

        return action