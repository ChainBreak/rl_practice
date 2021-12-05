import torch
import torch.nn as nn

class TransitionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.state_size = 3
        self.action_size = 1

        self.seq = nn.Sequential(
            nn.Linear(self.state_size+self.action_size,32),
            nn.SiLU(),
            nn.Linear(32,32),
            nn.SiLU(),
            nn.Linear(32,self.state_size),
        )

    def forward(self,state,action):

        x = torch.cat([state,action],dim=1)

        next_state = self.seq(x)

        return next_state

        