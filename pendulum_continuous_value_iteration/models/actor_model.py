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


import torch
import torch.nn as nn
import pytorch_lightning as pl


        
class LitActorModel(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        p = self.hparams

        self.learning_rate = p.learning_rate

        self.state_size = 3
        self.action_size = 1

        self.criterion = torch.nn.MSELoss()

        self.seq = nn.Sequential(
            nn.Linear(self.state_size,32),
            nn.SiLU(),
            nn.Linear(32,32),
            nn.SiLU(),
            nn.Linear(32,self.action_size),
        )

    

    def configure_optimizers(self):
        p = self.hparams
        optimizer = torch.optim.Adam( self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p.cosine_annealing_t_max )
        return [optimizer],[scheduler]


    def forward(self,state):
        action = self.seq(state)
        return action

    def forward_step(self, batch, batch_id):
        state = batch["state"]
        target_action = batch["best_action"]

        action = self.forward(state)

        loss = self.criterion(action,target_action)
  
        return loss


    def training_step(self, batch, batch_id):

        loss = self.forward_step(batch, batch_id)
  
        self.log("loss/train",loss.item())

        return loss


    def validation_step(self, batch, batch_id):

        loss = self.forward_step(batch, batch_id)
  
        self.log("loss/valid",loss.item())

        return loss