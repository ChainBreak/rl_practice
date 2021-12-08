import torch
import torch.nn as nn
import pytorch_lightning as pl


        
class LitTransitionModel(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        p = self.hparams

        self.learning_rate = p.learning_rate

        self.state_size = 3
        self.action_size = 1

        self.criterion = torch.nn.MSELoss()

        self.seq = nn.Sequential(
            nn.Linear(self.state_size+self.action_size,32),
            nn.SiLU(),
            nn.Linear(32,32),
            nn.SiLU(),
            nn.Linear(32,self.state_size),
        )

        

    def configure_optimizers(self):
        p = self.hparams
        optimizer = torch.optim.Adam( self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p.cosine_annealing_t_max )
        return [optimizer],[scheduler]


    def forward(self,state,action):

        x = torch.cat([state,action],dim=1)

        next_state = self.seq(x)

        return next_state

    def forward_step(self, batch, batch_id):
        state = batch["state"]
        action = batch["action"]
        target_next_state = batch["next_state"]

        next_state = self.forward(state, action)

        loss = self.criterion(next_state,target_next_state)
  
        return loss


    def training_step(self, batch, batch_id):

        loss = self.forward_step(batch, batch_id)
  
        self.log("loss/train",loss.item())

        return loss


    def validation_step(self, batch, batch_id):

        loss = self.forward_step(batch, batch_id)
  
        self.log("loss/valid",loss.item())

        return loss