import torch
import torch.nn as nn
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self,**kwargs):
    
        super().__init__()
        self.save_hyperparameters()
        p = self.hparams

        self.runtime = False
   
        self.state_size = p.state_size
        self.action_size = p.action_size

        self.criterion = torch.nn.MSELoss()

        self.transition_model = nn.Sequential(
            nn.Linear(self.state_size+self.action_size,32),
            nn.SiLU(),
            nn.Linear(32,32),
            nn.SiLU(),
            nn.Linear(32,self.state_size),
        )

        self.reward_model = nn.Sequential(
            nn.Linear(self.state_size+self.action_size,32),
            nn.SiLU(),
            nn.Linear(32,32),
            nn.SiLU(),
            nn.Linear(32,1),
        )

    def configure_optimizers(self):
        p = self.hparams
        optimizer = torch.optim.Adam(self.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=p.cosine_scheduler_max_epoch)
        return [optimizer],[scheduler]


    def forward(self,state,action):

        state_action = torch.cat([state,action],dim=1)

        next_state = state + self.transition_model(state_action)
        reward = self.reward_model(state_action)

        return next_state, reward


    def training_step(self, batch, batch_id):
        state = batch["state"]
        action = batch["action"]
        target_next_state = batch["next_state"]
        target_reward = batch["reward"]

        next_state, reward = self.forward(state, action)

        loss_state = self.criterion(next_state,target_next_state)
        loss_reward = self.criterion(reward,target_reward)
        loss = loss_state + loss_reward

        self.log("loss_state/train",loss_state.item())
        self.log("loss_reward/train",loss_reward.item())
        self.log("loss/train",loss.item())

        return loss

    def validation_step(self, batch, batch_id):
        state = batch["state"]
        action = batch["action"]
        target_next_state = batch["next_state"]
        target_reward = batch["reward"]

        next_state, reward = self.forward(state, action)

        loss_state = self.criterion(next_state,target_next_state)
        loss_reward = self.criterion(reward,target_reward)
        loss = loss_state + loss_reward
        
        self.log("loss_state/valid",loss_state.item())
        self.log("loss_reward/valid",loss_reward.item())
        self.log("loss/valid",loss.item())

        return loss