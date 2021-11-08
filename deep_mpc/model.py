
import pytorch_lightning as pl

class LitModel(pl):
    def __init__(self,**kwargs):
    
        super().__init__()
        self.save_hyperparameters()
        p = self.hparams


    def forward(self,state,action):

        a_matrix = self.a_model(state)
        b_matrix = self.b_model(state)

        next_state = torch.bmm(a_matrix,state) + torch.bmm(b_matrix,action)

        return next_state