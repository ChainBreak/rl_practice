import gym
import time
import pytorch_lightning as pl
from tqdm import trange, tqdm
import torch
import torch.nn as nn
from dataset import TransitionDataset
from model import LitModel
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, random_split

def main():
    hparams = {
        "gym_env_name": "Pendulum-v0",
        "state_size": 3,
        "action_size": 1,
        "num_collection_runs": 1000,
        "num_samples_per_run": 100,

        "max_epochs":10,
        "cosine_scheduler_max_epoch":10,
        "learning_rate":0.001,
        "weight_decay":0.0000,

        "horizon_length":20,
        "test_episode_length": 1000,

    }

    run_experiment(hparams)


def run_experiment(hparams):

    # Collect all the transition data
    full_dataset = collect_dataset(hparams)

    # Compute train and valid lenghts
    train_length = int(len(full_dataset) * 0.75)
    valid_length = len(full_dataset)-train_length

    # Split out train and valid datasets
    train_dataset,valid_dataset =  random_split(full_dataset, [train_length,valid_length])

    train_dataloader = DataLoader(train_dataset,batch_size=128, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=128, shuffle=False)

    model = LitModel(**hparams)
    p = model.hparams
    
    callback_list = [
        # ModelCheckpoint(monitor="loss/valid",save_top_k=3),
        LearningRateMonitor(logging_interval='step')
    ]

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=p.max_epochs,
        callbacks=callback_list,
    )
    trainer.fit(model,train_dataloader,valid_dataloader )

    run_mpc(model,hparams)



def collect_dataset(hparams):
    """Run the environment a few times and record all the transitions.
    Wrap the transitions in a dataset class so it can be used for training
    """
    gym_env_name = hparams["gym_env_name"]
    num_collection_runs = hparams["num_collection_runs"]
    num_samples_per_run = hparams["num_samples_per_run"]

    env = gym.make(gym_env_name)
    data = []

    for _ in trange(num_collection_runs):
        state = env.reset()
        run_transitions = []

        for _ in range(num_samples_per_run):
            action = env.action_space.sample()
            # env.render()
            next_state,reward, done, info = env.step(action) # take a random action

            run_transitions.append((state,action,next_state,reward))
            state = next_state

            if done:
                break
        data.append(run_transitions)
    env.close()

    return TransitionDataset(data)


def run_mpc(model,hparams):
    gym_env_name = hparams["gym_env_name"]
    
    test_episode_length = hparams["test_episode_length"]
    state_size = hparams["state_size"]

    model.eval()

    env = gym.make(gym_env_name)

    while True:
        state = env.reset()
        for step in range(test_episode_length):

            state_tensor = torch.tensor(state,dtype=torch.float).unsqueeze(0)
            action = get_optimal_action(model,state_tensor,hparams)

            env.render()
            # action = env.action_space.sample()
            next_state,reward, done, info = env.step(action) 


def get_optimal_action(model,current_state,hparams):
    horizon_length = hparams["horizon_length"]
    action_size = hparams["action_size"]
    solver_iterations = 20

    action_tensor = nn.Parameter(torch.zeros(horizon_length,action_size))
    optimizer = torch.optim.Adam([action_tensor],lr=0.1)

    model.runtime = True
    for i in range(solver_iterations):
        state = current_state

        reward_sum = 0
        for step in range(horizon_length):
            action = action_tensor[step].unsqueeze(0)
            state,reward = model(state,action)

            reward_sum += -reward

        optimizer.zero_grad()
        reward_sum.backward(retain_graph=True)
        optimizer.step()
    
    return action_tensor[0].cpu().detach().numpy()

        
    # time.sleep(5)


if __name__ == "__main__":
    main()
