import math
import torch
import torch.nn as nn
from env.pendulum import PendulumEnv
from tqdm import trange, tqdm
import gym

from transition_dataset import TransitionRecorder
from plotting import PendulumPhasePlot, plot_value_on_phase_plot
from batching import generate_batch_indicies

from models.transition_model import TransitionModel
from models.value_model import ValueModel


def main():
    hparams = {

        "num_collection_runs": 100,
        "num_samples_per_run": 100,

    }

    # Collect transition data from the simulator
    transition_dataset = collect_transition_dataset(hparams)
    num_states = len(transition_dataset["state"])
    transition_dataset["state_value"] = transition_dataset["state_reward"].clone()
    transition_dataset["best_action"] = torch.zeros(num_states,1)
    transition_dataset["best_next_state"] = torch.zeros(num_states,3)
    transition_dataset["best_next_value"] = torch.zeros(num_states,1)

    
    phase_plot = PendulumPhasePlot()
    phase_plot.plot_transitions_on_phase_plot(transition_dataset["state"],transition_dataset["next_state"],"figures/raw_transitions.png")
    
    # Instanciate models
    transition_model = TransitionModel()
    value_model = ValueModel()

    plot_value_on_phase_plot(value_model,"figures/value_random_weights.png")
    
    # fit the transition model to the data
    fit_transition_model_to_data(transition_model, transition_dataset)

    update_dataset_with_next_best_state(transition_model,value_model,transition_dataset)
  
    fit_value_model_to_data(value_model,transition_dataset)

    plot_value_on_phase_plot(value_model,"figures/value_rewards_only.png")

    iterate_value_through_state_space( transition_model, value_model,transition_dataset)
    
    plot_value_on_phase_plot(value_model,"figures/value_final.png")

    update_dataset_with_next_best_state(transition_model,value_model,transition_dataset)

    phase_plot = PendulumPhasePlot()
    phase_plot.plot_transitions_on_phase_plot(transition_dataset["state"],transition_dataset["best_next_state"],"figures/best_transitions.png")


def collect_transition_dataset(hparams):
    """Run the environment a few times and record all the transitions.
    """
    num_collection_runs = hparams["num_collection_runs"]
    num_samples_per_run = hparams["num_samples_per_run"]

    env = PendulumEnv()

    transition_recorder = TransitionRecorder()

    for _ in trange(num_collection_runs):
        state = env.reset()

        for _ in range(num_samples_per_run):
            action = env.action_space.sample()
            # env.render()
            next_state, state_reward, action_cost = env.step(action) # take a random action

            transition_recorder.add_transition(
                state=state,
                state_reward=state_reward,
                action=action,
                next_state=next_state,
                action_cost=action_cost
            )

            state = next_state

    env.close()

    return transition_recorder.get_dict_of_tensors()


def fit_transition_model_to_data(transition_model: nn.Module ,transition_tensors : dict):
    num_epochs = 50
    batch_size = 128
    num_transitions = len(transition_tensors["state"])

    state_tensor = transition_tensors["state"]
    action_tensor = transition_tensors["action"] 
    target_tensor = transition_tensors["next_state"]

    optimizer = torch.optim.Adam(transition_model.parameters(),lr=0.001)
    criterion = nn.MSELoss()

    for epoch_i in trange(num_epochs,desc="Transition Model Epochs"):
        loss_list = []
        for batch_indicies in generate_batch_indicies(batch_size, num_transitions):
            batch_state_tensor = state_tensor[batch_indicies]
            batch_action_tensor =action_tensor[batch_indicies]
            batch_target_tensor =target_tensor[batch_indicies]

            batch_pred_next_state = transition_model( batch_state_tensor, batch_action_tensor)

            loss = criterion(batch_pred_next_state, batch_target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        print("Loss",sum(loss_list) / len(loss_list))

    return transition_model
        

def fit_value_model_to_data(value_model,transition_tensors : dict):
    state_tensor = transition_tensors["state"]
    state_value_tensor = transition_tensors["state_value"]

    num_states = len(state_tensor)
    batch_size = 1024
    num_iteration_steps = 100

    optimizer = torch.optim.Adam(value_model.parameters(),lr=0.001)
    criterion = nn.MSELoss()

    for step_i in trange(num_iteration_steps,desc="Value fitting epochs"):

        for batch_indicies in generate_batch_indicies(batch_size, num_states):

            batch_state_tensor = state_tensor[batch_indicies]
            batch_value_target_tensor = state_value_tensor[batch_indicies]

            # compute the value for our current state
            batch_value_tensor = value_model(batch_state_tensor)

            loss = criterion(batch_value_tensor, batch_value_target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

def find_best_action_for_batch_of_states(transition_model, value_model, batch_state_tensor):
    num_optimizer_steps = 100

    batch_action_tensor = nn.Parameter(torch.zeros(len(batch_state_tensor),1 ))

    optimizer = torch.optim.Adam([batch_action_tensor],lr=0.01)

    for step_i in range(num_optimizer_steps):

        next_state = transition_model(batch_state_tensor, batch_action_tensor)
        next_state_value = value_model(next_state)

        loss = -next_state_value.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_action_tensor.data = batch_action_tensor.data.clamp(-1.,1.)

    return batch_action_tensor.data

def update_dataset_with_next_best_state(transition_model, value_model, transition_dataset):

    batch_size = 1024
    
    state_tensor = transition_dataset["state"]
    best_action_tensor = transition_dataset["best_action"]
    best_next_state_tensor = transition_dataset["best_next_state"]
    best_next_value_tensor = transition_dataset["best_next_value"]
    
    num_states = len(state_tensor)

    for batch_indicies in generate_batch_indicies(batch_size, num_states):
        
        batch_state_tensor = state_tensor[batch_indicies]

        batch_best_action_tensor = find_best_action_for_batch_of_states(transition_model, value_model, batch_state_tensor)
        
        with torch.no_grad():
            batch_best_next_state_tensor = transition_model(batch_state_tensor,batch_best_action_tensor)
            batch_best_next_value_tensor = value_model(batch_best_next_state_tensor)


        best_action_tensor[batch_indicies] = batch_best_action_tensor
        best_next_state_tensor[batch_indicies] = batch_best_next_state_tensor
        best_next_value_tensor[batch_indicies] = batch_best_next_value_tensor




def iterate_value_through_state_space( transition_model, value_model,transition_dataset : dict):
    
    state_tensor = transition_dataset["state"]
    reward_tensor = transition_dataset["state_reward"]

    num_transitions = len(state_tensor)
    batch_size = 1024
    num_iteration_steps = 20
    
    optimizer = torch.optim.Adam(value_model.parameters(),lr=0.001)
    criterion = nn.MSELoss()

    for step_i in trange(num_iteration_steps, desc="Value Iteration Steps"):

        update_dataset_with_next_best_state(transition_model, value_model,transition_dataset)

        transition_dataset["state_value"] = transition_dataset["state_reward"] + transition_dataset["best_next_value"]

        fit_value_model_to_data(value_model,transition_dataset)

        plot_value_on_phase_plot(value_model,f"figures/value_step_{step_i}.png")



if __name__ == "__main__":
    main()

