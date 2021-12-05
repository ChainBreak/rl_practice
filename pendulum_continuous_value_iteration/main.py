import math
import torch
import torch.nn as nn
from env.pendulum import PendulumEnv
from tqdm import trange, tqdm
import gym

from transition_dataset import TransitionRecorder
from plotting import PendulumPhasePlot
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
    transition_tensors = transition_dataset.get_transition_tensors_dict()
    
    phase_plot = PendulumPhasePlot()
    phase_plot.plot_transitions_on_phase_plot(transition_tensors["state"],transition_tensors["next_state"])
    
    # Instanciate models
    transition_model = TransitionModel()
    value_model = ValueModel()

    # fit the transition model to the data
    fit_transition_model_to_data(transition_model, transition_tensors)

    iterate_value_through_state_space( transition_model, value_model,transition_tensors)
    

    best_next_state_tensor = get_best_next_state(transition_model, value_model,transition_tensors["state"])

    phase_plot = PendulumPhasePlot()
    phase_plot.plot_transitions_on_phase_plot(transition_tensors["state"],best_next_state_tensor)

    pass

def collect_transition_dataset(hparams):
    """Run the environment a few times and record all the transitions.
    Wrap the transitions in a dataset class so it can be used for training
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

    return transition_recorder.get_transition_dataset()


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
        for batch_indicies in tqdm(generate_batch_indicies(batch_size, num_transitions),desc="Transition Model Batches"):
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
        

def find_best_action_for_every_state(state_tensor, transition_model, value_model):
    action_size = 1
    batch_size = 1024
    num_optimizer_steps = 20

    num_states = len(state_tensor)

    action_tensor = torch.zeros(num_states, action_size)

    for batch_indicies in generate_batch_indicies(batch_size, num_states):
        
        batch_state_tensor = state_tensor[batch_indicies]
        batch_action_tensor = nn.Parameter(action_tensor[batch_indicies])

        optimizer = torch.optim.Adam([batch_action_tensor],lr=0.01)

        for step_i in range(num_optimizer_steps):
            next_state = transition_model(batch_state_tensor, batch_action_tensor)
            next_state_value = value_model(next_state)

            loss = -next_state_value.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_action_tensor.data = batch_action_tensor.data.clamp(-1.,1.)

        action_tensor[batch_indicies] = batch_action_tensor.data

    return action_tensor


def iterate_value_through_state_space( transition_model, value_model,transition_tensors : dict):
    
    state_tensor = transition_tensors["state"]
    reward_tensor = transition_tensors["state_reward"]

    num_transitions = len(state_tensor)
    batch_size = 1024
    num_iteration_steps = 100
    
    optimizer = torch.optim.Adam(value_model.parameters(),lr=0.001)
    criterion = nn.MSELoss()

    for _ in trange(num_iteration_steps,desc="Value Iteration Steps"):

        # Find the best action for every state given our current value funciton
        best_action_tensor = find_best_action_for_every_state(state_tensor,transition_model,value_model)

        for batch_indicies in tqdm(generate_batch_indicies(batch_size, num_transitions),desc="Value Iteration Batches"):

            batch_state_tensor = state_tensor[batch_indicies]
            batch_best_action_tensor = best_action_tensor[batch_indicies]
            batch_reward_tensor =reward_tensor[batch_indicies]

            with torch.no_grad():
                # Use our best action to step forward to the next best state
                batch_best_next_state_tensor = transition_model(batch_state_tensor,batch_best_action_tensor)

                # Collect the value from the next best state
                batch_best_next_value_tensor = value_model(batch_best_next_state_tensor)

                batch_value_target_tensor = torch.max(batch_reward_tensor,batch_best_next_value_tensor)

            # compute the value for our current state
            batch_value_tensor = value_model(batch_state_tensor)

            # Try to make this value
            loss = criterion(batch_value_tensor, batch_value_target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def get_best_next_state(transition_model, value_model,state_tensor):

    
    num_transitions = len(state_tensor)
    batch_size = 1024
    # Find the best action for every state given our current value funciton
    best_action_tensor = find_best_action_for_every_state(state_tensor,transition_model,value_model)

    best_next_state_tensor = torch.zeros_like(state_tensor)
    with torch.no_grad():
        for batch_indicies in tqdm(generate_batch_indicies(batch_size, num_transitions),desc="Value Iteration Batches"):
            batch_state_tensor = state_tensor[batch_indicies]
            batch_best_action_tensor = best_action_tensor[batch_indicies]

            # Use our best action to step forward to the next best state
            batch_best_next_state_tensor = transition_model(batch_state_tensor,batch_best_action_tensor)

            best_next_state_tensor[batch_indicies] = batch_best_next_state_tensor

    return best_next_state_tensor

if __name__ == "__main__":
    main()

