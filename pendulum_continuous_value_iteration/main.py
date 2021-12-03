
from env.pendulum import PendulumEnv
from tqdm import trange, tqdm
import gym

from transition_dataset import TransitionRecorder
from plotting import PendulumPhasePlot


def main():
    hparams = {

        "num_collection_runs": 10,
        "num_samples_per_run": 50,

    }

    transition_dataset = collect_transition_dataset(hparams)
    
    phase_plot = PendulumPhasePlot()
    phase_plot.plot_transition_dataset(transition_dataset)
    # plot known transitions in phase diagram

    # train transition model

    # animate actions on phase diagram
        
    # iterate value
        # get best action
        # update value
        # animate value moving around the phase space

    # plot best actions
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

if __name__ == "__main__":
    main()

