import gym

from dataset import TransitionDataset
from model import LitModel

def main():
    hparams = {
        "gym_env_name": "Pendulum-v0",
        "num_collection_runs": 5,
        "num_samples_per_run": 1000,
    }

    run_experiment(hparams)


def run_experiment(hparams):

    train_dataset = collect_dataset(hparams)






def collect_dataset(hparams):
    """Run the environment a few times and record all the transitions.
    Wrap the transitions in a dataset class so it can be used for training
    """
    gym_env_name = hparams["gym_env_name"]
    num_collection_runs = hparams["num_collection_runs"]
    num_samples_per_run = hparams["num_samples_per_run"]

    env = gym.make(gym_env_name)
    data = []

    for _ in range(num_collection_runs):
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


if __name__ == "__main__":
    main()
