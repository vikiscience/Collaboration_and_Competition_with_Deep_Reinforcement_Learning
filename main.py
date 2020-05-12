import const
from src import agent, algo, hyperparameter_search, utils_env

import numpy as np
import random
import argparse
import warnings

random_seed = const.random_seed
np.random.seed(random_seed)
random.seed(random_seed)

state_size = const.state_size
action_size = const.action_size
N = const.rolling_mean_N

warnings.filterwarnings("ignore", category=UserWarning)


def try_random_agent():
    env = utils_env.Environment()
    brain_name = env.brain_names[0]

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(const.num_agents)  # initialize the score (for each agent)
    while True:
        actions = np.random.randn(const.num_agents, action_size)  # select an action (for each agent)
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished
        scores += rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
    env.close()


def train_default_algo():
    env = utils_env.Environment()
    # use default params
    ag = agent.DRLAgent()
    al = algo.DRLAlgo(env, ag)
    al.train()


def test_default_algo(use_ref_model: bool = False):
    env = utils_env.Environment()
    # use default params
    ag = agent.DRLAgent()
    if use_ref_model:
        print('... Test the agent using reference model ...')
        ag.set_model_path('ref')
    al = algo.DRLAlgo(env, ag)
    al.test()


def get_env_info():
    env = utils_env.Environment()
    env.get_info()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test a Deep '
                                                 'Reinforcement Learning agent '
                                                 'to navigate in a Reacher Environment',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--exec', choices=['train', 'test', 'grid', 'info'],
                        default='info', help='Train/test a DRL agent, '
                                             'perform grid search to find the best agent, '
                                             'or get the Environment info')
    parser.add_argument('-r', '--use_reference_model', action="store_true", default=False,
                        help='In Test Mode, use the pretrained reference model')

    args = parser.parse_args()
    exec = args.exec
    use_ref_model = args.use_reference_model

    if exec == 'train':
        train_default_algo()
    elif exec == 'test':
        test_default_algo(use_ref_model)
    elif exec == 'grid':
        hyperparameter_search.grid_search()
    else:
        get_env_info()
        # try_random_agent()
