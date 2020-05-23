import const
from src import agent, algo, hyperparameter_search, utils_env, utils_plot

import argparse
import numpy as np
from pathlib import Path
import random
import warnings

random_seed = const.random_seed
np.random.seed(random_seed)
random.seed(random_seed)

state_size = const.state_size
action_size = const.action_size
N = const.rolling_mean_N

warnings.filterwarnings("ignore", category=UserWarning)


def try_random_agent(num_episodes=const.num_episodes_test):
    env = utils_env.Environment()
    brain_name = env.brain_names[0]

    for i in range(num_episodes):
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

        print('Episode {}, score (max over agents): {}'.format(i, np.max(scores)))
        print('Episode {}, score of each agent: ['.format(i), '; '.join(['{:.3f}'.format(s) for s in scores]), ']')

    env.close()


def train_two_agents():
    env = utils_env.Environment()
    # use default params
    ag_1 = agent.DRLAgent()
    ag_1.set_model_path(1)
    ag_2 = agent.DRLAgent()
    ag_2.set_model_path(2)
    al = algo.DRLAlgo(env, ag_1, ag_2)
    history, best_e, best_score = al.train()
    print('\nFinal score: {:.3f}'.format(np.mean(history[-const.rolling_mean_N:])))
    print('Final memory length:', ag_1.memory.get_length())
    print('Best score in {:d} episodes, avg_score: {:.3f}'.format(best_e, best_score))

    # plot losses
    losses_lists = [ag_1.actor_loss_list, ag_2.actor_loss_list,
                    ag_1.critic_loss_list, ag_2.critic_loss_list]
    losses_labels = ['agent_1_actor', 'agent_2_actor', 'agent_1_critic', 'agent_2_critic']
    utils_plot.plot_loss(losses_lists, losses_labels)

    # plot noise
    utils_plot.plot_scatter(ag_1.noise_list, title_text='Noise', fp=const.file_path_img_noise)

    # plot memory actions
    memory_actions = np.array([t[1] for t in ag_1.memory.memory])
    utils_plot.plot_scatter(memory_actions, title_text='Actions', fp=const.file_path_img_actions)

    # show mean memory actions
    mean_a = np.mean(memory_actions, axis=0)
    std_a = np.std(memory_actions, axis=0)
    print('Mean/std actions agent_1:', mean_a[:2], std_a[:2])
    print('Mean/std actions agent_2:', mean_a[2:], std_a[2:])


def test_default_algo(use_ref_model: bool = False):
    env = utils_env.Environment()
    model_name_suffix = ''
    if use_ref_model:
        print('... Test the agent using reference model ...')
        model_name_suffix = 'ref_'

    # use default params
    ag_1 = agent.DRLAgent()
    ag_1.set_model_path(model_name_suffix + str(1))
    ag_2 = agent.DRLAgent()
    ag_2.set_model_path(model_name_suffix + str(2))
    al = algo.DRLAlgo(env, ag_1, ag_2)
    al.test()


def get_env_info():
    env = utils_env.Environment()
    env.get_info()


def plot_raw_noise():
    from src import noise, utils_plot
    n = noise.OrnsteinUhlenbeckActionNoise()
    n.reset()
    n_list = []
    for e in range(const.num_episodes):
        n_list.append(n())
    fp_noise = const.file_path_img_noise
    fp_noise_raw = Path(fp_noise.parent, fp_noise.stem + '_raw' + fp_noise.suffix)
    utils_plot.plot_scatter(n_list, title_text='Noise Raw', fp=fp_noise_raw)


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
        train_two_agents()
    elif exec == 'test':
        test_default_algo(use_ref_model)
    elif exec == 'grid':
        hyperparameter_search.grid_search()
    else:
        get_env_info()
        # try_random_agent()
        # plot_raw_noise()
