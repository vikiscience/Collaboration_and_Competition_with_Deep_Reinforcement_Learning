import const
from src import utils_plot

from pathlib import Path
import numpy as np


class DRLAlgo:
    image_path = const.file_path_img_score

    def __init__(self, env, agent,
                 num_episodes: int = const.num_episodes):
        self.env = env
        self.brain_name = env.brain_names[0]
        self.agent = agent
        self.num_states = agent.num_states
        self.num_actions = agent.num_actions

        # algo params
        self.num_episodes = num_episodes

    def train(self, with_close=True):
        print('Training ...')

        history = []

        for e in range(self.num_episodes):
            env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
            states = env_info.vector_observations  # get the current state (s_t)
            scores = np.zeros(const.num_agents)  # initialize the score

            t = 0

            while True:

                # choose a_t using epsilon-greedy policy
                actions = self.agent.act(states, t)  # todo: each agent receives its own, local observation

                # take action a_t, observe r_{t+1} and s_{t+1}
                env_info = self.env.step(actions)[self.brain_name]  # send the action to the environment
                next_states = env_info.vector_observations  # get the next state
                rewards = env_info.rewards  # get the reward
                dones = env_info.local_done  # see if episode has finished

                # Memorize new sample, replay, update target network
                if states[0][-4] < 0:  # 1. agent (right) was in turn
                    self.agent.do_stuff(states[0], actions[0], rewards[0], next_states[0], dones[0], t)  # todo [0]
                else:  # 2. agent (left) was in turn
                    self.agent.do_stuff(states[1], actions[1], rewards[1], next_states[1], dones[1], t)  # todo [1]

                states = next_states
                scores += rewards
                t += 1

                if np.any(dones):
                    break

            score = np.max(scores)  # max of scores over all agents for this episode
            print("\r -> Episode: {}/{}, score: {:.3f}".format(e + 1, self.num_episodes, score), end='')
            history.append(score)

            if (e + 1) % 100 == 0 or e + 1 == self.num_episodes:
                self.agent.save()

        const.myprint('History:', history)
        utils_plot.plot_history_rolling_mean(history, fp=self.image_path)

        if with_close:
            self.env.close()

        return history

    def test(self, num_episodes=const.num_episodes_test):
        self.agent.load()

        final_scores = []
        for i in range(num_episodes):
            env_info = self.env.reset(train_mode=False)[self.brain_name]  # reset the environment
            states = env_info.vector_observations  # get the current state (for each agent)
            scores = np.zeros(const.num_agents)  # initialize the score (for each agent)
            t = 0
            i = self.agent.start_policy_training_iter + 1  # set high i to avoid random actions in the beginning

            while True:
                actions = self.agent.act(states, i)  # select an action (for each agent)
                env_info = self.env.step(actions)[self.brain_name]  # send all actions to tne environment
                next_states = env_info.vector_observations  # get next state (for each agent)
                rewards = env_info.rewards  # get reward (for each agent)
                dones = env_info.local_done  # see if episode has finished
                scores += rewards  # update the score (for each agent)
                states = next_states  # roll over states to next time step
                t += 1
                if np.any(dones):  # exit loop if episode finished
                    break

            score = np.max(scores)  # max of scores over all agents
            final_scores.append(score)

        print("Final scores (per episode):", final_scores)
        self.env.close()

    def set_image_path(self, i):
        p = self.image_path
        self.image_path = Path(p.parent, 'score_' + str(i) + p.suffix)
