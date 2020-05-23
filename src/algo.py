import const
from src import utils_plot

from collections import deque
import numpy as np
from pathlib import Path


class DRLAlgo:
    image_path = const.file_path_img_score

    def __init__(self, env, agent_1, agent_2,
                 num_episodes: int = const.num_episodes):
        self.env = env
        self.brain_name = env.brain_names[0]
        self.agent_1 = agent_1
        self.agent_2 = agent_2

        # algo params
        self.num_episodes = num_episodes

    def train(self, with_close=True):
        print('Training ...')

        history = []
        rolling_window = deque(maxlen=const.rolling_mean_N)
        solved = False
        best_score = 0.
        best_e = 0
        best_found = False

        for e in range(self.num_episodes):
            env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
            states = env_info.vector_observations  # get the current state (s_t)
            scores = np.zeros(const.num_agents)  # initialize the score
            self.agent_1.reset()
            self.agent_2.reset()

            while True:
                # choose a_t using epsilon-greedy policy
                states_input = states.flatten()
                a_1 = self.agent_1.act(states_input)
                a_2 = self.agent_2.act(states_input)
                actions_input = np.array([a_1, a_2]).flatten()

                # take action a_t, observe r_{t+1} and s_{t+1}
                env_info = self.env.step(actions_input)[self.brain_name]  # send the action to the environment
                next_states = env_info.vector_observations  # get the next state
                rewards = env_info.rewards  # get the reward
                dones = env_info.local_done  # see if episode has finished

                # Memorize new sample, replay, update target network
                next_states_input = next_states.flatten()
                self.agent_1.do_stuff(states_input, actions_input, rewards[0], next_states_input, dones[0], 0)
                self.agent_2.do_stuff(states_input, actions_input, rewards[1], next_states_input, dones[1], 1)

                states = next_states
                scores += rewards

                if np.any(dones):
                    break

            score = np.max(scores)  # max of scores over all agents for this episode
            rolling_window.append(score)
            history.append(score)
            curr_best_score = np.mean(rolling_window)
            print("\r -> Episode: {}/{}, score: {:.3f}, avg_score: {:.3f}".format(e + 1,
                                                                                  self.num_episodes,
                                                                                  score,
                                                                                  curr_best_score), end='')

            # if (e + 1) % 100 == 0 or e + 1 == self.num_episodes:
            if curr_best_score > best_score or (e + 1 == self.num_episodes and not best_found):
                best_score = curr_best_score
                best_e = e + 1
                best_found = True
                self.agent_1.save()
                self.agent_2.save()

            if np.mean(rolling_window) >= const.high_score and not solved:
                print('\nEnv solved in {:d} episodes, avg_score: {:.3f}'.format(e + 1, np.mean(rolling_window)))
                solved = True

        # plot scores
        const.myprint('History:', history)
        utils_plot.plot_history_rolling_mean(history, fp=self.image_path)

        if with_close:
            self.env.close()

        return history, best_e, best_score

    def test(self, num_episodes=const.num_episodes_test):
        self.agent_1.load()
        self.agent_2.load()

        final_scores = []
        for i in range(num_episodes):
            env_info = self.env.reset(train_mode=False)[self.brain_name]  # reset the environment
            states = env_info.vector_observations  # get the current state (for each agent)
            scores = np.zeros(const.num_agents)  # initialize the score (for each agent)

            while True:
                states_input = states.flatten()
                a_1 = self.agent_1.act(states_input)  # select an action (for each agent)
                a_2 = self.agent_2.act(states_input)
                actions_input = np.array([a_1, a_2]).flatten()
                env_info = self.env.step(actions_input)[self.brain_name]  # send all actions to tne environment
                next_states = env_info.vector_observations  # get next state (for each agent)
                rewards = env_info.rewards  # get reward (for each agent)
                dones = env_info.local_done  # see if episode has finished
                scores += rewards  # update the score (for each agent)
                states = next_states  # roll over states to next time step
                if np.any(dones):  # exit loop if episode finished
                    break

            score = np.max(scores)  # max of scores over all agents
            final_scores.append(score)

        print("Final scores (per episode):", final_scores)
        self.env.close()

    def set_image_path(self, i):
        p = self.image_path
        self.image_path = Path(p.parent, 'score_' + str(i) + p.suffix)
