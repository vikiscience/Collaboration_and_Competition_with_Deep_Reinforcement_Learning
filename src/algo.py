import const
from src import utils_plot

from pathlib import Path
import numpy as np


class DRLAlgo:
    image_path = const.file_path_img_score

    def __init__(self, env, agent,
                 num_episodes: int = const.num_episodes,
                 policy_eval_freq: int = const.policy_eval_freq):
        self.env = env
        self.brain_name = env.brain_names[0]
        self.agent = agent
        self.num_states = agent.num_states
        self.num_actions = agent.num_actions

        # algo params
        self.num_episodes = num_episodes
        self.policy_eval_freq = policy_eval_freq

    def train(self, with_close=True):
        print('Training ...')

        history = []

        # Evaluate untrained policy
        evaluations = [self.eval_policy(self.agent.policy, const.random_seed)]

        for e in range(self.num_episodes):
            env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
            states = env_info.vector_observations  # get the current state (s_t)
            scores = np.zeros(const.num_agents)  # initialize the score

            t = 0

            while True:

                # choose a_t using epsilon-greedy policy
                actions = self.agent.act(states, t)

                # take action a_t, observe r_{t+1} and s_{t+1}
                env_info = self.env.step(actions)[self.brain_name]  # send the action to the environment
                next_states = env_info.vector_observations  # get the next state
                rewards = env_info.rewards  # get the reward
                dones = env_info.local_done  # see if episode has finished

                # Memorize new sample, replay, update target network
                self.agent.do_stuff(states[0], actions[0], rewards[0], next_states[0], dones[0], t)  # todo [0]

                states = next_states
                scores += rewards
                t += 1

                if np.any(dones):
                    break

            score = np.mean(scores)  # mean of scores over all agents
            print("\r -> Episode: {}/{}, score: {}".format(e + 1, self.num_episodes, score), end='')
            history.append(score)

            # Evaluate episode
            if (e + 1) % self.policy_eval_freq == 0:
                avg_reward = self.eval_policy(self.agent.policy, const.random_seed)
                evaluations.append(avg_reward)

            if (e + 1) % 100 == 0 or e + 1 == self.num_episodes:
                self.agent.save()

        print('History:', history)
        print('Evaluations:', evaluations)
        utils_plot.plot_history_rolling_mean(history, fp=self.image_path)

        if with_close:
            self.env.close()

        return history

    def test(self):
        self.agent.load()

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

        score = np.mean(scores)  # mean of scores over all agents
        print("Score: {}".format(score))

        self.env.close()

    def set_image_path(self, i):
        p = self.image_path
        self.image_path = Path(p.parent, 'score_' + str(i) + p.suffix)

    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment
    def eval_policy(self, policy, seed, eval_episodes=const.rolling_mean_N):
        history = []
        for _ in range(eval_episodes):
            scores = np.zeros(const.num_agents)
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations
            while True:
                actions = policy.select_action(np.array(states[0])).reshape(1, -1)  # todo
                env_info = self.env.step(actions)[self.brain_name]
                rewards = env_info.rewards
                dones = env_info.local_done
                scores += rewards
                if np.any(dones):
                    break
            history.append(np.mean(scores))  # mean of scores over all agents

        avg_score = np.mean(history)  # mean over all episodes
        return avg_score
