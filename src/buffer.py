import const

from collections import deque
import numpy as np
import random
import torch

device = 'cpu'  # torch.device("cuda" if torch.cuda.is_available() else "cpu") todo


class ReplayBuffer:

    def __init__(self, memory_size: int = const.memory_size):
        self.memory = deque(maxlen=memory_size)

    def get_length(self):
        return len(self.memory)

    def append(self, tupel):
        assert len(tupel) == 5
        # state, action, reward, next_state, done = tupel
        self.memory.append(tupel)

    def sample(self, batch_size: int):
        minibatch = random.sample(self.memory, batch_size)

        # states, actions and next_states are of shape (batch_size x n_dims)
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*minibatch))

        # reshape single values
        reward_batch = reward_batch.reshape(-1, 1)  # set n_dims to 1
        done_batch = done_batch.reshape(-1, 1)  # set n_dims to 1

        not_dones = 1. - done_batch  # invert bool values for formula use
        t_states = torch.FloatTensor(states_batch).to(device)
        t_actions = torch.FloatTensor(action_batch).to(device)
        t_rewards = torch.FloatTensor(reward_batch).to(device)
        t_next_states = torch.FloatTensor(next_states_batch).to(device)
        t_not_dones = torch.FloatTensor(not_dones).to(device)
        return t_states, t_actions, t_next_states, t_rewards, t_not_dones  # this order for usage in 'models.py'
