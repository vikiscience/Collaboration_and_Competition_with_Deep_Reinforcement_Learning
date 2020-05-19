import const

import numpy as np


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, stddev=const.ou_stddev,
                 theta=const.ou_theta, dt=const.ou_dt, x0=None):
        self.theta = theta
        self.mu = np.zeros(const.action_size)
        self.sigma = stddev * np.ones(const.action_size)
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class GLIE:
    def __init__(self, eps_start=const.eps_start, eps_end=const.eps_end,
                 num_episodes=const.num_episodes):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_start / num_episodes
        self.eps = self.eps_start

    def get_eps(self):
        return self.eps

    def update_eps(self):
        if self.eps < self.eps_end:
            self.eps = self.eps_end
        else:
            self.eps = self.eps - self.eps_decay
