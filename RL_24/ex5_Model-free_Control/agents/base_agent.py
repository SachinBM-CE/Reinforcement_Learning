import numpy as np


class BaseAgent:
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        self.env = env
        self.g = discount_factor
        self.lr = learning_rate
        self.eps = epsilon

        self.num_actions = env.action_space.n

        # Q[y, x, z] is value of action z for grid position y, x
        self.Q = np.zeros((*env.observation_space.high, self.num_actions), dtype=np.float32)

    def action(self, s, epsilon=None):
        # allows to set a small(er) value for testing
        eps = epsilon if epsilon is not None else self.eps

        # TODO 1.1: Implement epsilon-greedy action selection
        return self.env.action_space.sample()  # return dummy random action
