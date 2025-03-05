import numpy as np
from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(QLearningAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=200000):
        s, _ = self.env.reset()

        for i in range(n_timesteps):
            # TODO 1.3: Implement Q-learning training loop
            # You will have to call self.update_Q(...) at every step
            # Do not forget to reset the environment if you receive a 'terminated' signal
            pass

    def update_Q(self, s, a, r, s_):
        # TODO 1.3: Implement Q-learning update
        self.Q[*s, a] = 0.0
