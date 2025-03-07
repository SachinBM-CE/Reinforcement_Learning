from .base_agent import BaseAgent


class SARSAAgent(BaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(SARSAAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=200000):
        s, _ = self.env.reset()
        a = self.action(s)  # Epsilon-greedy action for initial state

        for i in range(n_timesteps):
            # TODO 1.2: Implement SARSA training loop
            # You will have to call self.update_Q(...) at every step
            # Do not forget to reset the environment and update the action if you receive a 'terminated' signal
            pass

    def update_Q(self, s, a, r, s_, a_):
        # TODO 1.2: Implement SARSA update
        self.Q[*s, a] = 0.0
