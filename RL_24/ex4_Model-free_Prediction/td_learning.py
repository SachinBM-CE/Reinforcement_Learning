import numpy as np

from gridworld import GridWorldEnv
from util import value_function_policy_plot


class TDAgent:
    def __init__(self, env, discount_factor, learning_rate):
        self.env = env
        self.g = discount_factor
        self.lr = learning_rate

        self.num_actions = env.action_space.n

        # V[y, x] is value for grid position y, x, initialize to all zeros
        self.V = np.zeros(env.observation_space.high, dtype=np.float32)

        # uniform random policy[y, x, z], i.e. probability of action z when in grid position y, x is 1 / num_actions
        self.policy = np.ones((*env.observation_space.high, self.num_actions), dtype=np.float32) / self.num_actions

        # TODO 3: experiment with different (not fully random) policies
        self.policy = np.full((*env.observation_space.high, self.num_actions), 0.25, dtype=np.float32)
        self.policy[:, :, 1] = 0.5  # Example: biasing towards moving right
        self.policy /= self.policy.sum(axis=-1, keepdims=True)  # Normalize to sum to 1

    def action(self, s):
        # TODO 2: Sample action following the policy
        # return self.env.action_space.sample()  # random action
        return np.random.choice(self.num_actions, p=self.policy[s[0], s[1]])

    def learn(self, n_timesteps=50000):
        s, _ = self.env.reset()

        for i in range(n_timesteps):
            # TODO 1: Implement the agent-interaction loop
            # You will have to call self.update(...) at every step
            # Do not forget to reset the environment if you receive a 'terminated' signal
            a = self.action(s)
            s_, r, done, _, _ = self.env.step(a)
            self.update(s, r, s_)
            if done:
                s, _ = self.env.reset()
            else:
                s = s_


    def update(self, s, r, s_):
        # TODO 1: Implement the TD estimation update rule
        # self.V[*s] = 0.0
        self.V[s[0], s[1]] += self.lr * (r + self.g * self.V[s_[0], s_[1]] - self.V[s[0], s[1]])


if __name__ == "__main__":
    # Create Agent and environment
    td_agent = TDAgent(GridWorldEnv(), discount_factor=0.9, learning_rate=0.01)

    # Learn the state-value function for 100000 steps
    td_agent.learn(n_timesteps=100000)

    # Visualize V
    value_function_policy_plot(td_agent.V, td_agent.policy, td_agent.env.map)
