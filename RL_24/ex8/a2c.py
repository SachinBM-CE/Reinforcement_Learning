import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym

from networks import ActorNetwork, CriticNetwork
from transition_memory import TransitionMemoryAdvantage
from utils import episode_reward_plot


class A2C:
    """The Actor-Critic approach."""

    def __init__(self, env, batch_size=500, gamma=0.99, lamb=0.99, lr_actor=0.005, lr_critic=0.001, use_gae=True):
        """ Constructor.

        Parameters
        ----------
        env : gym.Environment
            The object of the gym environment the agent should be trained in.
        batch_size : int, optional
            Number of transitions to use for one optimization step.
        gamma : float, optional
            Discount factor.
        lamb : float, optional
            Lambda parameters of GAE.
        lr_actor : float, optional
            Learning rate used for actor Adam optimizer.
        lr_critic : float, optional
            Learning rate used for critic Adam optimizer.
        use_gae : bool, optional
            Use generalized advantage estimation
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continuous actions not implemented!')

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.batch_size = batch_size
        self.env = env
        self.memory = TransitionMemoryAdvantage(gamma, lamb, use_gae)

        self.actor_net = ActorNetwork(self.obs_dim, self.act_dim)
        self.critic_net = CriticNetwork(self.obs_dim)
        self.optim_actor = optim.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.optim_critic = optim.Adam(self.critic_net.parameters(), lr=lr_critic)

    def learn(self, total_timesteps):
        """Train the actor-critic.

        Parameters
        ----------
        total_timesteps : int
            Number of timesteps to train the agent for.
        """
        obs, _ = self.env.reset()

        # For plotting
        overall_rewards = []
        episode_rewards = []

        episode_counter = 0
        for timestep in range(1, total_timesteps + 1):

            # TODO 1.7.a: Sample action and supplementary data, take step and save transition to buffer
            action, logprob, value = None, None, None
            obs_, reward, terminated, truncated, info = None, None, None, None, None
            episode_rewards.append(reward)

            # Update current obs
            obs = obs_

            if terminated or truncated:

                # TODO 1.7.b: Reset environment and call 'finish_trajectory' with correct 'next_value'
                overall_rewards.append(sum(episode_rewards))
                episode_rewards = []

            if (timestep - episode_counter) == self.batch_size:

                # TODO 1.7.c: Call 'finish_trajectory' with correct 'next_value', calculate losses, perform updates

                # Get transitions from memory -> used to perform update
                _, _, _, logprob_lst, return_lst, value_lst, adv_lst = self.memory.get()

                # Clear memory
                self.memory.clear()
                episode_counter = timestep

            # Episode reward plot
            if timestep % 500 == 0:
                episode_reward_plot(overall_rewards, timestep, window_size=5, step_size=1)

    @staticmethod
    def calc_critic_loss(value_lst, return_lst):
        """Calculate critic loss for one batch of transitions."""

        # TODO 1.5: Compute the MSE between state values and returns
        return None

    @staticmethod
    def calc_actor_loss(logprob_lst, adv_lst):
        """Calculate actor "loss" for one batch of transitions."""

        # TODO 1.6: Adjust Compute actor loss (Hint: Very similar to VPG version)
        return None

    def predict(self, obs, train_returns=False):
        """Sample the agents action based on a given observation.

        Parameters
        ----------
        obs : numpy.array
            Observation returned by gym environment
        train_returns : bool, optional
            Set to True to get log probability of decided action and predicted value of obs.
        """

        probs = self.actor_net(torch.Tensor(obs))
        policy = Categorical(probs=probs)
        action = policy.sample()
        logprob = policy.log_prob(action)

        # TODO 1.3 Evaluate the value function
        value = None

        if train_returns:
            return action.item(), logprob, value
        else:
            return action.item()
