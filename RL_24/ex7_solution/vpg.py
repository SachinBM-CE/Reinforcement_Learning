import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

from utils import episode_reward_plot


def compute_returns(rewards, next_value, discount):
    """ Compute returns based on episode rewards.

    Parameters
    ----------
    rewards : list of float
        Episode rewards.
    next_value : float
        Value of state the episode ended in. Should be 0.0 for terminal state, bootstrapped value otherwise.
    discount : float
        Discount factor.

    Returns
    -------
    list of float
        Episode returns.
    """

    # TODO (3.)
    return None


class TransitionMemory:
    """Datastructure to store episode transitions and perform return/advantage/generalized advantage calculations (GAE) at the end of an episode."""

    def __init__(self, gamma):

        # TODO (2.)
        pass

    def put(self, obs, action, reward, logprob):
        """Put a transition into the memory."""

        # TODO
        pass

    def get(self):
        """Get all stored transition attributes in the form of lists."""

        # TODO
        return None

    def clear(self):
        """Reset the transition memory."""

        # TODO
        pass

    def finish_trajectory(self, next_value):
        """Call on end of an episode. Will perform episode return or advantage or generalized advantage estimation (later exercise).
        
        Parameters
        ----------
        next_value:
            The value of the state the episode ended in. Should be 0.0 for terminal state.
        """

        # TODO
        pass


class ActorNetwork(nn.Module):
    """Neural Network used to learn the policy."""

    def __init__(self, num_observations, num_actions):
        super(ActorNetwork, self).__init__()

        # TODO (1.)
        pass

    def forward(self, obs):

        # TODO
        return None


class VPG:
    """The vanilla policy gradient (VPG) approach."""

    def __init__(self, env, episodes_update=5, gamma=0.99, lr=0.01):
        """ Constructor.
        
        Parameters
        ----------
        env : gym.Environment
            The object of the gym environment the agent should be trained in.
        episodes_update : int
            Number episodes to collect for every optimization step.
        gamma : float, optional
            Discount factor.
        lr : float, optional
            Learning rate used for actor and critic Adam optimizer.
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continuous actions not implemented!')
        
        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.memory = TransitionMemory(gamma)
        self.episodes_update = episodes_update

        self.actor_net = ActorNetwork(self.obs_dim, self.act_dim)
        self.optim_actor = optim.Adam(self.actor_net.parameters(), lr=lr)

    def learn(self, total_timesteps):
        """Train the VPG agent.
        
        Parameters
        ----------
        total_timesteps : int
            Number of timesteps to train the agent for.
        """

        # TODO (6.)
        obs, _ = self.env.reset()

        # For plotting
        overall_rewards = []
        episode_rewards = []

        episodes_counter = 0

        for timestep in range(1, total_timesteps + 1):

            # TODO Do one step, put into transition buffer, and store reward in episode_rewards for plotting
            terminated, truncated = None, None

            if terminated or truncated:

                # TODO reset environment

                # TODO finish trajectory

                episodes_counter += 1

                if episodes_counter == self.episodes_update:

                    # TODO optimize the actor

                    # Clear memory
                    episodes_counter = 0
                    self.memory.clear()

            # Episode reward plot
            if timestep % 500 == 0:
                episode_reward_plot(overall_rewards, timestep, window_size=5, step_size=1, wait=timestep == total_timesteps)

    @staticmethod
    def calc_actor_loss(logprob_lst, return_lst):
        """Calculate actor "loss" for one batch of transitions."""

        # TODO (5.)
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

        # TODO (4.)

        if train_returns:

            # TODO Return action, logprob
            return None
        else:

            # TODO Return action
            return None


if __name__ == '__main__':
    env_id = "CartPole-v1"
    _env = gym.make(env_id)
    vpg = VPG(_env)
    vpg.learn(100000)
