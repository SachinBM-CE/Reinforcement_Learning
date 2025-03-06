import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

from utils import episode_reward_plot

import time

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
    returns = []
    R = next_value
    for reward in reversed(rewards):
        # Return is computed in constant time O(T) without recalculating past steps
        R = reward + discount*R
        returns.insert(0,R)

    return returns


class TransitionMemory:
    """Datastructure to store episode transitions and perform return/advantage/generalized advantage calculations (GAE) at the end of an episode."""

    def __init__(self, gamma):

        # TODO (2.)
        self.gamma = gamma
        # Lists to store values for a full episode
        self.observations = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.returns = []


    def put(self, obs, action, reward, logprob):
        """Put a transition into the memory."""

        # TODO
        self.observations.append(obs)  # current observation
        self.actions.append(action)    # selected action
        self.rewards.append(reward)    # received reward
        self.logprobs.append(logprob)  # logprob of selecting the action


    def get(self):
        """Get all stored transition attributes in the form of lists."""

        # TODO
        return self.observations, self.actions, self.rewards, self.logprobs, self.returns


    def clear(self):
        """Reset the transition memory."""

        # TODO
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.logprobs.clear()
        self.returns.clear()


    def finish_trajectory(self, next_value):
        """Call on end of an episode. Will perform episode return or advantage or generalized advantage estimation (later exercise).
        
        Parameters
        ----------
        next_value:
            The value of the state the episode ended in. Should be 0.0 for terminal state.
        """

        # TODO
        self.returns = compute_returns(self.rewards, next_value, self.gamma)


class ActorNetwork(nn.Module):
    """Neural Network used to learn the policy."""

    def __init__(self, num_observations, num_actions):
        super(ActorNetwork, self).__init__()

        # TODO (1.)
        self.net = nn.Sequential(
            nn.Linear(num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):

        # TODO
        return self.net(obs)


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
            action, logprob = self.predict(obs, train_returns=True)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.memory.put(obs, action, reward, logprob)
            episode_rewards.append(reward) # Immediate reward
            obs = next_obs

            if terminated or truncated:

                # TODO reset environment
                overall_rewards.append(sum(episode_rewards))
                obs, _ = self.env.reset()

                # TODO finish trajectory
                self.memory.finish_trajectory(0.0)
                episode_rewards = []
                episodes_counter += 1

                if episodes_counter == self.episodes_update:

                    # TODO optimize the actor
                    obs_lst, act_lst, rew_lst, logprob_lst, ret_lst = self.memory.get()
                    loss = self.calc_actor_loss(logprob_lst, ret_lst)
                    self.optim_actor.zero_grad()
                    loss.backward()
                    self.optim_actor.step()

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
        logprobs = torch.stack([torch.tensor(lp, dtype=torch.float32, requires_grad=True) for lp in logprob_lst])
        returns = torch.tensor(return_lst, dtype=torch.float32)
        return -(logprobs*returns).mean()


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
        obs_tensor = torch.tensor(obs, dtype = torch.float32)
        action_probs = self.actor_net(obs_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        logprob = dist.log_prob(action)

        if train_returns:
            # TODO Return action, logprob
            return action.item(), logprob.item()

        else:
            # TODO Return action
            return action.item()


if __name__ == '__main__':
    env_id = "CartPole-v1"
    _env = gym.make(env_id)
    vpg = VPG(_env)
    vpg.learn(10000)
