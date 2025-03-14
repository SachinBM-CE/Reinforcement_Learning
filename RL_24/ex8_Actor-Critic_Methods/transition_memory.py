
######################
## For A2c and PPO  ##
######################

def compute_advantages(returns, values):
    """ Compute episode advantages based on precomputed episode returns.

    Parameters
    ----------
    returns : list of float
        Episode returns calculated with compute_returns.
    values: list of float
        Critic outputs for the states visited during the episode

    Returns
    -------
    list of float
        Episode advantages.
    """

    # TODO 1.4: Compute the advantages using equation 1.
    return [r - v for r, v in zip(returns, values)]


def compute_generalized_advantages(rewards, values, next_value, discount, lamb):
    """ Compute generalized advantages (GAE) of the episode.

    Parameters
    ----------
    rewards : list of float
        Episode rewards.
    values: list of float
        Episode state values.
    next_value : float
        Value of state the episode ended in. Should be 0.0 for terminal state, critic output otherwise.
    discount : float
        Discount factor.
    lamb: float
        Lambda parameter of GAE.

    Returns
    -------
    list of float
        Generalized advantages of the episode.
    """

    # TODO 1.8: Compute GAE using equation 3.
    # t: index for current time step [0, T-1]
    # l: index for steps ahead
    # For each t, l runs from 0 to T-t-1 (Summing over all future timesteps)
    GAE = 0
    advantages = []  # List to store computed GAEs in all the iterations
    # Loop from the end of the episode
    for r, v in zip(reversed(rewards), reversed(values)):
        TD_error = r + discount*next_value - v  # Temporal Difference Error
        GAE = TD_error + discount*lamb*GAE  # Recursive update of running total
        advantages.append(GAE)
        next_value = v  # value for next iteration
    return advantages[::-1]  # restore original time order


class TransitionMemoryAdvantage:
    """Datastructure to store episode transitions and perform return/advantage/generalized advantage calculations (GAE)
     at the end of an episode."""

    def __init__(self, gamma, lamb, use_gae):
        self.obs_lst, self.action_lst, self.reward_lst, self.logprob_lst, self.return_lst = [], [], [], [], []
        self.gamma = gamma
        self.traj_start = 0

        self.lamb = lamb
        self.use_gae = use_gae
        # TODO 1.2: Define additional datastructures

    def put(self, obs, action, reward, logprob, value):
        """Put a transition into the memory."""
        self.obs_lst.append(obs)
        self.action_lst.append(action)
        self.reward_lst.append(reward)
        self.logprob_lst.append(logprob)

        # TODO 1.2: append to new datastructures

    def get(self):
        """Get all stored transition attributes in the form of lists."""
        # TODO 1.2: Return new datastructures
        return self.obs_lst, self.action_lst, self.reward_lst, self.logprob_lst, self.return_lst, None, None

    def clear(self):
        """Reset the transition memory."""
        self.obs_lst, self.action_lst, self.reward_lst, self.logprob_lst, self.return_lst = [], [], [], [], []
        self.traj_start = 0

        # TODO 1.2: Clear new datastructures

    def finish_trajectory(self, next_value=0.0):
        """Call on end of an episode. Will perform episode return and advantage or generalized advantage estimation.

        Parameters
        ----------
        next_value:
            The value of the state the episode ended in. Should be 0.0 for terminal state, critic output otherwise.
        """
        reward_traj = self.reward_lst[self.traj_start:]
        return_traj = compute_returns(reward_traj, next_value, self.gamma)
        self.return_lst.extend(return_traj)

        # TODO 1.2: Extract values before updating trajectory termination counter
        value_traj = []

        self.traj_start = len(self.reward_lst)

        if self.use_gae:
            traj_adv = compute_generalized_advantages(reward_traj, value_traj, next_value, self.gamma, self.lamb)
        else:
            traj_adv = compute_advantages(return_traj, value_traj)

        # TODO 1.2: Append computed advantage to new datastructure


##############
## For VPG  ##
##############

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

    return_lst = []
    ret = next_value
    for reward in reversed(rewards):
        ret = reward + discount * ret
        return_lst.append(ret)
    return return_lst[::-1]


class TransitionMemory:
    """Datastructure to store episode transitions and perform return at the end of an episode."""

    def __init__(self, gamma):
        self.obs_lst, self.action_lst, self.reward_lst, self.logprob_lst, self.return_lst = [], [], [], [], []
        self.gamma = gamma
        self.traj_start = 0

    def put(self, obs, action, reward, logprob):
        """Put a transition into the memory."""
        self.obs_lst.append(obs)
        self.action_lst.append(action)
        self.reward_lst.append(reward)
        self.logprob_lst.append(logprob)

    def get(self):
        """Get all stored transition attributes in the form of lists."""
        return self.obs_lst, self.action_lst, self.reward_lst, self.logprob_lst, self.return_lst

    def clear(self):
        """Reset the transition memory."""
        self.obs_lst, self.action_lst, self.reward_lst, self.logprob_lst, self.return_lst = [], [], [], [], []
        self.traj_start = 0

    def finish_trajectory(self, next_value=0.0):
        """Call on end of an episode. Will perform episode return or advantage or generalized advantage estimation (later exercise).

        Parameters
        ----------
        next_value:
            The value of the state the episode ended in. Should be 0.0 for terminal state.
        """
        reward_traj = self.reward_lst[self.traj_start:]
        return_traj = compute_returns(reward_traj, next_value, self.gamma)
        self.return_lst.extend(return_traj)
        self.traj_start = len(self.reward_lst)
