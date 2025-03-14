from torch import nn


class CriticNetwork(nn.Module):
    """Neural Network used to learn the state-value function."""

    def __init__(self, num_observations, size_critic):
        super(CriticNetwork, self).__init__()

        # TODO 1.1 Implement network architecture (can be similar to Actor)
        self.net = nn.Sequential(
            nn.Linear(num_observations, size_critic),
            nn.ReLU(),
            nn.Linear(size_critic, 1)
            )

    def forward(self, obs):

        # TODO 1.1 Forward pass
        return self.net(obs)


class ActorNetwork(nn.Module):
    """Neural Network used to learn the policy."""

    def __init__(self, num_observations, num_actions, size_actor):
        super(ActorNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_observations, size_actor),
            nn.ReLU(),
            nn.Linear(size_actor, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        return self.net(obs)
