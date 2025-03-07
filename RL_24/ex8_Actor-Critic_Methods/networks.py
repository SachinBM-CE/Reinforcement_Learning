from torch import nn


class CriticNetwork(nn.Module):
    """Neural Network used to learn the state-value function."""

    def __init__(self, num_observations):
        super(CriticNetwork, self).__init__()

        # TODO 1.1 Implement network architecture (can be similar to Actor)
        self.net = None

    def forward(self, obs):

        # TODO 1.1 Forward pass
        return obs


class ActorNetwork(nn.Module):
    """Neural Network used to learn the policy."""

    def __init__(self, num_observations, num_actions):
        super(ActorNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        return self.net(obs)