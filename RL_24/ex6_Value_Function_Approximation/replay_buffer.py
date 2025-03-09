import random
from collections import deque


class ReplayBuffer:
    """A replay buffer as commonly used for off-policy Q-Learning methods."""

    def __init__(self, capacity):
        """Initializes replay buffer with certain capacity."""

        self.buffer = deque(maxlen=capacity)

    def put(self, s, a, r, s_, terminated):
        """Put a tuple of (obs, action, rewards, next_obs, terminated) into the replay buffer.
        The max length specified by capacity should never be exceeded.
        The oldest elements inside the replay buffer should be overwritten first.
        """

        # TODO 1.4: Implement replay buffer
        pass

    def get(self, batch_size):
        """Gives batch_size samples from the replay buffer."""

        # TODO 1.4: Implement replay buffer
        pass

    def __len__(self):
        """Returns the number of tuples inside the replay buffer."""

        # TODO 1.4: Implement replay buffer
        return 0