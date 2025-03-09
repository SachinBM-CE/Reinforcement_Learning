from abc import ABC, abstractmethod
from typing import Any, Dict
from copy import deepcopy
import random

import numpy as np
from tqdm import tqdm

from env import Connect4Env


def random_valid_action(env: Connect4Env):
    """Returns a random valid action for the current environment."""
    actions = np.arange(env.action_space.n)
    valid_actions = actions[env.action_mask()]
    return np.random.choice(valid_actions)


def randomized_argmax(x: np.ndarray):
    """Returns a random index from all the indices that have max value."""
    return np.random.choice(np.argwhere(x == x.max()).flatten())


class MCTSNode:
    def __init__(
            self,
            env: Connect4Env,
            state: Dict[str, Any],
            c_param: float = np.sqrt(2),
            action: int = None,  # We set these parameters to `None` for the root node
            reward: float = None,
            done: bool = None,
            parent: "MCTSNode" = None
    ):
        self.env = deepcopy(env)  # Copy, such that we have an encapsulated environment we can work with
        self.state = state
        self.c_param = c_param

        self.action = action
        self.reward = reward
        self.done = done
        self.parent = parent
        self.expanded = False

        self.num_actions = self.env.action_space.n

        self.valid_mask = self.state["valid_mask"]  # Array of `True` and `False` that can easily be used for indexing an array
        self.valid_moves = self.state["valid_moves"]  # Array of actions that are valid
        self.player = self.state["player"]

        self.child_num_visits = np.zeros(self.num_actions)
        self.child_value_sum = np.zeros(self.num_actions)
        self.children = [None] * self.num_actions

    @property
    def value(self):
        """Returns the value of this node."""
        return self.parent.child_value_sum[self.action]

    @value.setter
    def value(self, x):
        """Sets the value of this node."""
        self.parent.child_value_sum[self.action] = x

    @property
    def num_visit(self):
        """Return the visit counter for this state."""
        return self.parent.child_num_visits[self.action]

    @num_visit.setter
    def num_visit(self, x):
        """Sets the visit counter for this state."""
        self.parent.child_num_visits[self.action] = x

    def child_values(self):
        """Returns all the child values (i.e. value sum divided by visit counts).
        Take into special account the children that have not been visited yet (i.e. danger of division by zero).
        """
        # TODO 2)
        # Note: you have to deal specifically with visitation counts == 0 (i.e., division by zero)
        # You can just set those values to zero
        values = None
        return values

    def child_exploration(self):
        """Returns the exploration term (weighted by `c_param`).
        Take into special account the children that have not been visited yet (i.e. danger of division by zero).
        """
        # TODO 2)
        # Note: you have to deal specifically with N (total number of visits) == 0 (i.e., np.sqrt(0))
        exploration = None
        return exploration

    def uct(self):
        """Computes the UCT terms for all children."""
        uct = None
        # TODO 2)
        return uct

    def uct_action(self):
        """Samples the next action, based on current UCT values.
        Mask out the UCT values for invalid moves, i.e., by setting them to `-np.inf`. Use the `self.valid_mask` for this.
        """
        uct = self.uct()
        uct[~self.valid_mask] = -np.inf
        return randomized_argmax(uct)

    def selection(self) -> "MCTSNode":
        """Traverses the tree starting from the current node until a not expanded node is reached."""
        current = self
        while current.expanded:
            # TODO 3)
            pass
        return current

    def expansion(self) -> "MCTSNode":
        """Expands the node, i.e.:
        1) Create a `MCTSNode` object for every valid move.
        2) Take a random node child and return it.
        """
        if self.done:
            return self

        # TODO 3)
        for action in self.valid_moves:
            # Add empty child nodes

        # Pick a random child
        child_node = None
        return child_node

    def simulation(self) -> float:
        """Simulates a `num_rollouts_per_simulation` rollouts from this node.
        Returns the mean value.
        """
        if self.done:
            return self.reward

        # TODO 3)
        # Do a random simulation
        return reward

    def backpropagation(self, reward) -> None:
        """Back propagates the games' outcome up the search tree.
        At every step, you have to take into account which player's turn it was.
        A positive outcome for the player is a negative outcome for the opponent (hint: Flip the reward)
        """
        current = self
        if reward == 0.0:  # We have a draw
            reward = 0.5

        while current.parent is not None:
            # TODO 3) Implement the backpropagation
            # Remember to flip the win indicator every backprop step
            pass


class BaseAgent(ABC):
    @abstractmethod
    def compute_action(self, env: Connect4Env) -> int:
        ...


class RandomAgent(BaseAgent):
    def compute_action(self, env: Connect4Env) -> int:
        return random_valid_action(env)


class MCTSAgent(BaseAgent):
    def __init__(
            self,
            num_simulations: int,
            c_param: float = np.sqrt(2)
    ):
        self.num_simulations = num_simulations
        self.c_param = c_param

    def compute_action(self, env: Connect4Env) -> int:
        env = deepcopy(env)  # So that we don't alter the original environment
        root_node = MCTSNode(env, env.get_state(), c_param=self.c_param)  # Create root node

        for _ in range(self.num_simulations):  # Do N simulations
            leaf_node = root_node.selection()
            child_node = leaf_node.expansion()
            terminal_reward = child_node.simulation()
            child_node.backpropagation(terminal_reward)

        child_values = root_node.child_value_sum / root_node.child_num_visits
        return randomized_argmax(root_node.child_value_sum / root_node.child_num_visits)  # Select the action based on some criteria


def arena(player: BaseAgent, opponent: BaseAgent, env: Connect4Env, render: bool = False) -> int:
    env.reset()

    while True:
        player_action = player.compute_action(env)  # Calculate action
        _, reward, done, _, _ = env.step(player_action)  # Query player for action

        if render:  # Render
            env.render()

        # Return will be:
        # * 1 if player won
        # * 0 if we have a draw
        # * -1 if the player did an incorrect move -> opponent won
        if done:  # Check if done, report outcome if so
            return reward

        opponent_action = opponent.compute_action(env)
        _, reward, done, _, _ = env.step(opponent_action)

        if render:
            env.render()

        # Return will be:
        # * -1 if opponent won
        # * 0 if we have a draw
        # * 1 if the player did an incorrect move -> player won
        if done:
            return -reward


if __name__ == '__main__':
    env = Connect4Env()
    obs, _ = env.reset()

    # Initialize the two players
    player = MCTSAgent(
        num_simulations=200,
        c_param=1
    )
    opponent = RandomAgent()
    # opponent = MCTSAgent(
    #    num_simulations=200,
    #    c_param=1
    # )

    # Quantitative evaluation
    eval = False  # Just set this to `False` to skip the quantitative evaluation
    if eval:
        num_games = 10
        player_wins = 0
        opponent_wins = 0
        draws = 0
        for _ in tqdm(range(num_games)):
            outcome = arena(player, opponent, env)
            if outcome == 1:
                player_wins += 1
            elif outcome == -1:
                opponent_wins += 1
            else:
                draws += 1

        print(f"Outcome distribution of {num_games} games player:")
        print(f"Player wins: {player_wins}")
        print(f"Opponent wins: {opponent_wins}")
        print(f"Draws: {draws}")

    # Qualitative evaluation
    arena(player, opponent, env, render=True)
