import sys
import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldMDP:
    """A very simple gridworld MDP.
    
    Attributes
    ----------
    shape : list of int
        The shape of the gridworld
    num_states : int
        Number of states.
    num_actions : int
        Number of actions, always equal to 4. Actions are UP (0), RIGHT (1), DOWN (2), LEFT (3).
    P : dict
        P captures the state transition probabilities and the reward function. For every state s and every possible action a, 
        P[s][a] contains a list of tuples (p, s', r, is_terminal) with:
        - p: the probability of s' being the next state given s, a
        - s': the next state
        - r: the reward gained from this event
        - is_terminal: if s' is a terminal state

    Methods
    -------
    render()
        "Renders"/prints the gridworld to the terminal
    """

    def __init__(self, shape=[4, 4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape
        self.num_states = np.prod(shape)
        self.num_actions = 4
        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(self.num_states).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex # State number
            y, x = it.multi_index # (Row, Column)

            P[s] = {a: [] for a in range(self.num_actions)}
            
            # 1st & the last index are made terminal states,
            # all other states are non-terminal
            is_terminal = lambda s: s == 0 or s == (self.num_states - 1)
            reward = 0.0 if is_terminal(s) else -1.0

            # Rules to determine the agent's move during state transition s -> s'
            if is_terminal(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, reward, is_terminal(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_terminal(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_terminal(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_terminal(ns_left))]

            it.iternext()

        self.P = P

    def render(self):
        """Render the gridworld."""

        grid = np.arange(self.num_states).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index # y: row number, x: column number

            if s == 0 or s == self.num_states - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip() # Remove leading space for 1st column
            if x == self.shape[1] - 1:
                output = output.rstrip() # Remove trailing space for last column

            sys.stdout.write(output)

            if x == self.shape[1] - 1:
                sys.stdout.write("\n")

            it.iternext()


def print_value(x, mdp, form='.2f'):
    """ Print a value function array in a nice format."""
    # The value function x is a 1D array with length mdp.num_states.
    # It is reshaped into a 2D array matching the Gridworld’s shape.
    x = x.reshape(mdp.shape)
    print('\n'.join(' '.join(' ' + str(format(cell, form)) if cell >= 0 else str(format(cell, '.2f')) for cell in row) for row in x))


def print_deterministic_policy(policy, mdp):
    """ Print a policy array in a nice format."""
    # Mapping Actions to Characters
    action_dict = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
    # Determining the Best Action for Each State
    policy = np.array([action_dict[x] for x in np.argmax(policy, axis=1)]).reshape(mdp.shape)
    # Marking Terminal States
    policy[0, 0] = '-'
    policy[-1, -1] = '-'
    # Joins each row into a string with spaces between actions.
    # Joins rows into a multiline string.
    print('\n'.join(' '.join(str(cell) for cell in row) for row in policy))


def init_value(mdp):
    """ Returns a initialized value function array for given MDP."""
    # Used as a starting point before computing the value function.
    return np.zeros(mdp.num_states)


def random_policy(mdp):
    """ Returns the random policy for a given MDP.
    policy[x][y] is the probability of action with y for state x.
    """
    # equal probabilities (=1/num_actions) for each action.
    # For 4,4 Gridworld, it's size = (4*4 States, 4 Actions) = (16, 4)
    return np.ones([mdp.num_states, mdp.num_actions]) / mdp.num_actions
