import copy
from typing import Any, Tuple, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Used to prevent the agent from moving out of bounds in the grid.
def clamp(v, minimal_value, maximal_value):
    return min(max(v, minimal_value), maximal_value)


class GridWorldEnv(gym.Env):
    """
    This environment is a variation on common grid world environments.
    Observation:
        Type: Box(2)
        Num     Observation            Min                     Max
        0       x position             0                       self.map.shape[0] - 1
        1       y position             0                       self.map.shape[1] - 1
    Actions:
        Type: Discrete(4)
        Num   Action
        0     Go up
        1     Go right
        2     Go down
        3     Go left
        Each action moves a single cell.
    Reward:
        Reward is 0 at non-goal points, 1 at goal positions, and -1 at traps
    Starting State:
        The agent is placed at [0, 0]
    Episode Termination:
        Agent has reached a goal or a trap.
    Solved Requirements:
        
    """

    def __init__(self):
        self.map = [
            list("s   "),
            list("    "),
            list("    "),
            list("gt g"),
        ]

        # TODO: Define your action_space and observation_space here
        
        # https://www.gymlibrary.dev/api/spaces/#discrete
        # 0: Up, 1:Right, 2: Down, 3:Left
        self.action_space = spaces.Discrete(4)
        
        # https://www.gymlibrary.dev/api/spaces/#box
        # 2D coordinate [y,x] where y: row_num, x: col_num
        self.observation_space = spaces.Box(low=0, high=4, shape=(2,),
                                            dtype=np.int32)

        self.agent_position = [0, 0]


    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        
        # TODO: Write your implementation here
        
        # Reset agent to the start position
        self.agent_position = [0, 0]

        return self._observe(), {}


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        # TODO: Write your implementation here
        
        # Validates the action input (must be within [0,3]).
        assert self.action_space.contains(action)
        
        match action:
            case 0:  # To move up => y-1
                self.agent_position[0] -= 1
            case 1:  # To move right => x+1
                self.agent_position[1] += 1
            case 2:  # To move down => y+1
                self.agent_position[0] += 1
            case _:  # To move left => x-1
                self.agent_position[1] -= 1

        # Prevents the agent from moving out of the grid (coordinates are clamped between [0,3])                
        self.agent_position[0] = clamp(self.agent_position[0], 0, 3)
        self.agent_position[1] = clamp(self.agent_position[1], 0, 3)
        
        reward, done = 0, False
        if 't' == self.map[self.agent_position[0]][self.agent_position[1]]:
            reward, done = -1, True
        if 'g' == self.map[self.agent_position[0]][self.agent_position[1]]:
            reward, done = +1, True

        # no trap or no goal => 0 reward, continue episode
        reward, done = 0, False

        # trap => -1 reward, terminate episode 
        if 't' == self.map[self.agent_position[0]][self.agent_position[1]]:
            reward, done = -1, True
        # goal => +1 reward, terminate episode    
        if 'g' == self.map[self.agent_position[0]][self.agent_position[1]]:
            reward, done = +1, True

        # get the agent’s current position in the grid after taking an action
        observation = self._observe()
        
        return observation, reward, done, False, {}


    def render(self):
        rendered_map = copy.deepcopy(self.map)
        rendered_map[self.agent_position[0]][self.agent_position[1]] = "A"
        print("--------")
        for row in rendered_map:
            print("".join(["|", " "] + row + [" ", "|"]))
        print("--------")
        return None


    def close(self):
        pass


    def _observe(self):
        return np.array(self.agent_position)
