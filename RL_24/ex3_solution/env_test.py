import time

from gridworld import GridWorldEnv


if __name__ == "__main__":
    
    # Create the environment
    env = GridWorldEnv()

    # Reset
    obs = env.reset()
    env.render()

    for i in range(50):

        # Chooses random action (1 out of 4 directions)
        action = env.action_space.sample()

        # Executes the selected action
        obs, reward, terminated, truncated, info = env.step(action)

        # Updates the grid and renders the new state
        env.render()

        # Uncomment this to enable slow motion mode
        time.sleep(0.5)

        # If terminated = True (trap or goal) then,
        # resets the environment and places the agent back at [0,0]
        if terminated:
            env.reset()
            env.render()
            
    env.close()
