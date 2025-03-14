import time

import numpy as np
import matplotlib.pyplot as plt
import math

import gymnasium as gym
import neptune

def rolling_window(a, window, step_size):
    """Create a rolling window view of a numpy array.

    Parameters
    ----------
    a : numpy.array
    window : int
    step_size : int
    """

    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size + 1, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


fig = None
def episode_reward_plot(rewards, frame_idx, window_size=5, step_size=1, wait=False, saving_on=False, run_neptune=False):
    """Plot episode rewards rolling window mean, min-max range and standard deviation.

    Parameters
    ----------
    rewards : list
        List of episode rewards.
    frame_idx : int
        Current frame index.
    window_size : int
    step_size: int
    """

    global fig
    plt.ion()

    rewards_rolling = rolling_window(np.array(rewards), window_size, step_size)
    mean = np.mean(rewards_rolling, axis=1)
    std = np.std(rewards_rolling, axis=1)
    minimum = np.min(rewards_rolling, axis=1)
    maximum = np.max(rewards_rolling, axis=1)
    x = np.arange(math.floor(window_size/2), len(rewards) - math.floor(window_size/2), step_size)

    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111)

    ax.set_title('Timestep %s. Reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    ax.plot(x, mean, color='blue')
    ax.fill_between(x, mean-std, mean+std, alpha=0.3, facecolor='blue')
    ax.fill_between(x, minimum, maximum, alpha=0.1, facecolor='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    fig.canvas.draw()
    fig.canvas.flush_events()

    if saving_on:
        filename = "reward_episode.png"
        fig.savefig(filename)
        if run_neptune is not None:  # Ensure run exists globally before using it
            run_neptune["plots/reward"].upload(filename)  # Upload to Neptune

    if wait:
        time.sleep(1)


def visualize_agent(env, agent, timesteps=500):
    """ Visualize an agent performing inside a Gym environment. """

    obs, _ = env.reset()
    for timestep in range(1, timesteps + 1):
        env.render()
        action = agent.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

        # 30 FPS
        time.sleep(0.033)
