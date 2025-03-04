import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

# List to store episode rewards
episode_rewards = []

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)

# Compute discounted rewards
def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
    return discounted_rewards

# Train the model with an epsilon-greedy exploration strategy
def train(env, policy, optimizer, episodes=2000, epsilon=0.1):
    for episode in range(episodes):
        result = env.reset()  # Handle both old and new Gym versions
        state = result[0] if isinstance(result, tuple) else result  # Extract state safely
        log_probs = []
        rewards = []
        done = False

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state)
            m = Categorical(probs)

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore randomly
            else:
                action = m.sample().item()  # Exploit best known action

            result = env.step(action)  # Ensure compatibility
            state, reward, done, _ = result if len(result) == 4 else (result + (False,))

            log_probs.append(m.log_prob(torch.tensor(action)))
            rewards.append(reward)

            # Terminate if cart moves out of bounds
            if abs(state[0]) > 2.4 or abs(state[2]) > 0.2095:  # 12 degrees (0.2095 radians)
                done = True

            if done:
                episode_rewards.append(sum(rewards))
                discounted_rewards = compute_discounted_rewards(rewards)
                policy_loss = []
                for log_prob, Gt in zip(log_probs, discounted_rewards):
                    policy_loss.append(-log_prob * Gt)
                optimizer.zero_grad()
                policy_loss = torch.cat(policy_loss).sum()
                policy_loss.backward()
                optimizer.step()

                if episode % 100 == 0:
                    print(f"Episode {episode}, Total Reward: {sum(rewards)}")
                break

    # Save the trained model
    torch.save(policy.state_dict(), "cartpole_model.pth")
    print("Model saved as cartpole_model.pth")

# Load the trained model
def load_model(policy, model_path="cartpole_model.pth"):
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    print("Model loaded successfully.")

# Visualize and save simulation as GIF
def visualize_gif(env, policy, gif_path="cartpole_simulation.gif", fps=30):
    frames = []
    result = env.reset()
    state = result[0] if isinstance(result, tuple) else result  # Handle both old and new Gym versions
    done = False

    while not done:
        frame = env.render(mode="rgb_array")  # Should work in older Gym
        frames.append(Image.fromarray(frame))

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = policy(state_tensor)
        action = probs.argmax(dim=-1).item()

        result = env.step(action)
        state, _, done, _ = result if len(result) == 4 else (result + (False,))

        # Stop if the cart moves out of bounds
        if abs(state[0]) > 2.4 or abs(state[2]) > 0.2095:
            done = True

    env.close()
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=1000//fps, loop=0, optimize=True)
    print(f"Optimized GIF saved at: {gif_path}")


# Plot the training rewards
def plot_training_rewards():
    plt.plot(episode_rewards)
    plt.title('Training Reward Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

# Main function to execute the steps in order
def main():
    env = gym.make('CartPole-v1') 
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)  # Lower LR for stability

    # Train the model
    train(env, policy, optimizer, episodes=2000, epsilon=0.1)

    # Plot training rewards
    plot_training_rewards()

    # Load and visualize the trained model
    load_model(policy)
    visualize_gif(env, policy, gif_path="cartpole_simulation.gif")

if __name__ == "__main__":
    main()
