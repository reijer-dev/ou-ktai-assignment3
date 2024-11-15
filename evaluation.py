import os
import torch
import numpy as np
import gym
import torch.nn as nn
from torch.distributions.categorical import Categorical
from lunarcustom import LunarLanderCustom

agent_names = ["normal_game_from_scratch_agent.pt", "shifted_game_from_scratch_agent.pt", "shifted_game_transfer_agent.pt"]

# Define the layer initialization function
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Agent class matching the training setup
class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = np.array(env.observation_space.shape).prod()
        n_actions = env.action_space.n

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_action(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation function
def evaluate_policy(agent, env, episodes=1000):
    total_rewards = []
    success_count = 0

    for episode in range(episodes):
        print(f"Episode: {episode + 1}")
        obs = env.reset()  # Adjusted for Gym versions < 0.26
        done = False
        total_reward = 0

        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device) if len(obs) == 8 else torch.zeros((1, 8)).to(device)
            with torch.no_grad():
                action = agent.get_action(obs_tensor)
            action = action.cpu().numpy().item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        total_rewards.append(total_reward)

        # Consider an episode a success if total reward exceeds 200
        if total_reward >= 200:
            success_count += 1

    avg_reward = np.mean(total_rewards)
    success_rate = success_count / episodes * 100
    return avg_reward, success_rate

# Initialize the agent and load the trained model
for agent_name in agent_names:
    print(agent_name)

    # Load the environment
    env = LunarLanderCustom()
    if "shifted" not in agent_name:
        env = gym.make("LunarLander-v2")
    agent = Agent(env=env).to(device)
    state_dict = torch.load(agent_name, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()

    # Run evaluations
    episodes = 100  # Number of episodes to evaluate
    avg_reward, success_rate = evaluate_policy(agent, env, episodes)

    # Display the evaluation results
    print(f"Evaluation over {episodes} episodes:")
    print(f"Average Reward: {avg_reward}")
    print(f"Success Rate: {success_rate}%")

    # Close the environment after evaluation
    env.close()
    env.reset()
