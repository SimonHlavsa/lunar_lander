# utils.py - pomocné funkce

import numpy as np
import torch
import csv
import os
import matplotlib.pyplot as plt

def select_action(state, epsilon, q_net, action_size):
    if np.random.rand() < epsilon:
        return np.random.randint(action_size)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return q_net(state_tensor).argmax().item()

def plot_rewards_only(rewards, moving_avg, filename):
    episodes = np.arange(len(rewards))
    plt.figure()
    plt.plot(episodes, rewards, alpha=0.4, label="Reward per episode")
    plt.plot(episodes, moving_avg, linewidth=2, label=f"Moving avg ({len(moving_avg)})")
    plt.title("Reward and moving average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_success_rate(success_flags, filename):
    episodes = np.arange(len(success_flags))
    success_rate = np.cumsum(success_flags) / (episodes + 1)
    plt.figure()
    plt.plot(episodes, success_rate, linestyle="--")
    plt.title("Success rate over time")
    plt.xlabel("Episode")
    plt.ylabel("Success rate")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def write_csv_log(csv_path, episode, reward, moving_avg, epsilon, success):
    header = ["episode", "reward", "moving_avg", "epsilon", "success"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([episode, reward, moving_avg, epsilon, int(success)])

