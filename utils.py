# utils.py - pomocné funkce

import numpy as np
import torch
import matplotlib.pyplot as plt


def select_action(state, epsilon, q_net, action_size):
    if np.random.rand() < epsilon:
        return np.random.randint(action_size)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return q_net(state_tensor).argmax().item()


def plot_rewards(rewards, filename="plots/reward_curve.png"):
    plt.plot(rewards)
    plt.title("Vývoj odměny během tréninku")
    plt.xlabel("Epizoda")
    plt.ylabel("Celková odměna")
    plt.grid()
    plt.savefig(filename)
    plt.close()