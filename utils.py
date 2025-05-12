"""
Pomocné funkce: výběr akce, logování a vykreslování metrik tréninku.
"""

import csv
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


def select_action(state, eps, net, n_actions):
    """
    Epsilon-greedy výběr akce.

    Zvolí náhodnou akci s pravděpodobností *eps*; v opačném
    případě vrátí akci s nejvyšší Q-hodnotou podle aktuální politiky.
    """
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return net(state_t).argmax().item()


def plot_rewards_only(rewards, ma, path):
    """
    Uloží graf průběhu odměn a jejich klouzavého průměru.
    """
    episodes = np.arange(len(rewards))
    plt.figure()
    plt.plot(episodes, rewards, alpha=0.4, label="Reward")
    plt.plot(episodes, ma, linewidth=2, label=f"Moving avg ({len(ma)})")
    plt.title("Reward and moving average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_success_rate(success, path):
    """
    Vykreslí kumulativní úspěšnost.
    """
    episodes = np.arange(len(success))
    rate = np.cumsum(success) / (episodes + 1)
    plt.figure()
    plt.plot(episodes, rate, linestyle="--")
    plt.title("Success rate")
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.ylim(0, 1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_csv_log(path, ep, rew, ma, eps, suc):
    """
    Zapíše výsledky jedné epizody do CSV.
    """
    header = ["episode", "reward", "moving_avg", "epsilon", "success"]
    write_head = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_head:
            w.writerow(header)
        w.writerow([ep, f"{rew:.1f}", f"{ma:.1f}", f"{eps:.4f}", int(suc)])
