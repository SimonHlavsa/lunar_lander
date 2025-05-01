"""
Off-policy benchmark: načte natrénovaný model a odehraje pevný počet epizod
"""

import csv

import gymnasium as gym
import numpy as np
import torch

import config
from model import QNetwork

env = gym.make(config.ENV_NAME)

state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

net = QNetwork(state_dim, n_actions)
net.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
net.eval()

rewards, steps, successes, hard_landings = [], [], [], []

with open(config.BENCH_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward", "steps", "success"])

    for ep in range(1, config.BENCH_EPISODES + 1):
        state, _ = env.reset()
        done, ep_rew, ep_steps = False, 0.0, 0

        while not done:
            with torch.no_grad():
                action = net(
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                ).argmax().item()
            state, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_rew += r
            ep_steps += 1

        rewards.append(ep_rew)
        steps.append(ep_steps)
        success = int(ep_rew >= config.SUCCESS_THRESHOLD)
        successes.append(success)
        hard_landings.append(int(ep_rew < -100))

        writer.writerow([ep, f"{ep_rew:.1f}", ep_steps, success])

env.close()

print(f"Benchmark ({config.BENCH_EPISODES} episodes)")
print(f"Avg reward : {np.mean(rewards):7.1f}")
print(f"Success-rate (≥200) : {np.mean(successes):7.2%}")
print(f"Avg steps per episode : {np.mean(steps):7.1f}")
print(f"Hard landings (<-100) : {np.mean(hard_landings):7.2%}")
print(f"CSV saved to : {config.BENCH_CSV_PATH}")
