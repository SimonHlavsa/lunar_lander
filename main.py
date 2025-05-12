"""
Hlavní trénovací skript DQN pro LunarLander-v3
"""

import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import config
from buffer import ReplayBuffer
from model import QNetwork
from utils import (
    plot_rewards_only,
    plot_success_rate,
    select_action,
    write_csv_log,
)

# složky pro výstupy
os.makedirs("saved_models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("csv", exist_ok=True)

# deterministické prostředí a RNG
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

env = gym.make(config.ENV_NAME)
env.reset(seed=config.SEED)
env.action_space.seed(config.SEED)

state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = QNetwork(state_dim, n_actions)
target_net = QNetwork(state_dim, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

opt = optim.Adam(policy_net.parameters(), lr=config.LR)
buffer = ReplayBuffer(config.BUFFER_CAPACITY)

eps = config.EPS_START
rew_hist, suc_hist, ma_hist = [], [], []
rew_win = deque(maxlen=config.MOVING_AVG_WIN)
suc_win = deque(maxlen=200)

for ep in range(config.NUM_EPISODES):
    state, _ = env.reset()
    done, total_rew = False, 0.0

    while not done:
        action = select_action(state, eps, policy_net, n_actions)
        next_state, rew, term, trunc, _ = env.step(action)
        done = term or trunc

        buffer.add((state, action, rew, next_state, done))
        state = next_state
        total_rew += rew

        if len(buffer) >= config.BATCH_SIZE:
            s, a, r, s2, d = buffer.sample(config.BATCH_SIZE)
            s = torch.tensor(s, dtype=torch.float32)
            a = torch.tensor(a, dtype=torch.long)
            r = torch.tensor(r, dtype=torch.float32)
            s2 = torch.tensor(s2, dtype=torch.float32)
            d = torch.tensor(d, dtype=torch.float32)

            q = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            q_next = target_net(s2).max(1)[0]
            q_exp = r + config.GAMMA * q_next * (1 - d)

            loss = torch.nn.functional.smooth_l1_loss(q, q_exp)
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(policy_net.parameters(), 10)
            opt.step()

    # logování metrik
    rew_hist.append(total_rew)
    rew_win.append(total_rew)
    ma_val = np.mean(rew_win)
    ma_hist.append(ma_val)
    suc_flag = total_rew >= config.SUCCESS_THRESHOLD
    suc_hist.append(suc_flag)
    suc_win.append(suc_flag)

    write_csv_log(config.CSV_LOG_PATH, ep, total_rew, ma_val, eps, suc_flag)

    eps = max(config.EPS_END, eps * config.EPS_DECAY)

    if ep % config.TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if ep % 10 == 0:
        cur_sr = np.mean(suc_win)
        print(
            f"Ep {ep:4d} | reward {total_rew:6.1f} | "
            f"MA {ma_val:6.1f} | epsilon {eps:5.3f} | success-rate {cur_sr:4.2f}"
        )

    if len(ma_hist) >= 200:
        if np.mean(ma_hist[-200:]) >= 200 and np.mean(suc_hist[-200:]) >= 0.9:
            print(f"Kritérium splněno v epizodě {ep}.")
            break

# uložení modelu a grafů
torch.save(policy_net.state_dict(), config.MODEL_SAVE_PATH)
plot_rewards_only(rew_hist, ma_hist, config.REWARD_PLOT_PATH)
plot_success_rate(suc_hist, config.SRATE_PLOT_PATH)

env.close()
