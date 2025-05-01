# main.py - hlavní trénovací smyčka

import gymnasium as gym
import torch
import torch.optim as optim
import os, random
from model import QNetwork
from buffer import ReplayBuffer
from utils import plot_rewards_only, plot_success_rate, select_action, write_csv_log
from collections import deque
import config
import numpy as np
from torch.nn.utils import clip_grad_norm_

# Vytvoření složek
os.makedirs("saved_models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("csv", exist_ok=True)

# Inicializace prostředí
# 1) globální inicializace random seedů -------------------------------
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

# 2) prostředí se seedem a deterministickou action space -------------
env = gym.make(config.ENV_NAME)
env.reset(seed=config.SEED)
env.action_space.seed(config.SEED)
# --------------------------------------------------------------------

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = QNetwork(state_size, action_size)
target_net = QNetwork(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=config.LR)
replay_buffer = ReplayBuffer(capacity=config.BUFFER_CAPACITY)

epsilon = config.EPS_START
reward_history = []
success_history = []
moving_avg = []
reward_window = deque(maxlen=config.MOVING_AVG_WIN)
success_window = deque(maxlen=200)

for episode in range(config.NUM_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = select_action(state, epsilon, policy_net, action_size)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.add((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(replay_buffer) >= config.BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(config.BATCH_SIZE)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_net(next_states).max(1)[0]
            expected_q_values = rewards + (config.GAMMA * next_q_values * (1 - dones))

            loss = torch.nn.functional.smooth_l1_loss(q_values, expected_q_values)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(policy_net.parameters(), max_norm=10)
            optimizer.step()

    reward_history.append(total_reward)
    reward_window.append(total_reward)
    moving_avg_val = np.mean(reward_window)
    moving_avg.append(moving_avg_val)
    success_flag = total_reward >= config.SUCCESS_THRESHOLD
    success_history.append(success_flag)
    success_window.append(success_flag)
    
    write_csv_log(
        config.CSV_LOG_PATH,
        episode,
        total_reward,
        moving_avg_val,
        epsilon,
        success_flag,
    )

    epsilon = max(config.EPS_END, epsilon * config.EPS_DECAY)

    if episode % config.TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 10 == 0:
        current_sr = np.mean(success_window) if success_window else 0.0
        print(f"Epizoda {episode}, Reward: {total_reward:.2f}, MA(100): {moving_avg_val:7.1f}, Epsilon: {epsilon:.3f}, SR(200): {current_sr:5.2f}")

    if len(moving_avg) >= 200:
        ma_window = np.mean(moving_avg[-200:])
        sr_window = np.mean(success_history[-200:])
        if ma_window >= 200 and sr_window >= 0.9:
            print(f"Kritérium splněno v epizodě {episode}. Trénink ukončen.")
            break

# Uložení modelu
torch.save(policy_net.state_dict(), config.MODEL_SAVE_PATH)

# Plot odměn
plot_rewards_only(reward_history, moving_avg, config.REWARD_PLOT_PATH)
plot_success_rate(success_history, config.SRATE_PLOT_PATH)

env.close()
