# main.py - hlavní trénovací smyčka

import gymnasium as gym
import torch
import torch.optim as optim
import os
from model import QNetwork
from buffer import ReplayBuffer
from utils import select_action, plot_rewards
import config

# Vytvoření složek
os.makedirs("saved_models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Inicializace prostředí
env = gym.make(config.ENV_NAME)
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
            optimizer.step()

    reward_history.append(total_reward)

    epsilon = max(config.EPS_END, epsilon * config.EPS_DECAY)

    if episode % config.TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 10 == 0:
        print(f"Epizoda {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

# Uložení modelu
torch.save(policy_net.state_dict(), config.MODEL_SAVE_PATH)

# Plot odměn
plot_rewards(reward_history, config.PLOT_SAVE_PATH)

config.RUN_ID += 1

env.close()
