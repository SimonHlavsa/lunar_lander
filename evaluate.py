
# evaluate.py - skript na testování bez tréninku

import gymnasium as gym
import torch
from model import QNetwork
import config

# Inicializace prostředí
env = gym.make(config.ENV_NAME, render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Načtení modelu
policy_net = QNetwork(state_size, action_size)
policy_net.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
policy_net.eval()

state, _ = env.reset()
done = False

while not done:
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = policy_net(state_tensor).argmax().item()

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state

env.close()