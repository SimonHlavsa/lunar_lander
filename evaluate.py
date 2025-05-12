"""
Interaktivní demo: spustí natrénovaného agenta v renderovacím režimu
prostředí Gymnasium a zobrazí průběh jedné epizody.

Použití:
    python evaluate.py
"""

import gymnasium as gym
import torch

import config
from model import QNetwork

# příprava
env = gym.make(config.ENV_NAME, render_mode="human")

state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = QNetwork(state_dim, n_actions)
policy_net.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
policy_net.eval()

# jedna epizoda
state, _ = env.reset()
done = False

while not done:
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = policy_net(state_t).argmax().item()
    state, _, term, trunc, _ = env.step(action)
    done = term or trunc

env.close()
