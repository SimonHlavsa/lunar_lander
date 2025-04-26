
# config.py - nastavení hyperparametrů

RUN_ID = 5
ENV_NAME = "LunarLander-v3"
NUM_EPISODES = 1500
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.0005
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.998
TARGET_UPDATE = 10
BUFFER_CAPACITY = 100_000
MODEL_SAVE_PATH = f"saved_models/model_run{RUN_ID}.pth"
PLOT_SAVE_PATH = f"plots/reward_curve_run{RUN_ID}.png"

