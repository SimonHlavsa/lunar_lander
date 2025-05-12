"""
Centrální konfigurace
"""

RUN_ID = "4"
ENV_NAME = "LunarLander-v3"

# náhodný seed
SEED = 42

# trénink
NUM_EPISODES = 3000
BATCH_SIZE = 128

# DQN
GAMMA = 0.99
LR = 0.0005
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 5

# replay buffer
BUFFER_CAPACITY = 100_000

# metriky
SUCCESS_THRESHOLD = 200
MOVING_AVG_WIN = 100

# benchmark
BENCH_EPISODES = 500      

# cesty k výstupům
MODEL_SAVE_PATH = f"saved_models/model_run_{RUN_ID}.pth"
REWARD_PLOT_PATH = f"plots/rewards_run_{RUN_ID}.png"
SRATE_PLOT_PATH = f"plots/srate_run_{RUN_ID}.png"
CSV_LOG_PATH = f"csv/train_log_run_{RUN_ID}.csv"
BENCH_CSV_PATH = f"csv/benchmark_run_{RUN_ID}.csv"
