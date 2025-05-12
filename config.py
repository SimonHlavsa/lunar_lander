"""
Globální konfigurační soubor pro experimenty s DQN agentem v prostředí
gymnasium.LunarLander-v3.
"""

# === Identifikace runu =======================================================
RUN_ID = "4"  # ID runu
ENV_NAME = "LunarLander-v3"  # prostředí

# === SEED ====================================================================
SEED = 42  # globální seed pro random, NumPy i PyTorch

# === Parametry tréninku ======================================================
NUM_EPISODES = 3_000  # maximální počet epizod
BATCH_SIZE = 128      # velikost mini‑batche při aktualizaci sítě

# === Hyperparametry ==========================================================
GAMMA = 0.99          # diskontní faktor
LR = 5e-4             # rychlost učení.
EPS_START = 1.0       # počáteční hodnota epsilon‑greedy
EPS_END = 0.01        # minimální epsilon
EPS_DECAY = 0.995     # rozpad epsilonu po každé epizodě
TARGET_UPDATE = 5     # frekvence kopírování parametrů do target‑sítě

# === Replay buffer ===========================================================
BUFFER_CAPACITY = 100_000  # maximální velikost bufferu

# === Metriky úspěšnosti ======================================================
SUCCESS_THRESHOLD = 200    # reward > 200
MOVING_AVG_WIN = 100       # šířka klouzavého okna pro výpočet průměru rewardu

# === Benchmark ===============================================================
BENCH_EPISODES = 500       # počet epizod pro benchmark

# === Výstupní cesty ==========================================================
MODEL_SAVE_PATH = f"saved_models/model_run_{RUN_ID}.pth"
REWARD_PLOT_PATH = f"plots/rewards_run_{RUN_ID}.png"
SRATE_PLOT_PATH = f"plots/srate_run_{RUN_ID}.png"
CSV_LOG_PATH = f"csv/train_log_run_{RUN_ID}.csv"
BENCH_CSV_PATH = f"csv/benchmark_run_{RUN_ID}.csv"