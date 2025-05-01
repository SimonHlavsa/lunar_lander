# config.py – centrální konfigurace experimentu

# TEST RUN 1 - 33 minut
# TEST RUN 2 - 45 minut a 21 sekund
# TEST RUN 3a - 
# TEST RUN 3b - 
# TEST RUN 3c - 


RUN_ID            = "3a"                 # inkrementujte při každém novém tréninku
ENV_NAME          = "LunarLander-v3"

# --- náhodný seed ------------------------------------------------------------
SEED              = 42                # jediný zdroj pravdy pro replikovatelnost

# --- délka a granularita tréninku -------------------------------------------
NUM_EPISODES      = 3000              # delší trénink pro robustní konvergenci
BATCH_SIZE        = 128               # větší batch pro rychlejší sběr gradientů

# --- algoritmus DQN ----------------------------------------------------------
GAMMA             = 0.99              # diskontní faktor
LR                = 0.0005            # rychlost učení (ponecháno 5e-4)
EPS_START         = 1.0               # počáteční pravděpodobnost průzkumu
EPS_END           = 0.01              # minimální ε
EPS_DECAY         = 0.995             # pomalejší pokles ε
TARGET_UPDATE     = 5                 # frekvence synchronizace target-netu (v epizodách)

# --- replay buffer -----------------------------------------------------------
BUFFER_CAPACITY   = 100_000           # maximální velikost paměti

# --- metriky výkonu ----------------------------------------------------------
SUCCESS_THRESHOLD = 200               # hranice úspěšné epizody
MOVING_AVG_WIN    = 100               # okno klouzavého průměru odměn

# --- výstupy -----------------------------------------------------------------
MODEL_SAVE_PATH   = f"saved_models/model_run_{RUN_ID}.pth"
REWARD_PLOT_PATH  = f"plots/rewards_run_{RUN_ID}.png"
SRATE_PLOT_PATH   = f"plots/srate_run_{RUN_ID}.png"
CSV_LOG_PATH      = f"csv/train_log_run_{RUN_ID}.csv"
