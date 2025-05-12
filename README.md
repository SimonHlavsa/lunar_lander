# DQN LunarLander – stručný návod

Implementace agenta **Deep Q‑Network (DQN)**, který se v prostředí
`gymnasium.LunarLander‑v3` učí bezpečně přistát s lunárním modulem. Repo
obsahuje tréninkový skript, vizuální demo a skript pro měření výkonu.

## Požadavky

* Python ≥ 3.12
* pip (aktuální)
* `pip install -r requirements.txt`

## Jak spustit

```bash
# trénink
python main.py

# interaktivní demo (vykreslí jednu epizodu)
python evaluate.py

# benchmark natrénovaného modelu (500 epizod)
python benchmark.py
```

Hyperparametry a cesty je možné měnit v `config.py`.

## Struktura

```
buffer.py      – replay buffer
config.py      – nastavení hyperparametrů
main.py        – trénink DQN
model.py       – Q‑síť
utils.py       – logování, grafy, epsilon‑greedy
evaluate.py    – vizuální demo
benchmark.py   – měření výkonu
```

Výstupy jdou do `saved_models/`, `plots/` a `csv/`.
