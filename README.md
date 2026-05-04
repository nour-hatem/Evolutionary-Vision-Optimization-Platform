# Genetic Algorithm Optimizer

**Module owner:** Ahmad  
**Project:** Vision Feature Selection and Neural Evolution Platform  
**Course:** Cloud Computing — Spring 2026  

This module runs a Genetic Algorithm (GA) to find the best hyperparameter configuration for Yassin's CNN. The best config is saved as `best_config.json` and handed to Yassin for final training. The generation log is saved as `ga_log.csv` and sent to Nour for the dashboard.

---

## Results

The GA completed 10 generations with a population of 20. Best validation accuracy achieved during the 5-epoch search: **73.38%**

| Gene | Best value found |
|---|---|
| Learning rate | 0.0003 |
| Filters | 96 |
| Layers | 4 |
| Dropout | 0.3 |
| Batch size | 32 |

The population converged steadily from **70.02%** (Gen 0) to **73.38%** (Gen 10), with average fitness rising from **56.97%** to **69.54%** — showing the whole population improved, not just the single best individual.

---

## Files

| File | Description |
|---|---|
| `ga_optimizer_final.py` | Main script — runs the GA |
| `best_config.json` | Best hyperparameter config found — send to Yassin |
| `ga_log.csv` | Per-generation fitness stats — send to Nour |
| `ga_checkpoint.json` | Full population state saved after every generation |

---

## How It Works

The GA treats CNN hyperparameters as a **5-gene chromosome**:

```
[lr_index, filter_index, layer_index, dropout_index, batch_size_index]
```

Each gene is an index into a list of valid values. For example `[2, 2, 2, 0, 0]` decodes to `lr=0.0003, filters=96, layers=4, dropout=0.3, batch_size=32`.

Each generation the GA:
1. **Selects** parents via tournament selection (pick 3, keep the best)
2. **Crosses over** pairs of parents (two-point crossover, 70% chance)
3. **Mutates** individuals randomly (30% per individual, 30% per gene)
4. **Evaluates** only new/changed individuals by calling Yassin's `train()`
5. **Saves** a checkpoint so no work is lost if the run crashes

The best individual is always carried forward unchanged (elitism).

---

## Search Space

| Hyperparameter | Options |
|---|---|
| Learning rate | 0.01, 0.005, 0.003, 0.001, 0.0005, 0.0003, 0.0001 |
| Num filters | 32, 64, 96, 128 |
| Num layers | 2, 3, 4, 5 |
| Dropout | 0.3, 0.4, 0.5 |
| Batch size | 32, 64, 128 |

---

## Setup

**Requirements:**
```
Python 3.9+
deap
numpy
torch
```

**Install:**
```bash
pip install deap numpy torch
```

**Required folder structure** (Yassin's and Nour's modules must be present):
```
project/
├── ga_optimizer_final.py
├── model/
│   ├── __init__.py
│   ├── train.py          ← Yassin's module
│   └── cnn.py
└── data/
    ├── __init__.py
    └── loaders.py        ← Nour's module
```

---

## Usage

```bash
# Full run with Yassin's real model (~5–6 hours on CPU)
python ga_optimizer_final.py

# Resume after a crash — picks up from last saved checkpoint
python ga_optimizer_final.py --resume

# Quick sanity check — 1 generation x 3 individuals
python ga_optimizer_final.py --smoke-test
```

---

## Output Files

### `best_config.json`
The winning hyperparameter config, ready for Yassin to use for final retraining:
```json
{
  "best_config": {
    "lr": 0.0003,
    "num_filters": 96,
    "num_layers": 4,
    "dropout": 0.3,
    "batch_size": 32,
    "epochs": 15
  },
  "best_val_accuracy_during_search": 0.7338,
  "search_epochs_used": 5,
  "full_train_epochs": 15
}
```

### `ga_log.csv`
One row per generation — used by Nour's Streamlit dashboard to draw the convergence chart:

| Column | Description |
|---|---|
| `generation` | Generation number (0–10) |
| `best_fitness` | Highest validation accuracy this generation |
| `avg_fitness` | Average accuracy across all 20 individuals |
| `worst_fitness` | Lowest accuracy this generation |
| `best_lr` | Learning rate of the best individual |
| `best_num_filters` | Filter count of the best individual |
| `best_num_layers` | Layer count of the best individual |
| `best_dropout` | Dropout of the best individual |
| `best_batch_size` | Batch size of the best individual |
| `elapsed_seconds` | Total seconds elapsed since the GA started |

### `ga_checkpoint.json`
Saved after every generation. Contains the full population (genes + fitness + decoded config). Used by `--resume` to restart a crashed run without repeating any training.

---

## Integration with Other Modules

| Output | Goes to | Used for |
|---|---|---|
| `best_config.json` | Yassin | Re-train CNN with best config for 15 epochs |
| `ga_log.csv` | Nour | Plot convergence chart on Streamlit dashboard |

The fitness function calls Yassin's `train()` with `fast_mode=True` (5 epochs per call) during the search, then the winner is retrained for the full 15 epochs. Nour's `get_loaders()` is used internally by Yassin's `train()` for the data pipeline.

---

## Reproducibility

All random number generators are seeded with `SEED = 42` at the start of every run:
- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA (if available)
- `cudnn.deterministic = True` and `cudnn.benchmark = False`

Running with the same seed always produces the same population and evolution path.
