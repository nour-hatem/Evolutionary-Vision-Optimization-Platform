import argparse
import json
import random
import time
import csv

import numpy as np
import torch
from deap import base, creator, tools

# ─────────────────────────────────────────────
# SEARCH SPACE  (improvement 1 — dropout + batch_size added)
# ─────────────────────────────────────────────
LR_CHOICES         = [0.01, 0.005, 0.003, 0.001, 0.0005, 0.0003, 0.0001]
FILTER_CHOICES     = [32, 64, 96, 128]
LAYER_CHOICES      = [2, 3, 4, 5]
DROPOUT_CHOICES    = [0.3, 0.4, 0.5]   # NEW — regularisation strength
BATCH_SIZE_CHOICES = [32, 64, 128]      # NEW — gradient noise + speed trade-off

# ─────────────────────────────────────────────
# GA HYPERPARAMETERS
# ─────────────────────────────────────────────
POPULATION_SIZE = 20
N_GENERATIONS   = 10
CROSSOVER_PROB  = 0.7
MUTATION_PROB   = 0.3
GENE_MUTPB      = 0.3
EPOCHS_SEARCH   = 5
EPOCHS_FINAL    = 15
ELITE_SIZE      = 1
SEED            = 42

LOG_FILE        = "ga_log.csv"
CONFIG_FILE     = "best_config.json"
CHECKPOINT_FILE = "ga_checkpoint.json"  # NEW — resume file

# ─────────────────────────────────────────────
# FIXED CONFIG FIELDS  (not tuned by GA)
# ─────────────────────────────────────────────
BASE_CONFIG = {
    "dataset":        "cifar10",
    "num_classes":    10,
    "input_channels": 3,
    "weight_decay":   1e-4,
}


# ─────────────────────────────────────────────
# REPRODUCIBILITY  (improvement 2 — torch seeds)
# ─────────────────────────────────────────────
def set_all_seeds(seed: int = SEED):
    """Seed every RNG so the run is fully reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# FITNESS FUNCTION
# ─────────────────────────────────────────────
def real_train(config: dict) -> float:
    from model.train import train

    full_config = {**BASE_CONFIG, **config}
    result = train(full_config, fast_mode=True, save_checkpoint=None, verbose=False)

    if isinstance(result, dict):
        return float(result["accuracy"])
    return float(result)


# ─────────────────────────────────────────────
# CHROMOSOME HELPERS  (now 5 genes)
# ─────────────────────────────────────────────
def decode(individual) -> dict:
    """Convert 5 gene indices → config dict for train()."""
    lr_idx, filter_idx, layer_idx, dropout_idx, batch_idx = individual
    return {
        "lr":          LR_CHOICES        [int(lr_idx)],
        "num_filters": FILTER_CHOICES    [int(filter_idx)],
        "num_layers":  LAYER_CHOICES     [int(layer_idx)],
        "dropout":     DROPOUT_CHOICES   [int(dropout_idx)],
        "batch_size":  BATCH_SIZE_CHOICES[int(batch_idx)],
        "epochs":      EPOCHS_SEARCH,
    }


def random_individual():
    """One random 5-gene chromosome."""
    return [
        random.randrange(len(LR_CHOICES)),
        random.randrange(len(FILTER_CHOICES)),
        random.randrange(len(LAYER_CHOICES)),
        random.randrange(len(DROPOUT_CHOICES)),
        random.randrange(len(BATCH_SIZE_CHOICES)),
    ]


def mutate_individual(individual, indpb=GENE_MUTPB):
    """Randomly replace each gene with probability indpb."""
    limits = [
        len(LR_CHOICES), len(FILTER_CHOICES), len(LAYER_CHOICES),
        len(DROPOUT_CHOICES), len(BATCH_SIZE_CHOICES),
    ]
    for i, limit in enumerate(limits):
        if random.random() < indpb:
            individual[i] = random.randrange(limit)
    return (individual,)


# ─────────────────────────────────────────────
# CHECKPOINTING  (improvement 3)
# ─────────────────────────────────────────────
def save_checkpoint(generation: int, population, filepath: str = CHECKPOINT_FILE):
    """Save full population state to JSON after each generation."""
    data = {
        "generation": generation,
        "population": [
            {
                "genes":   list(ind),
                "fitness": ind.fitness.values[0] if ind.fitness.valid else None,
                "config":  decode(ind),
            }
            for ind in population
        ],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [Checkpoint] Saved generation {generation} → {filepath}")


def load_checkpoint(filepath: str = CHECKPOINT_FILE):
    """
    Load a saved checkpoint and rebuild the DEAP population from it.
    Returns (start_generation, population) or (0, None) if no checkpoint found.
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except FileNotFoundError:
        return 0, None

    population = []
    for entry in data["population"]:
        ind = creator.Individual(entry["genes"])
        if entry["fitness"] is not None:
            ind.fitness.values = (entry["fitness"],)
        population.append(ind)

    print(f"  [Checkpoint] Resumed from generation {data['generation']}  ({filepath})")
    return data["generation"], population


# ─────────────────────────────────────────────
# DEAP TOOLBOX
# ─────────────────────────────────────────────
def build_toolbox():
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate,
                     creator.Individual, random_individual)
    toolbox.register("population", tools.initRepeat,
                     list, toolbox.individual)

    def evaluate(individual):
        config = decode(individual)
        return (real_train(config),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("select",   tools.selTournament, tournsize=3)
    toolbox.register("mate",     tools.cxTwoPoint)
    toolbox.register("mutate",   mutate_individual, indpb=GENE_MUTPB)

    return toolbox


# ─────────────────────────────────────────────
# GENERATION LOGGER
# ─────────────────────────────────────────────
class GenerationLogger:
    def __init__(self, filepath: str = LOG_FILE, resume: bool = False):
        self.filepath   = filepath
        self.start_time = time.time()
        # Append to existing log if resuming, otherwise start fresh
        mode = "a" if resume else "w"
        with open(self.filepath, mode, newline="") as f:
            if not resume:
                csv.writer(f).writerow([
                    "generation", "best_fitness", "avg_fitness", "worst_fitness",
                    "best_lr", "best_num_filters", "best_num_layers",
                    "best_dropout", "best_batch_size", "elapsed_seconds",
                ])
        print(f"[Logger] {self.filepath} ready  ({'append' if resume else 'new'})")

    def log(self, gen: int, population):
        fitnesses = [ind.fitness.values[0] for ind in population]
        best_ind  = tools.selBest(population, 1)[0]
        best_cfg  = decode(best_ind)
        row = {
            "generation":       gen,
            "best_fitness":     round(max(fitnesses), 4),
            "avg_fitness":      round(np.mean(fitnesses), 4),
            "worst_fitness":    round(min(fitnesses), 4),
            "best_lr":          best_cfg["lr"],
            "best_num_filters": best_cfg["num_filters"],
            "best_num_layers":  best_cfg["num_layers"],
            "best_dropout":     best_cfg["dropout"],
            "best_batch_size":  best_cfg["batch_size"],
            "elapsed_seconds":  round(time.time() - self.start_time, 1),
        }
        with open(self.filepath, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=row.keys()).writerow(row)
        print(
            f"  Gen {gen:>2} | "
            f"best={row['best_fitness']:.4f}  "
            f"avg={row['avg_fitness']:.4f}  "
            f"worst={row['worst_fitness']:.4f}  "
            f"lr={best_cfg['lr']}  filters={best_cfg['num_filters']}  "
            f"layers={best_cfg['num_layers']}  dropout={best_cfg['dropout']}  "
            f"batch={best_cfg['batch_size']}"
        )


# ─────────────────────────────────────────────
# MAIN GA
# ─────────────────────────────────────────────
def run_ga(
    population_size: int  = POPULATION_SIZE,
    n_generations:   int  = N_GENERATIONS,
    resume:          bool = False,
):
    print("=" * 60)
    print("GA Optimizer")
    print(f"  Population : {population_size}")
    print(f"  Generations: {n_generations}")
    print(f"  Epochs/call: {EPOCHS_SEARCH}  (fast_mode=True)")
    print(f"  Elitism    : top {ELITE_SIZE} carried forward each gen")
    print(f"  Genes      : lr, filters, layers, dropout, batch_size")
    print(f"  Resume     : {resume}")
    print("=" * 60)

    # Improvement 2 — seed all RNGs including torch
    set_all_seeds(SEED)

    toolbox = build_toolbox()
    logger  = GenerationLogger(resume=resume)

    # ── Initial population or resume from checkpoint ──
    start_gen = 0
    if resume:
        start_gen, population = load_checkpoint()
        if population is None:
            print("[Resume] No checkpoint found — starting fresh.")
            resume = False

    if not resume:
        population = toolbox.population(n=population_size)
        print("\n[Gen 0] Evaluating initial population …")
        for ind, fit in zip(population, map(toolbox.evaluate, population)):
            ind.fitness.values = fit
        logger.log(0, population)
        save_checkpoint(0, population)

    # ── Evolution loop ──
    for gen in range(start_gen + 1, n_generations + 1):
        print(f"\n[Gen {gen}] Selecting & breeding …")

        # Elitism — carry best individual forward unchanged
        elite    = list(map(toolbox.clone, tools.selBest(population, ELITE_SIZE)))

        # Selection for the rest of the offspring
        offspring = list(map(toolbox.clone,
                             toolbox.select(population, population_size - ELITE_SIZE)))

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROB:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate only changed individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        print(f"         Evaluating {len(invalid)} new/mutated individuals …")
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        population[:] = elite + offspring
        logger.log(gen, population)

        # Improvement 3 — checkpoint after every generation
        save_checkpoint(gen, population)

    # ── Save winner ──
    best_individual = tools.selBest(population, 1)[0]
    best_config     = decode(best_individual)
    best_config["epochs"] = EPOCHS_FINAL
    best_accuracy   = best_individual.fitness.values[0]

    with open(CONFIG_FILE, "w") as f:
        json.dump({
            "best_config": best_config,
            "best_val_accuracy_during_search": round(best_accuracy, 4),
            "search_epochs_used": EPOCHS_SEARCH,
            "full_train_epochs":  EPOCHS_FINAL,
            "ga_params": {
                "population_size": population_size,
                "n_generations":   n_generations,
                "crossover_prob":  CROSSOVER_PROB,
                "mutation_prob":   MUTATION_PROB,
                "gene_mutpb":      GENE_MUTPB,
                "elite_size":      ELITE_SIZE,
                "seed":            SEED,
            },
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("GA COMPLETE")
    print(f"  Best config : {best_config}")
    print(f"  Best val acc: {best_accuracy:.4f}  (during {EPOCHS_SEARCH}-epoch search)")
    print(f"  Saved → {CONFIG_FILE}       (send to Yassin)")
    print(f"  Log    → {LOG_FILE}         (send to Nour)")
    print(f"  Check  → {CHECKPOINT_FILE}  (last generation state)")
    print("=" * 60)

    return best_config, best_accuracy


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA Optimizer")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from ga_checkpoint.json")
    parser.add_argument("--smoke-test", action="store_true",
                        help="1 gen x 3 pop — quick sanity check")
    args = parser.parse_args()

    if args.smoke_test:
        print(">>> SMOKE TEST (1 gen x 3 pop) <<<\n")
        run_ga(population_size=3, n_generations=1)
    else:
        run_ga(resume=args.resume)
