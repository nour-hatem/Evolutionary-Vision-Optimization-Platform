"""
experiments/manager.py
-----------------------
ExperimentManager: orchestrates GA experiment runs with full checkpointing,
DB logging, and graceful Spot-instance interruption/resume support.
"""

import time
from .logger import DBLogger
from .checkpoint import DBCheckpointManager
from .db_models import SessionLocal, Experiment


class ExperimentManager:
    def __init__(self):
        self.logger = DBLogger()
        self.checkpointer = DBCheckpointManager()

    def run(self, run_name: str, config: dict, interrupt_after: int = None) -> dict:
        """
        Run a GA experiment with checkpointing and DB logging.

        Parameters
        ----------
        run_name       : unique experiment name
        config         : dict with max_generations, population_size, mutation_rate, dataset
        interrupt_after: if set, simulate Spot interruption after this many generations
        """
        print(f"Starting Experiment: {run_name}")

        exp_id = self.logger.start_experiment(run_name, config)
        start_gen, pop_state = self.checkpointer.load_latest_checkpoint(exp_id)

        if pop_state:
            print(f"Resuming from generation {start_gen}...")
            start_gen = start_gen + 1
        else:
            print("Starting fresh training loop...")
            start_gen = 0

        for gen in range(start_gen, config["max_generations"]):
            if interrupt_after is not None and (gen - start_gen) >= interrupt_after:
                print(f"INTERRUPTED at generation {gen}! (Simulating Spot reclaim)")
                self.logger.interrupt_experiment(exp_id)
                return {
                    "status": "interrupted",
                    "experiment_id": exp_id,
                    "last_generation": gen - 1,
                }

            time.sleep(0.5)

            accuracy = round(0.85 + (gen * 0.012), 4)
            f1 = round(0.82 + (gen * 0.011), 4)
            mock_pop_state = {
                "best_fitness": accuracy,
                "generation": gen,
                "population_size": config.get("population_size", 10),
                "chromosomes": [f"chr_{i}" for i in range(config.get("population_size", 10))],
            }

            self.logger.log_metrics(exp_id, gen, accuracy, f1)
            self.checkpointer.save_checkpoint(exp_id, gen, mock_pop_state)
            print(f"  Gen {gen}: accuracy={accuracy}, f1={f1} [checkpointed]")

        self.logger.end_experiment(exp_id)
        print("Experiment completed successfully.")
        return {"status": "completed", "experiment_id": exp_id}

    def resume(self, run_name: str) -> dict:
        """Resume an interrupted experiment from its last DB checkpoint."""
        db = SessionLocal()
        try:
            exp = (
                db.query(Experiment)
                .filter(
                    Experiment.run_name == run_name,
                    Experiment.status == "interrupted",
                )
                .first()
            )

            if not exp:
                return {"error": f"No interrupted experiment found with name '{run_name}'"}

            exp_id = exp.id
            config = exp.config
            exp.status = "running"
            db.commit()
        finally:
            db.close()

        print(f"Resuming Experiment: {run_name} (ID: {exp_id})")

        start_gen, pop_state = self.checkpointer.load_latest_checkpoint(exp_id)

        if pop_state:
            print(f"Checkpoint found! Resuming from generation {start_gen + 1}")
            print(f"  Recovered state: best_fitness={pop_state.get('best_fitness')}")
            resume_gen = start_gen + 1
        else:
            print("No checkpoint found, starting from generation 0")
            resume_gen = 0

        for gen in range(resume_gen, config["max_generations"]):
            time.sleep(0.5)

            accuracy = round(0.85 + (gen * 0.012), 4)
            f1 = round(0.82 + (gen * 0.011), 4)
            mock_pop_state = {
                "best_fitness": accuracy,
                "generation": gen,
                "population_size": config.get("population_size", 10),
                "chromosomes": [f"chr_{i}" for i in range(config.get("population_size", 10))],
            }

            self.logger.log_metrics(exp_id, gen, accuracy, f1)
            self.checkpointer.save_checkpoint(exp_id, gen, mock_pop_state)
            print(f"  Gen {gen}: accuracy={accuracy}, f1={f1} [checkpointed]")

        self.logger.end_experiment(exp_id)
        print("Experiment RESUMED and completed successfully!")
        return {
            "status": "completed",
            "experiment_id": exp_id,
            "resumed_from_generation": resume_gen,
        }
