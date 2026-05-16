"""
experiments/checkpoint.py
--------------------------
DBCheckpointManager: saves and loads GA population state to/from the
AWS RDS PostgreSQL database so interrupted Spot instances can resume.
"""

from .db_models import SessionLocal, Checkpoint


class DBCheckpointManager:
    def __init__(self):
        self.db = SessionLocal()

    def save_checkpoint(self, experiment_id: int, generation: int,
                        population_state: dict) -> None:
        ckpt = (
            self.db.query(Checkpoint)
            .filter(Checkpoint.experiment_id == experiment_id)
            .first()
        )
        if ckpt:
            ckpt.generation = generation
            ckpt.population_state = population_state
        else:
            ckpt = Checkpoint(
                experiment_id=experiment_id,
                generation=generation,
                population_state=population_state,
            )
            self.db.add(ckpt)
        self.db.commit()

    def load_latest_checkpoint(self, experiment_id: int):
        """Returns (generation, population_state) or (None, None) if none exists."""
        ckpt = (
            self.db.query(Checkpoint)
            .filter(Checkpoint.experiment_id == experiment_id)
            .first()
        )
        if ckpt:
            return ckpt.generation, ckpt.population_state
        return None, None
