# experiments/checkpoint.py
from .db_models import SessionLocal, Checkpoint

class DBCheckpointManager:
    def __init__(self):
        self.db = SessionLocal()

    def save_checkpoint(self, experiment_id, generation, population_state):
        # Check if a checkpoint already exists for this run
        ckpt = self.db.query(Checkpoint).filter(Checkpoint.experiment_id == experiment_id).first()
        
        if ckpt:
            ckpt.generation = generation
            ckpt.population_state = population_state
        else:
            ckpt = Checkpoint(
                experiment_id=experiment_id, 
                generation=generation, 
                population_state=population_state
            )
            self.db.add(ckpt)
        self.db.commit()

    def load_latest_checkpoint(self, experiment_id):
        ckpt = self.db.query(Checkpoint).filter(Checkpoint.experiment_id == experiment_id).first()
        if ckpt:
            return ckpt.generation, ckpt.population_state
        return None, None