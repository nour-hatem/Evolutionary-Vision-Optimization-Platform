"""
experiments/logger.py
---------------------
DBLogger: persists experiment start/stop events and per-generation metrics
to the AWS RDS PostgreSQL database via SQLAlchemy.
"""

from datetime import datetime
from .db_models import SessionLocal, Experiment, RunLog


class DBLogger:
    def __init__(self):
        self.db = SessionLocal()

    def start_experiment(self, run_name: str, config_dict: dict) -> int:
        exp = Experiment(run_name=run_name, config=config_dict)
        self.db.add(exp)
        self.db.commit()
        self.db.refresh(exp)
        return exp.id

    def log_metrics(self, experiment_id: int, generation: int,
                    accuracy: float, f1_score: float) -> None:
        log = RunLog(
            experiment_id=experiment_id,
            generation=generation,
            accuracy=accuracy,
            f1_score=f1_score,
        )
        self.db.add(log)
        self.db.commit()

    def interrupt_experiment(self, experiment_id: int) -> None:
        """Mark an experiment as interrupted (e.g. Spot instance reclaimed)."""
        exp = self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if exp:
            exp.status = "interrupted"
            self.db.commit()
            print(f"Experiment {experiment_id} marked as INTERRUPTED")

    def end_experiment(self, experiment_id: int) -> None:
        exp = self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if exp:
            exp.status = "completed"
            exp.end_time = datetime.utcnow()
            exp.runtime_seconds = (exp.end_time - exp.start_time).total_seconds()
            self.db.commit()
