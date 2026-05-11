# experiments/logger.py
from .db_models import SessionLocal, Experiment, RunLog
from datetime import datetime

class DBLogger:
    def __init__(self):
        self.db = SessionLocal()

    def start_experiment(self, run_name, config_dict):
        exp = Experiment(run_name=run_name, config=config_dict)
        self.db.add(exp)
        self.db.commit()
        self.db.refresh(exp)
        return exp.id

    def log_metrics(self, experiment_id, generation, accuracy, f1_score):
        log = RunLog(
            experiment_id=experiment_id, 
            generation=generation, 
            accuracy=accuracy, 
            f1_score=f1_score
        )
        self.db.add(log)
        self.db.commit()

    def end_experiment(self, experiment_id):
        exp = self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if exp:
            exp.status = "completed"
            exp.end_time = datetime.utcnow()
            exp.runtime_seconds = (exp.end_time - exp.start_time).total_seconds()
            self.db.commit()