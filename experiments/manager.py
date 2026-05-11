# experiments/manager.py
from .logger import DBLogger
from .checkpoint import DBCheckpointManager
# import your GA module here

class ExperimentManager:
    def __init__(self):
        self.logger = DBLogger()
        self.checkpointer = DBCheckpointManager()

    def run(self, run_name, config):
        print(f"Starting Experiment: {run_name}")
        
        # 1. Initialize Logging
        exp_id = self.logger.start_experiment(run_name, config)
        
        # 2. Check for existing checkpoint to resume
        start_gen, pop_state = self.checkpointer.load_latest_checkpoint(exp_id)
        
        if pop_state:
            print(f"Resuming from generation {start_gen}...")
            # TODO: Initialize your GA with pop_state
        else:
            print("Starting fresh training loop...")
            start_gen = 0
            # TODO: Initialize fresh GA
            
        # 3. Training Loop Example
        for gen in range(start_gen, config['max_generations']):
            # ... run your evolution step ...
            
            # Mock results
            accuracy = 0.85 + (gen * 0.01)
            f1 = 0.82 + (gen * 0.01)
            mock_pop_state = {"best_fitness": accuracy, "chromosomes": []}
            
            # Log & Checkpoint
            self.logger.log_metrics(exp_id, gen, accuracy, f1)
            self.checkpointer.save_checkpoint(exp_id, gen, mock_pop_state)
            
        # 4. Finalize
        self.logger.end_experiment(exp_id)
        print("Experiment completed successfully.")