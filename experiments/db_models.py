# experiments/db_models.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# Replace with your actual RDS endpoint, username, and password
DB_URL = "postgresql://postgres:V3lvet%23T0rnado@clouddb.c1yu626sut0q.eu-north-1.rds.amazonaws.com:5432/postgres"

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    run_name = Column(String, unique=True, index=True)
    config = Column(JSON) # Store hyperparameters here
    status = Column(String, default="running") # running, completed, interrupted
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    runtime_seconds = Column(Float, default=0.0)

class RunLog(Base):
    __tablename__ = "run_logs"
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    generation = Column(Integer)
    accuracy = Column(Float)
    f1_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Checkpoint(Base):
    __tablename__ = "checkpoints"
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), unique=True)
    generation = Column(Integer)
    population_state = Column(JSON) # Storing the Genetic Algorithm state
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create all tables in the RDS database
Base.metadata.create_all(bind=engine)