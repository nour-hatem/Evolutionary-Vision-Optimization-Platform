"""
experiments/db_models.py
------------------------
SQLAlchemy ORM models for experiment tracking against AWS RDS PostgreSQL.

Database connection URL is read exclusively from the DATABASE_URL environment
variable (set via .env or container environment).  The hardcoded fallback has
been intentionally removed; if DATABASE_URL is not set the application will
raise a clear error at startup rather than silently using an exposed credential.
"""

import os
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DB_URL = os.environ.get("DATABASE_URL")

if not DB_URL:
    raise EnvironmentError(
        "DATABASE_URL environment variable is not set. "
        "Copy .env.example to .env and fill in your credentials."
    )

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Experiment(Base):
    __tablename__ = "experiments"
    id             = Column(Integer, primary_key=True, index=True)
    run_name       = Column(String, unique=True, index=True)
    config         = Column(JSON)
    status         = Column(String, default="running")   # running | completed | interrupted
    start_time     = Column(DateTime, default=datetime.utcnow)
    end_time       = Column(DateTime, nullable=True)
    runtime_seconds = Column(Float, default=0.0)


class RunLog(Base):
    __tablename__ = "run_logs"
    id            = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    generation    = Column(Integer)
    accuracy      = Column(Float)
    f1_score      = Column(Float)
    timestamp     = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id              = Column(Integer, primary_key=True, index=True)
    filename        = Column(String, nullable=True)
    predicted_class = Column(String)
    confidence      = Column(Float)
    timestamp       = Column(DateTime, default=datetime.utcnow)


class Checkpoint(Base):
    __tablename__ = "checkpoints"
    id               = Column(Integer, primary_key=True, index=True)
    experiment_id    = Column(Integer, ForeignKey("experiments.id"), unique=True)
    generation       = Column(Integer)
    population_state = Column(JSON)
    timestamp        = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables in the database. Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)


try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"[WARNING] Could not auto-create DB tables: {e}")
    print("  Tables will be created on first successful connection.")
