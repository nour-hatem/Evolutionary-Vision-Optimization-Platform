"""
check_db.py
-----------
Verify the AWS RDS database connection and list existing tables.

Usage:
    python check_db.py

Requires DATABASE_URL to be set in the environment (via .env).
"""

from sqlalchemy import inspect
from experiments.db_models import engine


def verify_database():
    print("Connecting to AWS RDS...")
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        print("\nConnection Successful!")
        print("-" * 30)
        print("Found the following tables:")

        if not tables:
            print("  (No tables found — run init_db() to create them.)")
        else:
            for table in tables:
                print(f"  {table}")

    except Exception as e:
        print(f"\nConnection Failed: {e}")


if __name__ == "__main__":
    verify_database()
