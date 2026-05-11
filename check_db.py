from sqlalchemy import inspect
from experiments.db_models import engine

def verify_database():
    print("Connecting to AWS RDS...")
    
    try:
        # Create an inspector object
        inspector = inspect(engine)
        
        # Get all table names
        tables = inspector.get_table_names()
        
        print("\n✅ Connection Successful!")
        print("-" * 30)
        print("Found the following tables:")
        
        if not tables:
            print("  (No tables found. Something went wrong.)")
        else:
            for table in tables:
                print(f"  📦 {table}")
                
    except Exception as e:
        print(f"\n❌ Connection Failed: {e}")

if __name__ == "__main__":
    verify_database()