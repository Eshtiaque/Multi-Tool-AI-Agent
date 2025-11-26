import pandas as pd
import sqlite3
import os

DATA_DIR = "data"
DB_DIR = "databases"

if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

def create_sqlite_db(csv_file, db_name, table_name):

    csv_path = os.path.join(DATA_DIR, csv_file)
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_file} not found in {DATA_DIR} folder!")
        return

    df = pd.read_csv(csv_path)

    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    db_path = os.path.join(DB_DIR, db_name)
    conn = sqlite3.connect(db_path)
    
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    
    conn.close()
    print(f"✅ Successfully created {db_name} with table '{table_name}'")

if __name__ == "__main__":
    print("🚀 Starting Database Creation...")
    
    # 1. Heart Disease
    create_sqlite_db("heart.csv", "heart_disease.db", "heart_disease")
    
    # 2. Cancer
    create_sqlite_db("cancer.csv", "cancer.db", "cancer_prediction")
    
    # 3. Diabetes
    create_sqlite_db("diabetes.csv", "diabetes.db", "diabetes")
    
    print("\n🎉 All databases are ready in 'databases' folder!")