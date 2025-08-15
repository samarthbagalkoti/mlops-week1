import os, pickle, pandas as pd, psycopg2

DATA = os.getenv("DATA_PATH", "/data/data.csv")
MODEL = os.getenv("MODEL_PATH", "/data/model.pkl")

DB_NAME = os.getenv("DB_NAME", "mldb")
DB_USER = os.getenv("DB_USER", "mluser")
DB_PASS = os.getenv("DB_PASSWORD", "mlpass")
DB_HOST = os.getenv("DB_HOST", "db")

df = pd.read_csv(DATA)
mean_score = float(df["score"].mean()) if "score" in df.columns else float(df.iloc[:,0].mean())
print(f"ℹ️ mean_score = {mean_score}")

with open(MODEL, "rb") as f:
    _ = pickle.load(f)  # proving we can load it

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS results (mean_score FLOAT)")
cur.execute("INSERT INTO results (mean_score) VALUES (%s)", (mean_score,))
conn.commit()
cur.close()
conn.close()
print("✅ Inserted result into Postgres")

