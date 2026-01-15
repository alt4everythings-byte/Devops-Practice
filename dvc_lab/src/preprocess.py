import pandas as pd
import os

raw_path = "data/raw/raw_data.csv"
processed_path = "data/processed/processed_data.csv"

df = pd.read_csv(raw_path)
df['age'].fillna(df['age'].mean(), inplace=True)
df['marks'].fillna(df['marks'].mean(), inplace=True)

os.makedirs("data/processed", exist_ok=True)
df.to_csv(processed_path, index=False)

print("Data preprocessing completed.")
