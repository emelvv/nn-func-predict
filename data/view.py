import os

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv


load_dotenv('data/.env')

DATA_NAME = os.getenv('DATA_NAME')
DATA_DIR = os.getenv('DATA_DIR')
DEFAULT_PATH = os.path.join(DATA_DIR, DATA_NAME)
LOCAL_PATH = DATA_NAME

if os.path.exists(LOCAL_PATH):
    file_path = LOCAL_PATH
elif os.path.exists(DEFAULT_PATH):
    file_path = DEFAULT_PATH
else:
    raise FileNotFoundError("The file generated_dataset.csv was not found either locally or in /mnt/data/. "
                            "Place the file alongside the script or in /mnt/data/.")

print(f"Loading from: {file_path}")
df = pd.read_csv(file_path)

print(df.head())

print(df.describe())

plt.figure(figsize=(7, 4))
plt.scatter(df['x'], df['y'], s=10, alpha=0.7, label="noisy points")
plt.plot(df['x'], df['y_true'], linewidth=2, label="true function")
plt.xlabel("x")
plt.ylabel("y")
plt.title("View of points from generated_dataset.csv")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
