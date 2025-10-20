import os

import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv


load_dotenv('.env')

DATA_NAME = os.getenv('DATA_NAME')
DATA_DIR = os.getenv('DATA_DIR')
POINTS = 2000


def generate_dataset(func, x_min, x_max, n_points=POINTS, noise_std=0.1, seed=None):
    """
    Generate a pandas DataFrame with points (x, y), where y = func(x) + noise
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(x_min, x_max, n_points)
    y_true = func(x)

    noise = np.random.normal(0, noise_std, size=n_points)
    y_noisy = y_true + noise

    df = pd.DataFrame({
        'x': x,
        'y': y_noisy,
        'y_true': y_true
    })
    return df

if __name__ == "__main__":
    def func(x): return np.sin(x) + 0.5 * x

    dataset = generate_dataset(
        func=func,
        x_min=-10,
        x_max=10,
        n_points=200,
        noise_std=0.2,
        seed=123
    )

    print(dataset.head())

    data_path = Path(DATA_DIR) / DATA_NAME
    dataset.to_csv(data_path, index=False)
