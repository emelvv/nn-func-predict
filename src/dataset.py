
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class DatasetConfig:
    csv_path: str
    x_col: str = "x"
    y_col: str = "y"
    use_y_true: bool = False
    add_x_noise_std: float = 0.0
    add_y_noise_std: float = 0.0
    fourier_dim: int = 0
    fourier_scale: float = 1.0
    normalize: bool = True


class GraphDataset(Dataset):
    """
    Simple 1D regression dataset: maps x -> y (denoised optional).
    Expects CSV with columns ['x','y'] and optionally ['y_true'].
    """

    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 123,
        stats: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        assert split in {"train", "val", "all"}
        self.cfg = config
        df = pd.read_csv(config.csv_path)
        if config.use_y_true and "y_true" in df.columns:
            target_col = "y_true"
        else:
            target_col = config.y_col

        rng = np.random.default_rng(seed)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        val_size = int(len(df) * val_fraction)
        if split == "train":
            sel = idx[val_size:]
        elif split == "val":
            sel = idx[:val_size]
        else:
            sel = idx
        df = df.iloc[sel].reset_index(drop=True)

        self.x = df[config.x_col].to_numpy(dtype=np.float32)
        self.y = df[target_col].to_numpy(dtype=np.float32)

        if stats is None:
            self.stats = {
                "x": (float(self.x.mean()), float(self.x.std() + 1e-8)),
                "y": (float(self.y.mean()), float(self.y.std() + 1e-8)),
            }
        else:
            self.stats = stats

        if self.cfg.normalize:
            xm, xs = self.stats["x"]
            ym, ys = self.stats["y"]
            self.x = (self.x - xm) / xs
            self.y = (self.y - ym) / ys

        self.freqs = None
        if self.cfg.fourier_dim > 0:
            k = np.arange(1, self.cfg.fourier_dim + 1, dtype=np.float32)
            self.freqs = (k * self.cfg.fourier_scale).astype(np.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def _to_features(self, x: float) -> np.ndarray:
        feat = [x]
        if self.freqs is not None:

            feat.extend(np.sin(self.freqs * x))
            feat.extend(np.cos(self.freqs * x))
        return np.asarray(feat, dtype=np.float32)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]

        if self.cfg.add_x_noise_std > 0.0:
            x = x + \
                np.random.normal(
                    0.0, self.cfg.add_x_noise_std).astype(np.float32)
        if self.cfg.add_y_noise_std > 0.0:
            y = y + \
                np.random.normal(
                    0.0, self.cfg.add_y_noise_std).astype(np.float32)

        feat = self._to_features(x)
        feat_t = torch.from_numpy(feat)
        y_t = torch.tensor([y], dtype=torch.float32)
        return feat_t, y_t
