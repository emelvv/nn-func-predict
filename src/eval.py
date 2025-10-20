
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataset import GraphDataset, DatasetConfig
from model import MLPRegressor
from utils import load_json


def evaluate(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg_dict = ckpt["cfg"]
    stats = ckpt["stats"]
    in_dim = ckpt["in_dim"]

    ds_cfg = DatasetConfig(**cfg_dict)
    # Use 'all' to report on the entire CSV
    ds = GraphDataset(ds_cfg, split="all", val_fraction=0.2,
                      seed=args.seed, stats=stats)
    dl = DataLoader(ds, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")

    model_hparams = ckpt.get("model_hparams")
    if model_hparams is None:
        raise RuntimeError(
            "Checkpoint missing 'model_hparams'. Re-train saving architecture or create model with identical args.")

    if model_hparams.get("in_dim") != in_dim:
        in_dim = model_hparams["in_dim"]

    model = MLPRegressor(**model_hparams)
    model.load_state_dict(ckpt["model_state"], strict=True)

    model.to(device)
    model.eval()

    mse_total, n = 0.0, 0
    preds, xs, ys = [], [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device).squeeze(-1)
            pred = model(xb)
            mse_total += torch.mean((pred - yb) ** 2).item() * xb.size(0)
            n += xb.size(0)
            preds.append(pred.cpu())
            # first feature is normalized x if normalize=True
            xs.append(xb[:, 0].cpu())
            ys.append(yb.cpu())
    mse = mse_total / max(1, n)
    print(f"MSE on full CSV: {mse:.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default='runs/exp1/model.pt',
                   help="Path to model.pt saved by train.py")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    evaluate(args)
