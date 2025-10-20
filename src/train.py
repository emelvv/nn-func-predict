
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import GraphDataset, DatasetConfig
from model import MLPRegressor
from utils import set_seed, save_json, load_json, count_params


def infer_in_dim(cfg: DatasetConfig) -> int:
    # base feature x + 2*fourier_dim
    return 1 + (2 * cfg.fourier_dim if cfg.fourier_dim > 0 else 0)


def train(args):
    set_seed(args.seed)

    ds_cfg = DatasetConfig(
        csv_path=args.csv,
        x_col=args.x_col,
        y_col=args.y_col,
        use_y_true=args.use_y_true,
        add_x_noise_std=args.aug_x,
        add_y_noise_std=args.aug_y,
        fourier_dim=args.fourier_dim,
        fourier_scale=args.fourier_scale,
        normalize=not args.no_normalize,
    )

    # First create train dataset to compute stats; then pass stats to val
    train_ds = GraphDataset(
        ds_cfg, split="train", val_fraction=args.val_fraction, seed=args.seed, stats=None)
    val_ds = GraphDataset(
        ds_cfg, split="val", val_fraction=args.val_fraction, seed=args.seed, stats=train_ds.stats)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")

    in_dim = infer_in_dim(ds_cfg)
    model = MLPRegressor(in_dim, hidden=args.hidden,
                         depth=args.depth, dropout=args.dropout).to(device)

    print(f"Model params: {count_params(model):,}")
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    patience_left = args.early_stop

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).squeeze(-1)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = total / max(1, n)

        # validation
        model.eval()
        with torch.no_grad():
            total = 0.0
            n = 0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).squeeze(-1)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                total += loss.item() * xb.size(0)
                n += xb.size(0)
            val_loss = total / max(1, n)

        sched.step(val_loss)
        print(
            f"Epoch {epoch:03d} | train MSE {train_loss:.6f} | val MSE {val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_left = args.early_stop
            model_hparams = {
                "in_dim": in_dim,
                "hidden": args.hidden,
                "depth": args.depth,
                "dropout": args.dropout
            }
            ckpt = {
                "model_state": model.state_dict(),
                "model_hparams": model_hparams,
                "model_class": "MLPRegressor@v1",
                "in_dim": in_dim,
                "cfg": vars(ds_cfg),
                "stats": train_ds.stats,
                "seed": args.seed,
            }
            torch.save(ckpt, out_dir / "model.pt")
            save_json({"best_val_mse": best_val, "epoch": epoch,
                      "stats": train_ds.stats}, out_dir / "meta.json")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default='data/processed/dataset.csv',
                   help="Path to CSV with columns x,y[,y_true]")
    p.add_argument("--x-col", type=str, default="x")
    p.add_argument("--y-col", type=str, default="y")
    p.add_argument("--use-y-true", action="store_true",
                   help="Train on y_true instead of y if available")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--fourier-dim", type=int, default=0,
                   help="Pairs of sin/cos Fourier features")
    p.add_argument("--fourier-scale", type=float, default=1.0)
    p.add_argument("--aug-x", type=float, default=0.0)
    p.add_argument("--aug-y", type=float, default=0.0)
    p.add_argument("--early-stop", type=int, default=30)
    p.add_argument("--out-dir", type=str, default="runs/exp1")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)
