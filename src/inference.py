
import argparse
import numpy as np
import torch
import pandas as pd

from dataset import DatasetConfig, GraphDataset
from model import MLPRegressor


def main(args):

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg_dict = ckpt["cfg"]
    stats = ckpt["stats"]
    in_dim = ckpt["in_dim"]
    ds_cfg = DatasetConfig(**cfg_dict)
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

    if args.x_csv is not None:
        df = pd.read_csv(args.x_csv)
        x = df[ds_cfg.x_col].to_numpy(dtype=np.float32)
    else:
        x = np.linspace(args.x_min, args.x_max,
                        args.n_points).astype(np.float32)

    xm, xs = stats["x"]
    if ds_cfg.normalize:
        x_norm = (x - xm) / xs
    else:
        x_norm = x

    feats = [x_norm.reshape(-1, 1)]
    if ds_cfg.fourier_dim and ds_cfg.fourier_dim > 0:
        k = np.arange(1, ds_cfg.fourier_dim + 1,
                      dtype=np.float32) * float(ds_cfg.fourier_scale)
        sin_feats = np.sin(np.outer(x_norm, k))
        cos_feats = np.cos(np.outer(x_norm, k))
        feats += [sin_feats, cos_feats]
    X = np.concatenate(feats, axis=1).astype(np.float32)

    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device)).cpu().numpy()

    ym, ys = stats["y"]
    if ds_cfg.normalize:
        pred = pred * ys + ym

    out = pd.DataFrame({ds_cfg.x_col: x, "y_pred": pred})
    out.to_csv(args.out_csv, index=False)
    print(f"Saved predictions to {args.out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default='runs/exp1/model.pt')
    p.add_argument("--x-csv", type=str, default=None,
                   help="Optional CSV with column x; if not given, uses a linspace")
    p.add_argument("--x-min", type=float, default=-10.0)
    p.add_argument("--x-max", type=float, default=10.0)
    p.add_argument("--n-points", type=int, default=200)
    p.add_argument("--out-csv", type=str, default="out/predictions.csv")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    main(args)
