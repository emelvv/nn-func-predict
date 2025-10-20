import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize model predictions and (optionally) original points."
    )
    parser.add_argument(
        "--pred-csv",
        type=str,
        default="out/predictions.csv",
        help="CSV with inference results (expects x and y_pred columns by default).",
    )
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default="data/processed/dataset.csv",
        help="Original dataset for comparison. Leave empty to skip plotting it.",
    )
    parser.add_argument(
        "--x-col",
        type=str,
        default="x",
        help="Name of the X column (used for both predictions and the dataset).",
    )
    parser.add_argument(
        "--y-col",
        type=str,
        default="y",
        help="Dataset column to compare with predictions (e.g. y or y_true).",
    )
    parser.add_argument(
        "--pred-col",
        type=str,
        default="y_pred",
        help="Name of the column with predicted values.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Inference vs ground truth",
        help="Plot title.",
    )
    return parser.parse_args()


def load_frame(path: str, expected_cols: list[str]) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File {csv_path} was not found.")
    df = pd.read_csv(csv_path)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"File {csv_path} is missing columns: {', '.join(missing)}."
        )
    return df


def main() -> None:
    args = parse_args()

    pred_df = load_frame(args.pred_csv, [args.x_col, args.pred_col])
    pred_df = pred_df.sort_values(args.x_col)

    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        pred_df[args.x_col],
        pred_df[args.pred_col],
        label="prediction",
        color="tab:blue",
        linewidth=2,
    )

    if args.dataset_csv:
        ds_df = load_frame(args.dataset_csv, [args.x_col, args.y_col])
        ax.scatter(
            ds_df[args.x_col],
            ds_df[args.y_col],
            label=f"dataset:{args.y_col}",
            s=15,
            color="tab:orange",
            alpha=0.7,
        )

    ax.set_title(args.title)
    ax.set_xlabel(args.x_col)
    ax.set_ylabel("value")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
