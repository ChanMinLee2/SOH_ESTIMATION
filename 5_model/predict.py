"""predict.py — Run inference on new pkl files using a trained DFRModel.

The input pkl files must follow the same schema as _4_data_hi/ (output of
4_hi_analysis/hi_correlation.py). All 411 HIs must be present.

Usage (from project root):

  # Predict on a single cell
  python 5_model/predict.py --run _5_data_model/0706_1430 --input _4_data_hi/MIT/b1c0.pkl

  # Predict on all files in a directory
  python 5_model/predict.py --run _5_data_model/0706_1430 --input _4_data_hi/HUST/

  # Save routing gates per cycle
  python 5_model/predict.py --run _5_data_model/0706_1430 --input _4_data_hi/MIT/b1c5.pkl --save-routing
"""

from __future__ import annotations
import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.hi_groups import build_hi_group_info_from_data
from utils.io_utils import load_config, load_checkpoint
from datasets.battery_dataset import BatteryHIDataset, collate_fn
from datasets.normalization import FeatureNormalizer
from models.dfr_model import DFRModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DFRModel inference on new data.")
    parser.add_argument("--run", required=True, help="Path to run directory.")
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument(
        "--input", required=True,
        help="Path to a pkl file or directory of pkl files (same format as _4_data_hi/).",
    )
    parser.add_argument("--output", default=None, help="CSV output path. Default: next to input.")
    parser.add_argument("--save-routing", action="store_true", help="Append gate columns to output.")
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def _collect_pkls(input_path: pathlib.Path) -> list[pathlib.Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.pkl"))


def _load_pkl(path: pathlib.Path) -> pd.DataFrame:
    import pickle
    with open(path, "rb") as f:
        df = pickle.load(f)
    if "cell_id" not in df.columns:
        df.insert(0, "cell_id", path.stem)
    if "dataset" not in df.columns:
        df.insert(0, "dataset", path.parent.name)
    return df


def main() -> None:
    args = parse_args()
    run_dir = pathlib.Path(args.run)
    cfg = load_config(run_dir / "config.yaml")
    data_dir = cfg["data"]["data_dir"]

    device = "cpu"
    if args.gpu is not None and torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    elif torch.cuda.is_available():
        device = "cuda:0"

    hi_info = build_hi_group_info_from_data(data_dir)
    normalizer = FeatureNormalizer.load(run_dir / "normalizer.pkl")
    model = DFRModel(hi_info, cfg)
    load_checkpoint(run_dir / "checkpoints" / args.checkpoint, model, device=device)
    model.eval()
    model.to(device)

    pkl_files = _collect_pkls(pathlib.Path(args.input))
    if not pkl_files:
        logger.error("No pkl files found at: %s", args.input)
        sys.exit(1)

    logger.info("Running inference on %d pkl file(s)...", len(pkl_files))

    all_frames = []
    for pkl_path in pkl_files:
        df = _load_pkl(pkl_path)
        dataset = BatteryHIDataset(df, hi_info, normalizer)
        loader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        preds, gates_list = [], []
        for batch in loader:
            cap, gates = model.predict(batch, device=device)
            preds.append(cap.numpy())
            gates_list.append(gates.numpy())

        y_pred = np.concatenate(preds)
        gates_arr = np.concatenate(gates_list, axis=0)

        out_df = pd.DataFrame({
            "cell_id": df["cell_id"].values,
            "dataset": df["dataset"].values,
            "cycle": df["cycle"].values,
            "capacity_pred": y_pred,
        })

        if "capacity_Ah" in df.columns:
            out_df["capacity_true"] = df["capacity_Ah"].values
            err = y_pred - df["capacity_Ah"].values
            out_df["error"] = err
            rmse = float(np.sqrt(np.mean(err ** 2)))
            logger.info("%s — RMSE=%.5f Ah", pkl_path.stem, rmse)

        if args.save_routing:
            group_names = model.group_names()
            for i, name in enumerate(group_names):
                out_df[f"gate_{name}"] = gates_arr[:, i]

        all_frames.append(out_df)

    combined = pd.concat(all_frames, ignore_index=True)

    if args.output:
        out_path = pathlib.Path(args.output)
    else:
        in_path = pathlib.Path(args.input)
        stem = in_path.stem if in_path.is_file() else in_path.name
        out_path = run_dir / "predictions" / f"predict_{stem}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    logger.info("Predictions saved to: %s", out_path)


if __name__ == "__main__":
    main()
