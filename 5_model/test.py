"""test.py — Evaluate a saved DFRModel checkpoint on a data split.

Usage (from project root):

  python 5_model/test.py --run _5_data_model/0706_1430
  python 5_model/test.py --run _5_data_model/0706_1430 --split val
  python 5_model/test.py --run _5_data_model/0706_1430 --checkpoint best.pth --gpu 0
"""

from __future__ import annotations
import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch

from utils.hi_groups import build_hi_group_info_from_data
from utils.io_utils import load_config, load_checkpoint
from datasets.battery_dataset import load_all_cells, split_cells, BatteryHIDataset, collate_fn
from datasets.normalization import FeatureNormalizer
from models.dfr_model import DFRModel
from evaluation.evaluator import Evaluator
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved DFRModel.")
    parser.add_argument(
        "--run", required=True,
        help="Path to run directory (e.g., _5_data_model/0706_1430).",
    )
    parser.add_argument(
        "--checkpoint", default="best.pth",
        help="Checkpoint filename inside run/checkpoints/.",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Which data split to evaluate.",
    )
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = pathlib.Path(args.run)

    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        sys.exit(1)

    cfg = load_config(run_dir / "config.yaml")
    data_dir = cfg["data"]["data_dir"]

    if args.gpu is not None and torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    logger.info("Device: %s", device)

    # Build HI info and load data
    hi_info = build_hi_group_info_from_data(data_dir)
    df = load_all_cells(data_dir, datasets=cfg["data"]["datasets"],
                        min_cycles=cfg["data"].get("min_cycles_per_cell", 10))
    train_df, val_df, test_df = split_cells(
        df,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["data"]["split_seed"],
    )
    split_map = {"train": train_df, "val": val_df, "test": test_df}
    eval_df = split_map[args.split]

    # Load normalizer fitted during training
    normalizer = FeatureNormalizer.load(run_dir / "normalizer.pkl")

    dataset = BatteryHIDataset(eval_df, hi_info, normalizer)
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Load model
    model = DFRModel(hi_info, cfg)
    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    load_checkpoint(ckpt_path, model, device=device)
    logger.info("Loaded checkpoint: %s", ckpt_path)

    # Evaluate
    evaluator = Evaluator(model, run_dir, device)
    results = evaluator.evaluate(
        loader,
        split_name=args.split,
        metric_names=cfg["evaluation"]["metrics"],
    )

    print("\n=== Evaluation Results ===")
    for k, v in results["metrics"].items():
        print(f"  {k:8s}: {v:.5f}")
    print(f"\n  Active groups (mean): {results['routing']['mean_active_groups']:.1f} / {hi_info.n_groups}")
    print(f"  Sparsity:             {results['routing']['sparsity']:.3f}")


if __name__ == "__main__":
    main()
