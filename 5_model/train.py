"""train.py — Entry point for training DFRModel.

Usage (from project root, in LFP_SOH_ESTIMATION conda env):

  python 5_model/train.py
  python 5_model/train.py --config 5_model/config/default.yaml
  python 5_model/train.py --lambda-sparse 0.05 --epochs 300
  python 5_model/train.py --gpu 0
"""

from __future__ import annotations
import argparse
import logging
import pathlib
import sys

# Allow running as `python 5_model/train.py` from the project root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch

from utils.hi_groups import build_hi_group_info_from_data
from utils.io_utils import load_config, save_config, save_json
from datasets.battery_dataset import make_dataloaders
from datasets.normalization import FeatureNormalizer
from models.dfr_model import DFRModel
from training.loss import DFRLoss
from training.trainer import Trainer
from evaluation.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DFRModel for battery capacity estimation."
    )
    parser.add_argument(
        "--config", default="5_model/config/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU index. None = 자동 (CUDA 가용 시 cuda:0 선택, 없으면 cpu).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda-sparse", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None,
                        help="Subdirectory name under output_dir for this run.")
    return parser.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    """Apply CLI argument overrides onto config dict in-place."""
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.lambda_sparse is not None:
        cfg["training"]["lambda_sparse"] = args.lambda_sparse
    if args.seed is not None:
        cfg["data"]["split_seed"] = args.seed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load and patch config
    cfg = load_config(args.config)
    apply_overrides(cfg, args)

    # Device
    if args.gpu is not None and torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    logger.info("Device: %s", device)

    # Output directory
    base_out = pathlib.Path(cfg["data"]["output_dir"])
    if args.run_name:
        out_dir = base_out / args.run_name
    else:
        import datetime
        out_dir = base_out / datetime.datetime.now().strftime("%m%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # Save config snapshot
    save_config(cfg, out_dir / "config.yaml")

    # HI group info (scans one pkl to discover columns)
    data_dir = cfg["data"]["data_dir"]
    hi_info = build_hi_group_info_from_data(data_dir)
    logger.info(
        "HI groups: %d global + %d segment-category groups (%d total features)",
        hi_info.n_global,
        hi_info.n_groups,
        hi_info.n_global + sum(len(v) for v in hi_info.groups.values()),
    )

    # DataLoaders
    train_loader, val_loader, test_loader, normalizer = make_dataloaders(
        data_dir, hi_info, cfg
    )
    normalizer.save(out_dir / "normalizer.pkl")

    # Model
    model = DFRModel(hi_info, cfg)
    logger.info(
        "DFRModel — %d trainable parameters", model.count_parameters()
    )

    # Loss
    criterion = DFRLoss(
        lambda_sparse=cfg["training"]["lambda_sparse"],
        group_costs=hi_info.group_costs,
        cost_weighted=cfg["training"].get("cost_weighted", True),
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Train
    trainer = Trainer(model, criterion, optimizer, cfg, out_dir, device)
    history = trainer.train(train_loader, val_loader)
    save_json(history, out_dir / "logs" / "history.json")

    # Evaluate on test set using best checkpoint
    from utils.io_utils import load_checkpoint
    load_checkpoint(out_dir / "checkpoints" / "best.pth", model, device=device)

    evaluator = Evaluator(model, out_dir, device)
    results = evaluator.evaluate(test_loader, split_name="test",
                                 metric_names=cfg["evaluation"]["metrics"])

    logger.info("=== Test Results ===")
    for k, v in results["metrics"].items():
        logger.info("  %s: %.5f", k, v)
    logger.info(
        "  mean_active_groups: %.1f / %d",
        results["routing"]["mean_active_groups"], hi_info.n_groups,
    )

    # Also evaluate val split
    evaluator.evaluate(val_loader, split_name="val",
                       metric_names=cfg["evaluation"]["metrics"])

    logger.info("Training complete. Outputs saved to: %s", out_dir)


if __name__ == "__main__":
    main()
