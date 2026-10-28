"""Extract learned hidden representations for the embedding figure (fig_embedding.py).

Runs the trained SCR model over the full test split under HARD routing (the
deployment condition: classifier argmax picks the scenario, no ground truth),
and records two penultimate-layer activations via forward hooks:
  - reg_h : input to cap_head's final Linear (what the SOH regressor "reads")
  - clf_h : input to probe_mlp's final Linear (what the routing classifier "reads")
plus true/predicted scenario and SOH per segment. A class-balanced subsample is
saved to docs/figures/_cache/embed_<tag>.npz; the heavy UMAP + rendering then
runs in the separate figure environment (fig_embedding.py).

Must run in the project env (torch). N_HI env vars are set from the run's own
config.yaml BEFORE importing project modules (same pattern as
9_eval/scenario_attribution.py) so parameters.py's current ACTIVE_N_HI (which
an in-flight sweep may have changed) never affects this extraction.

Run:
  python docs/figures/_extract_embedding.py --run-dir model_lib/results/p1v2_runs/1009_1342_p1v2_baseline_seed42 --tag lfp
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "8_train"))
sys.path.insert(0, str(PROJECT_ROOT / "9_eval"))
sys.path.insert(0, str(PROJECT_ROOT / "model_lib"))

_ap = argparse.ArgumentParser()
_ap.add_argument("--run-dir", required=True)
_ap.add_argument("--tag", required=True)
_ap.add_argument("--per-scenario", type=int, default=4000)
ARGS = _ap.parse_args()

import yaml  # noqa: E402
_run_dir = Path(ARGS.run_dir)
if not _run_dir.is_absolute():
    _run_dir = PROJECT_ROOT / _run_dir
_cfg = yaml.safe_load((_run_dir / "config.yaml").read_text(encoding="utf-8"))
os.environ["SOH_EXCLUDE_STAT_LEAK"] = "1" if _cfg["data"].get("exclude_stat_leak") else "0"
os.environ["SOH_EXCLUDE_DQDV_LEAK"] = "1" if _cfg["data"].get("exclude_dqdv_leak") else "0"

import numpy as np  # noqa: E402
import torch  # noqa: E402

import parameters as P  # noqa: E402
from test import (  # noqa: E402
    _resolve_run_and_config, _load_checkpoint, _build_dataset_and_kernel,
    _build_shared_mask, _build_and_load_model,
)
from evaluation.scr_evaluator import SCREvaluator  # noqa: E402

P.ACTIVE_DATA_DIR = None
P.ACTIVE_SEG_DATA_DIR = None

# A run dir renamed after training (e.g. ..._refact_... -> ..._baseline_...) keeps
# stale absolute paths in p1v2_summary.json; resolve each artifact by filename
# inside the run dir itself instead.
import json as _json  # noqa: E402
_summary = _json.loads((_run_dir / "p1v2_summary.json").read_text(encoding="utf-8"))


def _local(key):
    v = _summary.get(key)
    if not v:
        return None
    name = Path(v).name
    # run dir itself; the run family's shared results/ dir; the pre-reorg
    # 5_model/ -> legacy_results/ relocation.
    for cand in (_run_dir / name, _run_dir.parents[1] / name,
                 PROJECT_ROOT / v.replace("5_model/", "legacy_results/")):
        if cand.exists():
            return str(cand)
    print(f"[extract] WARNING: {key} not found anywhere ({v})")
    return None


P.ACTIVE_KERNEL_FEATURES_PKL = _local("kernel_features_pkl")
P.ACTIVE_INTERACTION_JSON = _local("interaction_json")
P.ACTIVE_COMBINED_REDUNDANCY_JSON = _local("combined_redundancy_json")


def _batch(ds, idx, device):
    b = {
        "x_hi": ds.x_hi[idx].to(device),
        "nan_mask": ds.nan_mask[idx].to(device),
        "direction": ds.direction[idx].to(device),
        "scen_idx": ds.scen_idx[idx].to(device),
        "cap_init": ds.cap_init[idx].to(device),
    }
    if hasattr(ds, "x_kernel"):
        b["x_kernel"] = ds.x_kernel[idx].to(device)
    if hasattr(ds, "x_kernel_probe"):
        b["x_kernel_probe"] = ds.x_kernel_probe[idx].to(device)
    return b


@torch.no_grad()
def main() -> None:
    run = _resolve_run_and_config(str(_run_dir))
    ckpt = _load_checkpoint(run.run_dir)
    data = _build_dataset_and_kernel(run.cfg, run.summary)
    shared = _build_shared_mask(data.spec, run.summary)
    model = _build_and_load_model(run.cfg, data, shared, run.device, ckpt.ckpt)
    model.eval()
    assert model.probe_mlp is not None, "classifier missing -- hard routing impossible"

    ev = SCREvaluator(model=model, normalizer=data.norm, device=run.device,
                      figures_dir=run.run_dir / "figures", rep_cells=[])
    ev.set_classifier(model.probe_mlp)
    routing_t = ev._build_routing_table()

    captured = {}
    model.cap_head.net[-1].register_forward_hook(
        lambda m, inp, out: captured.__setitem__("reg", inp[0].detach()))
    model.probe_mlp[-1].register_forward_hook(
        lambda m, inp, out: captured.__setitem__("clf", inp[0].detach()))

    ds = data.test_ds
    n = len(ds)
    reg_h, clf_h, scen_pred, soh_pred = [], [], [], []
    for start in range(0, n, 16384):
        idx = np.arange(start, min(start + 16384, n))
        b = _batch(ds, idx, run.device)
        dir_t = b["direction"]
        dir_idx = (dir_t <= 0).long()
        parts = [model.get_probe_x(b["x_hi"], dir_t, b["scen_idx"]), dir_t.unsqueeze(1)]
        if "x_kernel_probe" in b:
            pk = model.get_probe_kernel_x(b["x_kernel_probe"], dir_t)
            if pk is not None:
                parts.append(pk)
        logits = model.probe_mlp(torch.cat(parts, dim=1))
        clf_h.append(captured["clf"].cpu().numpy())
        b["scen_idx"] = routing_t[dir_idx, logits.argmax(1)]
        out = model(b)
        reg_h.append(captured["reg"].cpu().numpy())
        scen_pred.append(b["scen_idx"].cpu().numpy())
        soh_pred.append(out["cap_pred"].cpu().numpy())

    reg_h = np.concatenate(reg_h).astype(np.float32)
    clf_h = np.concatenate(clf_h).astype(np.float32)
    scen_pred = np.concatenate(scen_pred).astype(np.int16)
    soh_pred = np.concatenate(soh_pred).astype(np.float32)
    scen_true = ds.scen_idx.numpy().astype(np.int16)
    soh_true = ds.target.numpy().astype(np.float32) if hasattr(ds, "target") else None
    if soh_true is None:
        soh_true = np.array([ds[i]["target"].item() for i in range(n)], dtype=np.float32)
    cell = np.asarray(ds.cell_ids)
    cycle = np.asarray(ds.cycles)

    rng = np.random.default_rng(20261023)
    keep = []
    for s in np.unique(scen_true):
        pool = np.where(scen_true == s)[0]
        keep.append(rng.choice(pool, size=min(ARGS.per_scenario, len(pool)), replace=False))
    keep = np.sort(np.concatenate(keep))

    out_dir = Path(__file__).resolve().parent / "_cache"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"embed_{ARGS.tag}.npz"
    np.savez_compressed(
        out, reg_h=reg_h[keep], clf_h=clf_h[keep], scen_true=scen_true[keep],
        scen_pred=scen_pred[keep], soh_true=soh_true[keep], soh_pred=soh_pred[keep],
        cell=cell[keep], cycle=cycle[keep], scen_names=np.asarray(data.spec.scenario_names),
        acc=np.float32((scen_pred == scen_true).mean()), n_total=np.int64(n),
    )
    print(f"saved {out}  kept={len(keep)}/{n}  hard-route acc={(scen_pred == scen_true).mean():.4f}"
          f"  reg_h={reg_h.shape[1]}d clf_h={clf_h.shape[1]}d")


if __name__ == "__main__":
    main()
