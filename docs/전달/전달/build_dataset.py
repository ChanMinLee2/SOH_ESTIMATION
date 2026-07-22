"""build_dataset.py — cut random windows from _4_data_hi/clean and prepare training data.

Per window it stores: the raw resampled V/I curve (for the CNN), the HI values
(computed from the raw rows via the hi_compute registry), the observable descriptor,
and the SOH target. It also MEASURES each HI function's compute time here and writes
``hi_cost.json`` — that measured cost drives the model's cost-weighted L0 penalty.

Add an ``@hi`` function in hi_compute.py, re-run this, and the new HI (and its
measured cost) are picked up automatically — no schema to edit.

    python 5_model/build_dataset.py --datasets MIT HUST --per-phase 2

Output: _4_data_hi/samples/<DS>/<cell>.parquet  +  _4_data_hi/samples/hi_cost.json
"""

from __future__ import annotations
import argparse
import glob
import json
import pathlib
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils.io_utils import load_config
from hi_compute import compute_his, benchmark_cost, morph_reference

RAW_N = 48   # resample points for the CNN curve
BOL_CYCLES = 30   # fresh-state window: cell's first N cycles → morph BOL reference curve


def _resample(v, ai, q):
    qn = q / (q[-1] + 1e-9)
    grid = np.linspace(0.0, 1.0, RAW_N)
    return np.interp(grid, qn, v).astype(np.float32), np.interp(grid, qn, ai).astype(np.float32)


def _build_bol_ref(df, min_pts=15):
    """Cell's fresh-state (first BOL_CYCLES) full-phase shape curves → morph BOL reference
    per direction (+1 charge / -1 discharge). Factory BOL data — legitimate reference."""
    cmin = df["cycle"].min()
    fresh = df[df["cycle"] <= cmin + BOL_CYCLES]
    buckets = {1: [], -1: []}
    for _, cdf in fresh.groupby("cycle"):
        cdf = cdf.sort_values("time_s")
        for sign in (1, -1):
            ph = cdf[cdf.current_A * sign > 0.1]
            if len(ph) < min_pts:
                continue
            t = ph["time_s"].values.astype(float)
            buckets[sign].append((ph["voltage_V"].values.astype(float),
                                  ph["current_A"].values.astype(float), t - t[0]))
    return {s: (morph_reference(b) if b else None) for s, b in buckets.items()}


def sample_data(csv_path, soh_by_cycle, rng, per_phase, tmin, tmax, min_pts=15):
    df = pd.read_csv(csv_path)
    cell = df["cell_id"].iloc[0]
    bol_ref = _build_bol_ref(df)                                   # morph BOL reference (per direction)
    rows, raw_for_cost = [], []
    for cyc, cdf in df.groupby("cycle"):
        soh = soh_by_cycle.get(cyc)
        if soh is None or not np.isfinite(soh):
            continue
        cdf = cdf.sort_values("time_s")
        for sign, phase in ((1, cdf[cdf.current_A > 0.1]), (-1, cdf[cdf.current_A < -0.1])):
            if len(phase) < min_pts:
                continue
            t = phase["time_s"].values; v = phase["voltage_V"].values; i = phase["current_A"].values
            t0, t1 = t[0], t[-1]; span = t1 - t0
            for _ in range(per_phase):
                T = rng.uniform(tmin, min(tmax, span))
                if T < tmin * 0.9:
                    continue
                start = rng.uniform(t0, t1 - T)
                m = (t >= start) & (t <= start + T)
                if m.sum() < min_pts:
                    continue
                wv = v[m].astype(float); wi = i[m].astype(float); wt = t[m].astype(float) - t[m][0]
                ai = np.abs(wi); dt = np.clip(np.diff(wt, prepend=wt[0]), 0, None)
                q = np.cumsum(ai * dt) / 3600.0
                his = compute_his(wv, wi, wt, bol=bol_ref.get(sign))  # HIs (morph vs fresh curve)
                rv, ri = _resample(wv, ai, q)                       # CNN curve
                rows.append({
                    "cell_id": cell, "cycle": int(cyc), "soh": float(soh),
                    "direction": 1 if wi.mean() > 0 else 0,
                    "dur_s": float(wt[-1]), "dQ_obs": float(q[-1]),
                    "v_start": float(wv[0]), "v_end": float(wv[-1]), "i_signed_mean": float(wi.mean()),
                    **{f"hi_{k}": val for k, val in his.items()},
                    "raw_v": rv.tolist(), "raw_i": ri.tolist(),
                })
                if len(raw_for_cost) < 60:
                    raw_for_cost.append((wv, wi, wt))
    return pd.DataFrame(rows), raw_for_cost


def _build_cell(args):
    """Worker: sample one cell's windows, compute HIs, save parquet. Returns (n, sample raws)."""
    fp, out_dir, per_phase, tmin, tmax, seed = args
    df = pd.read_csv(fp)
    soh_by_cycle = df.groupby("cycle")["capacity_Ah"].first().to_dict()
    wdf, raws = sample_data(fp, soh_by_cycle, np.random.default_rng(seed), per_phase, tmin, tmax)
    if len(wdf):
        wdf.to_parquet(pathlib.Path(out_dir) / f"{wdf['cell_id'].iloc[0]}.parquet")
    return len(wdf), raws[:8]


def main():
    # settings come from the YAML; CLI is just --config and --limit (debug).
    ap = argparse.ArgumentParser(description="Cut windows + compute HIs + measure HI cost.")
    ap.add_argument("--config", default="5_model/config/default.yaml")
    ap.add_argument("--limit", type=int, default=None, help="cells per dataset (debug)")
    args = ap.parse_args()
    d = load_config(args.config)["data"]
    out_base = pathlib.Path(d["data_dir"]); out_base.mkdir(parents=True, exist_ok=True)

    jobs = []
    for ds in d["datasets"]:
        files = sorted(glob.glob(f"{d['clean_dir']}/{ds}/*.csv"))
        if args.limit:
            files = files[:args.limit]
        (out_base / ds).mkdir(parents=True, exist_ok=True)
        for k, fp in enumerate(files):
            jobs.append((fp, str(out_base / ds), d["per_phase"], d["tmin"], d["tmax"],
                         d["split_seed"] + hash(fp) % 100000))

    from multiprocessing import Pool
    tot = 0; cost_samples = []
    with Pool(min(8, len(jobs))) as pool:
        for n, raws in pool.imap_unordered(_build_cell, jobs):
            tot += n
            if len(cost_samples) < 200:
                cost_samples.extend(raws)
    print(f"{len(jobs)} cells → {tot} windows")

    # measure each HI's compute time on the collected windows → cost for the L0 penalty
    names, cost = benchmark_cost(cost_samples[:150])
    json.dump({"hi_names": names, "cost_us": [float(c) for c in cost]},
              open(out_base / "hi_cost.json", "w"), indent=2)
    print(f"측정된 HI 계산비용 → {out_base/'hi_cost.json'}  ({len(names)}개 HI)")


if __name__ == "__main__":
    main()
