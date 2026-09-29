#!/usr/bin/env python3
"""Build a simple JSON of binned telemetry vectors for one power run.

The inst_power vector comes from calculate_power_distribution() in
minos-analysis/dendrogram_plot/dendrogram.py itself (raw 1 ms-requested samples,
no smoothing, power/TDP, samples >= 0.5 TDP, bins np.arange(0.5, 2.05, bin_size)).
The other quantities are binned with the same algorithm over their own ranges
(the ranges used by build_workload_vectors.py).

Samples are kept only inside the kernel window [first kernel start, last kernel
end] from the rocprofv3 kernel trace of the same run; the sampler and rocprofv3
share the CLOCK_BOOTTIME/MONOTONIC time base (checked via clock_ref.txt).

Multi-GPU runs (one profiling_result_<tag>_<gpu>.csv per sampled GPU): vectors
pool the samples of all GPUs (each GPU-sample counts once); --per-gpu also writes
<name>_gpu<N> vectors. The kernel window spans all kernel-trace files.

  python3 build_sampling_json.py results/prof-433930 --label "LAMMPS 16x8x12"
"""
import argparse
import glob
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

DENDRO_DIR = "/work1/sinclair/sairajatg/minos-analysis/dendrogram_plot"   # --dendrogram-dir
TDP_W = 750.0     # MI300X; dendrogram.py default

# name in JSON -> (CSV column, reference, lo, hi, step, unit)
# hi is exclusive as in np.arange, so the last edge is hi - step.
SPECS = {
    "inst_power":    ("inst_power_W",           TDP_W, 0.5, 2.05, None,  "fraction of TDP (750 W)"),
    "socket_power":  ("current_socket_power_W", TDP_W, 0.5, 2.05, None,  "fraction of TDP (750 W)"),
    "gfx_frequency": ("gfx_clock_MHz",          1.0,   0.0, 2300.0, 100.0, "MHz"),
    "hotspot_temp":  ("hotspot_temp_C",         1.0,  30.0, 115.0,  5.0,  "degrees C"),
    "edge_temp":     ("edge_temp_C",            1.0,  30.0, 115.0,  5.0,  "degrees C"),
}


def bin_vector(series, ref, lo, hi, step):
    """dendrogram.py's binning, for any column: fraction of samples >= lo per bin."""
    x = pd.to_numeric(series, errors="coerce").dropna() / ref
    edges = np.arange(lo, hi, step)
    pop = x[x >= lo]
    if len(pop) == 0:
        return None, edges, 0, 0
    vec = [round(float(((pop >= edges[i]) & (pop < edges[i + 1])).sum() / len(pop)), 4)
           for i in range(len(edges) - 1)]
    return vec, edges, len(pop), int((pop >= edges[-1]).sum())


def dendrogram_power_vector(dendrogram, df, label):
    """Run dendrogram.calculate_power_distribution() on the trimmed samples.
    It reads a plain CSV and caches into a vectors file, so both go to a temp dir."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "metric_sampling.csv")
        df.to_csv(csv_path, index=False)
        dendrogram.filename_mapping = {csv_path: label}
        dendrogram.results_cache.clear()
        out = dendrogram.calculate_power_distribution(
            csv_path, tdp=TDP_W, vectors_file=os.path.join(tmp, "vectors.json"))
    if out is None:
        raise SystemExit("dendrogram.calculate_power_distribution failed")
    return [float(v) for v in out[1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="power run folder, e.g. results/prof-433930")
    ap.add_argument("--label", default="LAMMPS 16x8x12", help="workload label")
    ap.add_argument("--out", help="output JSON (default: <run_dir>/sampling_vectors.json)")
    ap.add_argument("--no-trim", action="store_true", help="keep all samples, not just the kernel window")
    ap.add_argument("--per-gpu", action="store_true", help="also write <name>_gpu<N> vectors (multi-GPU runs)")
    ap.add_argument("--dendrogram-dir", default=DENDRO_DIR, help=f"folder with dendrogram.py (default {DENDRO_DIR})")
    args = ap.parse_args()

    sys.path.insert(0, args.dendrogram_dir)
    import dendrogram  # noqa: E402

    run_dir = os.path.abspath(args.run_dir)
    tels = sorted(glob.glob(os.path.join(run_dir, "profiling_result_*.csv")))
    if not tels:
        raise SystemExit(f"no profiling_result_*.csv in {run_dir}")
    with open(tels[0]) as fh:
        preamble = fh.readline().strip()
    frames = []
    for t in tels:
        d = pd.read_csv(t, skiprows=1)
        d["gpu"] = os.path.splitext(os.path.basename(t))[0].rsplit("_", 1)[-1]
        frames.append(d)
    df_all = pd.concat(frames, ignore_index=True)
    ts = df_all["timestamp_ns"].to_numpy()

    kts = sorted(glob.glob(os.path.join(run_dir, "**", "*kernel_trace.csv"), recursive=True))
    if kts and not args.no_trim:
        k0 = k1 = None
        for kt in kts:
            k = pd.read_csv(kt, usecols=["Start_Timestamp", "End_Timestamp"])
            a, b = int(k["Start_Timestamp"].min()), int(k["End_Timestamp"].max())
            k0 = a if k0 is None else min(k0, a)
            k1 = b if k1 is None else max(k1, b)
        df = df_all[(ts >= k0) & (ts <= k1)].reset_index(drop=True)
        window = (f"kernel window only: first kernel start {k0} ns to last kernel end {k1} ns "
                  f"({(k1 - k0) / 1e9:.3f} s) over {len(kts)} kernel-trace file(s); "
                  f"{int((ts < k0).sum())} GPU-samples before (app/rocprofv3 startup) and "
                  f"{int((ts > k1).sum())} after (rocprofv3 output writing) dropped")
    else:
        df = df_all
        window = "all samples (not trimmed to the kernel window)"

    first = df[df["gpu"] == df["gpu"].iloc[0]]
    dt = np.diff(first["timestamp_ns"].to_numpy()) / 1e6
    info_path = os.path.join(run_dir, "run_info.txt")
    run_info = open(info_path).read().strip().splitlines() if os.path.exists(info_path) else []
    ref = open(os.path.join(run_dir, "clock_ref.txt")).read().strip() \
        if os.path.exists(os.path.join(run_dir, "clock_ref.txt")) else None

    vectors, bins, skipped = {}, {}, {}
    for name, (col, refv, lo, hi, step, unit) in SPECS.items():
        if col not in df.columns:
            skipped[name] = f"column {col} not in CSV"
            continue
        step = step if step is not None else dendrogram.bin_size
        vec, edges, n_pop, n_above = bin_vector(df[col], refv, lo, hi, step)
        if vec is None:
            skipped[name] = f"no valid samples in {col} (sensor not exposed: all nan)" \
                if df[col].isna().all() else f"no samples >= {lo}"
            continue
        if name == "inst_power":
            dvec = dendrogram_power_vector(dendrogram, df, args.label)
            if len(dvec) != len(vec) or max(abs(a - b) for a, b in zip(dvec, vec)) > 1e-4:
                raise SystemExit(f"inst_power mismatch vs dendrogram.py:\n{dvec}\n{vec}")
            vec = dvec
        vectors[name] = vec
        if args.per_gpu and df["gpu"].nunique() > 1:
            for g, dg in df.groupby("gpu", sort=True):
                v, _, _, _ = bin_vector(dg[col], refv, lo, hi, step)
                if v is not None:
                    vectors[f"{name}_gpu{g}"] = v
        bins[name] = {
            "column": col,
            "bin_edges": [round(float(e), 4) for e in edges],
            "unit": unit,
            "samples_in_population": n_pop,
            "samples_above_last_edge": n_above,
        }

    doc = {
        "description": {
            "label": args.label,
            "gpu": "AMD Instinct MI300X (gfx942), ROCm 7.2.0",
            "gpus_sampled": sorted(df_all["gpu"].unique().tolist()),
            "run": run_dir,
            "run_info": run_info,
            "telemetry_files": [os.path.basename(t) for t in tels],
            "sampling": (
                "rocprofwrap_lt wrapper.py + amd-smi-query sampled each listed GPU while the "
                "application (see run_info) ran under rocprofv3 --kernel-trace (no counters). "
                "Vectors pool the samples of all sampled GPUs. Requested interval 1 ms; achieved median "
                f"{np.median(dt):.3f} ms (mean {dt.mean():.3f} ms), limited by amd-smi query time. "
                "inst_power = energy-counter delta / time delta between samples; "
                "socket_power = amd-smi current_socket_power; gfx_frequency = current gfx clock; "
                "hotspot_temp = junction temperature. Sampler preamble: " + preamble),
            "window": window,
            "clock_alignment": (
                "sampler timestamps (amd-smi energy timestamp) and rocprofv3 kernel timestamps "
                "share the same time base (CLOCK_BOOTTIME/MONOTONIC); reference: " + str(ref)),
            "vector_creation": (
                "Same algorithm as calculate_power_distribution() in "
                "minos-analysis/dendrogram_plot/dendrogram.py: raw samples, no smoothing; "
                "value = sample / reference; keep samples >= the first bin edge; "
                "histogram over np.arange(lo, hi, step) (hi exclusive); each entry = fraction of "
                "kept samples in [edge_i, edge_i+1), rounded to 4 decimals. Samples at or above "
                "the last edge are kept but fall in no bin, so a vector can sum to < 1. "
                f"inst_power is produced by dendrogram.py itself (bin_size = {dendrogram.bin_size}, "
                f"TDP = {TDP_W:.0f} W), so it is directly comparable with app_vectors.json. "
                "gfx_frequency and hotspot_temp use the ranges from build_workload_vectors.py."),
            "samples": {"total_in_files": int(len(df_all)), "used": int(len(df))},
            "bins": bins,
            "skipped": skipped,
        },
    }
    doc.update(vectors)

    out = args.out or os.path.join(run_dir, "sampling_vectors.json")
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    print(f"wrote {out}")
    for name, vec in vectors.items():
        print(f"  {name:14s} {len(vec):2d} bins  sum={sum(vec):.4f}  {vec}")
    for name, why in skipped.items():
        print(f"  {name:14s} skipped: {why}")


if __name__ == "__main__":
    main()
