#!/usr/bin/env python3
"""Post-process rocprofv3 PMC passes for a workload on MI300X (gfx942).

Implements the formulas in post_processing_formulae.md (derived metrics in
counters.json).  Reads every pass listed in <pmc-dir>/counters.json, sums
counters and durations per kernel name within each pass, and combines passes
only through per-kernel rates (per ns).  Launch IDs are never matched across
passes: the QEq solver's iteration count differs slightly between runs.

Outputs, in <pmc-dir>/analysis/ unless --out-dir is given:
  per_kernel_metrics.csv   every kernel plus a TOTAL row, all metrics
  top_kernels.csv          kernels covering --time-pct (90%) of GPU time, key metrics, bound
  summary.json             whole-run metrics and data checks
  report.md                readable summary: whole run, top kernels, checks

Usage:
  python3 postprocess_pmc.py results/pmc-433875
  python3 postprocess_pmc.py results/pmc-433875 \\
      --power-ktrace results/prof-<jobid>/ktrace_hns16812

Multi-GPU runs (8x MI300X, one output set per process): every
*counter_collection.csv / *kernel_trace.csv in a pass folder is read and summed.
Launch IDs restart per process, so launches are keyed by (file, Dispatch_Id).
Rates are then per-GPU averages (sum of work / sum of kernel time over GPUs),
which is what % of a single GPU's peak needs.

Standard library only (Python >= 3.9).
"""
import argparse
import bisect
import collections
import csv
import glob
import json
import math
import os
import re
import sys

# --------------------------------------------------------------------------
# MI300X constants (post_processing_formulae.md, section 2)
# --------------------------------------------------------------------------
N_XCC = 8
N_CU = 304
N_SIMD = 1216
WAVE = 64
MAX_WAVES_CU = 32
MAX_WAVES_SIMD = 8
VGPR_PER_SIMD = 512
LDS_PER_CU = 65536
MOPS_UNIT = 512
F_BOOST_MHZ = 2100.0
PEAK_FP64_VALU = 81.7     # TFLOP/s
PEAK_FP32_VALU = 163.4
PEAK_FP64_MFMA = 163.4
PEAK_FP32_MFMA = 163.4
PEAK_FP16_VALU = 163.4    # assumed = packed FP32 rate (not on the datasheet)
PEAK_F16_MFMA = 1307.4    # FP16 and BF16 dense: 304 x 2048 FLOP/clk x 2.1 GHz
PEAK_F8_MFMA = 2614.9     # FP8 dense: 304 x 4096 FLOP/clk x 2.1 GHz
PEAK_I8_MFMA = 2614.9     # INT8 dense, TOP/s
PEAK_MEM_BW = 5.3         # TB/s
LATENCY_PCT = 10.0        # bound_mix = latency below this % of both peaks (--latency-pct)

NAN = float("nan")
TOTAL = "TOTAL"


def div(a, b):
    try:
        if b == 0 or b is None or a is None or math.isnan(a) or math.isnan(b):
            return NAN
    except TypeError:
        return NAN
    return a / b


def ceil_to(x, n):
    return int(math.ceil(x / n) * n)


# --------------------------------------------------------------------------
# Kernel display names (section 3.4)
# --------------------------------------------------------------------------
def short_name(full):
    if full.startswith("__amd_rocclr_"):
        return "rocclr:" + full[len("__amd_rocclr_"):]
    kind = "reduce " if "ParallelReduce<" in full else ("scan " if "ParallelScan<" in full else "")
    functor = re.search(r"LAMMPS_NS::(\w+)", full)
    tag = re.search(r"LAMMPS_NS::(Tag\w+(?:<[^<>]*>)?)", full)
    if functor:
        name = functor.group(1)
        if tag and tag.group(1) != name:
            name += "::" + tag.group(1).replace(" ", "")
        return kind + name
    kimpl = re.search(r"Parallel(?:For|Reduce|Scan)<Kokkos::Impl::(\w+)", full)
    if kimpl:
        return kind + "Kokkos::" + kimpl.group(1)
    return full[:80]


def unique_short_names(names):
    out, used = {}, collections.Counter()
    for full in names:
        s = short_name(full)
        used[s] += 1
        out[full] = s if used[s] == 1 else f"{s} #{used[s]}"
    return out


# --------------------------------------------------------------------------
# Reading one pass (section 3.1)
# --------------------------------------------------------------------------
class PassData:
    def __init__(self, name, label, counters):
        self.name, self.label, self.counters = name, label, counters
        self.C = collections.defaultdict(lambda: collections.defaultdict(float))
        self.D = collections.defaultdict(int)       # ns
        self.L = collections.Counter()
        self.regs = {}                              # kernel -> Counter of (vgpr, agpr, lds)
        self.wg = {}                                # kernel -> Counter of wg_threads
        self.trace_dispatches = 0
        self.counter_dispatches = 0
        self.missing_dispatches = 0
        self.missing_ns = 0
        self.trace_ns = 0
        self.loop_time_s = NAN
        self.present = set()
        self.files = 0
        self.agents = set()


def find_all(folder, pattern):
    return sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True))


def find_one(folder, pattern):
    hits = find_all(folder, pattern)
    return hits[0] if hits else None


PERF_REGEX = r"Loop time of ([0-9.eE+-]+)"


def read_pass(pmc_dir, p):
    pd = PassData(p["name"], p.get("label", ""), p["counters"])
    folder = os.path.join(pmc_dir, p["name"])
    ccs = find_all(folder, "*counter_collection.csv")
    if not ccs:
        raise SystemExit(f"error: no counter_collection.csv in {folder}")
    pd.files = len(ccs)

    seen = set()                     # (file index, Dispatch_Id): IDs restart per process
    for fi, cc in enumerate(ccs):
        with open(cc, newline="") as f:
            rd = csv.reader(f)
            h = {c: i for i, c in enumerate(next(rd))}
            iD, iK, iN, iV = h["Dispatch_Id"], h["Kernel_Name"], h["Counter_Name"], h["Counter_Value"]
            iS, iE, iA = h["Start_Timestamp"], h["End_Timestamp"], h.get("Agent_Id")
            iVG, iAG, iLDS = h["VGPR_Count"], h["Accum_VGPR_Count"], h["LDS_Block_Size"]
            for r in rd:
                k = r[iK]
                pd.C[k][r[iN]] += float(r[iV])
                pd.present.add(r[iN])
                d = (fi, r[iD])
                if d not in seen:
                    seen.add(d)
                    pd.D[k] += int(r[iE]) - int(r[iS])
                    pd.L[k] += 1
                    pd.regs.setdefault(k, collections.Counter())[(int(r[iVG]), int(r[iAG]), int(r[iLDS]))] += 1
                    if iA is not None:
                        pd.agents.add((fi, r[iA]))
    pd.counter_dispatches = len(seen)

    # match each trace file to its counter file by the rocprofv3 prefix
    prefix = {os.path.basename(c)[: -len("counter_collection.csv")]: i for i, c in enumerate(ccs)}
    for kt in find_all(folder, "*kernel_trace.csv"):
        fi = prefix.get(os.path.basename(kt)[: -len("kernel_trace.csv")])
        with open(kt, newline="") as f:
            rd = csv.reader(f)
            h = {c: i for i, c in enumerate(next(rd))}
            iD, iK, iS, iE = h["Dispatch_Id"], h["Kernel_Name"], h["Start_Timestamp"], h["End_Timestamp"]
            ix, iy, iz = h["Workgroup_Size_X"], h["Workgroup_Size_Y"], h["Workgroup_Size_Z"]
            for r in rd:
                pd.trace_dispatches += 1
                dur = int(r[iE]) - int(r[iS])
                pd.trace_ns += dur
                pd.wg.setdefault(r[iK], collections.Counter())[int(r[ix]) * int(r[iy]) * int(r[iz])] += 1
                if (fi, r[iD]) not in seen:
                    pd.missing_dispatches += 1
                    pd.missing_ns += dur

    # the application's own timing line, from its log or the captured stdout
    for path in find_all(folder, "log.*") + find_all(folder, "*.txt") + find_all(folder, "*.log"):
        with open(path, errors="replace") as f:
            for line in f:
                m = re.search(PERF_REGEX, line)
                if m:
                    pd.loop_time_s = float(m.group(1))
                    break
        if not math.isnan(pd.loop_time_s):
            break

    # TOTAL pseudo-kernel: whole-run sums (section 3.3)
    for k in list(pd.C):
        for c, v in pd.C[k].items():
            pd.C[TOTAL][c] += v
    pd.D[TOTAL] = sum(v for k, v in pd.D.items() if k != TOTAL)
    pd.L[TOTAL] = sum(v for k, v in pd.L.items() if k != TOTAL)
    return pd


# --------------------------------------------------------------------------
# Theoretical occupancy (section 5.5)
# --------------------------------------------------------------------------
def max_waves_cu(vgpr, agpr, lds, wg_threads):
    if agpr > 0:
        alloc = ceil_to(ceil_to(vgpr, 4) + agpr, 8)
    else:
        alloc = ceil_to(max(vgpr, 1), 8)
    w_simd = min(MAX_WAVES_SIMD, VGPR_PER_SIMD // max(alloc, 1))
    limit = min(4 * w_simd, MAX_WAVES_CU)
    if lds > 0 and wg_threads > 0:
        waves_per_wg = int(math.ceil(wg_threads / WAVE))
        limit = min(limit, (LDS_PER_CU // lds) * waves_per_wg)
    return limit


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
class Metrics:
    """Looks up each counter in whichever pass collected it."""

    def __init__(self, passes):
        self.passes = passes
        self.where = {}
        for pd in passes:
            for c in pd.counters:
                self.where.setdefault(c, pd)   # first pass holding it (GRBM_GUI_ACTIVE: pass 1)

    def pd(self, counter):
        return self.where.get(counter)

    def S(self, k, counter, pd=None):
        pd = pd or self.pd(counter)
        if pd is None or k not in pd.D or counter not in pd.present:
            return NAN
        return pd.C[k].get(counter, 0.0)

    def Dn(self, k, counter):
        pd = self.pd(counter)
        return pd.D.get(k, 0) if pd else 0

    def rate(self, k, counter):              # per ns, within the counter's pass
        return div(self.S(k, counter), self.Dn(k, counter))

    clock_mhz = None                         # sampled gfx clock (4.2); None -> counter fallback

    def gui_counter(self, k, pd):            # GRBM_GUI_ACTIVE cycles: kernel + per-launch overhead
        return div(self.S(k, "GRBM_GUI_ACTIVE", pd), N_XCC)

    def gui(self, k, pd):                    # cycles the kernel ran in that pass (section 4.2)
        if self.clock_mhz is None:
            return self.gui_counter(k, pd)
        if pd is None or k not in pd.D:
            return NAN
        return pd.D[k] * self.clock_mhz / 1000.0     # ns x MHz / 1000


def compute(k, M, pmax_waves, useful_total=None):
    """All metrics for kernel k (or TOTAL). Section references are to
    post_processing_formulae.md."""
    m = {}
    p1 = M.pd("SQ_INSTS_VALU_ADD_F64")
    p2 = M.pd("SQ_INSTS_VALU_ADD_F32")
    p3 = M.pd("SQ_INSTS_SALU")
    p4 = M.pd("SQ_WAVE_CYCLES")
    p5 = M.pd("TCC_EA0_RDREQ_sum")
    p6 = M.pd("TCC_EA0_WRREQ_sum")
    S, R = M.S, M.rate

    # ---- time (4.1) ---------------------------------------------------
    durs = [pd.D[k] for pd in M.passes if pd.D.get(k)]
    m["time_ms"] = div(sum(durs), len(durs)) / 1e6 if durs else NAN
    m["launches"] = p4.L.get(k, 0) if p4 else 0
    m["mean_launch_us"] = div(p4.D.get(k, 0), m["launches"]) / 1e3 if p4 else NAN

    # ---- clock (4.3, 4.4), from pass 1 ----------------------------------
    gui1 = M.gui(k, p1)
    m["eff_clock_MHz"] = div(M.gui_counter(k, p1), p1.D.get(k, 0)) * 1000 if p1 else NAN
    m["clock_MHz_used"] = M.clock_mhz if M.clock_mhz is not None else m["eff_clock_MHz"]
    clock_factor = div(m["clock_MHz_used"], F_BOOST_MHZ)
    m["clock_factor"] = clock_factor

    # ---- CU utilisation (5.1 - 5.9), pass 4 (MFMA busy: pass 1) ----------
    gui4 = M.gui(k, p4)
    m["CU_busy_frac"] = div(S(k, "SQ_BUSY_CU_CYCLES"), gui4 * N_CU)
    m["CU_busy_pct"] = 100 * m["CU_busy_frac"]
    m["VALUBusy_pct"] = 100 * div(S(k, "SQ_ACTIVE_INST_VALU"), gui4 * N_CU)
    m["VALU_threads"] = div(S(k, "SQ_THREAD_CYCLES_VALU"), S(k, "SQ_ACTIVE_INST_VALU"))
    m["VALUUtilization_pct"] = 100 * div(m["VALU_threads"], WAVE)
    m["waves_per_CU"] = div(4 * S(k, "SQ_WAVE_CYCLES"), gui4 * N_CU)
    m["Occupancy_pct"] = 100 * div(m["waves_per_CU"], MAX_WAVES_CU)
    m["max_waves_CU"] = pmax_waves.get(k, NAN)
    m["Occupancy_of_max_pct"] = 100 * div(m["waves_per_CU"], m["max_waves_CU"])
    m["Wave_issue_pct"] = 100 * div(S(k, "SQ_ACTIVE_INST_ANY"), S(k, "SQ_WAVE_CYCLES"))
    m["SALUBusy_pct"] = 100 * div(S(k, "SQ_INST_CYCLES_SALU"), gui4 * N_CU)
    m["MfmaUtil_pct"] = 100 * div(S(k, "SQ_VALU_MFMA_BUSY_CYCLES"), gui1 * N_SIMD)

    # ---- FLOPs (6.1 - 6.9) ---------------------------------------------
    add64, mul64 = S(k, "SQ_INSTS_VALU_ADD_F64"), S(k, "SQ_INSTS_VALU_MUL_F64")
    fma64, tr64 = S(k, "SQ_INSTS_VALU_FMA_F64"), S(k, "SQ_INSTS_VALU_TRANS_F64")
    m["FP64_VALU_FLOP"] = WAVE * (add64 + mul64 + 2 * fma64 + tr64)
    m["FP64_TRANS_FLOP"] = WAVE * tr64
    m["FP64_MFMA_FLOP"] = MOPS_UNIT * S(k, "SQ_INSTS_VALU_MFMA_MOPS_F64")
    m["FP64_FLOP"] = m["FP64_VALU_FLOP"] + m["FP64_MFMA_FLOP"]
    if useful_total is None:
        m["FP64_VALU_FLOP_useful"] = m["FP64_VALU_FLOP"] * m["VALUUtilization_pct"] / 100
    else:
        m["FP64_VALU_FLOP_useful"] = useful_total          # TOTAL: sum over kernels (6.3)
    d1 = p1.D.get(k, 0)
    m["TFLOPs_FP64_VALU"] = div(m["FP64_VALU_FLOP"], d1) / 1000
    m["TFLOPs_FP64_VALU_useful"] = div(m["FP64_VALU_FLOP_useful"], d1) / 1000
    m["TFLOPs_FP64_MFMA"] = div(m["FP64_MFMA_FLOP"], d1) / 1000
    m["TFLOPs_FP64"] = div(m["FP64_FLOP"], d1) / 1000
    m["FP64_VALU_peak_pct"] = 100 * div(m["TFLOPs_FP64_VALU"], PEAK_FP64_VALU)
    m["FP64_VALU_peak_clk_pct"] = 100 * div(m["TFLOPs_FP64_VALU"], PEAK_FP64_VALU * clock_factor)
    m["FP64_MFMA_peak_pct"] = 100 * div(m["TFLOPs_FP64_MFMA"], PEAK_FP64_MFMA)
    m["FP64_share_pct"] = 100 * div(add64 + mul64 + fma64 + tr64, S(k, "SQ_INSTS_VALU"))

    m["FP32_VALU_FLOP"] = WAVE * (S(k, "SQ_INSTS_VALU_ADD_F32") + S(k, "SQ_INSTS_VALU_MUL_F32")
                                  + 2 * S(k, "SQ_INSTS_VALU_FMA_F32") + S(k, "SQ_INSTS_VALU_TRANS_F32"))
    m["FP32_MFMA_FLOP"] = MOPS_UNIT * S(k, "SQ_INSTS_VALU_MFMA_MOPS_F32")
    m["FP32_FLOP"] = m["FP32_VALU_FLOP"] + m["FP32_MFMA_FLOP"]
    d2 = p2.D.get(k, 0) if p2 else 0
    m["TFLOPs_FP32"] = div(m["FP32_FLOP"], d2) / 1000
    m["FP32_MFMA_peak_pct"] = 100 * div(div(m["FP32_MFMA_FLOP"], d2) / 1000, PEAK_FP32_MFMA)
    m["FP16_VALU_FLOP"] = WAVE * (S(k, "SQ_INSTS_VALU_ADD_F16") + S(k, "SQ_INSTS_VALU_MUL_F16")
                                  + 2 * S(k, "SQ_INSTS_VALU_FMA_F16") + S(k, "SQ_INSTS_VALU_TRANS_F16"))

    # ---- instruction mix (7), via per-ns rates -------------------------
    mix = {"VALU": R(k, "SQ_INSTS_VALU"), "SALU": R(k, "SQ_INSTS_SALU"),
           "VMEM_RD": R(k, "SQ_INSTS_VMEM_RD"), "VMEM_WR": R(k, "SQ_INSTS_VMEM_WR"),
           "LDS": R(k, "SQ_INSTS_LDS")}
    tot = sum(mix.values())
    for key, v in mix.items():
        m[f"inst_{key}_pct"] = 100 * div(v, tot)

    # ---- memory (8.1 - 8.8) --------------------------------------------
    rd, rd32, bub = S(k, "TCC_EA0_RDREQ_sum"), S(k, "TCC_EA0_RDREQ_32B_sum"), S(k, "TCC_BUBBLE_sum")
    wr, wr64 = S(k, "TCC_EA0_WRREQ_sum"), S(k, "TCC_EA0_WRREQ_64B_sum")
    m["mem_read_bytes"] = 128 * bub + 64 * (rd - bub - rd32) + 32 * rd32
    m["read_128B_pct"] = 100 * div(bub, rd)
    m["read_32B_pct"] = 100 * div(rd32, rd)
    m["mem_write_bytes"] = 32 * (wr - wr64) + 64 * wr64
    m["atomic_req_pct"] = 100 * div(S(k, "TCC_EA0_ATOMIC_sum"), wr)
    m["atomic_bytes"] = 32 * S(k, "TCC_EA0_ATOMIC_sum")
    m["uncached_read_bytes"] = 32 * S(k, "TCC_EA0_RD_UNCACHED_32B_sum")
    m["uncached_write_bytes"] = 32 * S(k, "TCC_EA0_WR_UNCACHED_32B_sum")
    d5 = p5.D.get(k, 0) if p5 else 0
    d6 = p6.D.get(k, 0) if p6 else 0
    rbw = div(m["mem_read_bytes"], d5) / 1000
    wbw = div(m["mem_write_bytes"], d6) / 1000
    m["read_BW_TBps"], m["write_BW_TBps"] = rbw, wbw
    m["mem_BW_TBps"] = rbw + wbw
    m["mem_BW_peak_pct"] = 100 * div(m["mem_BW_TBps"], PEAK_MEM_BW)
    m["AI_FP64"] = div(m["TFLOPs_FP64"], m["mem_BW_TBps"])          # (FLOP/ns) / (B/ns)
    attain = min(PEAK_FP64_VALU, m["AI_FP64"] * PEAK_MEM_BW) if not math.isnan(m["AI_FP64"]) else NAN
    m["attainable_TFLOPs"] = attain
    m["roofline_pct"] = 100 * div(m["TFLOPs_FP64"], attain)
    m["bound"] = ("" if math.isnan(m["AI_FP64"]) else
                  ("memory" if m["AI_FP64"] < PEAK_FP64_VALU / PEAK_MEM_BW else "compute"))

    # ---- low-precision MFMA (pass 7, if collected) and all-precision roofline (8.9)
    for t in ("BF16", "F16", "F8", "I8"):
        m[f"{t}_MFMA_FLOP"] = MOPS_UNIT * S(k, f"SQ_INSTS_VALU_MFMA_MOPS_{t}")
    p7 = M.pd("SQ_INSTS_VALU_MFMA_MOPS_BF16")
    d3 = p3.D.get(k, 0) if p3 else 0
    d7 = p7.D.get(k, 0) if p7 else 0
    # (name, TFLOP/s, peak TFLOP/s); each rate over its own pass's kernel time
    work = [("FP64_VALU", m["TFLOPs_FP64_VALU"], PEAK_FP64_VALU),
            ("FP64_MFMA", m["TFLOPs_FP64_MFMA"], PEAK_FP64_MFMA),
            ("FP32_VALU", div(m["FP32_VALU_FLOP"], d2) / 1000, PEAK_FP32_VALU),
            ("FP32_MFMA", div(m["FP32_MFMA_FLOP"], d2) / 1000, PEAK_FP32_MFMA),
            ("FP16_VALU", div(m["FP16_VALU_FLOP"], d3) / 1000, PEAK_FP16_VALU),
            ("BF16_MFMA", div(m["BF16_MFMA_FLOP"], d7) / 1000, PEAK_F16_MFMA),
            ("F16_MFMA", div(m["F16_MFMA_FLOP"], d7) / 1000, PEAK_F16_MFMA),
            ("F8_MFMA", div(m["F8_MFMA_FLOP"], d7) / 1000, PEAK_F8_MFMA),
            ("I8_MFMA", div(m["I8_MFMA_FLOP"], d7) / 1000, PEAK_I8_MFMA)]
    # drop precisions no pass collected; a kernel missing from a pass that did gives nan
    has = {"FP64_VALU": p1, "FP64_MFMA": p1, "FP32_VALU": p2, "FP32_MFMA": p2, "FP16_VALU": p3,
           "BF16_MFMA": p7, "F16_MFMA": p7, "F8_MFMA": p7, "I8_MFMA": p7}
    work = [(n, r, p) for n, r, p in work if has[n] is not None]
    m["TFLOPs_lowp_MFMA"] = sum(r for n, r, _ in work if n in ("BF16_MFMA", "F16_MFMA", "F8_MFMA", "I8_MFMA"))
    m["TFLOPs_all"] = sum(r for _, r, _ in work)
    frac = sum(r / p for _, r, p in work)                   # share of peak compute time used
    m["compute_peak_pct"] = 100 * frac
    m["peak_TFLOPs_mix"] = div(m["TFLOPs_all"], frac)         # peak for this kernel's precision mix
    m["dominant_precision"] = (max(work, key=lambda w: w[1] / w[2])[0]
                               if frac > 0 else "")                # False for nan
    m["AI_all"] = div(m["TFLOPs_all"], m["mem_BW_TBps"])
    m["ridge_mix"] = div(m["peak_TFLOPs_mix"], PEAK_MEM_BW)
    ai_ok = frac > 0 and m["AI_all"] > 0                    # False for nan; no FLOPs -> no ceiling
    m["attainable_TFLOPs_mix"] = min(m["peak_TFLOPs_mix"], m["AI_all"] * PEAK_MEM_BW) if ai_ok else NAN
    m["attainable_BW_TBps"] = min(PEAK_MEM_BW, m["peak_TFLOPs_mix"] / m["AI_all"]) if ai_ok else NAN
    # near neither ceiling -> limited by latency / too little parallel work, not by a roof
    m["bound_mix"] = ("" if math.isnan(m["mem_BW_peak_pct"]) or math.isnan(frac) else
                      "latency" if max(m["compute_peak_pct"], m["mem_BW_peak_pct"]) < LATENCY_PCT else
                      "compute" if m["compute_peak_pct"] > m["mem_BW_peak_pct"] else "memory")

    # ---- per-pass clocks (check) ---------------------------------------
    for pd in M.passes:
        m[f"clock_MHz_{pd.name}"] = div(M.gui_counter(k, pd), pd.D.get(k, 0)) * 1000
    return m


# --------------------------------------------------------------------------
# Power run kernel trace (4.5)
# --------------------------------------------------------------------------
def _union_ns(iv):
    iv.sort()
    busy, cur_s, cur_e = 0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    return busy


def read_power_ktrace(path):
    """GPU busy % per GPU: kernel intervals grouped by (trace file, Agent_Id), so it
    works for one process per GPU and for one process driving several GPUs."""
    files = find_all(path, "*kernel_trace.csv") if os.path.isdir(path) else [path]
    if not files:
        raise SystemExit(f"error: no kernel_trace.csv under {path}")
    groups, per_k, n = collections.defaultdict(list), collections.Counter(), 0
    for fi, p in enumerate(files):
        with open(p, newline="") as f:
            rd = csv.reader(f)
            h = {c: i for i, c in enumerate(next(rd))}
            iK, iS, iE, iA = h["Kernel_Name"], h["Start_Timestamp"], h["End_Timestamp"], h.get("Agent_Id")
            for r in rd:
                s, e = int(r[iS]), int(r[iE])
                groups[(fi, r[iA] if iA is not None else "")].append((s, e))
                per_k[r[iK]] += e - s
                n += 1
    busy_pct, busy_tot, span_max, first = [], 0, 0, None
    for iv in groups.values():
        span = max(e for _, e in iv) - min(s for s, _ in iv)
        busy = _union_ns(iv)
        busy_pct.append(100 * div(busy, span))
        busy_tot += busy
        span_max = max(span_max, span)
        first = min(first, min(s for s, _ in iv)) if first is not None else min(s for s, _ in iv)
    return {"file": path, "trace_files": len(files), "gpus": len(groups), "dispatches": n,
            "span_s": span_max / 1e9, "busy_s": busy_tot / 1e9 / max(len(groups), 1),
            "sum_kernel_s": sum(per_k.values()) / 1e9,
            "GPU_busy_pct": sum(busy_pct) / len(busy_pct) if busy_pct else NAN,
            "GPU_busy_pct_per_gpu": [round(x, 2) for x in busy_pct],
            "per_kernel_ns": per_k, "first_start_ns": first,
            "intervals": [iv for g in groups.values() for iv in g]}


def sampled_clock(telemetry, intervals):
    """Mean sampled gfx clock over the samples taken while a kernel was running
    (4.2). Telemetry and kernel trace share the boot-time clock. Multi-GPU runs are
    pooled: a sample counts if any traced kernel was running at its timestamp."""
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    starts = [iv[0] for iv in merged]
    run, n_all, n_win = [], 0, 0
    k0, k1 = merged[0][0], merged[-1][1]
    for path in telemetry:
        with open(path, newline="") as fh:
            fh.readline()                                  # sampler preamble line
            rd = csv.reader(fh)
            h = {c: i for i, c in enumerate(next(rd))}
            iT, iC = h["timestamp_ns"], h["gfx_clock_MHz"]
            for r in rd:
                n_all += 1
                try:
                    t, c = int(r[iT]), float(r[iC])
                except ValueError:
                    continue
                if math.isnan(c) or not (k0 <= t <= k1):
                    continue
                n_win += 1
                i = bisect.bisect_right(starts, t) - 1
                if i >= 0 and t <= merged[i][1]:
                    run.append(c)
    if not run:
        raise SystemExit("error: no telemetry sample falls inside a traced kernel; "
                         "check that --telemetry and --power-ktrace are from the same run")
    run.sort()
    return {"files": telemetry, "samples_total": n_all, "samples_kernel_window": n_win,
            "samples_kernel_running": len(run), "clock_MHz": sum(run) / len(run),
            "clock_MHz_median": run[len(run) // 2],
            "clock_MHz_p5": run[int(0.05 * (len(run) - 1))],
            "clock_MHz_p95": run[int(0.95 * (len(run) - 1))]}


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
def f(x, spec=".3g"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, str):
        return x
    return format(x, spec)


WHOLE_RUN_ROWS = [
    ("Kernel time (mean over passes; summed over GPUs)", "time_ms", ".1f", "ms"),
    ("Kernel launches (pass 4)", "launches", "d", ""),
    ("Clock for utilisation denominators", "clock_MHz_used", ".0f", "MHz"),
    ("GRBM_GUI_ACTIVE / 8 per ns (check only)", "eff_clock_MHz", ".0f", "MHz"),
    ("CU busy", "CU_busy_frac", ".3f", "fraction"),
    ("VALU busy", "VALUBusy_pct", ".1f", "%"),
    ("Threads active per vector instruction", "VALU_threads", ".1f", "of 64"),
    ("VALU thread utilisation", "VALUUtilization_pct", ".1f", "%"),
    ("Occupancy", "waves_per_CU", ".1f", "waves/CU"),
    ("Occupancy", "Occupancy_pct", ".1f", "% of 32"),
    ("Occupancy of kernels' own limit (time-weighted)", "Occupancy_of_max_pct", ".1f", "%"),
    ("Wave issue fraction", "Wave_issue_pct", ".1f", "%"),
    ("SALU busy (lower bound)", "SALUBusy_pct", ".1f", "%"),
    ("MFMA busy", "MfmaUtil_pct", ".2f", "%"),
    ("FP64 FLOPs issued", "FP64_VALU_FLOP", ".3e", "FLOP"),
    ("FP64 FLOPs useful (estimate)", "FP64_VALU_FLOP_useful", ".3e", "FLOP"),
    ("FP64 transcendental FLOPs", "FP64_TRANS_FLOP", ".3e", "FLOP"),
    ("FP64 MFMA FLOPs", "FP64_MFMA_FLOP", ".3e", "FLOP"),
    ("FP64 throughput (issued)", "TFLOPs_FP64_VALU", ".2f", "TFLOP/s"),
    ("FP64 throughput (useful)", "TFLOPs_FP64_VALU_useful", ".2f", "TFLOP/s"),
    ("FP64 % of peak (2100 MHz)", "FP64_VALU_peak_pct", ".1f", "%"),
    ("FP64 % of peak (measured clock)", "FP64_VALU_peak_clk_pct", ".1f", "%"),
    ("FP64 share of vector instructions", "FP64_share_pct", ".1f", "%"),
    ("FP32 FLOPs", "FP32_FLOP", ".3e", "FLOP"),
    ("FP16 FLOPs", "FP16_VALU_FLOP", ".3e", "FLOP"),
    ("Instruction mix: VALU", "inst_VALU_pct", ".1f", "%"),
    ("Instruction mix: SALU", "inst_SALU_pct", ".1f", "%"),
    ("Instruction mix: VMEM read", "inst_VMEM_RD_pct", ".1f", "%"),
    ("Instruction mix: VMEM write", "inst_VMEM_WR_pct", ".1f", "%"),
    ("Instruction mix: LDS", "inst_LDS_pct", ".1f", "%"),
    ("Read bytes (L2 -> memory)", "mem_read_bytes", ".3e", "B"),
    ("Reads that are 128-byte", "read_128B_pct", ".2f", "%"),
    ("Write bytes (L2 -> memory)", "mem_write_bytes", ".3e", "B"),
    ("Atomics share of write requests", "atomic_req_pct", ".1f", "%"),
    ("Uncached write bytes", "uncached_write_bytes", ".3e", "B"),
    ("Read bandwidth", "read_BW_TBps", ".2f", "TB/s"),
    ("Write bandwidth", "write_BW_TBps", ".2f", "TB/s"),
    ("Total bandwidth", "mem_BW_TBps", ".2f", "TB/s"),
    ("Bandwidth % of 5.3 TB/s (upper bound)", "mem_BW_peak_pct", ".1f", "%"),
    ("Arithmetic intensity FP64", "AI_FP64", ".2f", "FLOP/B"),
    ("Roofline limit at this intensity", "attainable_TFLOPs", ".1f", "TFLOP/s"),
    ("% of roofline", "roofline_pct", ".1f", "%"),
]

TOP_COLS = [
    ("time %", "top_share_pct", ".1f"), ("cum %", "cum_share_pct", ".1f"),
    ("launches", "launches", "d"),
    ("CU util %", "CU_busy_pct", ".1f"), ("VALU busy %", "VALUBusy_pct", ".1f"),
    ("VALU lanes %", "VALUUtilization_pct", ".1f"), ("MFMA util %", "MfmaUtil_pct", ".1f"),
    ("FP64 TF/s", "TFLOPs_FP64", ".3f"), ("FP64 useful TF/s", "TFLOPs_FP64_VALU_useful", ".3f"),
    ("FP32 TF/s", "TFLOPs_FP32", ".3f"),
    ("low-prec MFMA TF/s", "TFLOPs_lowp_MFMA", ".2f"), ("total TF/s", "TFLOPs_all", ".2f"),
    ("dominant", "dominant_precision", ""), ("peak TF/s", "peak_TFLOPs_mix", ".1f"),
    ("compute %", "compute_peak_pct", ".1f"),
    ("rd TB/s", "read_BW_TBps", ".3f"), ("wr TB/s", "write_BW_TBps", ".3f"),
    ("BW util %", "mem_BW_peak_pct", ".1f"),
    ("FLOP/B", "AI_all", ".1f"), ("ridge FLOP/B", "ridge_mix", ".0f"),
    ("max TF/s", "attainable_TFLOPs_mix", ".1f"), ("max TB/s", "attainable_BW_TBps", ".2f"),
    ("bound", "bound_mix", ""),
]

KERNEL_COLS = [
    ("time %", "time_share_pct", ".1f"), ("launches", "launches", "d"),
    ("us/launch", "mean_launch_us", ".1f"), ("CU busy", "CU_busy_frac", ".2f"),
    ("VALU busy %", "VALUBusy_pct", ".1f"), ("thr/64", "VALU_threads", ".1f"),
    ("waves/CU", "waves_per_CU", ".1f"), ("max", "max_waves_CU", ".0f"),
    ("% of max", "Occupancy_of_max_pct", ".0f"), ("issue %", "Wave_issue_pct", ".1f"),
    ("FP64 TF/s", "TFLOPs_FP64_VALU", ".2f"), ("useful TF/s", "TFLOPs_FP64_VALU_useful", ".2f"),
    ("% peak", "FP64_VALU_peak_pct", ".1f"),
    ("rd TB/s", "read_BW_TBps", ".2f"), ("wr TB/s", "write_BW_TBps", ".2f"),
    ("atomic %", "atomic_req_pct", ".0f"), ("FLOP/B", "AI_FP64", ".2f"),
    ("% roofline", "roofline_pct", ".0f"),
]


def main():
    global PERF_REGEX, LATENCY_PCT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pmc_dir", help="results/pmc-<jobid> folder")
    ap.add_argument("--counters", help="counters.json (default: <pmc_dir>/counters.json)")
    ap.add_argument("--power-ktrace", help="power run kernel trace CSV or its folder (for GPU busy %%)")
    ap.add_argument("--telemetry", nargs="+",
                    help="power run profiling_result_*.csv (default: next to the --power-ktrace folder)")
    ap.add_argument("--clock-mhz", type=float,
                    help="gfx clock for the utilisation denominators, instead of the sampled one")
    ap.add_argument("--out-dir", help="output folder (default: <pmc_dir>/analysis)")
    ap.add_argument("--top", type=int, default=20, help="kernels listed in report.md (default 20)")
    ap.add_argument("--time-pct", type=float, default=90.0,
                    help="top_kernels.csv: fewest kernels covering this %% of GPU time (default 90)")
    ap.add_argument("--latency-pct", type=float, default=LATENCY_PCT,
                    help="bound = latency when both compute %% and BW util %% are below this (default 10)")
    ap.add_argument("--top-n", type=int,
                    help="top_kernels.csv: the N kernels with the most GPU time instead of --time-pct")
    ap.add_argument("--perf-regex", default=PERF_REGEX,
                    help="regex with one group capturing the app's own time in s "
                         "(default: LAMMPS 'Loop time of X')")
    args = ap.parse_args()
    PERF_REGEX = args.perf_regex
    LATENCY_PCT = args.latency_pct

    pmc_dir = os.path.abspath(args.pmc_dir)
    cfg = json.load(open(args.counters or os.path.join(pmc_dir, "counters.json")))
    out_dir = args.out_dir or os.path.join(pmc_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    passes = []
    for p in cfg["passes"]:
        print(f"reading {p['name']} ...", file=sys.stderr, flush=True)
        passes.append(read_pass(pmc_dir, p))
    M = Metrics(passes)
    need = ["SQ_INSTS_VALU_ADD_F64", "SQ_INSTS_VALU_ADD_F32", "SQ_INSTS_SALU",
            "SQ_WAVE_CYCLES", "TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum", "GRBM_GUI_ACTIVE"]
    lost = [c for c in need if M.pd(c) is None]
    if lost:
        raise SystemExit(f"error: counters.json has no pass with {lost}")
    p4 = M.pd("SQ_WAVE_CYCLES")

    kernels = sorted({k for pd in passes for k in pd.D if k != TOTAL})
    names = unique_short_names(kernels)

    # theoretical max waves per kernel (5.5); TOTAL = time-weighted over pass 4
    pmax = {}
    for k in kernels:
        regs = wg = None
        for pd in [p4] + passes:
            if k in pd.regs and regs is None:
                regs = pd.regs[k].most_common(1)[0][0]
            if k in pd.wg and wg is None:
                wg = pd.wg[k].most_common(1)[0][0]
        if regs is not None:
            pmax[k] = max_waves_cu(regs[0], regs[1], regs[2], wg or 0)
    wsum = sum(p4.D.get(k, 0) * pmax[k] for k in pmax)
    pmax[TOTAL] = div(wsum, sum(p4.D.get(k, 0) for k in pmax))

    # power run (4.5) and sampled clock (4.2), needed before the metrics
    power = clock = None
    if args.power_ktrace:
        power = read_power_ktrace(args.power_ktrace)
        tel = args.telemetry
        if not tel:
            pk_dir = os.path.abspath(args.power_ktrace)
            run_dir = os.path.dirname(pk_dir if os.path.isdir(pk_dir) else os.path.dirname(pk_dir))
            tel = sorted(glob.glob(os.path.join(run_dir, "profiling_result_*.csv")))
        if tel and args.clock_mhz is None:
            print("reading telemetry for the sampled clock ...", file=sys.stderr, flush=True)
            clock = sampled_clock(tel, power["intervals"])
    if args.clock_mhz is not None:
        M.clock_mhz, clock_src = args.clock_mhz, "--clock-mhz"
    elif clock:
        M.clock_mhz = clock["clock_MHz"]
        clock_src = "sampled gfx clock while kernels ran (power run)"
    else:
        clock_src = "GRBM_GUI_ACTIVE / 8 (no telemetry: includes per-launch profiler overhead)"

    rows = {k: compute(k, M, pmax) for k in kernels}
    useful_total = sum(v["FP64_VALU_FLOP_useful"] for v in rows.values()
                       if not math.isnan(v["FP64_VALU_FLOP_useful"]))
    total = compute(TOTAL, M, pmax, useful_total=useful_total)

    tot_ms = sum(v["time_ms"] for v in rows.values() if not math.isnan(v["time_ms"]))
    for v in rows.values():
        v["time_share_pct"] = 100 * div(v["time_ms"], tot_ms)
    total["time_share_pct"] = 100.0

    if power:
        pk = power["per_kernel_ns"]
        psum = sum(pk.values())
        for k, v in rows.items():
            v["unprof_time_ms"] = pk.get(k, 0) / 1e6
            v["unprof_time_share_pct"] = 100 * div(pk.get(k, 0), psum)
            v["profiled_over_unprof"] = div(v["time_ms"], v["unprof_time_ms"])
        total["unprof_time_ms"] = psum / 1e6
        total["unprof_time_share_pct"] = 100.0
        total["GPU_busy_pct"] = power["GPU_busy_pct"]

    # ---- kernels covering --time-pct of GPU time (9.1) -------------------
    # time basis: the unprofiled power run if given (counter collection inflates
    # short kernels), else the PMC passes
    share_key = "unprof_time_share_pct" if power else "time_share_pct"
    by_share = sorted(kernels, key=lambda k: -(rows[k].get(share_key) or 0))
    top, cum = [], 0.0
    total["top_share_pct"] = 100.0
    for k in by_share:
        if (len(top) >= args.top_n) if args.top_n else (cum >= args.time_pct):
            break
        rows[k]["top_share_pct"] = rows[k].get(share_key) or 0
        cum += rows[k]["top_share_pct"]
        rows[k]["cum_share_pct"] = cum
        top.append(k)

    # ---- checks ---------------------------------------------------------
    checks = {"passes": []}
    for pd in passes:
        missing_ctr = [c for c in pd.counters if c not in pd.present]
        checks["passes"].append({
            "pass": pd.name, "label": pd.label,
            "counter_files": pd.files,
            "gpus": len(pd.agents),
            "launches_with_counters": pd.counter_dispatches,
            "launches_in_trace": pd.trace_dispatches,
            "launches_missing_counters": pd.missing_dispatches,
            "missing_time_pct": 100 * div(pd.missing_ns, pd.trace_ns),
            "kernel_time_s": pd.D[TOTAL] / 1e9,
            "eff_clock_MHz": total[f"clock_MHz_{pd.name}"],
            "gui_excess_pct": (100 * (1 - pd.D[TOTAL] * M.clock_mhz / 1000 / M.gui_counter(TOTAL, pd))
                               if M.clock_mhz is not None else NAN),
            "lammps_loop_time_s": pd.loop_time_s,
            "counters_missing": missing_ctr,
        })
    warn = []
    for c in checks["passes"]:
        if c["counters_missing"]:
            warn.append(f"{c['pass']}: counters with no data: {c['counters_missing']}")
        # GRBM_GUI_ACTIVE/8 per ns sits a little above the real clock (per-launch profiler
        # cycles); a missing /8 would put it near 8x
        ref = M.clock_mhz
        if ref is None and not (1000 <= c["eff_clock_MHz"] <= 2200):
            warn.append(f"{c['pass']}: GRBM_GUI_ACTIVE clock {c['eff_clock_MHz']:.0f} MHz outside 1000-2200 (check /8)")
        if ref is not None and not (0.8 * ref <= c["eff_clock_MHz"] <= 2.0 * ref):
            warn.append(f"{c['pass']}: GRBM_GUI_ACTIVE clock {c['eff_clock_MHz']:.0f} MHz vs sampled "
                        f"{ref:.0f} MHz: outside 0.8-2x (check /8)")
        if c["missing_time_pct"] > 0.5:
            warn.append(f"{c['pass']}: {c['missing_time_pct']:.2f}% of kernel time has no counters")
    times = [c["kernel_time_s"] for c in checks["passes"]]
    spread = 100 * (max(times) - min(times)) / min(times)
    checks["kernel_time_spread_pct"] = spread
    if spread > 5:
        warn.append(f"kernel time differs by {spread:.1f}% across passes; cross-pass metrics less reliable")
    for k, v in rows.items():
        if v["time_share_pct"] < 0.1:
            continue
        if v["CU_busy_frac"] > 1.05:
            warn.append(f"{names[k]}: CU busy {v['CU_busy_frac']:.2f} > 1")
        if v["VALU_threads"] > 64.5:
            warn.append(f"{names[k]}: {v['VALU_threads']:.1f} threads > 64")
        if v["waves_per_CU"] > 1.05 * v["max_waves_CU"]:
            warn.append(f"{names[k]}: {v['waves_per_CU']:.1f} waves/CU > limit {v['max_waves_CU']}")
    checks["warnings"] = warn

    # ---- per_kernel_metrics.csv ----------------------------------------
    order = sorted(kernels, key=lambda k: -(rows[k]["time_ms"] if not math.isnan(rows[k]["time_ms"]) else 0))
    keys = ["time_share_pct", "time_ms"] + [c for c in total if c not in ("time_share_pct", "time_ms")]
    csv_path = os.path.join(out_dir, "per_kernel_metrics.csv")
    with open(csv_path, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["kernel", "kernel_full_name"] + keys)
        w.writerow([TOTAL, ""] + [total.get(c, "") for c in keys])
        for k in order:
            w.writerow([names[k], k] + [rows[k].get(c, "") for c in keys])

    # ---- top_kernels.csv ------------------------------------------------
    top_path = os.path.join(out_dir, "top_kernels.csv")
    with open(top_path, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["kernel"] + [c[0] for c in TOP_COLS] + ["kernel_full_name"])
        for k in top + [TOTAL]:
            v = total if k == TOTAL else rows[k]
            w.writerow([names.get(k, k)] + [v.get(c[1], "") for c in TOP_COLS] + ["" if k == TOTAL else k])

    # ---- summary.json ---------------------------------------------------
    summary = {"pmc_dir": pmc_dir, "whole_run": total, "checks": checks}
    if power:
        summary["power_run"] = {k: v for k, v in power.items() if k not in ("per_kernel_ns", "intervals")}
    summary["clock"] = {"source": clock_src, "clock_MHz": M.clock_mhz, "sampled": clock}
    def clean(x):
        if isinstance(x, float) and math.isnan(x):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x
    with open(os.path.join(out_dir, "summary.json"), "w") as fo:
        json.dump(clean(summary), fo, indent=2)

    # ---- report.md ------------------------------------------------------
    L = []
    L.append(f"# PMC Analysis: {os.path.basename(pmc_dir)}\n")
    L.append(f"Source: `{pmc_dir}`. Formulas: `post_processing_formulae.md`. "
             f"Full per-kernel table: `per_kernel_metrics.csv`.\n")
    L.append("## Whole run\n")
    L.append(f"Utilisation denominators (CU, VALU, SALU, MFMA busy, occupancy) use "
             f"kernel time x {f(M.clock_mhz, '.0f')} MHz ({clock_src})"
             + (f"; {clock['samples_kernel_running']} of {clock['samples_total']} samples, "
                f"median {clock['clock_MHz_median']:.0f}, p5-p95 {clock['clock_MHz_p5']:.0f}-"
                f"{clock['clock_MHz_p95']:.0f} MHz" if clock else "") + ".\n")
    L.append("| Metric | Value | Unit |\n|---|---|---|")
    for label, key, spec, unit in WHOLE_RUN_ROWS:
        L.append(f"| {label} | {f(total.get(key), spec)} | {unit} |")
    L.append(f"| Bound (roofline) | {total['bound']} | ridge {PEAK_FP64_VALU / PEAK_MEM_BW:.1f} FLOP/B |")
    if power:
        L.append(f"| GPU busy % of wall time (power run) | {power['GPU_busy_pct']:.1f} | % |")
    L.append("")
    if power:
        L.append("## Power run kernel trace\n")
        L.append(f"- File: `{power['file']}`")
        L.append(f"- Trace files: {power['trace_files']}, GPUs: {power['gpus']}, kernel launches: {power['dispatches']}")
        L.append(f"- First kernel start to last kernel end: {power['span_s']:.3f} s")
        L.append(f"- GPU running at least one kernel: {power['busy_s']:.3f} s per GPU "
                 f"({power['GPU_busy_pct']:.1f}% mean; per GPU {power['GPU_busy_pct_per_gpu']})")
        L.append(f"- Sum of kernel times (overlaps counted twice): {power['sum_kernel_s']:.3f} s\n")
    L.append(f"## Top {args.top} kernels by GPU time\n")
    L.append("| Kernel | " + " | ".join(c[0] for c in KERNEL_COLS) + " |")
    L.append("|---|" + "---|" * len(KERNEL_COLS))
    for k in order[:args.top]:
        v = rows[k]
        L.append(f"| {names[k]} | " + " | ".join(f(v.get(c[1]), c[2]) for c in KERNEL_COLS) + " |")
    rest = order[args.top:]
    if rest:
        L.append(f"\n{len(rest)} more kernels: "
                 f"{sum(rows[k]['time_share_pct'] for k in rest):.1f}% of GPU time together.")
    L.append("\nColumns: `thr/64` threads active per vector instruction; `max` the kernel's own "
             "waves/CU limit from registers and LDS; `issue %` share of a wave's resident time spent "
             "issuing; `FP64 TF/s` issued (all 64 lanes); `useful TF/s` issued x thread utilisation; "
             "`% peak` FP64 issued FLOP/s as % of 81.7 TFLOP/s; `rd/wr TB/s` L2 -> memory "
             "traffic (upper bound on HBM); `atomic %` share of write requests.\n")
    basis = "unprofiled power run" if power else "PMC passes (serialized, profiled)"
    L.append(f"## Top {len(top)} kernels by GPU time\n" if args.top_n else
             f"## Kernels covering {args.time_pct:g}% of GPU time\n")
    L.append(f"{len(top)} of {len(kernels)} kernels cover {cum:.1f}% of GPU time "
             f"(time share from the {basis}). Also in `top_kernels.csv`.\n")
    L.append("| Kernel | " + " | ".join(c[0] for c in TOP_COLS) + " |")
    L.append("|---|" + "---|" * len(TOP_COLS))
    for k in top + [TOTAL]:
        v = total if k == TOTAL else rows[k]
        L.append(f"| {names.get(k, k)} | " + " | ".join(f(v.get(c[1]), c[2]) for c in TOP_COLS) + " |")
    L.append("\nColumns: `CU util` SQ_BUSY_CU_CYCLES share; `VALU busy` share of CU cycles issuing "
             "vector instructions; `VALU lanes` share of the 64 lanes active per vector instruction; "
             "`MFMA util` matrix-core busy share; FLOP rates are achieved TFLOP/s (INT8 as TOP/s) over "
             "the kernel's own time; `FP64 useful` = issued x lanes active; `low-prec MFMA` = "
             "BF16+FP16+FP8+INT8 matrix; `dominant` = precision using the largest share of its peak; "
             "`peak TF/s` = peak for this kernel's precision mix (total FLOP/s / sum of achieved/peak "
             "per precision); `compute %` = sum of achieved/peak per precision; `BW util` = "
             "(read+write) / 5.3 TB/s (L2 -> memory traffic, an upper bound on HBM); `FLOP/B` = all "
             "FLOPs per byte; `ridge` = `peak TF/s` / 5.3; `max TF/s` = roofline ceiling at this "
             "intensity, min(peak, FLOP/B x 5.3); `max TB/s` = min(5.3, peak / FLOP/B); `bound` = "
             f"latency if both `compute %` and `BW util` are below {LATENCY_PCT:g}% (near neither "
             "ceiling: too little parallel work, or waves waiting on dependent memory accesses), "
             "otherwise compute if FLOP/B >= ridge (equivalently `compute %` > `BW util`), else "
             "memory. Kernels with no FLOPs have no ceiling (`-`).\n")
    L.append("## Data checks\n")
    L.append("| Pass | Files / GPUs | Launches | Missing counters (launches, % time) | Kernel time (s) | GRBM_GUI_ACTIVE clock (MHz) | GUI cycles outside kernel time | App time (s) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in checks["passes"]:
        L.append(f"| {c['pass']} | {c['counter_files']} / {c['gpus']} | {c['launches_with_counters']} | {c['launches_missing_counters']}, "
                 f"{f(c['missing_time_pct'], '.3f')}% | {c['kernel_time_s']:.3f} | "
                 f"{c['eff_clock_MHz']:.0f} | {f(c['gui_excess_pct'], '.1f')}% | {f(c['lammps_loop_time_s'], '.2f')} |")
    L.append(f"\nKernel time spread across passes: {spread:.2f}%.\n")
    L.append("**Warnings:**\n")
    L.extend([f"- {x}" for x in warn] or ["- none"])
    with open(os.path.join(out_dir, "report.md"), "w") as fo:
        fo.write("\n".join(L) + "\n")

    # ---- stdout ----------------------------------------------------------
    print(f"\nwrote {csv_path}\n      {os.path.join(out_dir, 'summary.json')}\n"
          f"      {os.path.join(out_dir, 'report.md')}\n")
    for label, key, spec, unit in WHOLE_RUN_ROWS:
        print(f"  {label:48s} {f(total.get(key), spec):>12s} {unit}")
    if power:
        print(f"  {'GPU busy % of wall time (power run)':48s} {power['GPU_busy_pct']:12.1f} %")
    print(f"\n  {len(top)} kernels cover {cum:.1f}% of GPU time -> {top_path}")
    print("\nwarnings:" if warn else "\nwarnings: none")
    for x in warn:
        print("  " + x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
