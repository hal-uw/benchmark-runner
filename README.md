# Benchmark Runner and Profiling Flow (Tailored for MI300x)

A Claude Code skill and set of scripts for characterizing one GPU workload on
AMD Instinct MI300X (gfx942, ROCm 7.2) Slurm nodes. For a given application it
produces:

- a **plain run** for correctness and reference timing,
- **hardware-counter (PMC) profiles** collected with `rocprofv3`, turned into
  per-kernel metrics: FLOP rates, % of peak, CU and VALU utilization, thread
  divergence, occupancy, memory bandwidth, and a roofline bound label,
- a **power, frequency and temperature trace** sampled every ~2 ms while the
  app runs, aligned with a kernel trace, and binned into telemetry vectors that
  the `minos-analysis` dendrogram can use.

The flow was built and validated on LAMMPS ReaxFF/HNS (see
[`references/lammps_case_study.md`](references/lammps_case_study.md)) and has
since been used on other workloads, including OSPREY. Everything that depends
on the workload goes in one `workload.conf`. The sbatch scripts and the
post-processing are the same for every workload.

---

## Contents

1. [Two ways to use it](#two-ways-to-use-it)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [The pipeline, step by step](#the-pipeline-step-by-step)
6. [Configuring a workload (`workload.conf`)](#configuring-a-workload-workloadconf)
7. [Counter set](#counter-set)
8. [Outputs](#outputs)
9. [Metrics and how to read them](#metrics-and-how-to-read-them)
10. [Multi-GPU (8x MI300X)](#multi-gpu-8x-mi300x)
11. [ML / LLM workloads](#ml--llm-workloads)
12. [Pitfalls (verified on MI300X)](#pitfalls-verified-on-mi300x)
13. [Troubleshooting](#troubleshooting)
14. [File layout](#file-layout)

---

## Two ways to use it

**With Claude Code (recommended).** This repo *is* the skill: `SKILL.md` sits
at its root. Once it is linked into `~/.claude/skills/` (see
[Installation](#installation)), Claude Code loads it when you ask it something like *"profile my app on MI300X"*, *"write the
rocprofv3 sbatch for X"* or *"build power vectors for this run"*. Claude asks
for the GPU count, input, paths and which stages you want. It then writes the
`workload.conf`, checks it with dry runs, and gives you a runbook of `sbatch`
commands. **Claude does not submit jobs.** You submit each one and tell Claude
when it finishes. Claude then reads the log, checks the result and moves on.

**By hand.** The scripts don't need Claude. Follow the
[Quick start](#quick-start). `SKILL.md` is the detailed guide (it's written for
Claude, but people can read it too), and `references/` explains every counter
and formula.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Slurm cluster with MI300X nodes | Partitions `mi3001x` (1 GPU per node) and `mi3008x` (8 GPUs per node). The default partition on the cluster this was built on is `mi3501x` (MI350), so **always pass `-p`**. |
| ROCm 7.2 | `module load rocm/7.2.0`, `ROCM_PATH=/opt/rocm-7.2.0`. Tested with `rocprofv3` 1.1.0. |
| A gfx942 build of your app | HIP: `--offload-arch=gfx942`. Kokkos: `Kokkos_ARCH_AMD_GFX942=ON`. See `SKILL.md` Phase 1. |
| `rocprofwrap_lt` (power sampler) | **Bundled** in [`third_party/rocprofwrap_lt/`](third_party/rocprofwrap_lt/UPSTREAM.md): `wrapper.py` plus the `amd-smi-query` sampler, from [`hal-uw/rocprofwrap`](https://github.com/hal-uw/rocprofwrap) with edge/hotspot temperature columns added. Build `amd-smi-query` once (see [Installation](#installation)) and point `PROF_LT` in the conf at that folder. Needed only for the power run. |
| Python ≥ 3.9 | `postprocess_pmc.py` and `check_counters.py` use only the standard library. `build_sampling_json.py` needs `numpy` and `pandas`. |
| *Optional:* [`hal-uw/minos-analysis`](https://github.com/hal-uw/minos-analysis) (private; needs `hal-uw` access) | Not needed to build the power vectors. `build_sampling_json.py` implements the binning of [`dendrogram_plot/dendrogram.py`](https://github.com/hal-uw/minos-analysis/blob/main/dendrogram_plot/dendrogram.py) itself. If you have the repo, pass `--dendrogram-dir <clone>/dendrogram_plot` (or set `$DENDROGRAM_DIR`) to also recompute `inst_power` with `dendrogram.py` as a cross-check. That needs a `dendrogram.py` that reads the `inst_power_W` column (older versions read only `power_from_e`). |

---

## Installation

Clone the repo and link it into your Claude Code user skills. Claude Code looks
for `~/.claude/skills/<name>/SKILL.md`, so the link name becomes the skill's
folder name:

```bash
git clone https://github.com/hal-uw/benchmark-runner.git
mkdir -p ~/.claude/skills
ln -s "$PWD/benchmark-runner" ~/.claude/skills/mi300x-workload-profiling
```

Start a new Claude Code session afterwards so it picks up the skill. A
`git pull` in the clone updates the skill in place. (Cloning directly into
`~/.claude/skills/mi300x-workload-profiling` works too.) If you only use the
scripts by hand, the link isn't needed.

For the power run, build the bundled sampler once. It needs ROCm's amd-smi
library, so build it where ROCm 7.2 is installed:

```bash
module load rocm/7.2.0
make -C benchmark-runner/third_party/rocprofwrap_lt ROCM_DIR=/opt/rocm-7.2.0
# -> benchmark-runner/third_party/rocprofwrap_lt/amd-smi-query
```

Then set `PROF_LT=/path/to/benchmark-runner/third_party/rocprofwrap_lt` in each
`workload.conf`. The binary is not tracked in git (`.gitignore`). Rebuild it
after a ROCm upgrade.

For each workload, copy the scripts into a profiling folder next to it:

```bash
SKILL=/path/to/benchmark-runner
mkdir -p <workload>/profiling
cp $SKILL/scripts/* <workload>/profiling/
cp <workload>/profiling/workload.conf.template <workload>/profiling/<tag>.conf
```

Slurm copies an sbatch script into its spool directory when you submit it, so
the scripts find everything through the absolute paths in the conf
(`PROFILING_DIR`, `RESULTS_DIR`), not through their own location.

---

## Quick start

```bash
cd <workload>/profiling
$EDITOR <tag>.conf                        # fill in; see "Configuring a workload"

# 0. Check the conf on the login node: prints every command, runs nothing
DRY_RUN=1 bash run_plain.sbatch <tag>.conf
DRY_RUN=1 bash run_pmc.sbatch   <tag>.conf
DRY_RUN=1 bash run_power.sbatch <tag>.conf

# Write RUNBOOK_<tag>.md: every command below, filled in for this workload
bash make_runbook.sh <tag>.conf

# 2. List the counters the GPU actually exposes, then validate counters.json
sbatch -p mi3001x list_counters.sbatch $PWD/counters_list_gfx942.txt
python3 check_counters.py counters.json counters_list_gfx942.txt      # must print OK

# 3. Plain run
sbatch -p mi3001x -J <tag>-plain run_plain.sbatch $PWD/<tag>.conf

# 4. PMC profiling (6 passes; allow ~passes x 3 x plain wall time)
sbatch -p mi3001x -J <tag>-pmc -t 01:00:00 run_pmc.sbatch $PWD/<tag>.conf

# 5. Power / frequency / temperature + kernel trace
sbatch -p mi3001x -J <tag>-power run_power.sbatch $PWD/<tag>.conf

# 6. Post-process on the login node (no GPU needed)
python3 postprocess_pmc.py <RESULTS_DIR>/pmc-<jobid> \
        --power-ktrace <RESULTS_DIR>/prof-<jobid>/ktrace_<tag>
python3 build_sampling_json.py <RESULTS_DIR>/prof-<jobid> --label "<Display name>"
```

For 8 GPUs, use `-p mi3008x` and add `-n <NTASKS>` when the app runs more than
one task (see [Multi-GPU](#multi-gpu-8x-mi300x)). A worked, filled-in example
is in [`examples/RUNBOOK_hns16812.md`](examples/RUNBOOK_hns16812.md).

---

## The pipeline, step by step

| # | Step | Command | Check before moving on |
|---|---|---|---|
| 0 | Dry run (login node) | `DRY_RUN=1 bash run_{plain,pmc,power}.sbatch <tag>.conf` | Right binary and arguments, `srun -n <NTASKS>`, no `Error:` lines |
| 1 | Build (if needed) | your build job, for gfx942 | Every path in `REQUIRED_FILES` exists |
| 2 | Counter list | `list_counters.sbatch` → `check_counters.py` | The list shows only `gfx942`; the checker prints `OK` |
| 3 | Plain run | `run_plain.sbatch` | Exit 0, correct output, the app's timing line; note the wall time |
| 4 | PMC run | `run_pmc.sbatch` | Every pass reports `rc=0` with counter CSVs; app timing consistent across passes |
| 5 | Power run | `run_power.sbatch` | Exit 0; kernels traced; first sample and first kernel both come just after the same `clock_ref` |
| 6 | Post-process | `postprocess_pmc.py`, `build_sampling_json.py` | The warnings list in `report.md` says `none` |

Step 4 depends on step 2. Steps 3 and 5 don't depend on anything. Running
step 3 first confirms the app works before you queue the long PMC job. The
generated runbook also gives an optional `--dependency=afterok` chain that
submits steps 3–5 together.

**What each job does:**

- **`run_plain.sbatch`** creates `plain-<jobid>/`, links the inputs in with
  `stage_inputs`, runs the app under `srun` and prints the line matched by
  `PERF_GREP`.
- **`list_counters.sbatch`** runs `rocprofv3 --list-avail` **on a GPU node**.
  Don't do this with `salloc` or `srun --pty` inside a script: the rest of the
  script then runs on the login node, which reports a different GPU (gfx90a).
- **`run_pmc.sbatch`** runs the whole app once per counter pass in
  `counters.json`, each pass under
  `rocprofv3 --pmc <counters> --kernel-trace`. rocprofv3 serializes kernels
  while it collects counters, so each pass takes about 2.5–4x the plain run. If
  one pass fails, the job reports it and runs the remaining passes.
- **`run_power.sbatch`** runs
  `wrapper.py -- srun rocprofv3 --kernel-trace -- <app>`. The sampler records
  power, clock and temperature every ~2.3 ms (1 ms is requested). The kernel
  trace, with no counters, costs about 4% and gives kernel timestamps on the
  same clock as the samples. `clock_ref.txt` records that clock so the two can
  be aligned.

---

## Configuring a workload (`workload.conf`)

The conf is a bash file that the sbatch scripts `source`. Copy
`scripts/workload.conf.template`. The fields:

| Field | Meaning |
|---|---|
| `WORKLOAD` | Short tag used in file and folder names (no spaces) |
| `WORKLOAD_DESC` | One-line description: input size, steps, precision |
| `PROFILING_DIR` | Absolute path of your copy of `scripts/` |
| `RESULTS_DIR` | Where `plain-<jobid>`, `pmc-<jobid>` and `prof-<jobid>` are created |
| `NGPUS` | `1` → `mi3001x`, `8` → `mi3008x` |
| `NTASKS` | `srun` tasks: `NGPUS` for MPI with one rank per GPU; `1` for a single process driving all GPUs (torchrun, accelerate, …) |
| `BIND_GPU_PER_TASK` | `1` gives each task its own GPU (`ROCR_VISIBLE_DEVICES=$SLURM_LOCALID`). The partitions have no GPU GRES, so Slurm doesn't bind GPUs itself. |
| `SAMPLE_GPUS` | GPUs the power sampler reads, e.g. `0` or `0,1,2,3,4,5,6,7` |
| `MODULES`, `ROCM_PATH` | Environment (`rocm/7.2.0`) |
| `PROF_LT`, `SAMPLE_MS` | Location of `rocprofwrap_lt`; requested sampling interval in ms |
| `REQUIRED_FILES` | Files that must exist before any run (binary, inputs); checked first |
| `PERF_GREP` | `grep -E` pattern for the app's own timing line (LAMMPS `Loop time of`, PyTorch `it/s`, …) |
| `stage_inputs()` | Runs inside the per-job run folder: symlink inputs here so nothing is written into the source tree |
| `app_cmd()` | Sets the `APP_CMD` array. Don't include `srun`; the scripts add it. Use `$OUT_DIR` for any log path. |
| `app_env()` | Optional environment for the app (`OMP_NUM_THREADS`, …) |

See [`examples/lammps_hns16812.conf`](examples/lammps_hns16812.conf) for a real
one.

**Choosing the input.** Use a problem large enough to fill all 304 compute
units (CUs). A small input mostly measures kernel-launch latency rather than
the GPU. Use the same input sizes as earlier results so the numbers stay
comparable, and use the **same input for the plain, PMC and power runs**.

---

## Counter set

`scripts/counters.json` defines six passes for gfx942. A pass is one full run
of the app. Only a limited number of counters can be read in each run. The
per-run limits that worked were SQ 8, TCC 4, TCP 4 and GRBM 2.

| Pass | Purpose | Key counters |
|---|---|---|
| `01_fp64_flops` | FP64 FLOPs (VALU + MFMA) | `SQ_INSTS_VALU_{ADD,MUL,FMA,TRANS}_F64`, `SQ_INSTS_VALU_MFMA_MOPS_F64`, `SQ_VALU_MFMA_BUSY_CYCLES` |
| `02_fp32_int_ops` | FP32 FLOPs, integer ops | `SQ_INSTS_VALU_*_F32`, `SQ_INSTS_VALU_MFMA_MOPS_F32`, `SQ_INSTS_VALU_INT{32,64}` |
| `03_fp16_instmix` | FP16 FLOPs, instruction mix | `SQ_INSTS_VALU_*_F16`, `SQ_INSTS_{SALU,VMEM_RD,VMEM_WR,LDS}` |
| `04_cu_util` | CU busy, VALU busy, divergence, occupancy | `SQ_BUSY_CU_CYCLES`, `SQ_WAVE_CYCLES`, `SQ_ACTIVE_INST_VALU`, `SQ_THREAD_CYCLES_VALU` |
| `05_hbm_read` | L2 → memory reads | `TCC_EA0_RDREQ{,_32B}_sum`, `TCC_EA0_RD_UNCACHED_32B_sum` |
| `06_hbm_write` | L2 → memory writes, atomics | `TCC_EA0_WRREQ{,_64B}_sum`, `TCC_EA0_WR_UNCACHED_32B_sum`, `TCC_EA0_ATOMIC_sum` |

`GRBM_GUI_ACTIVE` is in every pass so each one has its own count of GPU-busy
cycles. The file also holds the derived-metric formulas and the MI300X peak
numbers (FP64 VALU 81.7 TFLOP/s, FP32 VALU/MFMA and FP64 MFMA 163.4 TFLOP/s,
HBM 5.3 TB/s at 2.1 GHz).

Change the set only for a specific reason. The reasoning behind each counter,
and the counters deliberately left out (MFMA instruction counts, L1→L2 traffic,
L2 hit rate, stalls, LDS), are in
[`references/counters_gfx942.md`](references/counters_gfx942.md). Always
re-run `check_counters.py` after editing.

---

## Outputs

```
RESULTS_DIR/
  plain-<jobid>/                   app output of the plain run
  pmc-<jobid>/
    counters.json, workload.conf   copies of what was used
    <pass>/<pass>_counter_collection.csv, <pass>_kernel_trace.csv, LABEL, rocprof.<pass>.txt
    analysis/                      written by postprocess_pmc.py
      per_kernel_metrics.csv       every kernel + a TOTAL row, all metrics
      top_kernels.csv              fewest kernels covering 90% of GPU time, key metrics + bound
      summary.json                 whole-run metrics and data checks
      report.md                    readable summary: whole run, top kernels, warnings
  prof-<jobid>/
    profiling_result_<tag>_<gpu>.csv   timestamp_ns, current_socket_power_W, inst_power_W,
                                       gfx_clock_MHz, edge_temp_C, hotspot_temp_C
    ktrace_<tag>/..._kernel_trace.csv  kernel trace of the power run
    clock_ref.txt                      clock value used to align samples and kernels
    sampling_vectors.json              written by build_sampling_json.py
```

A counter CSV can be large, about 160 MB for a 5-second run with 68k kernel
launches. Keep `RESULTS_DIR` **outside** the git repo.

### `postprocess_pmc.py` options

| Option | Effect |
|---|---|
| `--power-ktrace <dir or csv>` | Uses the unprofiled power run for GPU busy %, for ranking kernels by real time, and for the clock used in utilization math. **Always pass it.** |
| `--telemetry …` / `--clock-mhz …` | Override where the clock comes from |
| `--time-pct 90` | Coverage target for `top_kernels.csv` |
| `--top-n N` | List the N biggest kernels instead of the 90% set |
| `--latency-pct 10` | Threshold for the "latency" bound label |
| `--perf-regex` | The app's timing line (defaults to the LAMMPS one; set it for other apps) |
| `--counters`, `--out-dir`, `--top` | Alternate counters.json, output folder, rows in `report.md` |

### `build_sampling_json.py` options

| Option | Effect |
|---|---|
| `--label` | Display name stored in the JSON |
| `--per-gpu` | Also write one vector per GPU (multi-GPU runs; the default pools all GPUs) |
| `--no-trim` | Keep samples from outside the kernel window (normally dropped) |
| `--dendrogram-dir` | Optional. Folder containing `minos-analysis` `dendrogram.py`, used to cross-check `inst_power` (default `$DENDROGRAM_DIR`; unset means no cross-check) |
| `--out` | Output path |

The JSON has a `description` header (sampling logic, window, clock alignment,
bin edges) and one key per vector: `inst_power`, `socket_power`,
`gfx_frequency` and `hotspot_temp`. All of them are binned with the algorithm
of `calculate_power_distribution()` in `minos-analysis`'s `dendrogram.py`: raw
samples with no smoothing and, for power, a fraction of TDP (the rated power
limit, 750 W) in 0.1-wide bins from 0.5 to 2.0 × TDP. That keeps `inst_power`
directly comparable with `app_vectors.json`. The `cross_check` field says
whether `dendrogram.py` was also run and matched. Without `--dendrogram-dir`
it reads `none`; the vectors are the same either way.

---

## Metrics and how to read them

Every formula is derived and explained in
[`references/formulae.md`](references/formulae.md), with LAMMPS values to use
as a sanity check. The main ones:

| Metric | What it means |
|---|---|
| **CU busy** | Fraction of kernel time the compute units had any work (≤ 1) |
| **VALU busy %** | Fraction of time the vector ALUs were issuing instructions |
| **VALU lanes active** | Average threads active per vector instruction, out of 64. Low values mean branch divergence. |
| **Occupancy** | Average waves (groups of 64 threads) resident per CU, compared with the hardware maximum of 32 and with each kernel's own register-limited maximum |
| **MFMA utilization** | Fraction of time the matrix cores were busy |
| **FP64 issued / useful TFLOP/s** | Issued counts all 64 lanes of every instruction. Useful multiplies by the lanes actually active (an estimate). |
| **L2 → memory bandwidth** | Read and write traffic leaving the L2. Measured before the 256 MB Infinity Cache, so it is an **upper bound** on HBM traffic. |
| **Arithmetic intensity** | FLOPs per byte of L2 → memory traffic. Compared with the roofline ridge point (peak FLOP/s ÷ peak bandwidth). |
| **Bound label** | `compute` or `memory` depending on which side of the roofline the kernel is on. `latency` means under 10% of both peak compute and peak bandwidth: the kernel is limited by neither, usually by memory latency or launch overhead. |
| **GPU busy %** | From the power run: fraction of the kernel window during which some kernel was running |

---

## Multi-GPU (8x MI300X)

| | 1x MI300X | 8x MI300X |
|---|---|---|
| Partition | `mi3001x` | `mi3008x` |
| Submit | `sbatch -p mi3001x …` | `sbatch -p mi3008x -n <NTASKS> …` |
| `NGPUS` / `SAMPLE_GPUS` | `1` / `0` | `8` / `0,1,2,3,4,5,6,7` |

- **MPI with one rank per GPU:** `NTASKS=8`. Set `BIND_GPU_PER_TASK=1` unless
  the app picks its own device per rank (LAMMPS `-k on g 8` does).
- **One process driving all GPUs** (e.g. torchrun inside `app_cmd`):
  `NTASKS=1`.
- Each process writes its own counter files. `postprocess_pmc.py` reads all of
  them and reports **per-GPU averages**, which is what a comparison against
  one GPU's peak needs.
- By default the power vectors pool the samples from all GPUs. Add `--per-gpu`
  to also get one vector per GPU.

---

## ML / LLM workloads

The bundled six-pass set covers FP64, FP32 and FP16 only. For kernels that do
BF16, FP16, FP8 or INT8 matrix math, **add a `07_lowp_mfma` pass**
(`SQ_INSTS_VALU_MFMA_MOPS_{BF16,F16,F8,I8}`, `SQ_VALU_MFMA_BUSY_CYCLES`, …) so
those FLOPs are counted and the kernels aren't mislabelled memory-bound. The
bound label compares each precision with its own peak (BF16/FP16 MFMA 1307.4,
FP8/INT8 2614.9 TFLOP/s). The OSPREY profiling folder has a `counters.json`
with this pass. Without it, the script falls back to FP64/FP32/FP16.

---

## Pitfalls (verified on MI300X)

- `rocminfo | grep "Marketing Name"` shows the **CPU** first. To identify the
  GPU, use `rocminfo | grep -m1 "  Name:  *gfx"`.
- gfx942 counter names are `TCC_EA0_*`, not `TCC_EA_*` as on older GPUs.
  Validate names with `check_counters.py`.
- `GRBM_GUI_ACTIVE` is summed over the 8 XCCs (the chiplets that make up one
  MI300X), so divide it by 8.
- **Utilization denominators use kernel time × the gfx clock sampled in the
  power run**, not `GRBM_GUI_ACTIVE`. In counter mode `GRBM_GUI_ACTIVE` adds a
  fixed ~14k–18k cycles to every launch, which understates utilization by
  about 10% overall and by several times for microsecond-long kernels. Take the
  clock from samples taken while kernels were running. Don't use the
  whole-file mean, which includes startup and the time rocprofv3 spends
  writing its output.
- `SQ_WAVE_CYCLES` is counted in quad-cycles and needs ×4. `SQ_BUSY_CU_CYCLES`
  must **not** get ×4 (that would give 299% busy). The counter descriptions are
  unreliable about units, so the script checks results against physical limits
  (CU busy ≤ 1, threads ≤ 64, occupancy ≤ the register limit).
- `GRBM_GUI_ACTIVE / GRBM_COUNT` is always 100% in per-kernel counter mode.
  Take GPU busy % from the power run's kernel trace instead.
- Kernel launch IDs differ between runs (iteration counts vary slightly), so
  passes are combined **per kernel name**, never per launch.
- Kernel durations in a profiled run are serialized per-kernel times. For
  wall-clock behaviour, use the power run.
- The sampler achieves ~2.3 ms, not the requested 1 ms. The edge-temperature
  sensor isn't exposed (always `nan`). Samples before the first kernel and
  after the last are trimmed.
- Workgroups can be 2-D or 3-D: size = X·Y·Z.
- Keep `#!/bin/bash` shebangs. A zsh shebang in a batch job doesn't load the
  module system.
- Compute derived numbers with the script, not by hand. A hand calculation in
  the LAMMPS study was off by 30%.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Counter list shows `gfx90a` | It was generated on the login node. Use `list_counters.sbatch`. |
| `check_counters.py` reports unknown counters | Wrong generation's names (`TCC_EA_*`), or a typo. Compare with the list. |
| A PMC pass fails with `rc≠0` on a block limit | Too many counters from one hardware block in that pass. Split the pass. |
| Job lands on MI350 nodes | `-p` was left out. The default partition is `mi3501x`. |
| All 8 ranks use GPU 0 | Set `BIND_GPU_PER_TASK=1`. |
| CU busy > 1 or waves/CU above the limit in `report.md` warnings | The clock from the power run differs from the PMC runs (power-throttled workload). Read the warning and treat those kernels' utilization as approximate. |
| `run_power.sbatch` reports `amd-smi-query` missing | Build it: `make -C third_party/rocprofwrap_lt ROCM_DIR=/opt/rocm-7.2.0`, and check `PROF_LT` points at that folder. |
| No `hotspot_temp` vector (`skipped: column hotspot_temp_C not in CSV`) | The run used an upstream `rocprofwrap_lt` without temperature columns. Point `PROF_LT` at the bundled copy. |
| `no dendrogram.py in …` | `--dendrogram-dir` or `$DENDROGRAM_DIR` points at the wrong folder. Fix it, or unset it to skip the cross-check. |
| `KeyError: 'power_from_e'` from `dendrogram.py` | That `dendrogram.py` predates `inst_power_W` support. Drop `--dendrogram-dir` (the vectors don't need it) or use a version that reads `inst_power_W`. |
| Power samples and kernels don't line up | Check `clock_ref.txt` and the job summary: both should start a few seconds after the same reference value. |

---

## File layout

All paths are relative to the repo root.

```
SKILL.md                         instructions Claude Code loads (also a full human guide)
README.md                        this file
scripts/
  workload.conf.template         per-workload settings
  run_plain.sbatch               plain run
  list_counters.sbatch           rocprofv3 --list-avail on a GPU node
  check_counters.py              validate counters.json against that list
  counters.json                  6-pass counter set, derived formulas, peaks
  run_pmc.sbatch                 one rocprofv3 --pmc run per pass, + kernel trace
  run_power.sbatch               rocprofwrap_lt sampling + rocprofv3 kernel trace
  postprocess_pmc.py             per-kernel / whole-run metrics -> CSV, JSON, report.md
  build_sampling_json.py         binned power/frequency/temperature vectors
  make_runbook.sh                RUNBOOK_<tag>.md with every command filled in
third_party/
  rocprofwrap_lt/                vendored power sampler: wrapper.py, power_query.cpp, Makefile,
                                 UPSTREAM.md (source + local changes), local-changes.patch
examples/
  lammps_hns16812.conf           worked config (LAMMPS HNS 16x8x12, 466,944 atoms)
  RUNBOOK_hns16812.md            runbook generated from it
references/
  counters_gfx942.md             why each counter, pass design, unit checks
  formulae.md                    every derived metric, explained, with LAMMPS values
  lammps_case_study.md           the original study: jobs, results, lessons
```

The paths inside `examples/` and `references/` are from the original LAMMPS
study on the author's account. They are left as they were for reference.
Replace them with your own when you copy a config.
