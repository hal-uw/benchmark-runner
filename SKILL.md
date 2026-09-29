---
name: mi300x-workload-profiling
description: End-to-end GPU workload characterization on AMD MI300X (gfx942, ROCm 7.2) Slurm nodes -- build a runnable workload, write sbatch scripts for a plain run, rocprofv3 hardware-counter (PMC) profiling from a counters.json, and power/frequency/temperature sampling with rocprofwrap_lt plus a rocprofv3 kernel trace, then post-process into per-kernel metric CSVs (FLOPs, % of peak, CU/VALU utilization, divergence, occupancy, memory bandwidth, roofline) and dendrogram-compatible telemetry vector JSON. Use this whenever the user wants to profile, characterize, benchmark or collect counters/power/telemetry for any HPC or ML workload on MI300X (or mi3001x / mi3008x partitions), port an existing profiling flow (e.g. the LAMMPS one) to another application, choose rocprofv3 counters, compute FLOPs/bandwidth/utilization from counter CSVs, or build power/frequency vectors for the minos-analysis dendrogram -- even if they only mention one piece, such as "sbatch for rocprofv3", "power trace for my app", or "which PMCs should I collect".
---

# MI300X workload profiling

A tested, reusable flow for characterizing one workload on 1x or 8x AMD
Instinct MI300X. It was built on LAMMPS ReaxFF/HNS (see
`references/lammps_case_study.md`) and generalized: everything
workload-specific lives in one `workload.conf`; the sbatch scripts and
post-processing are generic.

```
scripts/
  workload.conf.template   per-workload settings: GPUs, paths, inputs, command
  run_plain.sbatch         plain run (correctness + reference timing)
  list_counters.sbatch     rocprofv3 --list-avail ON the GPU node
  check_counters.py        validate counters.json against that list
  counters.json            6-pass counter set + derived formulas (gfx942)
  run_pmc.sbatch           one rocprofv3 --pmc run per pass, + kernel trace
  run_power.sbatch         rocprofwrap_lt power/clock/temp sampling + kernel trace
  postprocess_pmc.py       per-kernel + whole-run metrics -> CSV / JSON / report.md,
                           + top_kernels.csv (kernels covering 90% of GPU time)
  build_sampling_json.py   power/frequency/temperature vectors (dendrogram.py binning)
  make_runbook.sh          RUNBOOK_<tag>.md: the user's commands, in order, filled in
examples/lammps_hns16812.conf   worked config
examples/RUNBOOK_hns16812.md    runbook generated from it
references/counters_gfx942.md   why each counter, pass design, unit checks
references/formulae.md          every derived formula, explained, with LAMMPS values
references/lammps_case_study.md what was run, results, and lessons
```

## How to work with this user

- **Do not submit jobs.** Write the sbatch scripts and give the exact
  `sbatch ...` commands; the user runs them and tells you when they finish.
  Running local analysis (post-processing CSVs, dry runs, syntax checks) is
  fine.
- **Keep profiling minimal.** Fewer passes and fewer runs are preferred; only
  collect counters that feed a metric the user wants. Do not add extra
  validation runs (e.g. a small-input rehearsal before the real run) unless
  asked -- the user considers them redundant.
- Explain choices in plain terms and show evidence from the data (the unit
  factors below were settled that way).
- Capture explanations the user asks for in markdown files next to the scripts.

## The command flow (what the user runs, in order)

Always end setup by giving the user this sequence, filled in for their
workload. Generate it rather than typing it:
`bash make_runbook.sh <tag>.conf` -> `$PROFILING_DIR/RUNBOOK_<tag>.md`
(partition and `-n` come from `NGPUS`/`NTASKS`). Paste the commands into the
reply as well, and say what to check after each step.

| # | Who / where | Command | Check before moving on |
|---|---|---|---|
| 0 | Claude, login node | `DRY_RUN=1 bash run_{plain,pmc,power}.sbatch <tag>.conf` | right binary/args, `srun -n <NTASKS>`, no `Error:` |
| 1 | user (if not built) | build job for gfx942 | every `REQUIRED_FILES` path exists |
| 2 | user | `sbatch -p <part> list_counters.sbatch <list.txt>` then `python3 check_counters.py counters.json <list.txt>` | list is only `gfx942`; checker says `OK` |
| 3 | user | `sbatch -p <part> [-n N] -J <tag>-plain run_plain.sbatch <tag>.conf` | exit 0, correct output, app timing; note wall time |
| 4 | user | `sbatch -p <part> [-n N] -J <tag>-pmc -t <~passes x 3 x plain> run_pmc.sbatch <tag>.conf` | every pass `rc=0` with counter CSVs; app timing consistent across passes |
| 5 | user | `sbatch -p <part> [-n N] -J <tag>-power run_power.sbatch <tag>.conf` | exit 0; kernels traced; first sample and first kernel just after the same `clock_ref` |
| 6 | user or Claude, login node | `python3 postprocess_pmc.py <pmc-dir> --power-ktrace <prof-dir>/ktrace_<tag>` and `python3 build_sampling_json.py <prof-dir> --label "<name>"` | `report.md` warnings list says `none`; present `top_kernels.csv` (kernels covering 90% of time) |

`<part>` is `mi3001x` for 1 GPU and `mi3008x` for 8. Step 4 depends on step 2;
steps 3 and 5 do not. Running step 3 first shows the app works before the long
PMC job; the runbook also gives an optional `--dependency=afterok` chain that
submits 3-5 at once. When the user reports a job finished, read its `.out` and
the run folder and confirm the check before moving to the next step.

## Phase 0 -- ask before building anything

Use AskUserQuestion (batch them) for what cannot be inferred:

1. **GPU count: single MI300X or 8x MI300X?** This decides the partition and
   launch layout:
   | | 1x MI300X | 8x MI300X |
   |---|---|---|
   | partition | `mi3001x` | `mi3008x` (2 nodes) |
   | submit | `sbatch -p mi3001x ...` | `sbatch -p mi3008x -n <NTASKS> ...` |
   | `NGPUS` / `SAMPLE_GPUS` | `1` / `0` | `8` / `0,1,2,3,4,5,6,7` |
   The cluster default partition is `mi3501x` (MI350), so always pass `-p`.
2. **If 8 GPUs: how does the app use them?** MPI with one rank per GPU
   (`NTASKS=8`; set `BIND_GPU_PER_TASK=1` unless the app picks its device per
   rank itself, as LAMMPS `-k on g 8` does) or one process driving all GPUs /
   launching its own workers (`NTASKS=1`, e.g. torchrun inside the command).
   The partitions have no GPU GRES, so Slurm does not bind GPUs.
3. **Which input / problem size** is the representative one, and is there a
   reference output to check correctness against?
4. **Where are the source, binary and inputs**, and is it already built?
5. **Which stages**: plain run, PMC profiling, power sampling (default: all).

## Phase 1 -- a runnable workload

Goal: a binary built for gfx942 and a command that runs on one node.

- Build with ROCm 7.2 for `gfx942` (HIP: `--offload-arch=gfx942`; Kokkos:
  `Kokkos_ARCH_AMD_GFX942=ON`, `Kokkos_ENABLE_HIP=ON`, `CMAKE_CXX_COMPILER=hipcc`).
  For codes with floating-point atomics add `-munsafe-fp-atomics` (hardware FP
  atomics on MI300X; LAMMPS used it). LAMMPS example (reconstructed from
  `build-mi300x/CMakeCache.txt`; adapt paths and packages):
  ```bash
  module load rocm/7.2.0
  cmake -S cmake -B build-mi300x -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=hipcc \
    -DCMAKE_CXX_FLAGS=-munsafe-fp-atomics -DPKG_KOKKOS=on -DPKG_REAXFF=on \
    -DKokkos_ENABLE_HIP=on -DKokkos_ARCH_AMD_GFX942=on -DKokkos_ENABLE_OPENMP=off \
    -DBUILD_MPI=on -DFFT_KOKKOS=KISS
  cmake --build build-mi300x -j 16
  ```
  Build on a compute node or a node with the same ROCm; offer the user the
  build as an sbatch too.
- Pick a problem **large enough to fill 304 CUs** (small inputs measure launch
  latency, not the GPU). Match sizes/inputs used in prior results so numbers
  stay comparable.
- Stage inputs by symlinking into a per-job run folder (`stage_inputs` in the
  conf) so nothing is written into the source tree.

## Phase 2 -- set up the profiling folder

```bash
mkdir -p <workload>/profiling && cp <skill>/scripts/* <workload>/profiling/
cp <workload>/profiling/workload.conf.template <workload>/profiling/<tag>.conf   # fill it in
```
The sbatch scripts are copied into Slurm's spool when submitted, so they find
everything through absolute paths in the conf (`PROFILING_DIR`, `RESULTS_DIR`).
Validate the conf on the login node, then generate the runbook
(`bash make_runbook.sh <tag>.conf`) and hand the user the command flow above:
```bash
DRY_RUN=1 bash run_plain.sbatch <tag>.conf    # prints every command, runs nothing
DRY_RUN=1 bash run_pmc.sbatch   <tag>.conf
DRY_RUN=1 bash run_power.sbatch <tag>.conf
```
Scripts start with `#!/bin/bash` on purpose: a zsh shebang in a batch job does
not load the module system.

## Phase 3 -- plain run

`sbatch -p <partition> [-n NTASKS] run_plain.sbatch <tag>.conf` ->
`results/plain-<jobid>/`. Check correctness against the reference output and
record the app's own timing (`PERF_GREP`); profiled runs are compared to it.

## Phase 4 -- choose and validate counters

1. Generate the counter list **on the GPU node** (never with `salloc`/`srun
   --pty` inside a script -- those open interactive shells and the rest runs on
   the login node, which reported gfx90a):
   `sbatch list_counters.sbatch <workload>/mi300x_rocm72_counters.txt`
2. `python3 check_counters.py counters.json <list.txt>` -- must print `OK`.
3. The bundled `counters.json` (6 passes) is the default: FP64/FP32/FP16 VALU
   and MFMA MOPS (FLOPs), instruction mix, CU utilization (busy, divergence,
   occupancy), and L2 -> memory reads/writes incl. uncached and atomics. Change
   it only for a reason the user agrees with; read `references/counters_gfx942.md`
   for why each counter is there and what was dropped (MFMA instruction
   counts, L1->L2, L2 hit rate, stalls, LDS, separate occupancy pass).
4. Per-pass limits that worked on gfx942: SQ 8, TCC 4, TCP 4, GRBM 2.
   Keep `GRBM_GUI_ACTIVE` in every pass (each pass needs its own busy-cycle
   denominator).

## Phase 5 -- PMC profiling

`sbatch -p <partition> [-n NTASKS] -t <time> run_pmc.sbatch <tag>.conf` ->
`results/pmc-<jobid>/<pass>/` with `<pass>_counter_collection.csv`,
`<pass>_kernel_trace.csv`, `LABEL`, `rocprof.<pass>.txt`, plus `counters.json`
and `workload.conf` copies. Each pass is a full app run and takes ~2.5-4x the
plain run (kernels are serialized). With several tasks each process writes its
own files (`<pass>_<pid>_...`). A failed pass is reported and the rest
continue; if a pass fails on a block limit, split it.

Check after the job: every pass `rc=0 csv=1` (or one per process), the app's
results identical across passes (fixed seeds), kernel time within ~1% across
passes.

## Phase 6 -- power, frequency, temperature

`sbatch -p <partition> [-n NTASKS] run_power.sbatch <tag>.conf` ->
`results/prof-<jobid>/`: `profiling_result_<tag>_<gpu>.csv` per sampled GPU
(`timestamp_ns, current_socket_power_W, inst_power_W, gfx_clock_MHz,
edge_temp_C, hotspot_temp_C`), `ktrace_<tag>/..._kernel_trace.csv`,
`clock_ref.txt`. The command is `wrapper.py -- srun rocprofv3 --kernel-trace --
<app>` with no counters, so overhead stays small (~4% for LAMMPS). Check the
summary: first sample and first kernel should both be a few seconds after the
same `clock_ref` value (both use the boot-time clock on MI300X/ROCm 7.2).

## Phase 7 -- post-process (login node, no GPU)

```bash
python3 postprocess_pmc.py results/pmc-<jobid> --power-ktrace results/prof-<jobid>/ktrace_<tag>
#   -> results/pmc-<jobid>/analysis/{per_kernel_metrics.csv, top_kernels.csv, summary.json, report.md}
python3 build_sampling_json.py results/prof-<jobid> --label "<Display name>"
#   -> results/prof-<jobid>/sampling_vectors.json
```
- `postprocess_pmc.py` sums counters per kernel **name** within a pass and
  combines passes only through per-kernel rates (launch IDs differ between
  runs). Non-LAMMPS apps: pass `--perf-regex` for the app's timing line. Read
  its warnings; they encode the physical sanity checks below.
- `build_sampling_json.py` produces a simple JSON: a `description` header
  (sampling logic, window, clock alignment, vector creation, bin edges) and one
  key per vector (`inst_power`, `socket_power`, `gfx_frequency`,
  `hotspot_temp`). `inst_power` comes from `calculate_power_distribution()` in
  `minos-analysis/dendrogram_plot/dendrogram.py` itself, so it is comparable
  with `app_vectors.json`. Samples are trimmed to the kernel window.
  Multi-GPU: vectors pool all GPUs; `--per-gpu` adds per-GPU vectors.
- **Always do the top-kernel analysis.** `top_kernels.csv` (and the "Kernels
  covering 90% of GPU time" section of `report.md`) lists the fewest kernels
  whose time shares add up to 90% (`--time-pct` to change it), ranked by the
  unprofiled power run's kernel time when `--power-ktrace` is given. Per
  kernel: CU utilization, VALU busy and VALU lanes active, MFMA utilization,
  FP64 (issued and useful) and FP32 TFLOP/s, low-precision MFMA TFLOP/s, read /
  write bandwidth, bandwidth utilization (% of 5.3 TB/s), max FLOPs and max
  bandwidth (the roofline ceilings at the kernel's arithmetic intensity), and a
  compute / memory / latency bound label (latency = under 10% of both peak
  compute and peak bandwidth, `--latency-pct`). `--top-n N` lists the N
  biggest kernels instead of the 90% set. Present this table to the user with each
  column defined (`references/formulae.md` 8.9, 9.1).
- **The bound label counts every precision** (8.9): each precision's FLOP rate
  is measured against its own peak (FP64 81.7, BF16/FP16 MFMA 1307.4, FP8/INT8
  2614.9 TFLOP/s, ...), so ML kernels doing BF16/FP8 matrix math are not
  mislabelled memory-bound. This needs the `07_lowp_mfma` pass
  (`SQ_INSTS_VALU_MFMA_MOPS_{BF16,F16,F8,I8}`, `SQ_VALU_MFMA_BUSY_CYCLES`, ...);
  the bundled 6-pass `counters.json` lacks it, so add it for any ML/LLM
  workload (see `workloads/OSPREY/mi300x/profiling/counters.json`). Without it
  the script falls back to FP64/FP32/FP16 only. When a kernel is far below
  both its compute % and bandwidth %, say so: it is limited by neither
  (latency or launch overhead), whatever the binary label says.
- Give the user the whole-run numbers and the notable kernels, then offer to
  write them up.

## Facts that are easy to get wrong (verified on MI300X, job 433875)

- `rocminfo | grep -m1 "Marketing Name"` prints the **CPU**; identify the GPU
  with `grep -m1 "  Name:  *gfx"`.
- Counter names differ per generation: gfx942 uses `TCC_EA0_*`, not `TCC_EA_*`.
- `GRBM_GUI_ACTIVE` is summed over the 8 XCCs: divide by 8 (gives ~1815 MHz
  effective clock; without it ~14,500 MHz).
- **Utilisation denominators use kernel time × the sampled gfx clock, not
  `GRBM_GUI_ACTIVE`.** In counter-collection mode `GRBM_GUI_ACTIVE` adds a
  fixed ~14k-18k cycles to every launch, which understates CU/VALU/SALU/MFMA busy
  and occupancy by ~10% overall and several times for µs-long kernels.
  `postprocess_pmc.py --power-ktrace` takes the clock from the power run's
  telemetry, averaging only samples taken while a kernel ran (OSPREY 2070 MHz).
  Never quote the whole-file mean from the job log: it includes startup and the
  rocprofv3 output write (2052 MHz). Frequency always comes from the
  telemetry; `GRBM_GUI_ACTIVE / 8 / time` is only a check of the ÷8. Limit:
  the clock is from the power run, the counters from unsampled PMC runs. On a
  power-throttled workload they can differ (LAMMPS 1733 MHz sampled gave CU busy
  1.15 on five kernels), so read the CU-busy and waves/CU warnings. Details:
  `references/formulae.md` 4.2-4.4.
- `SQ_WAVE_CYCLES` needs x4 (quad-cycles); `SQ_BUSY_CU_CYCLES` must **not**
  get x4 (it would give CUs 299% busy). Counter descriptions are unreliable on
  units -- check against physical limits (occupancy under each kernel's
  register limit, CU busy <= 1, threads <= 64). The script does this.
- `GRBM_GUI_ACTIVE / GRBM_COUNT` is always 100% in per-kernel counter mode;
  take GPU busy % from the power run's kernel trace.
- FLOPs need only MFMA **MOPS** counters (x512); instruction counts add nothing.
  VALU FLOPs from instruction counts are *issued* (x64 lanes); useful FLOPs
  = issued x thread utilization (estimate, cross-pass).
- `TCC_EA0_*` traffic is measured before the 256 MB Infinity Cache: an upper
  bound on HBM traffic; name it "L2 -> memory".
- Workgroups can be 2-D (e.g. 1 x 256): size = X*Y*Z.
- Profiled kernel durations are serialized per-kernel times; wall-clock
  behaviour comes from the power run.
- Sampler: 1 ms requested, ~2.3 ms achieved; the edge-temperature sensor is
  not exposed (all `nan`); samples before the first kernel (startup) and after
  the last (rocprofv3 writing output, ~3 s) must be trimmed.
- rocprofv3 1.1.0's counter CSV already carries start/end timestamps; each
  counter CSV is ~160 MB for a 5 s, 68k-launch run.

## References

- `references/counters_gfx942.md` -- counter-by-counter rationale, pass layout,
  design decisions and cuts.
- `references/formulae.md` -- every derived metric, why it is written that
  way, and LAMMPS whole-run values (use them to sanity-check a new workload's
  magnitudes).
- `references/lammps_case_study.md` -- the full LAMMPS run: commands, job IDs,
  results, lessons.
