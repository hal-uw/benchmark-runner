# Case study: LAMMPS ReaxFF/HNS 16x8x12 on 1x MI300X

The run this skill was built from. Use it as a template for a new workload and
as a magnitude check for its results.

## Workload

| | |
|---|---|
| Code | LAMMPS develop (patch 2 Sep 2026), KOKKOS + REAXFF, HIP, double precision |
| Build | `lammps/build-mi300x/` (settings from its CMakeCache: hipcc, `-munsafe-fp-atomics`, `Kokkos_ARCH_AMD_GFX942`, `Kokkos_ENABLE_HIP`, `BUILD_MPI=on`, `FFT_KOKKOS=KISS`, Release) |
| Input | `in.reaxc.hns` (MINOS artifact), `-v x 16 -v y 8 -v z 12` = 466,944 atoms, 100 steps, fixed velocity seed |
| Kokkos flags | `-k on g 1 device 0 -sf kk -pk kokkos neigh half neigh/qeq full newton on` |
| Correctness | 2x2x2 run (2,432 atoms) against `examples/reaxff/HNS/log.30Nov23.reaxff.hns.g++.4` |
| Folder | `/work1/sinclair/sairajatg/workloads/lammps/` |
| Config | `examples/lammps_hns16812.conf` |

## Jobs

| Job | Script (original) | What | Result |
|---|---|---|---|
| 431866 | `hns-mi300x.sbatch` | plain runs (2x2x2 + 16x8x12) | loop time 5.33 s |
| 431893 | `hns-mi300x-prof.sbatch` (first version) | power sampling, no kernel trace | power timeline only |
| 433875 | `hns-mi300x-pmc.sbatch` | 6 PMC passes | all rc=0; loop 14-20 s per pass; kernel time 5.08 s +-0.25% |
| 433930 | `hns-mi300x-prof.sbatch` | power sampling + kernel trace | loop 5.55 s (+4%); 67,483 launches; clocks aligned |

Post-processing: `postprocess_pmc.py results/pmc-433875 --power-ktrace
results/prof-433930/ktrace_hns16812`; `build_sampling_json.py
results/prof-433930 --label "LAMMPS 16x8x12"`.

## Whole-run results (per GPU)

| Metric | Value |
|---|---|
| Effective clock | 1815 MHz (boost 2100) |
| CU busy / VALU busy | 0.748 / 25.6% |
| Threads active per vector instruction | 32.4 of 64 (50.6%) |
| Occupancy | 13.6 waves/CU (42% of 32; 61% of kernels' own limits) |
| Wave issue fraction | 11.4% (waves wait ~89% of the time) |
| FP64 issued / useful | 5.67 / 2.32 TFLOP/s (6.9% of 81.7 peak issued) |
| MFMA | 0 (not used) |
| L2 -> memory bandwidth | 1.31 read + 0.52 write = 1.84 TB/s (35% of 5.3) |
| Atomics | 63.6% of write requests (Kokkos force accumulation) |
| Arithmetic intensity | 3.09 FLOP/B (ridge 15.4) -> memory side, 35% of roofline |
| GPU busy % of kernel window (power run) | 66.7% |
| Socket power | at 675-750 W (the cap) 92% of the kernel window |
| gfx clock | 1700-1800 MHz 73% of the time |
| Hotspot temperature | 50-75 C |

Top kernels: QEq sparse matvec (26% of time; bandwidth-bound at 3.0 TB/s, 57%
of roofline), LJCoulomb (21%; 11.4 TFLOP/s issued, all writes atomic),
BuildListsHalfPreview (12%; 18.1 issued but 1.8 useful TFLOP/s -- 6.3/64
threads active), Bond/Torsion/Angular (15%; VALU 3-5% busy, waiting on memory
and atomics).

## Lessons (what went wrong or nearly did)

1. The first counter list was generated with `salloc`/`srun --pty` inside a
   script: the commands after them ran on the login node and listed gfx90a.
   Fixed with an sbatch job (`list_counters.sbatch`).
2. The earlier `counters_memory.json` used `TCC_EA_*_sum`, which do not exist
   on gfx942 (`TCC_EA0_*`).
3. Counter-set design started at 12 passes; the user cut it to 6 (MFMA MOPS
   only, no L1->L2, no L2 hit rate/stalls/LDS, no separate occupancy pass).
4. Unit factors were settled with data, not descriptions: /8 for GRBM, x4 for
   `SQ_WAVE_CYCLES`, none for `SQ_BUSY_CU_CYCLES`.
5. A hand calculation of FP64 FLOPs was off by 30% (3.74e13 vs 2.886e13); the
   script caught it. Compute numbers with code, not by hand.
6. QEq iteration counts differ ~0.5% between runs, so launch IDs cannot be
   matched across passes -- aggregate per kernel name.
7. The first power run had no kernel trace, so power could not be attributed
   to kernels; adding `rocprofv3 --kernel-trace` inside `wrapper.py`'s command
   costs ~4% and gives aligned timestamps.
8. `calculate_power_distribution()` in dendrogram.py cannot read the sampler
   CSV directly (metadata first line) and writes into `app_vectors.json`; the
   builder feeds it a clean temporary CSV instead.

## Documents written during the study (in the workload folder)

`gfx942_pmc_counters_info.md` (counter set), `post_processing_formulae.md`
(formulas), `sbatch_lammps_exp.md` (walkthrough of the PMC sbatch).
