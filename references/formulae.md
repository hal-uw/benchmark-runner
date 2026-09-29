# Post-Processing Formulae: LAMMPS ReaxFF/HNS PMC Data on MI300X

Every formula the post-processing script uses to turn the rocprofv3 counter
data into metrics, with what it means, why it is written that way, and its
whole-run value from job 433875 as a check.

| | |
|---|---|
| Data | `results/pmc-433875/` (6 passes, 16×8×12, 466,944 atoms, 100 steps) |
| Counter set | `counters.json` (see `gfx942_pmc_counters_info.md`) |
| Run script | `hns-mi300x-pmc.sbatch` (see `sbatch_lammps_exp.md`) |
| GPU | AMD Instinct MI300X (`gfx942`), ROCm 7.2.0, rocprofv3 1.1.0 |

The formulas include five changes from the original `counters.json`, agreed
and applied to `counters.json` (see [Section 12](#12-decisions)).

---

## 1. Inputs

Each pass folder `results/pmc-<jobid>/<pass>/` provides:

| File | Columns used |
|---|---|
| `<pass>_counter_collection.csv` | `Dispatch_Id`, `Kernel_Name`, `Counter_Name`, `Counter_Value`, `Start_Timestamp`, `End_Timestamp`, `VGPR_Count`, `Accum_VGPR_Count`, `LDS_Block_Size` |
| `<pass>_kernel_trace.csv` | `Dispatch_Id`, `Kernel_Name`, `Start_Timestamp`, `End_Timestamp`, `Workgroup_Size_X/Y/Z`, `Grid_Size_X/Y/Z` |

The counter CSV has one row per kernel launch per counter. The value in each
row is already summed over every hardware instance (all 8 XCCs, all shader
engines, all L2 channels).

---

## 2. Constants

| Symbol | Value | Meaning |
|---|---|---|
| `N_XCC` | 8 | Accelerator chiplets (XCCs) |
| `N_CU` | 304 | Compute units |
| `N_SIMD` | 1216 | SIMDs (4 per CU) |
| `WAVE` | 64 | Threads per wave |
| `MAX_WAVES_CU` | 32 | Max resident waves per CU (8 per SIMD × 4) |
| `VGPR_PER_SIMD` | 512 | Vector registers per thread slot per SIMD (VGPR + AGPR unified) |
| `LDS_PER_CU` | 65,536 B | Shared memory (LDS) per CU |
| `MOPS_UNIT` | 512 | FLOPs per MFMA MOPS count |
| `F_BOOST` | 2100 MHz | Peak engine clock |
| `PEAK_FP64_VALU` | 81.7 TFLOP/s | 304 × 128 FLOP/clk × 2.1 GHz |
| `PEAK_FP32_VALU` | 163.4 TFLOP/s | Packed FP32 (unpacked: 81.7) |
| `PEAK_FP64_MFMA` | 163.4 TFLOP/s | 304 × 256 FLOP/clk × 2.1 GHz |
| `PEAK_FP32_MFMA` | 163.4 TFLOP/s | 304 × 256 FLOP/clk × 2.1 GHz |
| `PEAK_MEM_BW` | 5.3 TB/s | HBM3 peak |

---

## 3. Aggregation and joining passes

### 3.1 Within one pass: sum per kernel

For every kernel name `k` in a pass `p`:

```
C[p][k][counter] = Σ over launches of k   Counter_Value
D[p][k]          = Σ over launches of k   (End_Timestamp − Start_Timestamp)      [ns]
L[p][k]          = number of launches of k
```

**Why sum per kernel instead of per launch:** launch IDs don't line up across
passes. The QEq charge solver iterates until it converges, and floating-point
differences from atomics change its iteration count by about 0.5% per run.
Launches per pass were 67,591–68,059.

### 3.2 Across passes: combine rates, never raw sums

A metric that uses counters from two passes combines **per-kernel rates**
(per ns) or **per-launch averages**, each computed within its own pass:

```
rate[p][k][counter] = C[p][k][counter] / D[p][k]          (per ns)
```

Example: arithmetic intensity uses FLOP/ns from pass 1 and bytes/ns from
passes 5 and 6. Summing FLOPs from one run and dividing by bytes from another
would be wrong whenever the launch counts differ.

**Assumption:** each kernel behaves the same in every pass. This holds well:
same fixed velocity seed, thermo output identical to 7–8 significant figures,
total kernel time 5.076–5.101 s in every pass.

### 3.3 Whole-run values

```
whole-run count  = Σ_k C[p][k][counter]
whole-run time   = Σ_k D[p][k]
whole-run rate   = whole-run count / whole-run time          (within one pass)
```

Cross-pass whole-run metrics combine whole-run rates, the same way as 3.2.
Setup kernels (building the replicated system, the first neighbour list, the
first QEq solve) are included; per-kernel reporting shows them separately.

### 3.3a Multi-GPU runs (8x MI300X)

With several processes each writes its own CSVs; `postprocess_pmc.py` reads
all of them, keys launches by (file, `Dispatch_Id`) because IDs restart per
process, and sums counters and durations over GPUs. Rates are therefore
**per-GPU averages** (total work / total GPU-kernel time), which is what "% of
one GPU's peak" needs; multiply by the GPU count for whole-node rates. GPU busy
% from the power run is computed per (trace file, `Agent_Id`) and averaged.

### 3.4 Kernel names

Kokkos kernel names all start with the same
`hip_parallel_launch_constant_memory<Kokkos::Impl::ParallelFor<...>>` wrapper.
The display name is the functor class plus tag extracted from the full name,
e.g. `PairReaxFFKokkos::TagPairReaxComputeLJCoulomb`. The full name stays the
join key.

---

## 4. Time and clock

### 4.1 Kernel duration

```
duration_ns = End_Timestamp − Start_Timestamp
```

Per launch, from the counter CSV or the kernel trace (they match). Under
counter collection rocprofv3 runs kernels one at a time, so this is each
kernel's own runtime, not a slice of an overlapping timeline.

### 4.2 Cycles available to the kernel (utilisation denominator)

Every busy-cycle metric (CU busy 5.1, VALU busy 5.2, occupancy 5.4, SALU busy
5.8, MFMA busy 5.9) is `busy cycles / (cycles that elapsed during the kernel ×
units)`. The numerators are hardware counters. The elapsed cycles are:

```
cycles = D[k] (ns, kernel trace of that pass) × clock_MHz / 1000
clock_MHz = mean sampled gfx clock over the power-run samples taken while a kernel ran
```

**Why the sampled clock, not a counter.** The obvious counter,
`GRBM_GUI_ACTIVE / N_XCC` (what ROCm's own `MfmaUtil` derived metric divides
by), adds a fixed number of cycles to every launch in counter-collection mode:
about 13,700 per launch for OSPREY gpt-oss-120b, 17,700 for LAMMPS (a straight-line fit of
per-launch cycles against duration). The fitted slope is the real clock (2059
and 2074 MHz on OSPREY passes 1 and 4, against a sampled 2070 MHz); the constant
is profiler overhead. It is about 10% of the whole-run count and several
times a 3 µs kernel's own cycles, so it understates utilisation, most of all for
short kernels. The numerators do not carry it: launches under 5 µs averaged
20,800 `GRBM_GUI_ACTIVE` cycles but only 402 `SQ_BUSY_CU_CYCLES` per CU.

**Which samples.** Only samples taken while a traced kernel was running (the
telemetry and the kernel trace share the boot-time clock). A mean over the whole
telemetry file is wrong: OSPREY's file was 1.03M samples, of which only 89k
fell on a running kernel; startup ran at 138 MHz and the long rocprofv3 output
write after the last kernel dominated the rest (whole-file mean 2052 MHz,
running mean 2070 MHz). Per-kernel clocks for OSPREY were within 0.6%
(2066–2078 MHz), so one clock per run is enough there.

**How the script gets it.** `postprocess_pmc.py --power-ktrace
<prof-dir>/ktrace_<tag>` reads `<prof-dir>/profiling_result_*.csv` (or
`--telemetry <csv>...`); `--clock-mhz X` overrides it. Without telemetry it
falls back to `GRBM_GUI_ACTIVE / N_XCC` and says so. `report.md` states the clock
and its source at the top of "Whole run", and `summary.json` has it under
`clock`.

**Limit: throttled workloads.** The clock comes from the power run, but the
counters come from the PMC runs, which were not sampled. The two match when the
GPU runs near its boost clock (OSPREY: 2070 MHz, no check fails). LAMMPS is
power-throttled (sampled 1733 MHz under load). With that clock, five kernels
came out at 1.13–1.15 CU busy and two above their register limit on waves/CU. The
likely reason is that PMC runs, which serialize kernels with gaps, let the GPU
clock higher than the power run. For such workloads, sample the clock during the
PMC passes, or read the results as bounds. The script's sanity warnings (CU busy
> 1, waves/CU > limit) catch the mismatch, so read them before using the numbers.

### 4.3 `GRBM_GUI_ACTIVE` clock (check only)

```
eff_clock_MHz = (GRBM_GUI_ACTIVE / N_XCC) / duration_ns × 1000
```

It is no longer a denominator. It checks the ÷8: without it the value would be
about 8× the sampled clock (LAMMPS ~14,500 MHz). With the per-launch overhead it
sits a little above the real clock (LAMMPS 1815 against 1733 sampled; OSPREY
2250–2280 against 2070), and more for workloads of many short launches. The script
warns outside 0.8–2× the sampled clock. The data-checks table also gives, per
pass, the share of `GRBM_GUI_ACTIVE` cycles outside kernel time × clock (OSPREY
about 8–10%).

### 4.4 Clock factor

```
clock_factor = clock_MHz / F_BOOST
```

Scales datasheet peaks to the clock the GPU actually ran at (the sampled clock
of 4.2). OSPREY: 2070 / 2100 = 0.986.

### 4.5 GPU busy % of wall time (from the power run, not the PMC run)

```
GPU_busy_pct = 100 × (union of kernel intervals) / (last kernel end − first kernel start)
```

From `ktrace_hns16812/hns16812_kernel_trace.csv` in the power run. That run
has no counters, so kernels overlap normally and the timing is realistic.
"Union of intervals" merges overlapping kernels so time is not counted twice.

**Replaces** `GPU_UTIL_pct = 100 × GRBM_GUI_ACTIVE / GRBM_COUNT`. In
per-kernel counter mode both counters only count while the kernel runs, so
that ratio is always 100% (measured: both 7.4176e10).

---

## 5. CU utilisation (pass 4)

### 5.1 CU busy fraction

```
CU_busy_frac = SQ_BUSY_CU_CYCLES / (cycles × N_CU)
```

Share of CU-cycles in which a CU had work. 1.0 = all 304 CUs busy for the
whole kernel.

**No ×4.** The counter description says quad-cycles, but with ×4 the whole
run gives 2.99, i.e. CUs 299% busy, which is impossible. Without ×4 it is
0.748, and the busiest kernels read 0.97. This matches AMD's own
`SIMD_UTILIZATION` expression, which has no ×4.

Whole run: **0.748**.

### 5.2 VALU busy

```
VALUBusy_pct = 100 × SQ_ACTIVE_INST_VALU / (cycles × N_CU)
```

Share of time the vector units (VALUs) are executing instructions.
`SQ_ACTIVE_INST_VALU` is in quad-cycles summed over the 4 SIMDs of each CU.
A SIMD executes a 64-thread wave instruction in 4 cycles (16 lanes × 4), i.e.
one per quad-cycle, so the SIMD-level maximum is `cycles / 4` per SIMD, or
`N_SIMD × cycles / 4 = N_CU × cycles` in total. That is why the
denominator is `cycles × N_CU` with no extra factor. This is AMD's
`VALUBusy` expression.

Check: `SQ_ACTIVE_INST_VALU` (7.21e11) ≈ `SQ_INSTS_VALU` (6.98e11), i.e. about
one quad-cycle per vector instruction, as expected. Highest kernel: 0.68.

Whole run: **25.6%**.

### 5.3 Thread utilisation (divergence)

```
VALU_threads        = SQ_THREAD_CYCLES_VALU / SQ_ACTIVE_INST_VALU          (0–64)
VALUUtilization_pct = 100 × VALU_threads / WAVE
```

Average number of the 64 threads active per vector instruction. When an `if`
splits a wave, both branches run one after the other with the inactive threads
switched off, so fewer threads are active. 100% = no divergence.

Per kernel it ranges from 2.1 to 61.5 threads, never above 64.

Whole run: **32.4 threads = 50.6%**.

### 5.4 Occupancy (mean resident waves)

```
waves_per_CU  = 4 × SQ_WAVE_CYCLES / (cycles × N_CU)
Occupancy_pct = 100 × waves_per_CU / MAX_WAVES_CU
```

`SQ_WAVE_CYCLES` adds 1 per resident wave every 4 clock cycles (quad-cycles),
so `4 × SQ_WAVE_CYCLES` is total wave-cycles: resident waves summed over every
clock cycle. Dividing by kernel cycles and by CUs gives the average number of
waves resident on a CU. AMD's `OccupancyPercent` writes this as
`400 × SQ_WAVE_CYCLES / GUI / CU_NUM / 32` (100 for percent × 4 for
quad-cycles).

**×4 checked:** with ×4, every top kernel sits just under its register limit
(5.5); without ×4 they would be at ~25% of it while their CUs are 85–97% busy,
and anything above ×4 would exceed the limits.

| GPU time | Register limit (waves/CU) | With ×4 | Without ×4 |
|---|---|---|---|
| 25.9% | 32 | 28.9 | 7.2 |
| 20.6% | 16 | 12.2 | 3.0 |
| 11.5% | 16 | 11.6 | 2.9 |
| 7.0% | 24 | 21.3 | 5.3 |
| 6.1% | 12 | 10.7 | 2.7 |

Whole run: **13.6 waves/CU = 42%**.

### 5.5 Theoretical (maximum) occupancy per kernel

```
vgpr_alloc      = ceil8( ceil4(VGPR_Count) + Accum_VGPR_Count )    if Accum_VGPR_Count > 0
                = ceil8( VGPR_Count )                              otherwise
waves_SIMD_reg  = min(8, floor(VGPR_PER_SIMD / vgpr_alloc))
wg_threads      = Workgroup_Size_X × Workgroup_Size_Y × Workgroup_Size_Z
waves_per_WG    = ceil(wg_threads / WAVE)
waves_CU_LDS    = floor(LDS_PER_CU / LDS_Block_Size) × waves_per_WG     (no limit if LDS_Block_Size = 0)
max_waves_CU    = min(4 × waves_SIMD_reg, waves_CU_LDS, MAX_WAVES_CU)
```

`ceilN(x)` rounds up to a multiple of N.

The most waves a kernel can ever have resident on a CU, set by its register
and LDS use:
- **Registers:** each SIMD has 512 registers per thread slot, shared between
  ordinary vector registers (VGPRs) and accumulation registers (AGPRs), and
  allocated in blocks of 8. A kernel using 128 fits 4 waves per SIMD =
  16 per CU.
- **LDS:** each workgroup reserves `LDS_Block_Size` bytes of the CU's 64 KB.
  Here LDS use is at most 512 B per workgroup, so LDS never limits.
- **Workgroup size:** workgroups are 2-D for most kernels (e.g. 1 × 256), so
  the size is X × Y × Z, not X alone.

Examples: 28 VGPR + 4 AGPR → 32 registers → 8 waves/SIMD → **32/CU**;
20 VGPR + 132 AGPR → 152 → 3 waves/SIMD → **12/CU**; 128 VGPR → 4 → **16/CU**.

Scalar-register limits are not modelled; they rarely limit on gfx942.

### 5.6 Occupancy relative to the kernel's own limit

```
Occupancy_of_max_pct = 100 × waves_per_CU / max_waves_CU
```

Distinguishes "occupancy is low because registers cap it" (fix: fewer
registers) from "occupancy is low even though registers allow more" (fix:
more work per launch, or launch overheads). A 128-register kernel at 12.2
waves/CU is 38% of 32 but **76% of its own limit of 16**.

### 5.7 Wave issue fraction

```
Wave_issue_pct = 100 × SQ_ACTIVE_INST_ANY / SQ_WAVE_CYCLES
```

Share of a resident wave's lifetime spent issuing any instruction; the rest is
waiting (on memory, dependencies, barriers). Both counters are in quad-cycles,
so the ratio needs no unit factor.

Whole run: 1.0894e12 / 9.5624e12 = **11.4%**. Waves spend ~89% of their time
waiting.

### 5.8 Scalar (SALU) busy

```
SALUBusy_pct = 100 × SQ_INST_CYCLES_SALU / (cycles × N_CU)
```

AMD's `SALUBusy` expression: share of time the scalar unit is executing.
One scalar unit per CU, hence `N_CU`. The quad-cycle units in the description
are not verified for this counter; treat the value as a lower bound
(×4 would be the upper bound).

Whole run: **6.8%** (27% if ×4).

### 5.9 MFMA busy

```
MfmaUtil_pct = 100 × SQ_VALU_MFMA_BUSY_CYCLES / (cycles × N_SIMD)        (cycles: 4.2)
```

Share of time the matrix units are occupied (the counter is per SIMD).
Pass 1. Whole run: **0** — LAMMPS does not use MFMA.

---

## 6. FLOPs (passes 1–3)

Instruction counters count **wave instructions**: one instruction executed
for a whole wave, whatever number of threads is active.

### 6.1 FP64 vector (VALU) FLOPs, issued

```
FP64_VALU_FLOP = WAVE × (ADD_F64 + MUL_F64 + 2 × FMA_F64 + TRANS_F64)
```

(`ADD_F64` = `SQ_INSTS_VALU_ADD_F64`, and so on.)

- **× WAVE (64):** each wave instruction covers 64 lanes.
- **2 × FMA:** fused multiply-add is a multiply and an add.
- **TRANS as 1 FLOP:** exp, log, sqrt, reciprocal. They run at a quarter of the
  normal rate, so they occupy the vector unit longer than their FLOP count
  suggests (VALU busy, 5.2, captures that). AMD's `TOTAL_64_OPS` leaves them
  out. They are also reported separately (6.2).

These are **issued** FLOPs: they assume all 64 lanes are active. Right for
"% of hardware peak", because inactive lanes still take up the issue slot.

Whole run: 64 × (1.3241e11 + 4.009e10 + 2 × 1.3605e11 + 6.347e9) = 64 × 4.510e11 = **2.886e13**.

### 6.2 FP64 transcendental FLOPs

```
FP64_TRANS_FLOP = WAVE × TRANS_F64
```

Whole run: 4.06e11, 1.4% of FP64 FLOPs (6.3e9 instructions, 2% of FP64
instructions).

### 6.3 FP64 useful FLOPs (estimate)

```
FP64_VALU_FLOP_useful[k] = FP64_VALU_FLOP[k] × VALUUtilization[k] / 100
```

Per kernel `k`, then summed. Counts only lanes that were switched on. Uses the
kernel's thread utilisation from pass 4, applied to pass 1, and assumes FP64
instructions diverge like the kernel's vector instructions on average, so it
is an estimate.

Whole run: the per-kernel sum is **1.18e13** (2.32 TFLOP/s), 41% of issued.
Applying the whole-run utilisation instead (2.886e13 × 0.506 ≈ 1.46e13) gives
a different answer, because utilisation varies from 3% to 96% by kernel and
some FP64-heavy kernels are highly divergent (e.g. `BuildListsHalfPreview`:
6.3 of 64 threads active).

### 6.4 FP64 matrix (MFMA) FLOPs

```
FP64_MFMA_FLOP = MOPS_UNIT × SQ_INSTS_VALU_MFMA_MOPS_F64
```

MOPS counts matrix operations divided by 512. Whole run: **0**.

### 6.5 FP64 total

```
FP64_FLOP = FP64_VALU_FLOP + FP64_MFMA_FLOP
```

### 6.6 FP32 and FP16

```
FP32_VALU_FLOP = WAVE × (ADD_F32 + MUL_F32 + 2 × FMA_F32 + TRANS_F32)
FP32_MFMA_FLOP = MOPS_UNIT × SQ_INSTS_VALU_MFMA_MOPS_F32
FP32_FLOP      = FP32_VALU_FLOP + FP32_MFMA_FLOP
FP16_VALU_FLOP = WAVE × (ADD_F16 + MUL_F16 + 2 × FMA_F16 + TRANS_F16)
```

Same structure as FP64. Packed FP32 instructions (2 operations per lane) are
counted once, so FP32 would be under-counted if they were used.

Whole run: FP32 = 64 × (1.349e9 + 1.349e9) = 1.7e11 (0.6% of FP64);
FP16 = 0; FP32 MFMA = 0.

### 6.7 Throughput

```
TFLOPs = FLOP / duration_ns / 1000
```

FLOP per ns is GFLOP/s; ÷ 1000 gives TFLOP/s. Per kernel: `C / D` within
pass 1 (or 2).

Whole run FP64: 2.886e13 / 5.090e9 ns / 1000 = **5.67 TFLOP/s**.

### 6.8 % of peak

```
FP64_VALU_peak_pct       = 100 × TFLOPs_FP64_VALU / PEAK_FP64_VALU
FP64_VALU_peak_clk_pct   = 100 × TFLOPs_FP64_VALU / (PEAK_FP64_VALU × clock_factor)
FP64_MFMA_peak_pct       = 100 × TFLOPs_FP64_MFMA / PEAK_FP64_MFMA
FP32_MFMA_peak_pct       = 100 × TFLOPs_FP32_MFMA / PEAK_FP32_MFMA
```

- **At the 2100 MHz boost clock:** % of the datasheet peak.
- **At the measured clock** (`clock_factor` from pass 1's own
  `GRBM_GUI_ACTIVE`): % of what the GPU could do at the clock it actually ran
  at. The gap between the two is the clock drop.

Whole run FP64 vector: **6.9%** of 81.7 TFLOP/s; **8.0%** of 70.6 TFLOP/s at
1815 MHz.

### 6.9 FP64 share of vector instructions

```
FP64_share_pct = 100 × (ADD_F64 + MUL_F64 + FMA_F64 + TRANS_F64) / SQ_INSTS_VALU
```

Pass 1 only. The rest is integer, compare, move, conversion and similar.
Whole run: 3.149e11 / 6.984e11 = **45%**.

---

## 7. Instruction mix (passes 1 and 3)

```
total_insts = SQ_INSTS_VALU + SQ_INSTS_SALU + SQ_INSTS_VMEM_RD + SQ_INSTS_VMEM_WR + SQ_INSTS_LDS
share_X_pct = 100 × SQ_INSTS_X / total_insts
```

`SQ_INSTS_VALU` is from pass 1, the rest from pass 3; combine as per-kernel
per-ns rates (3.2). There is no total-instruction counter, so this is the share
among these five types; branch and scalar-memory instructions are not
included.

Whole run:

| Type | Count | Share |
|---|---|---|
| Vector (VALU) | 6.98e11 | 72.4% |
| Scalar (SALU) | 1.90e11 | 19.7% |
| Vector memory reads | 3.96e10 | 4.1% |
| Vector memory writes | 6.77e9 | 0.7% |
| Shared memory (LDS) | 2.97e10 | 3.1% |

---

## 8. Memory (passes 5–6)

**Where this is measured:** at the L2 → memory interface (`TCC_EA0`). On
MI300X a 256 MB Infinity Cache (MALL) sits between that interface and HBM, and
some requests are served there without reaching HBM. So these bytes are an
**upper bound on HBM traffic**, named `mem_*` ("L2 → memory") rather than
`HBM_*`.

### 8.1 Read bytes

```
mem_read_bytes = 128 × TCC_BUBBLE_sum
               + 64  × (TCC_EA0_RDREQ_sum − TCC_BUBBLE_sum − TCC_EA0_RDREQ_32B_sum)
               + 32  × TCC_EA0_RDREQ_32B_sum
```

Read requests come in 32, 64 and 128 bytes. `RDREQ` counts all of them,
`BUBBLE` the 128-byte ones, `RDREQ_32B` the 32-byte ones, and the remainder are
64-byte. AMD's `FETCH_SIZE`, in bytes instead of KB.

Whole run: 128 × 5.2157e10 + 64 × 2.5e7 + 0 = **6.68e12 B**.

### 8.2 Read request size mix

```
read_128B_pct = 100 × TCC_BUBBLE_sum / TCC_EA0_RDREQ_sum
read_32B_pct  = 100 × TCC_EA0_RDREQ_32B_sum / TCC_EA0_RDREQ_sum
read_64B_pct  = 100 − read_128B_pct − read_32B_pct
```

Whole run: **99.95% 128-byte**, 0% 32-byte. Reads are full cache lines.

### 8.3 Write bytes

```
mem_write_bytes = 32 × (TCC_EA0_WRREQ_sum − TCC_EA0_WRREQ_64B_sum)
                + 64 × TCC_EA0_WRREQ_64B_sum
```

Write requests are 32 or 64 bytes. AMD's `WRITE_SIZE`, in bytes. Atomics sent
to memory travel on the same interface and are included as 32-byte requests:
their payload is 8 bytes (one double), but 32 bytes is what moves.

Whole run: 32 × 4.4077e10 + 64 × 1.9337e10 = **2.65e12 B**.

### 8.4 Atomics

```
atomic_req_pct   = 100 × TCC_EA0_ATOMIC_sum / TCC_EA0_WRREQ_sum
atomic_bytes     = 32 × TCC_EA0_ATOMIC_sum
```

Share of write requests that are atomics (Kokkos force accumulation).

Whole run: 4.03e10 of 6.34e10 = **63.6%** of write requests; 1.29e12 B,
49% of write bytes.

Atomics that return a value also send data back; those bytes are not in
`mem_read_bytes`.

### 8.5 Uncached traffic

```
uncached_read_bytes  = 32 × TCC_EA0_RD_UNCACHED_32B_sum
uncached_write_bytes = 32 × TCC_EA0_WR_UNCACHED_32B_sum
```

The counters are in 32-byte units (a 64-byte request counts as 2) and are
subsets of the totals above.

Whole run: reads 1.3e8 B (negligible); writes 1.29e12 B. The uncached write
count equals the atomic count exactly (4.0322e10), so **all uncached writes are
the atomics**.

### 8.6 Bandwidth

```
read_BW_TBps  = mem_read_bytes  / D[pass 5] / 1000
write_BW_TBps = mem_write_bytes / D[pass 6] / 1000
mem_BW_TBps   = read_BW_TBps + write_BW_TBps
mem_BW_peak_pct = 100 × mem_BW_TBps / PEAK_MEM_BW
```

Bytes per ns is GB/s; ÷ 1000 gives TB/s. **Each direction uses its own pass's
durations**, then the rates are added. This replaces
`(read + write) / one duration`.

Whole run: 6.68e12 / 5.082e9 / 1000 = 1.31 TB/s read; 2.65e12 / 5.076e9 / 1000
= 0.52 TB/s write; **1.84 TB/s = 35%** of 5.3 TB/s.

### 8.7 Arithmetic intensity

```
AI_FP64 = (FP64_FLOP / D[pass 1]) / ((mem_read_bytes / D[pass 5]) + (mem_write_bytes / D[pass 6]))
```

FP64 FLOPs per byte of L2 → memory traffic, as a ratio of rates from three
passes.

Whole run: 5.67 TFLOP/s / 1.84 TB/s = **3.09 FLOP/byte**.

### 8.8 Roofline

```
ridge_FP64       = PEAK_FP64_VALU / PEAK_MEM_BW                     (15.4 FLOP/byte)
attainable_TFLOPs = min(PEAK_FP64_VALU, AI_FP64 × PEAK_MEM_BW)
roofline_pct     = 100 × TFLOPs_FP64 / attainable_TFLOPs
```

Below the ridge a kernel is limited by memory bandwidth, above it by compute.
`roofline_pct` says how close the kernel gets to the limit that applies to it.

Whole run: AI 3.09 < 15.4, so memory-side; attainable = 3.09 × 5.3 = 16.4
TFLOP/s; achieved 5.67 → **34.6% of the roofline**. The upper-bound caveat
(Infinity Cache) makes the true HBM intensity higher than 3.09.

### 8.9 All-precision roofline (mixed precision, ML workloads)

The FP64 roofline (8.8) calls every LLM kernel memory-bound, because their work
is BF16/FP8 matrix math. This version counts every precision against its own
peak. Low-precision MFMA needs pass `07_lowp_mfma`
(`SQ_INSTS_VALU_MFMA_MOPS_{BF16,F16,F8,I8}`, x512 like the other MOPS).

```
rate_i            = FLOP_i / D[pass of FLOP_i]      for i in FP64 VALU/MFMA, FP32 VALU/MFMA,
                                                    FP16 VALU, BF16/F16/F8/I8 MFMA
compute_peak_pct  = 100 × Σ rate_i / peak_i          share of peak compute time used
TFLOPs_all        = Σ rate_i
peak_TFLOPs_mix   = TFLOPs_all / Σ (rate_i / peak_i) peak for this kernel's precision mix
AI_all            = TFLOPs_all / mem_BW_TBps
ridge_mix         = peak_TFLOPs_mix / 5.3
attainable_TFLOPs_mix = min(peak_TFLOPs_mix, AI_all × 5.3)      "Max FLOPs"
attainable_BW_TBps    = min(5.3, peak_TFLOPs_mix / AI_all)      "Max BW"
bound_mix         = latency if compute_peak_pct < 10 and mem_BW_peak_pct < 10   (--latency-pct)
                    else compute if compute_peak_pct > mem_BW_peak_pct, else memory
                    (same as AI_all >= ridge_mix)
```

Peaks (dense, 2100 MHz): FP64 VALU 81.7; FP64/FP32 MFMA and FP32 VALU 163.4;
BF16/FP16 MFMA 1307.4 (304 × 2048 FLOP/clk); FP8 and INT8 MFMA 2614.9
(304 × 4096). FP16 VALU is not on the datasheet; it is taken as 163.4 (the packed
FP32 rate). For an FP64-only code this reduces to 8.8. Kernels with no FLOPs
(copies, element-wise) get no ceiling.

**Latency-bound.** A roofline label only means something for a kernel near one of
its ceilings. Below 10% of both peak compute and peak bandwidth, the kernel is
labelled `latency`: it has too few waves to fill the GPU, or its waves wait on
dependent memory accesses, so neither ceiling applies. Check it with occupancy
and wave issue % (5.4-5.7). Example, OSPREY `kernel_paged_attention_2d` (80% of
GPU time): 0.35% of compute and 0.9% of bandwidth, CUs busy 6.3%, 0.25 waves per
CU against a limit of 8, waves issuing only 26% of the time, 978 µs per launch.
That is decode attention over at most 5 sequences, whose small grid leaves most
CUs idle. The 10% cutoff is a judgment call; change it with `--latency-pct`.

OSPREY gpt-oss-120b (MXFP4 weights): the `matmul_ogs_*_bf16xbf16xmxfp4_*`
kernels run as BF16 MFMA (gfx942 has no FP4 matrix units), so their peak is
about 1300 TFLOP/s and their ridge about 240 FLOP/B.

---

## 9. Per-kernel report

For each kernel, sorted by share of GPU time:

| Column | Formula | Pass |
|---|---|---|
| Time share | `100 × D[k] / Σ D` | 4 |
| Launches | `L[k]` | 4 |
| Mean time per launch | `D[k] / L[k]` | 4 |
| Effective clock | 4.3 | 1 |
| CU busy | 5.1 | 4 |
| VALU busy | 5.2 | 4 |
| Threads active / 64 | 5.3 | 4 |
| Waves/CU, limit, % of limit | 5.4–5.6 | 4 + trace |
| Wave issue % | 5.7 | 4 |
| FP64 TFLOP/s, % peak (boost, measured clock) | 6.7–6.8 | 1 |
| FP64 useful TFLOP/s | 6.3 / time | 1 + 4 |
| Read / write TB/s, % peak | 8.6 | 5, 6 |
| Atomic % of writes | 8.4 | 6 |
| Arithmetic intensity, % of roofline | 8.7–8.8 | 1, 5, 6 |

Kernels under 0.1% of GPU time are grouped as "other".

### 9.1 Kernels covering 90% of GPU time (`top_kernels.csv`)

The fewest kernels, taken by descending time share, whose shares add up to at
least `--time-pct` (default 90). The time share comes from the power run's
kernel trace when `--power-ktrace` is given (unprofiled, so short kernels are
not inflated by counter collection), otherwise from the PMC passes. One row per
kernel plus a TOTAL row, also written as a section of `report.md`:

| Column | Key | Formula |
|---|---|---|
| time %, cum % | `top_share_pct`, `cum_share_pct` | above |
| CU util % | `CU_busy_pct` | 5.1 |
| VALU busy % | `VALUBusy_pct` | 5.2 |
| VALU lanes % | `VALUUtilization_pct` | 5.3 |
| MFMA util % | `MfmaUtil_pct` | 5.9 |
| FP64 TF/s, FP64 useful TF/s | `TFLOPs_FP64`, `TFLOPs_FP64_VALU_useful` | 6.7, 6.3 |
| FP32 TF/s | `TFLOPs_FP32` | 6.6 |
| low-prec MFMA TF/s, total TF/s | `TFLOPs_lowp_MFMA`, `TFLOPs_all` | 8.9 |
| dominant, peak TF/s, compute % | `dominant_precision`, `peak_TFLOPs_mix`, `compute_peak_pct` | 8.9 |
| rd / wr TB/s, BW util % | `read_BW_TBps`, `write_BW_TBps`, `mem_BW_peak_pct` | 8.6 |
| FLOP/B, ridge | `AI_all`, `ridge_mix` | 8.9 |
| max TF/s, max TB/s | `attainable_TFLOPs_mix`, `attainable_BW_TBps` | 8.9 |
| bound | `bound_mix` | 8.9 |

On LAMMPS job 433875 this reproduces the hand-made
`results/pmc-433875/analysis/top90_kernels.md` (13 kernels, 91.0%).

---

## 10. Whole-run values from job 433875 (hand-computed check)

| Metric | Value |
|---|---|
| Total kernel time per pass | 5.076–5.101 s |
| Effective clock | 1815 MHz |
| CU busy | 0.748 |
| VALU busy | 25.6% |
| Threads active per vector instruction | 32.4 / 64 (50.6%) |
| Occupancy | 13.6 waves/CU (42% of 32; 61% of kernels' own limits, time-weighted) |
| Wave issue fraction | 11.4% |
| SALU busy | 6.8% (lower bound) |
| MFMA busy / MFMA FLOPs | 0 / 0 |
| FP64 issued FLOPs | 2.886e13 |
| FP64 useful FLOPs (per-kernel estimate) | 1.18e13 (2.32 TFLOP/s) |
| FP64 throughput (issued) | 5.67 TFLOP/s |
| FP64 % of peak | 6.9% (boost), 8.0% (measured clock) |
| FP64 share of vector instructions | 45% |
| Read bytes / bandwidth | 6.68e12 B / 1.31 TB/s |
| Write bytes / bandwidth | 2.65e12 B / 0.52 TB/s |
| Total memory bandwidth | 1.84 TB/s (35% of 5.3) |
| Atomics | 63.6% of write requests |
| Arithmetic intensity | 3.09 FLOP/byte (ridge 15.4) |
| % of roofline | 34.6% |

`postprocess_pmc.py` reproduces all of these (checked on job 433875). An earlier
hand calculation of FP64 FLOPs (3.74e13) was an arithmetic slip; the values
above are the corrected ones.

---

## 11. Caveats

1. **Cross-pass metrics assume identical runs.** Supported by matching thermo
   output and kernel times, but the QEq launch counts differ by ~0.5%; hence
   per-kernel rates.
2. **Kernel durations are from profiled runs** (kernels serialized). Rates are
   per-kernel rates; for wall-clock behaviour use the power run.
3. **Memory is measured before the Infinity Cache.** Bytes and bandwidth are
   upper bounds on HBM traffic; arithmetic intensity is a lower bound for HBM.
4. **Issued vs useful FLOPs.** Issued FLOPs count all 64 lanes; useful FLOPs
   are an estimate using pass 4's thread utilisation.
5. **Unit factors were checked against data, not only descriptions.**
   `SQ_WAVE_CYCLES` needs ×4; `SQ_BUSY_CU_CYCLES` must not have it;
   `SQ_ACTIVE_INST_VALU` is consistent with one quad-cycle per instruction;
   `SQ_INST_CYCLES_SALU` is unverified (5.8).
6. **Pass 2 lost counters for 66 launches** (IDs 65556–65621, 1.85 ms, 0.036%
   of GPU time). FP32/INT totals are low by a negligible amount.
7. **Setup kernels are included** in whole-run totals.

---

## 12. Decisions

These changes are agreed and applied to `counters.json` and `gfx942_pmc_counters_info.md`:

| # | Change | Status |
|---|---|---|
| 1 | Drop `GPU_UTIL_pct`; take GPU busy % from the power run's kernel trace (4.5) | Agreed |
| 2 | Remove the ×4 from `CU_busy_frac` (5.1) | Agreed |
| 3 | Report issued and useful FP64 FLOPs; % of peak at boost and measured clock (6.3, 6.8) | Agreed |
| 4 | Rename `HBM_*` → `mem_*` (L2 → memory), with the Infinity Cache caveat (8) | Agreed |
| 5 | Keep setup kernels; report per kernel plus whole run (3.3, 9) | Agreed |

Added since `counters.json` was written: theoretical occupancy and occupancy of
limit (5.5–5.6), wave issue fraction (5.7), SALU busy (5.8), FP64 share of
vector instructions (6.9), instruction mix (7), read size mix (8.2), atomic
share (8.4), per-pass bandwidth (8.6), roofline (8.8).

---

## 13. Running the script

`postprocess_pmc.py` (in this directory) implements every formula above.
Standard library only, Python ≥ 3.9; no GPU needed; about 20 s for job 433875.

```bash
cd /work1/sinclair/sairajatg/workloads/lammps
python3 postprocess_pmc.py results/pmc-433875
# with GPU busy % and unprofiled kernel times from the power run:
python3 postprocess_pmc.py results/pmc-433875 --power-ktrace results/prof-<jobid>/ktrace_hns16812
```

Options: `--out-dir` (default `<pmc-dir>/analysis`), `--counters` (default
`<pmc-dir>/counters.json`, the copy saved by the run), `--top` (kernels in the
report, default 20).

Outputs in `<pmc-dir>/analysis/`:

| File | Contents |
|---|---|
| `per_kernel_metrics.csv` | One row per kernel plus a `TOTAL` row, every metric; short and full kernel names |
| `summary.json` | Whole-run metrics, per-pass data checks, warnings, power-run summary |
| `report.md` | Whole-run table, top kernels, data checks and warnings |

The script finds each counter's pass from `counters.json`, so it does not
depend on pass names. Its data checks warn if: a counter has no data, a pass's
effective clock is outside 1000–2200 MHz (÷8 assumption), more than 0.5% of
kernel time has no counters, kernel time differs by more than 5% across
passes, or a kernel exceeds a physical limit (CU busy > 1, threads > 64,
waves above its register limit).
