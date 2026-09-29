# gfx942 (MI300X) PMC Counter Set for LAMMPS ReaxFF/HNS

Reference for `counters.json` in this directory: which hardware performance
counters (PMCs) we collect with `rocprofv3`, why, how they are grouped into
passes, and how the raw counts become metrics.

| | |
|---|---|
| GPU | AMD Instinct MI300X (`gfx942`), partition `mi3001x` |
| ROCm | 7.2.0 (`module load rocm/7.2.0`) |
| Counter list checked against | `/work1/sinclair/sairajatg/workloads/mi300x_rocm72_counters.txt` (`rocprofv3 --list-avail`, 479 counters, `gfx942` only) |
| Config file | `/work1/sinclair/sairajatg/workloads/lammps/counters.json` |
| Run script | `/work1/sinclair/sairajatg/workloads/lammps/hns-mi300x-pmc.sbatch` (16×8×12, 466,944 atoms) |
| Validation | All counter names in `counters.json` exist in the gfx942 list (0 missing) |
| Size | 6 passes, 6 LAMMPS runs |

---

## 1. Findings from checking the counter list

1. **The first counter list was for the wrong GPU.** The first
   `mi300x_rocm72_counters.txt` reported `gfx90a` on GPUs 0 and 1 (MI210/MI250).
   It was produced by running `salloc` / `srun --pty` inside a script: those
   commands open interactive shells, so the remaining lines ran on the login
   node after the allocation ended. It was regenerated with an sbatch job on
   `mi3001x` and now reports only `gfx942`.
2. **`rocminfo | grep -m1 "Marketing Name"` prints the CPU.** `rocminfo` lists
   the CPU agent first (AMD EPYC 9684X on these nodes). Use the `gfx` name line,
   or filter for `Device Type: GPU`, to identify the GPU.
3. **The old memory counter names do not exist on gfx942.** The earlier
   `utilization_profiling/lammps/counters_memory.json` uses `TCC_EA_*_sum`.
   On gfx942 these are named `TCC_EA0_*_sum`. The old file would make rocprofv3
   fail on MI300X.
4. **gfx942 has counters the old files did not use:** `SQ_INSTS_VALU_MFMA_BF16`,
   `SQ_INSTS_VALU_MFMA_F8`, their `_MOPS_` versions, and the full set of
   FP64/FP32/FP16 vector (VALU) counters.
5. **The old `counters.json` only covered MFMA.** For LAMMPS, the vector
   counters matter more: ReaxFF runs almost entirely as ordinary FP64 vector
   instructions, not matrix-core (MFMA) instructions.
6. **`GRBM_GUI_ACTIVE` and `GRBM_COUNT` are per XCC.** MI300X has 8 XCCs
   (chiplets) and rocprofv3 sums the counter over them, so the CSV value is
   8 × the GPU's busy cycles. The formulas below divide by 8 (confirmed with
   job 433875: 1815 MHz).
7. **Counter descriptions are not reliable about quad-cycle units.** Each unit
   factor was checked against the job 433875 data (see Section 6).

---

## 2. Design decisions (what was cut, and why)

The set was reduced from 12 passes to 6.

### 2.1 MFMA: only the FP64 and FP32 MOPS counters are kept

FLOPs only need the **MOPS** counters: `FLOPs = 512 × SQ_INSTS_VALU_MFMA_MOPS_<type>`.
The MFMA **instruction** counters (`SQ_INSTS_VALU_MFMA_<type>`) count
instructions, not work, and add nothing to a FLOP count. They were dropped, as
were the separate MFMA passes.

LAMMPS runs in FP64 (and possibly some FP32), so only `MOPS_F64` and `MOPS_F32`
are collected. They sit in the same pass as the matching vector counters, so
vector and matrix FLOPs for a precision come from one run.

`SQ_VALU_MFMA_BUSY_CYCLES` is kept as a check for any other MFMA use. If it is
> 0 while `MOPS_F64` and `MOPS_F32` are both 0, a low-precision MFMA
(F16/BF16/I8/F8) ran, and those MOPS counters would need their own pass.

Dropped: `SQ_INSTS_VALU_MFMA_{F64,F32,F16,BF16,I8,F8}`,
`SQ_INSTS_VALU_MFMA_MOPS_{F16,BF16,I8,F8}`, `SQ_INSTS_MFMA`.

### 2.2 MFMA utilisation: from FLOPs vs. from busy cycles

Yes, MFMA utilisation can be estimated from FLOPs:

```
MFMA % of peak = (FP64_MFMA_FLOP / kernel_duration) / peak FP64 MFMA FLOP/s
```

The two ways of measuring it answer different questions:

| Method | Formula | What it measures |
|---|---|---|
| FLOP-based | `MFMA_FLOP / duration / peak` | Useful matrix work as a share of the datasheet peak |
| Busy-cycle based (`MfmaUtil`) | `SQ_VALU_MFMA_BUSY_CYCLES / (cycles × 1216 SIMDs)` | Share of time the matrix units were occupied |

Notes on the FLOP-based version:
- **Kernel duration** comes from `<pass>_kernel_trace.csv` (end − start, ns).
- **The peak assumes the 2100 MHz boost clock.** The power run averaged about
  1690 MHz under load, so a FLOP-based % against the datasheet peak includes
  the clock drop. For a clock-adjusted %, scale the peak by
  `measured clock / 2100`. The effective clock of each kernel is
  `cycles / duration`.
- The same method gives vector (VALU) utilisation: `FP64_VALU_FLOP / duration / 81.7 TFLOP/s`.

### 2.3 Memory: HBM reads and writes only

For HBM bandwidth and arithmetic intensity, the `TCC_EA0_*` counters (L2 ↔
memory interface) are enough: read and write requests, request sizes (for
bytes), and uncached traffic. Everything else on the memory side was dropped:

- **L1 → L2 traffic** (`TCP_TCC_*`, `TCP_TOTAL_CACHE_ACCESSES_sum`): only needed
  for an L2-level analysis.
- **L2 hit rate and DRAM-vs-remote split** (old `08_l2_dram`: `TCC_HIT_sum`,
  `TCC_MISS_sum`, `TCC_EA0_RDREQ_DRAM_sum`, `TCC_EA0_WRREQ_DRAM_sum`). On a
  single GPU with device memory, EA requests are effectively HBM requests.
- **Stalls, L2 atomics and LDS** (old `09_mem_stalls_lds`:
  `TCP_TCP_TA_DATA_STALL_CYCLES_sum`, `TCP_PENDING_STALL_CYCLES_sum`,
  `TCC_EA0_RDREQ_DRAM_CREDIT_STALL_sum`, `TCC_ATOMIC_sum`, `SQ_LDS_IDX_ACTIVE`,
  `SQ_LDS_BANK_CONFLICT`).
- Also dropped earlier: `TCC_READ_sum`, `TCC_WRITE_sum`, `TCC_TAG_STALL_sum`.

Kept or added: `TCC_EA0_RDREQ_sum`, `TCC_EA0_RDREQ_32B_sum`, `TCC_BUBBLE_sum`,
`TCC_EA0_RD_UNCACHED_32B_sum`, `TCC_EA0_WRREQ_sum`, `TCC_EA0_WRREQ_64B_sum`,
`TCC_EA0_WR_UNCACHED_32B_sum`, `TCC_EA0_ATOMIC_sum`.

---

## 3. File layout

`counters.json` has five sections:

1. **Header** (`arch`, `gpu`, `rocm`, `source`, `notes`): which GPU and ROCm
   version the names were checked against, plus unit conventions.
2. **`passes`**: 6 counter groups. Each has a `name` (numbered so output
   folders sort in order), a one-line `label` describing what it measures, a
   `category`, and its `counters`. Each pass is one `rocprofv3 --pmc ...` run,
   which is one full LAMMPS run.
3. **`derived`**: formulas that turn raw counts into metrics, applied in
   post-processing.
4. **`peak_MI300X`**: datasheet peaks used for % of peak.

`counters.json` is our own format, not rocprofv3's `-i` input format. The run
script loops over `passes`, prints each pass's label, writes it to
`<pass>/LABEL`, and calls `rocprofv3 --pmc <counters>` once per pass.

---

## 4. Why the counters are split into passes

Each hardware block has a fixed number of counter registers. rocprofv3 fails
if one pass asks for more than a block can provide
(*"job will fail if entire set of counters cannot be collected in single pass"*).
Each pass is kept within these limits:

| Block | What it is | Max per pass |
|---|---|---|
| `SQ` | Instruction issue and wave scheduling in each compute unit (CU) | 8 |
| `TCC` | L2 cache and its link to HBM | 4 |
| `TCP` | L1 vector cache | 4 |
| `GRBM` | Global busy/idle counters | 2 |

These are the usual gfx942 limits, but only a real run proves a pass is
collectable. Passes 1–4 use exactly 8 SQ counters, so if a pass fails it is
most likely one of these. The script reports a failed pass and moves on to the
next one.

`GRBM_GUI_ACTIVE` (cycles the GPU is busy) is in every pass. Each pass is a
separate run, so each needs its own denominator to turn counts into rates and
percentages.

### Pass summary

| # | Pass | Label | Counters | Blocks used |
|---|---|---|---|---|
| 1 | `01_fp64_flops` | FP64 FLOPs: vector (VALU) + matrix (MFMA) FP64 ops, MFMA busy | 9 | SQ 8, GRBM 1 |
| 2 | `02_fp32_int_ops` | FP32 FLOPs (VALU + MFMA) and integer / conversion ops | 9 | SQ 8, GRBM 1 |
| 3 | `03_fp16_instmix` | FP16 vector ops + instruction mix (scalar, memory, LDS) | 9 | SQ 8, GRBM 1 |
| 4 | `04_cu_util` | CU utilization: GPU/CU/VALU busy, thread divergence, wave residency | 10 | SQ 8, GRBM 2 |
| 5 | `05_hbm_read` | HBM reads: L2 → memory read requests and bytes, incl. uncached | 5 | TCC 4, GRBM 1 |
| 6 | `06_hbm_write` | HBM writes: L2 → memory write requests and bytes, incl. uncached and atomics | 5 | TCC 4, GRBM 1 |

The 16×8×12 run has a loop time of about 5.3 s unprofiled, so 6 passes are
cheap. Profiling adds overhead because kernels are serialized during counter
collection.

---

## 5. The passes in detail

### 5.1 Compute, VALU and MFMA: passes 1–3

#### Pass 1: `01_fp64_flops`
*FP64 FLOPs: vector (VALU) + matrix (MFMA) FP64 ops, MFMA busy*

| Counter | Meaning |
|---|---|
| `SQ_INSTS_VALU_ADD_F64` | FP64 add instructions |
| `SQ_INSTS_VALU_MUL_F64` | FP64 multiply instructions |
| `SQ_INSTS_VALU_FMA_F64` | FP64 fused multiply-add instructions (2 FLOPs each) |
| `SQ_INSTS_VALU_TRANS_F64` | FP64 exp, log, sqrt, reciprocal. ReaxFF bond-order and Coulomb terms use many of these. AMD's built-in `TOTAL_64_OPS` leaves them out; we include them. |
| `SQ_INSTS_VALU_MFMA_MOPS_F64` | FP64 matrix operations; 1 unit = 512 FLOPs |
| `SQ_VALU_MFMA_BUSY_CYCLES` | Cycles the matrix units are busy (per SIMD). Gives `MfmaUtil` and catches any MFMA use. |
| `SQ_INSTS_VALU` | All vector instructions; the total the others are compared against |
| `SQ_WAVES` | Waves (groups of 64 threads) launched; used for per-wave counts |
| `GRBM_GUI_ACTIVE` | GPU busy cycles (× 8 XCCs) |

#### Pass 2: `02_fp32_int_ops`
*FP32 FLOPs (VALU + MFMA) and integer / conversion ops*

| Counter | Meaning |
|---|---|
| `SQ_INSTS_VALU_ADD_F32` / `_MUL_F32` / `_FMA_F32` / `_TRANS_F32` | FP32 vector instructions; show whether any kernel runs in single precision |
| `SQ_INSTS_VALU_MFMA_MOPS_F32` | FP32 matrix operations; 1 unit = 512 FLOPs |
| `SQ_INSTS_VALU_INT32` | 32-bit integer operations, mostly neighbour-list indexing |
| `SQ_INSTS_VALU_INT64` | 64-bit integer operations (address and index arithmetic) |
| `SQ_INSTS_VALU_CVT` | Type conversions |
| `GRBM_GUI_ACTIVE` | GPU busy cycles (× 8 XCCs) |

#### Pass 3: `03_fp16_instmix`
*FP16 vector ops + instruction mix (scalar, memory, LDS)*

| Counter | Meaning |
|---|---|
| `SQ_INSTS_VALU_ADD_F16` / `_MUL_F16` / `_FMA_F16` / `_TRANS_F16` | FP16 vector instructions (expected ≈ 0 for LAMMPS; included for completeness) |
| `SQ_INSTS_SALU` | Scalar instructions (control flow, per-wave constants) |
| `SQ_INSTS_VMEM_RD` | Vector memory load instructions |
| `SQ_INSTS_VMEM_WR` | Vector memory store instructions |
| `SQ_INSTS_LDS` | Shared-memory (LDS) instructions |
| `GRBM_GUI_ACTIVE` | GPU busy cycles (× 8 XCCs) |

Passes 1–3 together give the instruction mix: the share of compute, memory,
scalar and LDS instructions.

### 5.2 CU utilisation: pass 4

#### Pass 4: `04_cu_util`
*CU utilization: GPU/CU/VALU busy, thread divergence, wave residency*

| Counter | Meaning |
|---|---|
| `GRBM_GUI_ACTIVE` | GPU busy cycles (× 8 XCCs) |
| `GRBM_COUNT` | Total cycles (× 8 XCCs); with the above gives the share of time the GPU is busy |
| `SQ_BUSY_CU_CYCLES` | How busy the CUs are, i.e. whether all 304 CUs have work. Units of 4 cycles (quad-cycles). |
| `SQ_BUSY_CYCLES` | Cycles with at least one wave running on a shader engine |
| `SQ_WAVE_CYCLES` | Total cycles waves are resident on CUs; used for occupancy. Units of 4 cycles. |
| `SQ_ACTIVE_INST_VALU` | Cycles issuing vector instructions (VALU busy) |
| `SQ_THREAD_CYCLES_VALU` | Same, weighted by active threads out of 64. Divided by `SQ_ACTIVE_INST_VALU × 64` it measures thread divergence, which ReaxFF's many branches will show. |
| `SQ_ACTIVE_INST_ANY` | Cycles issuing any instruction |
| `SQ_INST_CYCLES_SALU` | Scalar busy cycles |
| `SQ_WAVES` | Waves launched |

#### What occupancy and thread divergence are, and why pass 4 is needed

**Occupancy (wave residency).** Each CU runs *waves*: groups of 64 threads
that execute the same instruction together. A CU holds up to 32 waves at once
(8 per SIMD × 4 SIMDs); waves loaded on a CU are *resident*. Occupancy is the
average number of resident waves per CU as a share of 32. It is limited by
registers and LDS used per wave (a register-heavy kernel fits fewer waves) and
by how many waves the kernel launches.

It matters because the GPU hides memory latency by switching between waves.
An HBM load takes hundreds of cycles; while one wave waits, the CU runs
another that is ready. With many resident waves there is usually one ready;
with 2 or 3, all may be waiting and the CU idles even though it has work.

`OccupancyPercent` is computed from `SQ_WAVE_CYCLES` in this pass. The
separate `MeanOccupancyPerCU` pass (built on `SQ_LEVEL_WAVES`) measures the
same thing by a different method and was dropped to save a run.

**Thread divergence.** The 64 threads of a wave share one instruction stream.
When an `if` splits them, the wave runs both branches one after the other,
with the threads not on the current branch switched off. The vector unit is
busy, but only some of its 64 lanes do useful work.
`VALUUtilization = SQ_THREAD_CYCLES_VALU / (SQ_ACTIVE_INST_VALU × 64)` is the
share of lanes active: 100% means no divergence.

**Why they are needed.** The FLOP and HBM passes show how far a kernel is from
peak compute and peak bandwidth, not why. Pass 4 separates the causes:

| What pass 4 shows | Reading |
|---|---|
| Vector units busy (high `VALUBusy`), low `VALUUtilization` | **Divergence.** Units busy but most lanes switched off; work is wasted inside each wave. |
| Vector units idle, low occupancy | **Latency-bound.** Too few waves to hide memory waits; neither compute nor bandwidth saturated. |
| Vector units idle, high occupancy, bandwidth near peak | **Bandwidth-bound** (the usual roofline case) |
| CUs idle (low `CU_busy_frac`) | **Not enough work** to fill 304 CUs: small kernels or launch gaps |

ReaxFF is likely to show the first two: bond, angle and torsion terms check
distance cutoffs and bond-order thresholds per atom pair (divergence), and the
Kokkos ReaxFF kernels are large and register-heavy (low occupancy). Both also
help explain power: a vector unit with most lanes off, or a CU idle waiting on
memory, draws less power than a fully used one.

### 5.3 Memory: passes 5–6

All memory traffic is measured at the L2 ↔ memory interface (`TCC_EA0`).

#### Pass 5: `05_hbm_read`
*HBM reads: L2 → memory read requests and bytes, incl. uncached*

| Counter | Meaning |
|---|---|
| `TCC_EA0_RDREQ_sum` | All L2 read requests to memory (32 B, 64 B or 128 B) |
| `TCC_EA0_RDREQ_32B_sum` | 32-byte read requests |
| `TCC_BUBBLE_sum` | 128-byte read requests |
| `TCC_EA0_RD_UNCACHED_32B_sum` | Uncached read traffic, in 32-byte units (a 64-byte request counts as 2). A subset of the reads above. |
| `GRBM_GUI_ACTIVE` | GPU busy cycles (× 8 XCCs) |

The first three are the inputs to AMD's `FETCH_SIZE` (bytes read).

#### Pass 6: `06_hbm_write`
*HBM writes: L2 → memory write requests and bytes, incl. uncached and atomics*

| Counter | Meaning |
|---|---|
| `TCC_EA0_WRREQ_sum` | All L2 write requests to memory (32 B or 64 B). Atomics sent to memory travel on the same interface and are counted here. |
| `TCC_EA0_WRREQ_64B_sum` | 64-byte write requests |
| `TCC_EA0_WR_UNCACHED_32B_sum` | Uncached write/atomic traffic, in 32-byte units. A subset of the writes above. |
| `TCC_EA0_ATOMIC_sum` | The part of `TCC_EA0_WRREQ_sum` that is atomics sent to memory |
| `GRBM_GUI_ACTIVE` | GPU busy cycles (× 8 XCCs) |

`TCC_EA0_WRREQ_sum` and `TCC_EA0_WRREQ_64B_sum` are the inputs to AMD's
`WRITE_SIZE` (bytes written). Reads and writes are in separate passes because
together they need 8 TCC counters, over the limit of 4.

---

## 6. Derived metrics

The full explanation of every formula (what it measures, why it is written that
way, and its whole-run value from job 433875) is in
**`post_processing_formulae.md`**. This section is the summary; the formulas
match `derived` in `counters.json`.

**MI300X constants:** 8 XCCs, 304 CUs, 1216 SIMDs, 64 threads per wave,
32 max waves per CU, 512 registers per SIMD thread slot (VGPR + AGPR), 64 KB LDS
per CU, 1 MFMA MOPS = 512 FLOPs, peak engine clock 2100 MHz.
`duration_ns` is each launch's `End_Timestamp − Start_Timestamp`.

**How passes are combined:** counters and durations are summed per kernel name
within a pass; passes are combined only through per-kernel rates (per ns) or
per-launch averages. Launch IDs do not match across passes, because the QEq
solver's iteration count varies by ~0.5%.

### Unit factors checked against the data (job 433875)

| Counter | Factor | Evidence |
|---|---|---|
| `GRBM_GUI_ACTIVE` | ÷ 8 (summed over 8 XCCs) | Gives 1815 MHz; without it ~14,500 MHz |
| `SQ_WAVE_CYCLES` | × 4 (quad-cycles) | Every top kernel lands just under its register limit on waves per CU |
| `SQ_BUSY_CU_CYCLES` | none | With × 4, CUs would be 299% busy |
| `SQ_ACTIVE_INST_VALU` | none (one quad-cycle per wave instruction) | ≈ `SQ_INSTS_VALU` (7.21e11 vs 6.98e11) |
| `SQ_INST_CYCLES_SALU` | none, unverified | Treated as a lower bound |

### Cycles and utilisation

| Metric | Formula | Meaning |
|---|---|---|
| `cycles` | `GRBM_GUI_ACTIVE / 8` | GPU busy cycles during the kernel |
| `eff_clock_MHz` | `cycles / duration_ns * 1000` | Effective engine clock |
| `clock_factor` | `eff_clock_MHz / 2100` | Scales datasheet peaks to the measured clock |
| `GPU_busy_pct` | `100 * union(kernel intervals) / (last end − first start)` | Share of wall time the GPU runs kernels, **from the power run's kernel trace** |
| `CU_busy_frac` | `SQ_BUSY_CU_CYCLES / (cycles * 304)` | Share of CU-cycles with work |
| `VALUBusy_pct` | `100 * SQ_ACTIVE_INST_VALU / (cycles * 304)` | Share of time the vector units execute |
| `VALU_threads` | `SQ_THREAD_CYCLES_VALU / SQ_ACTIVE_INST_VALU` | Threads active per vector instruction (0–64) |
| `VALUUtilization_pct` | `100 * VALU_threads / 64` | 100% = no divergence |
| `waves_per_CU` | `4 * SQ_WAVE_CYCLES / (cycles * 304)` | Mean resident waves per CU |
| `Occupancy_pct` | `100 * waves_per_CU / 32` | Occupancy as % of the hardware maximum |
| `max_waves_CU` | `min(4 * min(8, floor(512 / vgpr_alloc)), LDS limit, 32)` | Kernel's own maximum from registers and LDS |
| `Occupancy_of_max_pct` | `100 * waves_per_CU / max_waves_CU` | Occupancy as % of what the kernel can reach |
| `Wave_issue_pct` | `100 * SQ_ACTIVE_INST_ANY / SQ_WAVE_CYCLES` | Share of a wave's resident time spent issuing |
| `SALUBusy_pct` | `100 * SQ_INST_CYCLES_SALU / (cycles * 304)` | Scalar unit busy (lower bound) |
| `MfmaUtil_pct` | `100 * SQ_VALU_MFMA_BUSY_CYCLES / (cycles * 1216)` | Matrix units occupied |

`GPU_UTIL_pct = GRBM_GUI_ACTIVE / GRBM_COUNT` was **dropped**: in per-kernel
counter mode both count only while the kernel runs, so it is always 100%.

### FLOPs and % of peak

Instruction counter names are shortened (`ADD_F64` = `SQ_INSTS_VALU_ADD_F64`,
`MFMA_MOPS_F64` = `SQ_INSTS_VALU_MFMA_MOPS_F64`, and so on).

| Metric | Formula | Meaning |
|---|---|---|
| `FP64_VALU_FLOP` | `64 * (ADD_F64 + MUL_F64 + 2*FMA_F64 + TRANS_F64)` | FP64 vector FLOPs, **issued** (all 64 lanes) |
| `FP64_TRANS_FLOP` | `64 * TRANS_F64` | Transcendental part, reported separately |
| `FP64_VALU_FLOP_useful` | per kernel: `FP64_VALU_FLOP * VALUUtilization_pct / 100` | Estimate of FLOPs on active lanes (pass 1 × pass 4) |
| `FP64_MFMA_FLOP` | `512 * MFMA_MOPS_F64` | FP64 matrix FLOPs |
| `FP64_FLOP` | `FP64_VALU_FLOP + FP64_MFMA_FLOP` | All FP64 FLOPs |
| `FP32_VALU_FLOP` | `64 * (ADD_F32 + MUL_F32 + 2*FMA_F32 + TRANS_F32)` | FP32 vector FLOPs |
| `FP32_MFMA_FLOP` | `512 * MFMA_MOPS_F32` | FP32 matrix FLOPs |
| `FP32_FLOP` | `FP32_VALU_FLOP + FP32_MFMA_FLOP` | All FP32 FLOPs |
| `FP16_VALU_FLOP` | `64 * (ADD_F16 + MUL_F16 + 2*FMA_F16 + TRANS_F16)` | FP16 vector FLOPs |
| `TFLOPs` | `FLOP / duration_ns / 1000` | Throughput |
| `FP64_VALU_peak_pct` | `100 * TFLOPs_FP64_VALU / 81.7` | % of peak at the 2100 MHz boost clock |
| `FP64_VALU_peak_clk_pct` | `100 * TFLOPs_FP64_VALU / (81.7 * clock_factor)` | % of peak at the measured clock |
| `FP64_MFMA_peak_pct` | `100 * TFLOPs_FP64_MFMA / 163.4` | FLOP-based MFMA utilisation |
| `FP32_MFMA_peak_pct` | `100 * TFLOPs_FP32_MFMA / 163.4` | FP32 matrix % of peak |
| `FP64_share_pct` | `100 * (ADD_F64 + MUL_F64 + FMA_F64 + TRANS_F64) / SQ_INSTS_VALU` | FP64 share of vector instructions |
| `inst_share_pct` | `100 * SQ_INSTS_X / (VALU + SALU + VMEM_RD + VMEM_WR + LDS)` | Instruction mix (pass 1 × pass 3, via rates) |

### Memory (L2 → memory, before the Infinity Cache)

The `TCC_EA0` counters sit on the L2 side of MI300X's 256 MB Infinity Cache, so
these bytes are an **upper bound on HBM traffic**. Metrics are named `mem_*`,
not `HBM_*`.

| Metric | Formula | Meaning |
|---|---|---|
| `mem_read_bytes` | `128*TCC_BUBBLE_sum + 64*(TCC_EA0_RDREQ_sum - TCC_BUBBLE_sum - TCC_EA0_RDREQ_32B_sum) + 32*TCC_EA0_RDREQ_32B_sum` | Bytes read (AMD's `FETCH_SIZE`, in bytes) |
| `read_128B_pct`, `read_32B_pct` | `100 * TCC_BUBBLE_sum / TCC_EA0_RDREQ_sum`, `100 * TCC_EA0_RDREQ_32B_sum / TCC_EA0_RDREQ_sum` | Read request size mix |
| `mem_write_bytes` | `32*(TCC_EA0_WRREQ_sum - TCC_EA0_WRREQ_64B_sum) + 64*TCC_EA0_WRREQ_64B_sum` | Bytes written, atomics included (AMD's `WRITE_SIZE`, in bytes) |
| `atomic_req_pct` | `100 * TCC_EA0_ATOMIC_sum / TCC_EA0_WRREQ_sum` | Share of write requests that are atomics |
| `atomic_bytes` | `32 * TCC_EA0_ATOMIC_sum` | Bytes moved by atomics |
| `uncached_read_bytes` | `32 * TCC_EA0_RD_UNCACHED_32B_sum` | Uncached bytes read (part of reads) |
| `uncached_write_bytes` | `32 * TCC_EA0_WR_UNCACHED_32B_sum` | Uncached bytes written (part of writes) |
| `read_BW_TBps` | `mem_read_bytes / duration_ns[pass 5] / 1000` | Read bandwidth, pass 5's own durations |
| `write_BW_TBps` | `mem_write_bytes / duration_ns[pass 6] / 1000` | Write bandwidth, pass 6's own durations |
| `mem_BW_TBps` | `read_BW_TBps + write_BW_TBps` | Total bandwidth |
| `mem_BW_peak_pct` | `100 * mem_BW_TBps / 5.3` | % of HBM peak (upper bound on HBM use) |
| `AI_FP64` | `(FP64_FLOP / duration_ns[1]) / (mem_read_bytes / duration_ns[5] + mem_write_bytes / duration_ns[6])` | FP64 FLOPs per byte, as a ratio of rates |
| `ridge_FP64` | `81.7 / 5.3` (15.4 FLOP/byte) | Roofline ridge point |
| `attainable_TFLOPs` | `min(81.7, AI_FP64 * 5.3)` | Roofline limit at this intensity |
| `roofline_pct` | `100 * TFLOPs_FP64 / attainable_TFLOPs` | How close the kernel gets to its roofline limit |

### MI300X peaks (`peak_MI300X` in `counters.json`)

| Peak | Value | Derivation |
|---|---|---|
| Engine clock | 2100 MHz | Datasheet boost clock |
| FP64 vector (VALU) | 81.7 TFLOP/s | 304 CUs × 128 FLOP/clk × 2.1 GHz |
| FP32 vector (VALU) | 163.4 TFLOP/s | Assumes packed FP32; unpacked FP32 is 81.7 |
| FP64 matrix (MFMA) | 163.4 TFLOP/s | 304 CUs × 256 FLOP/clk × 2.1 GHz |
| FP32 matrix (MFMA) | 163.4 TFLOP/s | 304 CUs × 256 FLOP/clk × 2.1 GHz |
| HBM3 bandwidth | 5.3 TB/s | Datasheet |

For clock-adjusted peaks, multiply by `clock_factor`.

---

## 7. Caveats

- **Formulas that combine passes assume the runs behave the same.** Checked for
  job 433875: thermo output matches to 7–8 significant figures and total kernel
  time is 5.076–5.101 s in every pass. QEq launch counts differ by ~0.5%, hence
  per-kernel rates rather than raw sums.
- **Rates use profiled kernel durations.** rocprofv3 serializes kernels during
  counter collection, so each duration is that kernel's own runtime. Wall-clock
  behaviour and GPU busy % come from the power run.
- **Memory is measured before the Infinity Cache.** Bytes and bandwidth are
  upper bounds on HBM traffic; arithmetic intensity is a lower bound for HBM.
- **Issued vs useful FLOPs.** Issued FLOPs count all 64 lanes; useful FLOPs are
  an estimate that applies pass 4's thread utilisation to pass 1.
- **Low-precision MFMA is not counted.** If `SQ_VALU_MFMA_BUSY_CYCLES` > 0 while
  `MOPS_F64` and `MOPS_F32` are 0, add a pass with
  `SQ_INSTS_VALU_MFMA_MOPS_{F16,BF16,I8,F8}`. In job 433875 it is 0.
- **Job 433875, pass 2 lost counters for 66 launches** (IDs 65556–65621,
  1.85 ms, 0.036% of GPU time). FP32/INT totals are low by a negligible amount.

---

## 8. Commands used

Regenerate the counter list on an MI300X node (sbatch; do not put
`salloc`/`srun --pty` in a script):

```bash
#!/usr/bin/bash
#SBATCH --job-name=list-ctrs-mi300x
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=/work1/sinclair/sairajatg/workloads/list-ctrs-%j.log

module load rocm/7.2.0
echo "host: $(hostname)"
rocminfo | grep -m1 "  Name:  *gfx"       # expect gfx942
rocprofv3 --list-avail > /work1/sinclair/sairajatg/workloads/mi300x_rocm72_counters.txt 2>&1
```

Check the architecture in the counter list:

```bash
grep -E "^Name" /work1/sinclair/sairajatg/workloads/mi300x_rocm72_counters.txt | sort -u   # only gfx942
```

Look up a counter's description or expression:

```bash
grep -A3 "Counter_Name *:\s*FETCH_SIZE$" /work1/sinclair/sairajatg/workloads/mi300x_rocm72_counters.txt
```

Run all passes on 16×8×12:

```bash
cd /work1/sinclair/sairajatg/workloads/lammps
sbatch hns-mi300x-pmc.sbatch
```

Each pass runs (from the script):

```bash
srun -n 1 rocprofv3 --pmc <pass counters> --kernel-trace --output-format csv \
     -d results/pmc-<jobid>/<pass> -o <pass> \
     -- lmp <kokkos flags> -v x 16 -v y 8 -v z 12 -in in.reaxc.hns -nocite -log <pass>/log.<pass>
```

Output per pass in `results/pmc-<jobid>/<pass>/`: `LABEL`,
`<pass>_counter_collection.csv`, `<pass>_kernel_trace.csv`, `log.<pass>`,
`rocprof.<pass>.txt`.
