# Runbook: hns16812

LAMMPS ReaxFF/HNS 16x8x12 (466,944 atoms, 100 steps), KOKKOS HIP double precision, 1x MI300X

| | |
|---|---|
| GPUs | 1 x MI300X, partition `mi3001x`, 1 srun task(s), sampled GPUs `0` |
| Config | `/home1/sairajatg/.claude/skills/mi300x-workload-profiling/examples/lammps_hns16812.conf` |
| Scripts | `/work1/sinclair/sairajatg/workloads/lammps/profiling` |
| Results | `/work1/sinclair/sairajatg/workloads/lammps/results` (`plain-<jobid>`, `pmc-<jobid>`, `prof-<jobid>`) |

Run the steps in order. Only step 4 needs step 2 (a valid counter set); steps
3 and 5 do not, and 3, 4, 5 can run at the same time. Running the plain run
first is safer: it shows the app works before the long PMC job. After each job,
check the item listed and send the job ID (or the tail of its `.out` file) back.

All commands assume:
```bash
cd /work1/sinclair/sairajatg/workloads/lammps/profiling
```

## 0. Check the config (login node, no GPU)

```bash
DRY_RUN=1 bash run_plain.sbatch /home1/sairajatg/.claude/skills/mi300x-workload-profiling/examples/lammps_hns16812.conf
DRY_RUN=1 bash run_pmc.sbatch   /home1/sairajatg/.claude/skills/mi300x-workload-profiling/examples/lammps_hns16812.conf
DRY_RUN=1 bash run_power.sbatch /home1/sairajatg/.claude/skills/mi300x-workload-profiling/examples/lammps_hns16812.conf
```
Check: each prints `[dry-run] ... srun -n 1 ...` with the right binary and
arguments and no `Error:` line.

## 1. Build (only if the binary is not built yet)

Build for gfx942 with ROCm 7.2 on a GPU node (see Phase 1 of the skill).
Check: every path in `REQUIRED_FILES` exists.

## 2. Counter list for this GPU, and check counters.json

```bash
sbatch -p mi3001x list_counters.sbatch /work1/sinclair/sairajatg/workloads/lammps/profiling/counters_list_gfx942.txt
# when it finishes:
grep -E "^Name" /work1/sinclair/sairajatg/workloads/lammps/profiling/counters_list_gfx942.txt | sort -u          # must print only gfx942
python3 check_counters.py counters.json /work1/sinclair/sairajatg/workloads/lammps/profiling/counters_list_gfx942.txt
```
Check: only `gfx942`; the checker prints `OK`. (Skip this step if a valid
list for this ROCm version already exists.)

## 3. Plain run

```bash
sbatch -p mi3001x -J hns16812-plain run_plain.sbatch /home1/sairajatg/.claude/skills/mi300x-workload-profiling/examples/lammps_hns16812.conf
```
Check `hns16812-plain-<jobid>.out`: `exit status: 0`, the app's timing line
(`Loop time of`), and correct results in `/work1/sinclair/sairajatg/workloads/lammps/results/plain-<jobid>/`.
Note the wall time: step 4 needs about (number of passes) x 2.5-4 x that.

## 4. PMC profiling (rocprofv3, one run per pass)

```bash
sbatch -p mi3001x -J hns16812-pmc -t 01:00:00 run_pmc.sbatch /home1/sairajatg/.claude/skills/mi300x-workload-profiling/examples/lammps_hns16812.conf
```
Raise `-t` if the plain run is long. Check `hns16812-pmc-<jobid>.out`: every
pass `rc=0` with counter CSV file(s), and the app's timing similar across passes.

## 5. Power, frequency and temperature (rocprofwrap_lt + kernel trace)

```bash
sbatch -p mi3001x -J hns16812-power run_power.sbatch /home1/sairajatg/.claude/skills/mi300x-workload-profiling/examples/lammps_hns16812.conf
```
Check `hns16812-power-<jobid>.out`: `exit status: 0`, kernel launches traced,
and the clock-alignment lines (first sample and first kernel both a few seconds
after the same `clock_ref` value).

## 6. Post-process (login node, no GPU)

Fill in the job IDs from steps 4 and 5:
```bash
PMC=/work1/sinclair/sairajatg/workloads/lammps/results/pmc-<jobid-step4>
PROF=/work1/sinclair/sairajatg/workloads/lammps/results/prof-<jobid-step5>
python3 postprocess_pmc.py $PMC --power-ktrace $PROF/ktrace_hns16812
python3 build_sampling_json.py $PROF --label "<display name>"
```
For an app that does not print a LAMMPS-style `Loop time of X`, add
`--perf-regex '<regex with one group capturing seconds>'` to the first command.
For 8 GPUs add `--per-gpu` to the second to get per-GPU vectors too.

## 7. Outputs

| File | Contents |
|---|---|
| `$PMC/analysis/per_kernel_metrics.csv` | every kernel + TOTAL row, all metrics |
| `$PMC/analysis/summary.json` | whole-run metrics, data checks, warnings |
| `$PMC/analysis/report.md` | readable summary and top kernels |
| `$PROF/sampling_vectors.json` | inst_power, socket_power, gfx_frequency, hotspot_temp vectors |
| `$PMC/<pass>/*_counter_collection.csv` | raw counters per kernel launch |
| `$PROF/profiling_result_hns16812_<gpu>.csv` | raw power / clock / temperature samples |

Check: the `Warnings` list at the end of `report.md` says `none`; if not, report them before
using the numbers.

## Optional: submit 3-5 as one chain

```bash
j1=$(sbatch --parsable -p mi3001x -J hns16812-plain run_plain.sbatch /home1/sairajatg/.claude/skills/mi300x-workload-profiling/examples/lammps_hns16812.conf)
j2=$(sbatch --parsable -p mi3001x -J hns16812-pmc -t 01:00:00 --dependency=afterok:$j1 run_pmc.sbatch /home1/sairajatg/.claude/skills/mi300x-workload-profiling/examples/lammps_hns16812.conf)
j3=$(sbatch --parsable -p mi3001x -J hns16812-power --dependency=afterok:$j1 run_power.sbatch /home1/sairajatg/.claude/skills/mi300x-workload-profiling/examples/lammps_hns16812.conf)
echo "plain=$j1 pmc=$j2 power=$j3"
```
Steps 4 and 5 only start if the plain run succeeds.
