#!/bin/bash
# Write the ordered list of commands the user runs for one workload, filled in
# from its workload.conf (partition and -n from NGPUS/NTASKS, real paths).
#
# Usage: bash make_runbook.sh /path/<tag>.conf [out.md]
#        (default out: $PROFILING_DIR/RUNBOOK_<WORKLOAD>.md)

CONF=${1:?usage: bash make_runbook.sh <workload.conf> [out.md]}
[ -f "$CONF" ] || { echo "Error: no config file $CONF"; exit 1; }
CONF=$(readlink -f "$CONF")
source "$CONF"
NGPUS=${NGPUS:-1}; NTASKS=${NTASKS:-1}; SAMPLE_GPUS=${SAMPLE_GPUS:-0}
if [ "$NGPUS" -gt 1 ]; then PART=mi3008x; else PART=mi3001x; fi
NOPT=""; [ "$NTASKS" -gt 1 ] && NOPT=" -n $NTASKS"
P=$PROFILING_DIR
R=$RESULTS_DIR
LIST=$P/counters_list_gfx942.txt
PTRN=${PERF_GREP:-Time}
OUT=${2:-$P/RUNBOOK_${WORKLOAD}.md}

cat > "$OUT" <<EOF
# Runbook: $WORKLOAD

$WORKLOAD_DESC

| | |
|---|---|
| GPUs | $NGPUS x MI300X, partition \`$PART\`, $NTASKS srun task(s), sampled GPUs \`$SAMPLE_GPUS\` |
| Config | \`$CONF\` |
| Scripts | \`$P\` |
| Results | \`$R\` (\`plain-<jobid>\`, \`pmc-<jobid>\`, \`prof-<jobid>\`) |

Run the steps in order. Only step 4 needs step 2 (a valid counter set); steps
3 and 5 do not, and 3, 4, 5 can run at the same time. Running the plain run
first is safer: it shows the app works before the long PMC job. After each job,
check the item listed and send the job ID (or the tail of its \`.out\` file) back.

All commands assume:
\`\`\`bash
cd $P
\`\`\`

## 0. Check the config (login node, no GPU)

\`\`\`bash
DRY_RUN=1 bash run_plain.sbatch $CONF
DRY_RUN=1 bash run_pmc.sbatch   $CONF
DRY_RUN=1 bash run_power.sbatch $CONF
\`\`\`
Check: each prints \`[dry-run] ... srun -n $NTASKS ...\` with the right binary and
arguments and no \`Error:\` line.

## 1. Build (only if the binary is not built yet)

Build for gfx942 with ROCm 7.2 on a GPU node (see Phase 1 of the skill).
Check: every path in \`REQUIRED_FILES\` exists.

## 2. Counter list for this GPU, and check counters.json

\`\`\`bash
sbatch -p $PART list_counters.sbatch $LIST
# when it finishes:
grep -E "^Name" $LIST | sort -u          # must print only gfx942
python3 check_counters.py counters.json $LIST
\`\`\`
Check: only \`gfx942\`; the checker prints \`OK\`. (Skip this step if a valid
list for this ROCm version already exists.)

## 3. Plain run

\`\`\`bash
sbatch -p $PART$NOPT -J ${WORKLOAD}-plain run_plain.sbatch $CONF
\`\`\`
Check \`${WORKLOAD}-plain-<jobid>.out\`: \`exit status: 0\`, the app's timing line
(\`$PTRN\`), and correct results in \`$R/plain-<jobid>/\`.
Note the wall time: step 4 needs about (number of passes) x 2.5-4 x that.

## 4. PMC profiling (rocprofv3, one run per pass)

\`\`\`bash
sbatch -p $PART$NOPT -J ${WORKLOAD}-pmc -t 01:00:00 run_pmc.sbatch $CONF
\`\`\`
Raise \`-t\` if the plain run is long. Check \`${WORKLOAD}-pmc-<jobid>.out\`: every
pass \`rc=0\` with counter CSV file(s), and the app's timing similar across passes.

## 5. Power, frequency and temperature (rocprofwrap_lt + kernel trace)

\`\`\`bash
sbatch -p $PART$NOPT -J ${WORKLOAD}-power run_power.sbatch $CONF
\`\`\`
Check \`${WORKLOAD}-power-<jobid>.out\`: \`exit status: 0\`, kernel launches traced,
and the clock-alignment lines (first sample and first kernel both a few seconds
after the same \`clock_ref\` value).

## 6. Post-process (login node, no GPU)

Fill in the job IDs from steps 4 and 5:
\`\`\`bash
PMC=$R/pmc-<jobid-step4>
PROF=$R/prof-<jobid-step5>
python3 postprocess_pmc.py \$PMC --power-ktrace \$PROF/ktrace_$WORKLOAD
python3 build_sampling_json.py \$PROF --label "<display name>"
\`\`\`
For an app that does not print a LAMMPS-style \`Loop time of X\`, add
\`--perf-regex '<regex with one group capturing seconds>'\` to the first command.
For 8 GPUs add \`--per-gpu\` to the second to get per-GPU vectors too.

## 7. Outputs

| File | Contents |
|---|---|
| \`\$PMC/analysis/per_kernel_metrics.csv\` | every kernel + TOTAL row, all metrics |
| \`\$PMC/analysis/summary.json\` | whole-run metrics, data checks, warnings |
| \`\$PMC/analysis/report.md\` | readable summary and top kernels |
| \`\$PROF/sampling_vectors.json\` | inst_power, socket_power, gfx_frequency, hotspot_temp vectors |
| \`\$PMC/<pass>/*_counter_collection.csv\` | raw counters per kernel launch |
| \`\$PROF/profiling_result_${WORKLOAD}_<gpu>.csv\` | raw power / clock / temperature samples |

Check: the \`Warnings\` list at the end of \`report.md\` says \`none\`; if not, report them before
using the numbers.

## Optional: submit 3-5 as one chain

\`\`\`bash
j1=\$(sbatch --parsable -p $PART$NOPT -J ${WORKLOAD}-plain run_plain.sbatch $CONF)
j2=\$(sbatch --parsable -p $PART$NOPT -J ${WORKLOAD}-pmc -t 01:00:00 --dependency=afterok:\$j1 run_pmc.sbatch $CONF)
j3=\$(sbatch --parsable -p $PART$NOPT -J ${WORKLOAD}-power --dependency=afterok:\$j1 run_power.sbatch $CONF)
echo "plain=\$j1 pmc=\$j2 power=\$j3"
\`\`\`
Steps 4 and 5 only start if the plain run succeeds.
EOF
echo "wrote $OUT"
