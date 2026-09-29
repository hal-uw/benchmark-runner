#!/bin/bash
# Power, frequency and temperature sampling with rocprofwrap_lt (wrapper.py +
# amd-smi-query, ~1 ms requested) and a rocprofv3 kernel trace of the SAME run.
# No --pmc: without counters kernels are not serialized, so power/clock behave
# like an unprofiled run (overhead ~4% for LAMMPS).
#
# wrapper.py runs everything after "--" as the application, so rocprofv3 sits
# between the wrapper and the app:
#   wrapper.py -- srun rocprofv3 --kernel-trace -- <app>
#
# Clock alignment: a clock_ref line (CLOCK_BOOTTIME/MONOTONIC/REALTIME) is
# written right before launch. On MI300X/ROCm 7.2 both the sampler and rocprofv3
# timestamps are on the boot-time base, so kernels line up with samples directly;
# the summary prints the evidence (first sample and first kernel a few s after it).
#
# Usage:   sbatch [-p PART -t TIME -J NAME] run_power.sbatch /path/workload.conf
# Dry run: DRY_RUN=1 bash run_power.sbatch /path/workload.conf
#
# Output in $RESULTS_DIR/prof-<jobid>/:
#   profiling_result_<WORKLOAD>_<gpu>.csv   (one per sampled GPU)
#                                            timestamp_ns, current_socket_power_W,
#                                           inst_power_W, gfx_clock_MHz, edge_temp_C, hotspot_temp_C
#   ktrace_<WORKLOAD>/<WORKLOAD>_kernel_trace.csv
#   clock_ref.txt, run_info.txt, app.txt
# Then:    python3 $PROFILING_DIR/build_sampling_json.py $RESULTS_DIR/prof-<jobid>
#
#SBATCH --job-name=wl-power
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

CONF=${1:?usage: sbatch run_power.sbatch <workload.conf>}
[ -f "$CONF" ] || { echo "Error: no config file $CONF"; exit 1; }
source "$CONF"
set -u
run() { if [ -n "${DRY_RUN:-}" ]; then printf '[dry-run]'; printf ' %q' "$@"; echo; else "$@"; fi; }

# ---- launch layout (NGPUS / NTASKS from the config) -------------------------
NGPUS=${NGPUS:-1}; NTASKS=${NTASKS:-1}; BIND_GPU_PER_TASK=${BIND_GPU_PER_TASK:-0}
SAMPLE_GPUS=${SAMPLE_GPUS:-${GPU:-0}}
SRUN=(srun -n "$NTASKS")
BIND=()
if [ "$BIND_GPU_PER_TASK" = 1 ]; then
    BIND=(bash -c 'export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID; exec "$@"' bind)
fi
check_layout() {
    [ -n "${DRY_RUN:-}" ] && return 0
    local have
    have=$(rocminfo 2>/dev/null | grep -c "  Name:  *gfx")
    if [ "$have" -lt "$NGPUS" ]; then
        echo "Error: job sees $have GPU(s) but NGPUS=$NGPUS. For 8 GPUs submit with: sbatch -p mi3008x -n $NTASKS ..."
        exit 1
    fi
    if [ "${SLURM_NTASKS:-1}" -lt "$NTASKS" ]; then
        echo "Error: allocation has ${SLURM_NTASKS:-1} task(s) but NTASKS=$NTASKS. Submit with: sbatch -n $NTASKS ..."
        exit 1
    fi
}

for m in "${MODULES[@]}"; do run module load "$m"; done
ROCPROF=$ROCM_PATH/bin/rocprofv3
export LD_LIBRARY_PATH=$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}   # amd-smi-query links libamd_smi
for f in "${REQUIRED_FILES[@]}" "$PROF_LT/wrapper.py" "$PROF_LT/amd-smi-query"; do
    [ -e "$f" ] || { echo "Error: missing required file: $f"; exit 1; }
done
[ -n "${DRY_RUN:-}" ] || [ -x "$ROCPROF" ] || { echo "Error: no rocprofv3 at $ROCPROF"; exit 1; }

RUNDIR=$RESULTS_DIR/prof-${SLURM_JOB_ID:-manual}
[ -n "${DRY_RUN:-}" ] && RUNDIR=$(mktemp -d)/prof-dryrun   # dry runs never touch RESULTS_DIR
KTRACE_DIR=$RUNDIR/ktrace_$WORKLOAD
OUT_DIR=$RUNDIR
mkdir -p "$RUNDIR" "$KTRACE_DIR"
cp "$CONF" "$RUNDIR/workload.conf"

{
    echo "workload    : $WORKLOAD"
    echo "description : $WORKLOAD_DESC"
    echo "host        : $(hostname)"
    echo "date        : $(date)"
    echo "gpu arch    : $(rocminfo 2>/dev/null | grep -m1 '  Name:  *gfx' | awk '{print $2}')"
    echo "rocprofv3   : $("$ROCPROF" --version 2>&1 | head -n 1 | xargs)"
    echo "sampler     : $PROF_LT/amd-smi-query (interval ${SAMPLE_MS} ms, devices ${SAMPLE_GPUS})"
    echo "layout      : NGPUS=$NGPUS NTASKS=$NTASKS BIND_GPU_PER_TASK=$BIND_GPU_PER_TASK partition=${SLURM_JOB_PARTITION:-none}"
    echo "gpus seen   : $(rocminfo 2>/dev/null | grep -c '  Name:  *gfx')"
    echo "run dir     : $RUNDIR"
} | tee "$RUNDIR/run_info.txt"

check_layout
cd "$RUNDIR" || exit 1
stage_inputs
app_env
app_cmd
if [ "$NTASKS" -gt 1 ]; then ONAME="${WORKLOAD}_%pid%"; else ONAME="$WORKLOAD"; fi
printf 'command     :'; printf ' %q' "${APP_CMD[@]}"; echo
printf 'command     :' >> run_info.txt; printf ' %q' "${APP_CMD[@]}" >> run_info.txt; echo >> run_info.txt

echo
echo ">>> telemetry + kernel trace"
python3 -c 'import time; print("clock_ref_ns boottime=%d monotonic=%d realtime=%d" % (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.clock_gettime_ns(time.CLOCK_MONOTONIC), time.clock_gettime_ns(time.CLOCK_REALTIME)))' \
    | tee "$RUNDIR/clock_ref.txt"
if [ -n "${DRY_RUN:-}" ]; then
    run python3 "$PROF_LT/wrapper.py" -d "$SAMPLE_GPUS" -p "$WORKLOAD" --interval-ms "$SAMPLE_MS" -- \
        "${SRUN[@]}" "${BIND[@]}" "$ROCPROF" --kernel-trace --output-format csv -d "$KTRACE_DIR" -o "$ONAME" \
        -- "${APP_CMD[@]}"
    rc=0
else
    python3 "$PROF_LT/wrapper.py" -d "$SAMPLE_GPUS" -p "$WORKLOAD" --interval-ms "$SAMPLE_MS" -- \
        "${SRUN[@]}" "${BIND[@]}" "$ROCPROF" --kernel-trace --output-format csv -d "$KTRACE_DIR" -o "$ONAME" \
        -- "${APP_CMD[@]}" > "$RUNDIR/app.txt" 2>&1
    rc=$?
fi
echo "<<< exit status: $rc" | tee -a run_info.txt

TELEM="$RUNDIR/profiling_result_${WORKLOAD}_${SAMPLE_GPUS%%,*}.csv"   # first sampled GPU
KTRACE=$(find "$KTRACE_DIR" -name "*kernel_trace.csv" 2>/dev/null | head -n 1)

echo
echo "=============================================================="
echo "App timing ($PERF_GREP):"
grep -hE "$PERF_GREP" "$RUNDIR"/* 2>/dev/null | head -n 3
echo
echo "Telemetry files: $(ls "$RUNDIR"/profiling_result_*.csv 2>/dev/null | wc -l)  (summary below: $TELEM)"
if [ -s "$TELEM" ]; then
    # line 1 = sampler metadata, line 2 = header; nan = sensor not exposed
    awk -F, 'NR > 2 && NF >= 6 {
        n++; p += $2; if ($2 > pm) pm = $2; c += $4; if ($4 > cm) cm = $4
        if ($6 ~ /^[0-9.]+$/) { t += $6; tn++; if ($6 > tm) tm = $6 }
    } END {
        if (!n) { print "  no data rows"; exit }
        printf "  %d samples\n  socket power mean %6.1f W   max %6.1f W\n  gfx clock    mean %6.1f MHz max %6.1f MHz\n", n, p/n, pm, c/n, cm
        if (tn) printf "  hotspot temp mean %6.1f C   max %6.1f C\n", t/tn, tm
    }' "$TELEM"
else
    echo "  no samples written"
fi
echo
echo "Kernel trace: ${KTRACE:-none found under $KTRACE_DIR}"
[ -n "$KTRACE" ] && echo "  trace files: $(find "$KTRACE_DIR" -name "*kernel_trace.csv" | wc -l); kernel launches in first: $(( $(wc -l < "$KTRACE") - 1 ))"
echo
echo "Clock alignment (ns) -- first sample and first kernel should both be a few s after one reference:"
cat "$RUNDIR/clock_ref.txt"
echo "  first sampler timestamp : $(awk -F, 'NR == 3 { print $1; exit }' "$TELEM" 2>/dev/null)"
[ -n "$KTRACE" ] && python3 - "$KTRACE" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print("  first kernel start      : %d" % min(int(r["Start_Timestamp"]) for r in rows) if rows else "  no kernels")
EOF
echo
echo "Next: python3 $PROFILING_DIR/build_sampling_json.py $RUNDIR"
echo "      python3 $PROFILING_DIR/postprocess_pmc.py <pmc run dir> --power-ktrace $KTRACE_DIR"
echo "=============================================================="
exit $rc
