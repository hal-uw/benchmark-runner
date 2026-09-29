#!/bin/bash
# Hardware-counter (PMC) profiling with rocprofv3: one full application run per
# pass in counters.json, each with --pmc <that pass's counters> --kernel-trace.
# A failed pass is reported and the script continues with the next one.
#
# Usage:   sbatch [-p PART -t TIME -J NAME] run_pmc.sbatch /path/workload.conf [counters.json]
#          (default counters file: $PROFILING_DIR/counters.json)
# Dry run: DRY_RUN=1 bash run_pmc.sbatch /path/workload.conf
#
# Output, per pass, in $RESULTS_DIR/pmc-<jobid>/<pass>/:
#   <pass>_counter_collection.csv   one row per kernel launch per counter (+ start/end timestamps)
#   <pass>_kernel_trace.csv         kernel names, launch geometry, start/end timestamps
#   rocprof.<pass>.txt              rocprofv3 + application stdout/stderr
#   LABEL                           one-line description of the pass
# Then:    python3 $PROFILING_DIR/postprocess_pmc.py $RESULTS_DIR/pmc-<jobid>
#
# Time: each pass takes ~2.5-4x the plain run (kernels are serialized while
# counters are collected). Size --time accordingly.
#
#SBATCH --job-name=wl-pmc
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

CONF=${1:?usage: sbatch run_pmc.sbatch <workload.conf> [counters.json]}
[ -f "$CONF" ] || { echo "Error: no config file $CONF"; exit 1; }
source "$CONF"
COUNTERS=${2:-$PROFILING_DIR/counters.json}
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
for f in "${REQUIRED_FILES[@]}" "$COUNTERS"; do
    [ -e "$f" ] || { echo "Error: missing required file: $f"; exit 1; }
done
[ -n "${DRY_RUN:-}" ] || [ -x "$ROCPROF" ] || { echo "Error: no rocprofv3 at $ROCPROF"; exit 1; }

RUNDIR=$RESULTS_DIR/pmc-${SLURM_JOB_ID:-manual}
[ -n "${DRY_RUN:-}" ] && RUNDIR=$(mktemp -d)/pmc-dryrun   # dry runs never touch RESULTS_DIR
mkdir -p "$RUNDIR"
cp "$COUNTERS" "$RUNDIR/counters.json"      # record exactly what was collected
cp "$CONF" "$RUNDIR/workload.conf"

{
    echo "workload    : $WORKLOAD"
    echo "description : $WORKLOAD_DESC"
    echo "host        : $(hostname)"
    echo "date        : $(date)"
    echo "gpu arch    : $(rocminfo 2>/dev/null | grep -m1 '  Name:  *gfx' | awk '{print $2}')"
    echo "rocprofv3   : $("$ROCPROF" --version 2>&1 | head -n 1 | xargs)"
    echo "counters    : $COUNTERS"
    echo "layout      : NGPUS=$NGPUS NTASKS=$NTASKS BIND_GPU_PER_TASK=$BIND_GPU_PER_TASK partition=${SLURM_JOB_PARTITION:-none}"
    echo "gpus seen   : $(rocminfo 2>/dev/null | grep -c '  Name:  *gfx')"
    echo "run dir     : $RUNDIR"
} | tee "$RUNDIR/run_info.txt"

check_layout
cd "$RUNDIR" || exit 1
stage_inputs
app_env

mapfile -t PASSES < <(python3 -c '
import json, sys
for p in json.load(open(sys.argv[1]))["passes"]:
    print(p["name"], ",".join(p["counters"]), p.get("label", ""), sep="|")
' "$RUNDIR/counters.json")
[ "${#PASSES[@]}" -gt 0 ] || { echo "Error: no passes in $COUNTERS"; exit 1; }
echo "passes      : ${#PASSES[@]}"

declare -a STATUS
i=0
for line in "${PASSES[@]}"; do
    i=$((i + 1))
    IFS='|' read -r name ctrs label <<< "$line"
    OUT_DIR=$RUNDIR/$name
    mkdir -p "$OUT_DIR"
    echo "$label" > "$OUT_DIR/LABEL"
    app_cmd
    # one output set per process when several tasks run (Agent_Id in the CSV = GPU)
    if [ "$NTASKS" -gt 1 ]; then ONAME="${name}_%pid%"; else ONAME="$name"; fi

    echo
    echo "=============================================================="
    echo ">>> Pass $i/${#PASSES[@]}: $name"
    echo "    what    : $label"
    echo "    counters: $ctrs"
    t0=$(date +%s)
    if [ -n "${DRY_RUN:-}" ]; then
        run "${SRUN[@]}" "${BIND[@]}" "$ROCPROF" --pmc "$ctrs" --kernel-trace --output-format csv \
            -d "$OUT_DIR" -o "$ONAME" -- "${APP_CMD[@]}"
        rc=0
    else
        "${SRUN[@]}" "${BIND[@]}" "$ROCPROF" --pmc "$ctrs" --kernel-trace --output-format csv \
            -d "$OUT_DIR" -o "$ONAME" -- "${APP_CMD[@]}" \
            > "$OUT_DIR/rocprof.$name.txt" 2>&1
        rc=$?
    fi
    t1=$(date +%s)
    ncsv=$(find "$OUT_DIR" -name "*counter_collection.csv" | wc -l)
    perf=$(grep -hE "$PERF_GREP" "$OUT_DIR"/* 2>/dev/null | head -n 1)
    echo "<<< Pass $name exit status: $rc   wall: $((t1 - t0)) s   counter csv files: $ncsv"
    echo "    ${perf:-no '$PERF_GREP' line in outputs}"
    if [ -z "${DRY_RUN:-}" ] && { [ "$rc" -ne 0 ] || [ "$ncsv" -eq 0 ]; }; then
        echo "    --- last lines of rocprofv3 output ---"
        tail -n 15 "$OUT_DIR/rocprof.$name.txt" | sed 's/^/    /'
    fi
    STATUS+=("$(printf '%-20s rc=%-3s csv=%-2s %s' "$name" "$rc" "$ncsv" "$label")")
done

echo
echo "=============================================================="
echo "Pass summary:"
printf '  %s\n' "${STATUS[@]}"
echo
echo "App timing per pass ($PERF_GREP) -- should agree across passes:"
for line in "${PASSES[@]}"; do
    name=${line%%|*}
    echo "  $name: $(grep -hE "$PERF_GREP" "$RUNDIR/$name"/* 2>/dev/null | head -n 1)"
done
echo
echo "Counter files:"
find "$RUNDIR" -name "*counter_collection.csv" -printf '  %P  (%s bytes)\n' | sort
echo
echo "Next: python3 $PROFILING_DIR/postprocess_pmc.py $RUNDIR"
echo "=============================================================="
