#!/bin/bash
# Plain (unprofiled) run of a workload on one MI300X: correctness and the
# reference timing that profiled runs are compared against.
#
# Usage:   sbatch [-p PART -t TIME -J NAME] run_plain.sbatch /path/workload.conf
# Dry run: DRY_RUN=1 bash run_plain.sbatch /path/workload.conf   (prints, runs nothing)
#
# Output:  $RESULTS_DIR/plain-<jobid>/  app.txt (stdout+stderr), run_info.txt, app logs
#
#SBATCH --job-name=wl-plain
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

CONF=${1:?usage: sbatch run_plain.sbatch <workload.conf>}
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
for f in "${REQUIRED_FILES[@]}"; do
    [ -e "$f" ] || { echo "Error: missing required file: $f"; exit 1; }
done

RUNDIR=$RESULTS_DIR/plain-${SLURM_JOB_ID:-manual}
[ -n "${DRY_RUN:-}" ] && RUNDIR=$(mktemp -d)/plain-dryrun   # dry runs never touch RESULTS_DIR
OUT_DIR=$RUNDIR
mkdir -p "$RUNDIR"
cp "$CONF" "$RUNDIR/workload.conf"

{
    echo "workload    : $WORKLOAD"
    echo "description : $WORKLOAD_DESC"
    echo "host        : $(hostname)"
    echo "date        : $(date)"
    echo "gpu arch    : $(rocminfo 2>/dev/null | grep -m1 '  Name:  *gfx' | awk '{print $2}')"
    echo "layout      : NGPUS=$NGPUS NTASKS=$NTASKS BIND_GPU_PER_TASK=$BIND_GPU_PER_TASK partition=${SLURM_JOB_PARTITION:-none}"
    echo "gpus seen   : $(rocminfo 2>/dev/null | grep -c '  Name:  *gfx')"
    echo "run dir     : $RUNDIR"
} | tee "$RUNDIR/run_info.txt"

check_layout
cd "$RUNDIR" || exit 1
stage_inputs
app_env
app_cmd
printf 'command     :'; printf ' %q' "${APP_CMD[@]}"; echo
printf 'command     :' >> run_info.txt; printf ' %q' "${APP_CMD[@]}" >> run_info.txt; echo >> run_info.txt

echo ">>> plain run"
t0=$(date +%s)
if [ -n "${DRY_RUN:-}" ]; then
    run "${SRUN[@]}" "${BIND[@]}" "${APP_CMD[@]}"; rc=0
else
    "${SRUN[@]}" "${BIND[@]}" "${APP_CMD[@]}" > "$RUNDIR/app.txt" 2>&1; rc=$?
fi
t1=$(date +%s)
echo "<<< exit status: $rc   wall: $((t1 - t0)) s" | tee -a run_info.txt

echo
echo "App timing ($PERF_GREP):"
grep -hE "$PERF_GREP" "$RUNDIR"/*.txt "$RUNDIR"/*.log "$RUNDIR"/log.* 2>/dev/null | sort -u | head -n 5
echo "Files:"
ls -la "$RUNDIR"
exit $rc
