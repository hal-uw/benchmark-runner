#!/bin/bash
# List every hardware counter rocprofv3 can collect, ON THE TARGET GPU.
# Must run as a batch job on the GPU partition: salloc/srun --pty inside a
# script open interactive shells, and the lines after them end up running on
# the login node (which reported gfx90a instead of gfx942).
#
# Usage: sbatch [-p PART] list_counters.sbatch /path/out/counters_list.txt [rocm module]
# Check: grep -E "^Name" <out> | sort -u     -> must print only gfx942
#
#SBATCH --job-name=list-ctrs
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out

OUT=${1:?usage: sbatch list_counters.sbatch <output.txt> [rocm module]}
module load "${2:-rocm/7.2.0}"
echo "host: $(hostname)"
# rocminfo lists the CPU agent first; the gfx line identifies the GPU
rocminfo | grep -m1 "  Name:  *gfx"
rocprofv3 --list-avail > "$OUT" 2>&1
echo "wrote $OUT"
grep -E "^Name" "$OUT" | sort -u
