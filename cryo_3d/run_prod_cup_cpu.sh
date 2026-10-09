#!/usr/bin/env bash
#
# Production LE-cup run on CPU, started by hand on elbrus (local version of
# run_prod_cup_cpu.sbatch).  Run it inside tmux.
#
# Build first:
#     make -j16 USE_MPI=TRUE USE_OMP=TRUE USE_CUDA=FALSE TINY_PROFILE=TRUE
#
# Usage:
#   ./run_prod_cup_cpu.sh
#   NRANKS=64 INPUT=inputs.cryo_cup_eb_l0 ./run_prod_cup_cpu.sh
#
# Writes to the run dir: run.log, host_usage.csv, disk_usage.csv,
# step_times.csv, cells.csv, summary.txt  (for results/diag/run_time_usage_diag.py)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXECUTABLE="${EXECUTABLE:-${SCRIPT_DIR}/incflo3d.gnu.TPROF.MPI.OMP.EB.ex}"
input_name="${INPUT:-inputs.cryo_prod_run_cup}"
case_name="${input_name#inputs.cryo_}"
NRANKS="${NRANKS:-32}"
export OMP_NUM_THREADS=1
# Machine forbids CMA single-copy (kernel.yama.ptrace_scope = 3).
export OMPI_MCA_btl_vader_single_copy_mechanism=none

[[ -x "${EXECUTABLE}" ]] || { echo "Executable not found: ${EXECUTABLE} (see make line above)" >&2; exit 1; }
[[ -f "${SCRIPT_DIR}/${input_name}" ]] || { echo "Missing input: ${input_name}" >&2; exit 1; }

run_dir="${SCRIPT_DIR}/${case_name}_cpu_$(date +%m%d_%H%M%S)"
mkdir -p "${run_dir}"
cp "${SCRIPT_DIR}/${input_name}" "${run_dir}/"
cd "${run_dir}"
echo "run dir: ${run_dir}"

# --- background monitors: per-rank memory/CPU every 30 s, disk every 5 min ---
exe_name="$(basename "${EXECUTABLE}")"
( echo "timestamp,pid,rss_mb,pcpu"
  while sleep 30; do
      ps -u "$(id -u)" -o pid=,rss=,pcpu=,comm= | awk -v ts="$(date +%FT%T)" -v exe="${exe_name:0:15}" \
          '$4 == exe { printf "%s,%s,%.0f,%s\n", ts, $1, $2/1024, $3 }'
  done ) > host_usage.csv &
mon1=$!
( echo "timestamp,bytes,plotfiles,checkpoints"
  while :; do
      echo "$(date +%FT%T),$(du -sb . | cut -f1),$(ls -d plt* 2>/dev/null | wc -l),$(ls -d chk* 2>/dev/null | wc -l)"
      sleep 300
  done ) > disk_usage.csv &
mon2=$!
trap 'kill ${mon1} ${mon2} 2>/dev/null || true' EXIT

# --- run: one rank per physical core, spread over both sockets --------------
set +e
mpirun -np "${NRANKS}" --map-by numa --bind-to core --report-bindings \
    -x OMP_NUM_THREADS -x OMPI_MCA_btl_vader_single_copy_mechanism \
    "${EXECUTABLE}" "${input_name}" \
    incflo.cryo_grid_file="${REPO_ROOT}/data/grids/grid_Au_carbon_realAu300QuantifoilR14_v260930.hmap" \
    incflo.cryo_sample_place="${REPO_ROOT}/data/samples/cells.samples_mixed3" \
    incflo.cryo_temp_stats_file="${run_dir}/temperature_stats_${case_name}.dat" \
    2>&1 | tee -i run.log
status=${PIPESTATUS[0]}
set -e

# --- post-run diagnostics -----------------------------------------------------
awk 'BEGIN            { print "step,new_time_ms,dt_ms,wall_s" }
     /^Step [0-9]+:/  { step = $2; sub(":", "", step); t = $9; dt = $NF; sub(/\.$/, "", dt) }
     /^Time per step/ { if (step != "") print step "," t "," dt "," $NF }' run.log > step_times.csv

awk 'BEGIN            { print "step,level,grids,cells"; step = 0 }
     /^Step [0-9]+:/  { step = $2; sub(":", "", step) }
     /^  Level [0-9]+ +[0-9]+ grids +[0-9]+ cells/ { print step "," $2 "," $3 "," $5 }' run.log > cells.csv

{
    echo "exit status   ${status}"
    grep -E "Time spent in (InitData|Evolve)" run.log || true
    awk -F, 'NR > 1 { n++; s += $4; if ($4 > m) m = $4 }
             END { if (n) printf "steps         %d   wall s/step mean %.3f max %.3f\n", n, s/n, m }' step_times.csv
    awk -F, 'NR > 1 { if ($3 > p) p = $3; t[$1] += $3 } END { for (k in t) if (t[k] > q) q = t[k];
             printf "peak RSS      %d MB per rank, %d MB all ranks\n", p, q }' host_usage.csv
    echo "output size   $(du -sh . | cut -f1)"
} | tee summary.txt

exit "${status}"
