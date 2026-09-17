#!/usr/bin/env bash
#
# Production L3 run: inputs.cryo_gold_grid_sample_full_mix_L3 to stop_time = 5 ms.
#
# Unlike run_L3.sh (a bounded smoke test), this script overrides NOTHING that the
# input file controls -- no max_step, no plot cadence, no check_int.  The input
# owns the physics and the I/O; this script only supplies absolute paths that
# break when running from a subdirectory, plus a fail-fast memory guard.
#
# Run it inside tmux -- expect 5-15 hours.  Ctrl-b d to detach.
#
# Usage:
#   ./run_L3_prod.sh                      start a fresh run
#   ./run_L3_prod.sh /abs/path/to/chk01000   restart from a checkpoint
#   CUDA_VISIBLE_DEVICES=0 ./run_L3_prod.sh  use the other GPU
#   FORCE=1 ./run_L3_prod.sh              start even if the GPU looks busy
#
set -euo pipefail

# --- configuration --------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXECUTABLE="${SCRIPT_DIR}/incflo3d.gnu.CUDA.ex"

CASE="gold_grid_sample_full_mix_L3"
RESTART="${1:-}"

# GPU 0 is usually taken by another user's job; default to GPU 1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
GPU="${CUDA_VISIBLE_DEVICES}"
# --------------------------------------------------------------------------

input_name="inputs.cryo_${CASE}"
template="${SCRIPT_DIR}/${input_name}"

[[ -x "${EXECUTABLE}" ]] || { echo "Executable not found/executable: ${EXECUTABLE}" >&2; exit 1; }
[[ -f "${template}"   ]] || { echo "Missing input: ${template}" >&2; exit 1; }

if [[ -n "${RESTART}" ]]; then
    [[ -d "${RESTART}" ]] || { echo "Checkpoint not found: ${RESTART}" >&2; exit 1; }
    RESTART="$(cd "${RESTART}" && pwd)"          # must be absolute; we cd below
fi

# Refuse to land on a GPU somebody else is using.  nvidia-smi indexes physically,
# and CUDA_VISIBLE_DEVICES does not remap it, so GPU is the right index here.
#
# Check utilisation as well as memory: the long-running job on GPU 0 of this node
# holds only ~590 MiB but pins the SMs at 100%, so a memory-only test waves it
# through and both jobs then crawl.  Sample utilisation a few times -- a single
# reading catches idle gaps between kernels.
gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}")
gpu_util=0
for _ in 1 2 3; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${GPU}")
    (( u > gpu_util )) && gpu_util=$u
    sleep 1
done
if { (( gpu_mem > 1024 )) || (( gpu_util > 10 )); } && [[ "${FORCE:-0}" != "1" ]]; then
    echo "GPU ${GPU} looks busy: ${gpu_mem} MiB used, ${gpu_util}% utilisation." >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
    echo "Pick another GPU with CUDA_VISIBLE_DEVICES, or re-run with FORCE=1." >&2
    exit 1
fi

# Plotfiles are ~0.5 GB each at NATIVE_32 and 50 of them are expected.
avail_gb=$(df -BG --output=avail "${SCRIPT_DIR}" | tail -1 | tr -dc '0-9')
(( avail_gb < 100 )) && echo "WARNING: only ${avail_gb} GB free; expect ~25-30 GB of output." >&2

run_dir="${SCRIPT_DIR}/L3_prod_$(date +%m%d_%H%M%S)"
mkdir -p "${run_dir}"
cp "${template}" "${run_dir}/${input_name}"
cd "${run_dir}"

restart_arg=()
[[ -n "${RESTART}" ]] && restart_arg=(amr.restart="${RESTART}")

cat <<INFO

  run dir     ${run_dir}
  GPU         ${GPU} (${gpu_mem} MiB, ${gpu_util}% busy before start)
  restart     ${RESTART:-none, fresh start}
  input       $(grep -c . "${input_name}") lines, stop_time=$(grep -oP '^stop_time\s+=\s+\K[0-9.]+' "${input_name}")
  expect      5-15 h, ~50 plotfiles, ~25-30 GB

  monitor from another shell:
    tail -f ${run_dir}/run.log | grep --line-buffered -E "Step |WARNING|nan"
    grep -E '^  Level 3' ${run_dir}/run.log | tail -3      # watch against the ~24M cell ceiling
    nvidia-smi -i ${GPU} -l 5

INFO

# cryo_sample_place must be absolute: the input's "../../data/samples/..." is
# written relative to cryo_no_eb_3d and we are one level deeper.
# abort_on_out_of_gpu_memory makes an overrun fail immediately instead of
# crawling -- the checkpoints in the input let you restart from the last one.
"${EXECUTABLE}" "${input_name}" \
    "${restart_arg[@]}" \
    amrex.abort_on_out_of_gpu_memory=1 \
    incflo.cryo_sample_place="${REPO_ROOT}/data/samples/cells.samples_mixed" \
    incflo.cryo_temp_stats_file="${run_dir}/stats_L3.dat" \
    2>&1 | tee "${run_dir}/run.log"

echo
echo "Finished. Results in ${run_dir}"
grep -E "Time spent in Evolve|max space used" "${run_dir}/run.log" || true
