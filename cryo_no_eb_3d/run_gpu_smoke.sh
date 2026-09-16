#!/usr/bin/env bash
#
# Single-GPU smoke test for the CUDA build.
#
# Answers "does the GPU build actually run", not "how fast is it": plotfiles
# and checkpoints are disabled, and so is plot_per_exact, which otherwise
# bends dt to land on plot times and would corrupt the per-step timings.
#
# Usage:  ./run_gpu_smoke.sh [n_steps]        (default 100)
#         CUDA_VISIBLE_DEVICES=0 ./run_gpu_smoke.sh 10
#
set -euo pipefail

# --- configuration --------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXECUTABLE="${SCRIPT_DIR}/incflo3d.gnu.CUDA.ex"

CASE="gold_grid_sample_full_mix_coarse"
STEPS="${1:-100}"

# GPU 0 is often busy with another user's job; default to GPU 1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
# --------------------------------------------------------------------------

if [[ ! -x "${EXECUTABLE}" ]]; then
	    echo "Executable not found/executable: ${EXECUTABLE}" >&2
	        exit 1
fi

input_name="inputs.cryo_${CASE}"
template="${SCRIPT_DIR}/${input_name}"
if [[ ! -f "${template}" ]]; then
	    echo "Missing input: ${template}" >&2
	        exit 1
fi

run_dir="${SCRIPT_DIR}/gpu_smoke_$(date +%m%d_%H%M%S)"
mkdir -p "${run_dir}"
cp "${template}" "${run_dir}/${input_name}"

cd "${run_dir}"

# cryo_sample_place must be absolute: the input file's "../../data/samples/..."
# is written relative to cryo_no_eb_3d, and we are one level deeper.
"${EXECUTABLE}" "${input_name}" \
	    max_step="${STEPS}" \
	        amr.plot_per_exact=-1 amr.plot_int=-1 amr.check_int=-1 \
		    amrex.abort_on_out_of_gpu_memory=1 \
		        incflo.cryo_sample_place="${REPO_ROOT}/data/samples/cells.samples_mixed" \
			    incflo.cryo_temp_stats_file="${run_dir}/stats_gpu.dat" \
			        2>&1 | tee "${run_dir}/run.log"

echo
echo "Run directory: ${run_dir}"
echo "  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 2"

