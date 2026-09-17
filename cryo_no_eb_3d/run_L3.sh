#!/usr/bin/env bash
#
# L3 diagnostic run for the CUDA build (dx = 3.906 um at the finest level).
#
# This is NOT a production run.  stop_time = 5 ms at L3 is tens of thousands of
# steps; this script is bounded by max_step and exists to answer one question:
# is the L3 timestep advection-limited by the ~0.95 mm/ms plunge, or is it being
# throttled ~11x by spurious velocities at the solid stamp?  Watch magvel in the
# plotfiles and the reported dt.
#
# Memory: ~1777 bytes/cell measured from the L2 run's arena high-water, so the
# default arena (3/4 of 48.5 GB = 36.4 GB) holds ~20.5M cells.  The input refines
# L3 on cell_type (10.06M) plus gradtemperr[2] = 10 K for the thermal boundary
# layer.  The original temperr=95 / gradtemperr=1 pair gives 32.1M cells = ~53 GB
# and does NOT fit; abort_on_out_of_gpu_memory below makes an overrun fail fast
# rather than crawl.
#
# CALIBRATE FIRST after changing any tagging threshold:
#     ./run_L3.sh 1 -1        # one step, no plotfiles
# then read the "Level 3 ... cells" line and "max space used (MB)" from the log.
# Note the t~0 count is a LOWER BOUND -- the field is still a two-value step and
# the tagged set grows as the plume develops.  Leave real headroom.
#
# Usage:  ./run_L3.sh [n_steps] [plot_int]      (defaults: 50, 25)
#         ./run_L3.sh 1 -1                      (calibration, no plotfiles)
#         CUDA_VISIBLE_DEVICES=0 ./run_L3.sh 20 10
#
set -euo pipefail

# --- configuration --------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXECUTABLE="${SCRIPT_DIR}/incflo3d.gnu.CUDA.ex"

CASE="gold_grid_sample_full_mix_L3"
STEPS="${1:-50}"
PLOT_INT="${2:-25}"

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

run_dir="${SCRIPT_DIR}/L3_$(date +%m%d_%H%M%S)"
mkdir -p "${run_dir}"
cp "${template}" "${run_dir}/${input_name}"

cd "${run_dir}"

echo "L3 run  ->  ${run_dir}"
echo "  GPU ${CUDA_VISIBLE_DEVICES}, ${STEPS} steps, plotfile every ${PLOT_INT}"
echo "  cap is ~20.5M cells / 36.4 GB arena; plotfiles ~1 GB each"
echo

# cryo_sample_place must be absolute: the input file's "../../data/samples/..."
# is written relative to cryo_no_eb_3d, and we are one level deeper.
# plot_per_exact is left off -- it bends dt to land on plot times, which would
# corrupt the very dt measurement this run is for.  plot_int is step-based and
# does not touch dt.
"${EXECUTABLE}" "${input_name}" \
    max_step="${STEPS}" \
    amr.plot_int="${PLOT_INT}" amr.plot_per_exact=-1 amr.check_int=-1 \
    amrex.abort_on_out_of_gpu_memory=1 \
    incflo.cryo_sample_place="${REPO_ROOT}/data/samples/cells.samples_mixed" \
    incflo.cryo_temp_stats_file="${run_dir}/stats_L3.dat" \
    2>&1 | tee "${run_dir}/run.log"

echo
echo "Run directory: ${run_dir}"
echo
echo "dt vs the physical plunge (expect ~2.1e-3 ms if clean, ~1.8e-4 if throttled):"
echo "  grep 'with dt' ${run_dir}/run.log | tail -5"
echo "Peak device memory actually used:"
echo "  grep 'max space used' ${run_dir}/run.log"
echo "Cell counts per level (watch L3 against the ~20.5M ceiling):"
echo "  grep -E '^  Level' ${run_dir}/run.log | tail -4"
