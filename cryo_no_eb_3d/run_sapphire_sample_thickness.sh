#!/usr/bin/env bash
#
# Launch the sample-layer thickness sweep on the server, one detached `screen`
# session per run, using OpenMP.
#
# Run the input-file generator FIRST:
#   ./gen_sapphire_sample_thickness.sh
# then launch with this script.
#
set -euo pipefail

# --- configuration --------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTABLE="${SCRIPT_DIR}/incflo3d.gnu.OMP.ex"

# Sample-layer thickness values to sweep (mm). Must match the generator.
THICKNESSES=(0 0.01 0.02 0.1)

# OpenMP threads per run. Override from the shell, e.g.:
#   OMP_NUM_THREADS=8 ./run_sapphire_sample_thickness.sh
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# --------------------------------------------------------------------------

if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Executable not found/executable: ${EXECUTABLE}" >&2
    exit 1
fi
if ! command -v screen >/dev/null 2>&1; then
    echo "screen is not installed/available on this host." >&2
    exit 1
fi

for t in "${THICKNESSES[@]}"; do
    run_dir="${SCRIPT_DIR}/sapphire_disk_sample_layer_${t}"
    input_name="inputs.cryo_sapphire_disk_sample_layer_${t}"
    log_file="${run_dir}/run.log"
    session="sapphire_t${t}"

    if [[ ! -f "${run_dir}/${input_name}" ]]; then
        echo "Missing input: ${run_dir}/${input_name}" >&2
        echo "Run ./gen_sapphire_sample_thickness.sh first." >&2
        exit 1
    fi

    # Launch in a detached screen session running OpenMP.
    screen -dmS "${session}" bash -c "
        cd '${run_dir}'
        export OMP_NUM_THREADS=${OMP_NUM_THREADS}
        '${EXECUTABLE}' '${input_name}' 2>&1 | tee '${log_file}'
    "

    echo "Launched screen session '${session}'  ->  ${run_dir}  (OMP_NUM_THREADS=${OMP_NUM_THREADS})"
done

echo
echo "All runs launched. Useful screen commands:"
echo "  screen -ls                 # list running sessions"
echo "  screen -r sapphire_t0.01   # attach to a session (Ctrl-a d to detach)"
echo "  tail -f sapphire_disk_sample_layer_0.01/run.log   # follow a run's output"
