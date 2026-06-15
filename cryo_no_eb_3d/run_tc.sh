#!/usr/bin/env bash
#
# Launch the three test cases tc1, tc2, tc3 on the server, one detached
# `screen` session per run, using OpenMP.
#
# Nothing is varied here: each case uses its existing input file
# (inputs.cryo_tc1, inputs.cryo_tc2, inputs.cryo_tc3) and runs in its own
# folder so plt/chk output from different runs never overwrite each other.
#
set -euo pipefail

# --- configuration --------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTABLE="${SCRIPT_DIR}/incflo3d.gnu.OMP.ex"

# Test cases to run. Each must have a matching inputs.cryo_<case> file.
CASES=(tc1 tc2 tc3)

# OpenMP threads per run. Override from the shell, e.g.:
#   OMP_NUM_THREADS=8 ./run_tc.sh
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

for c in "${CASES[@]}"; do
    run_dir="${SCRIPT_DIR}/${c}"
    input_name="inputs.cryo_${c}"
    template="${SCRIPT_DIR}/${input_name}"
    log_file="${run_dir}/run.log"
    session="${c}"

    if [[ ! -f "${template}" ]]; then
        echo "Missing input: ${template}" >&2
        exit 1
    fi

    # Drop the input file into its own run folder.
    mkdir -p "${run_dir}"
    cp "${template}" "${run_dir}/${input_name}"

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
echo "  screen -ls            # list running sessions"
echo "  screen -r tc1         # attach to a session (Ctrl-a d to detach)"
echo "  tail -f tc1/run.log   # follow a run's output"
