#!/usr/bin/env bash
#
# Generate input files + run folders for a sweep over sample-layer thickness.
#
# For each thickness value it:
#   1. copies the template input, replacing the sample-layer thickness value
#   2. drops the new input file into its own run folder (so plt/chk output from
#      different runs never overwrite each other)
#
# This script ONLY creates files. Launch the runs with:
#   ./run_sapphire_sample_thickness.sh
#
set -euo pipefail

# --- configuration --------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/inputs.cryo_sapphire_disk_sample_layer_template"

# Sample-layer thickness values to sweep (mm).
THICKNESSES=(0 0.01 0.02 0.1)

# Key in the input file whose value we replace.
KEY="incflo.cryo_sample_layer_thickness"
# --------------------------------------------------------------------------

if [[ ! -f "${TEMPLATE}" ]]; then
    echo "Template not found: ${TEMPLATE}" >&2
    exit 1
fi

for t in "${THICKNESSES[@]}"; do
    run_dir="${SCRIPT_DIR}/sapphire_disk_sample_layer_${t}"
    input_name="inputs.cryo_sapphire_disk_sample_layer_${t}"
    input_path="${run_dir}/${input_name}"

    mkdir -p "${run_dir}"

    # Replace the thickness value, preserving any trailing comment on the line.
    sed -E "s|^([[:space:]]*${KEY}[[:space:]]*=[[:space:]]*)[^#[:space:]]+|\1${t}|" \
        "${TEMPLATE}" > "${input_path}"

    echo "Wrote ${input_path}"
done
