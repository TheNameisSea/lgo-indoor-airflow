#!/bin/bash
# Export every completed case to the ML loader's CSV layout.
#
# Two steps per case: `postProcess -func writeCellCentres` writes 0/C (the
# exporter needs cell centres and the solver does not write them), then
# export_ml.py writes the 21 CSVs.
#
# purgeWrite is 0 in every case, which is what makes the postProcess step safe:
# under purgeWrite 2 it once destroyed the converged solution by writing a time
# directory and pushing the real one out of the retention window.
#
# Only cases that reached t=3000 are exported; a case still being solved by the
# sweep is skipped and picked up on the next run (existing outputs are skipped
# too, so this is resumable).

set -u
cd "$(dirname "$0")/.." || exit 1
# Run inside the CFD environment (conda env create -f environment-cfd.yml),
# which puts OpenFOAM on PATH; or set CFD_ENV to its prefix.
if [ -n "${CFD_ENV:-}" ]; then
  export PATH="$CFD_ENV/bin:$PATH"; export WM_PROJECT_DIR="$CFD_ENV"
fi
PY=${CPY:-python}
OUT=cfd/dataset
JOBS=${JOBS:-3}

mkdir -p $OUT

one() {
    case_dir=$1
    name=$(basename "$case_dir")
    [ -f "$case_dir/3000/U" ] || { echo "SKIP  $name (not finished)"; return; }
    [ -f "$OUT/$name/Fluid_data.csv" ] && { echo "have  $name"; return; }
    # Reaching t=3000 is NOT the same as being usable. `cfd/sweep.py` runs
    # sanity per case and reports it, but nothing stopped a failing case from
    # being exported afterwards: this script used to gate on 3000/U alone.
    # D002 (tier D) is the case that exposed it -- snappyHexMesh lost the
    # Ceiling and HVAC_01 patches to a thin ceiling sliver, so the return BC was
    # never applied and it ran to t=3000 with supply_flux -0.0016 against an
    # expected 0.6019. It has a valid 3000/U and would have entered the dataset
    # as training data. A bad case must cost solver time, never corrupt data.
    if ! $PY cfd/sanity.py --case "$case_dir" > "$OUT/$name.sanity.log" 2>&1; then
        echo "REJECT $name (sanity):$(sed -n 's/^    ! /  /p' \
            "$OUT/$name.sanity.log" | head -2 | tr '\n' ';')"
        return
    fi
    if [ ! -f "$case_dir/0/C" ]; then
        ( cd "$case_dir" && nice -n 15 postProcess -func writeCellCentres -time 0 \
            > log.cellcentres 2>&1 ) || { echo "FAIL  $name (cell centres)"; return; }
    fi
    if nice -n 15 $PY cfd/export_ml.py --case "$case_dir" --out "$OUT/$name" \
         > "$OUT/$name.log" 2>&1; then
        n=$(wc -l < "$OUT/$name/Fluid_data.csv")
        echo "ok    $name  $((n-1)) interior points"
    else
        echo "FAIL  $name (export)"; tail -3 "$OUT/$name.log"
    fi
}
export -f one
export OUT PY

ls -d cfd/run/*/ | sort | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {}
echo "=== done: $(ls -d $OUT/*/ 2>/dev/null | wc -l) case directories exported ==="
