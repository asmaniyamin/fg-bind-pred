#!/usr/bin/env bash
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# Run the fetch -> preprocess -> extract pipeline against one or many PDB IDs.
#
# Usage:
#   bash scripts/pdb_to_pocket.sh <PDB_ID> [bio-flags]               # single PDB
#   bash scripts/pdb_to_pocket.sh -l <pdb_list.txt> [bio-flags]      # batch from file
#
# Examples:
#   bash scripts/pdb_to_pocket.sh 1stp                          # auto-bio (the default)
#   bash scripts/pdb_to_pocket.sh 1stp --bio                    # force biological assembly 1
#   bash scripts/pdb_to_pocket.sh 1stp --bio 2                  # force biological assembly 2
#   bash scripts/pdb_to_pocket.sh 1stp --no-bio                 # skip auto-bio; stay on AU
#   bash scripts/pdb_to_pocket.sh -l pdbs.txt                   # batch
#   bash scripts/pdb_to_pocket.sh -l pdbs.txt --no-bio          # batch with bio-unit disabled
#   bash scripts/pdb_to_pocket.sh -l pdbs.txt --no-cache        # batch + force re-fetch + re-prep
#
# Bio-unit options (auto-bio is the default):
#   --bio [N]       Force expansion to assembly N (default 1).
#   --no-bio        Force AU extraction; skip auto-bio check.
#   --auto-bio [N]  Explicit auto-bio (same as default).
#
# Auto-bio mode calls `check_assembly` on the raw CIF and expands to the
# biological assembly only when the bio-unit's chain count differs from
# the asymmetric unit. In batch mode, the bio flag applies to every PDB.
#
# List file format (used with -l):
#   - One PDB ID per line.
#   - Blank lines and lines starting with `#` are skipped.
#   - Inline `# ...` comments are stripped.
#
# Cache control:
#   FORCE_FETCH=1   Re-fetch even when the raw CIF is on disk.
#   FORCE_PREP=1    Re-run PrepWizard even when the prepped CIF is on disk.
#   --no-cache      Sets both of the above (most useful in batch mode).
#   NO_CACHE=1      Same as --no-cache, env-var form.
#
# PrepWizard is the dominant cost (~minutes per PDB on bio-units); the
# default-on cache lets you iterate on extract-side code without re-paying
# it. Use --no-cache when verifying that an upstream change still produces
# correct outputs from a clean slate.
#
# Debug:
#   DEBUG=1   Preserve the Schrodinger job archive (-SAVE), stream prepwizard
#             output live, and export SCHRODINGER_PYTHON_DEBUG=1 so protassign
#             tracebacks identify the offending residue by name.
#
# Failure semantics:
#   Single mode  - errors propagate; non-zero exit on any step failure.
#   Batch mode   - per-PDB failures are isolated (the sweep continues) and a
#                  passed/failed summary is printed at the end. Exit status
#                  is 0 if every PDB succeeded, 1 otherwise.
#
# Output layout (per-PDB):
#   data/<pdb_id>/proteins/<pdb_id>.cif            raw fetched mmCIF
#   data/<pdb_id>/processed/<pdb_id>_prepped.cif   Schrodinger-prepped PDBx
#   data/<pdb_id>/pockets/                         ligand SDFs + pocket JSONs
#
# Notes:
#   - Assumes the project venv lives at ./.venv.
#   - Requires $SCHRODINGER to point at a Schrodinger install, or uses
#     /opt/schrodinger/suites2026-1 as a default.
#   - Bio-unit expansion (auto and forced) requires gemmi.

set -euo pipefail

DEBUG="${DEBUG:-0}"
FORCE_FETCH="${FORCE_FETCH:-0}"
FORCE_PREP="${FORCE_PREP:-0}"
NO_CACHE="${NO_CACHE:-0}"

# --- Argument parsing ------------------------------------------------------
PDB_ID=""
PDB_LIST=""
BIO_FLAGS=()

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <PDB_ID|-l pdb_list.txt> [--bio [N] | --no-bio | --auto-bio [N]] [--no-cache]" >&2
    exit 2
fi

case "$1" in
    -l|--list)
        if [[ $# -lt 2 ]]; then
            echo "$0: -l requires a file path" >&2
            exit 2
        fi
        PDB_LIST="$2"
        shift 2
        ;;
    -*)
        echo "$0: first arg must be a PDB ID or '-l <file>'" >&2
        exit 2
        ;;
    *)
        PDB_ID="$1"
        shift
        ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cache)
            NO_CACHE=1
            shift
            ;;
        --bio|--no-bio|--auto-bio)
            BIO_FLAGS+=("$1")
            shift
            # --bio and --auto-bio may take an optional numeric assembly id.
            if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
                BIO_FLAGS+=("$1")
                shift
            fi
            ;;
        *)
            echo "$0: unknown arg '$1'" >&2
            exit 2
            ;;
    esac
done

if [[ "$NO_CACHE" == "1" ]]; then
    echo "=== --no-cache: forcing re-fetch + re-PrepWizard for every PDB ==="
    export FORCE_FETCH=1
    export FORCE_PREP=1
fi

# --- Setup -----------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
export SCHRODINGER="${SCHRODINGER:-/opt/schrodinger/suites2026-1}"

# --- run_one: full pipeline for a single PDB ID ---------------------------
# Uses BIO_FLAGS, DEBUG, FORCE_FETCH, FORCE_PREP, PROJECT_DIR from the
# enclosing scope. Called once in single mode and once per list entry in
# batch mode (wrapped in a subshell so set -e isolates failures).
run_one() {
    local PDB_ID_LC USE_BIO_ASSEMBLY AUTO_BIO_ASSEMBLY ASSEMBLY_ID
    local PDB_DIR RAW_DIR PROCESSED_DIR POCKETS_DIR
    local RAW_CIF PREPPED_CIF CIF_FOR_PREP EXPANDED_RAW
    local LIGAND_RESNAMES INCLUDE_FLAGS PREPROCESS_FLAGS AUTO_RC

    PDB_ID_LC="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"

    USE_BIO_ASSEMBLY=0
    AUTO_BIO_ASSEMBLY=1   # default; overridden by --bio or --no-bio below.
    ASSEMBLY_ID="1"
    case "${BIO_FLAGS[0]:-}" in
        --bio)
            USE_BIO_ASSEMBLY=1
            AUTO_BIO_ASSEMBLY=0
            ASSEMBLY_ID="${BIO_FLAGS[1]:-1}"
            ;;
        --no-bio)
            AUTO_BIO_ASSEMBLY=0
            ;;
        --auto-bio)
            ASSEMBLY_ID="${BIO_FLAGS[1]:-1}"
            ;;
    esac

    PDB_DIR="data/${PDB_ID_LC}"
    RAW_DIR="${PDB_DIR}/proteins"
    PROCESSED_DIR="${PDB_DIR}/processed"
    POCKETS_DIR="${PDB_DIR}/pockets"
    mkdir -p "$RAW_DIR" "$PROCESSED_DIR" "$POCKETS_DIR"
    rm -f "${PDB_ID_LC}.log"

    RAW_CIF="${PROJECT_DIR}/${RAW_DIR}/${PDB_ID_LC}.cif"
    PREPPED_CIF="${PROJECT_DIR}/${PROCESSED_DIR}/${PDB_ID_LC}_prepped.cif"

    # --- 1. Fetch ----------------------------------------------------------
    # X-ray only, reject structures worse than 3.0 A.
    # Skip when raw CIF is already on disk unless FORCE_FETCH=1 is set.
    if [[ -s "$RAW_CIF" && "$FORCE_FETCH" == "0" ]]; then
        echo "=== [1/4] Fetch: reusing cached ${RAW_CIF#${PROJECT_DIR}/} (set FORCE_FETCH=1 to refetch) ==="
    else
        echo "=== [1/4] Fetching ${PDB_ID_LC} ==="
        python -m protein.fetch.fetch "$PDB_ID_LC" \
            --output-dir "$RAW_DIR" \
            --require-xray \
            --min-resolution 3.0 \
            --overwrite
    fi

    # --- 1b. Auto-decide bio-unit vs AU (only when --auto-bio is in effect)
    # check_assembly exits 1 iff the named assembly has a different chain
    # count than the AU (i.e. expansion would change the extraction).
    # Exit 0 means AU == bio-unit, so we stay on the AU.
    if [[ "$AUTO_BIO_ASSEMBLY" -eq 1 ]]; then
        echo "--- auto-bio: checking assembly ${ASSEMBLY_ID} against AU ---"
        AUTO_RC=0
        python -m protein.extract.check_assembly "$RAW_CIF" --assembly-id "$ASSEMBLY_ID" || AUTO_RC=$?
        if [[ "$AUTO_RC" -eq 1 ]]; then
            echo "--- auto-bio: bio-unit differs from AU -> enabling expansion ---"
            USE_BIO_ASSEMBLY=1
        else
            echo "--- auto-bio: bio-unit matches AU -> staying on asymmetric unit ---"
        fi
    fi

    # --- 1c. Biological-assembly expansion (pre-PrepWizard) ----------------
    # Expansion has to happen BEFORE PrepWizard, not after. Schrodinger
    # strips the _pdbx_struct_assembly_* tables from its output, so those
    # operators are gone by the time we'd want to apply them. Running
    # PrepWizard on the expanded structure is also the scientifically
    # correct order: protonation and H-bond optimization should see
    # interfacial contacts between symmetry partners.
    CIF_FOR_PREP="$RAW_CIF"
    if [[ "$USE_BIO_ASSEMBLY" -eq 1 ]]; then
        EXPANDED_RAW="${PROJECT_DIR}/${RAW_DIR}/${PDB_ID_LC}_assembly_${ASSEMBLY_ID}.cif"
        echo "--- bio-unit expansion: ${PDB_ID_LC} -> assembly ${ASSEMBLY_ID} ---"
        python - "$RAW_CIF" "$EXPANDED_RAW" "$ASSEMBLY_ID" <<'PY'
import sys
from pathlib import Path
from protein.extract._core import expand_to_biological_assembly

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
assembly_id = sys.argv[3]
out = expand_to_biological_assembly(src, assembly_id=assembly_id, output_path=dst)
print(f"Expanded CIF written to: {out}")
PY
        CIF_FOR_PREP="$EXPANDED_RAW"
    fi

    # --- 2. Detect candidate ligands from raw CIF --------------------------
    # Schrodinger's PDBx output drops _atom_site.group_PDB, so the
    # extractor's HETATM-based heuristic can't identify ligands post-prep.
    # Workaround: scan the RAW cif (which has group_PDB) for HETATM
    # residues with >= 6 heavy atoms (excluding common buffers/ions/
    # solvents/PTMs) and pass each resname to the extractor as
    # --include-resname. Logic lives in protein.extract.detect_resnames so
    # it's importable + unit-testable.
    echo "=== [2/4] Detecting ligands in ${PDB_ID_LC} ==="
    LIGAND_RESNAMES="$(python -m protein.extract.detect_resnames "$RAW_CIF")"

    if [[ -z "$LIGAND_RESNAMES" ]]; then
        echo "No candidate ligands detected in raw CIF. Extraction would be empty."
        echo "To extract anyway (e.g. to force a specific resname), edit this script."
        return 0
    fi

    echo "Detected ligand resnames: ${LIGAND_RESNAMES}"

    INCLUDE_FLAGS=()
    for resname in $LIGAND_RESNAMES; do
        INCLUDE_FLAGS+=(--include-resname "$resname")
    done

    # --- 3. Preprocess with Schrodinger PrepWizard -------------------------
    # Flags passed to protein.clean.preprocess:
    #   --disulfides          : perceive S-S bridges geometrically.
    #   --use-mae-intermediate: write .maegz, then structconvert -PDBx to
    #                           real PDBx/mmCIF. Without this, PrepWizard
    #                           emits old-style core-CIF with fractional
    #                           coords.
    #
    # ProtAssign (H-bond network optimization + HIS tautomer / ASN-GLN
    # flip resolution) runs automatically. If it crashes on a specific
    # structure, pass --skip-protassign to protein.clean.preprocess
    # manually to bypass it for that run.
    #
    # Absolute paths are required because Schrodinger's job system runs
    # in a staging directory and resolves relative paths against the
    # wrong cwd.
    #
    # DEBUG=1 adds --debug which implies -SAVE (preserves the job
    # archive with protassign.log), streams prepwizard output live, and
    # exports SCHRODINGER_PYTHON_DEBUG=1 / SCHRODINGER_MMLIBS_DEBUG=1 /
    # PYTHONUNBUFFERED=1 / PYTHONFAULTHANDLER=1 so protassign tracebacks
    # identify the offending residue by name. prepwizard's toplevel.py
    # does NOT accept -DEBUG as a CLI flag, so verbosity comes from env
    # vars.
    PREPROCESS_FLAGS=(--disulfides --use-mae-intermediate --overwrite)
    if [[ "$DEBUG" != "0" ]]; then
        PREPROCESS_FLAGS+=(--debug --save-job-archive --keep-intermediate)
        echo "--- DEBUG=1: Schrodinger job archive and intermediate .maegz will be preserved ---"
    fi

    # Skip when prepped CIF is already on disk unless FORCE_PREP=1 is set.
    # PrepWizard is the dominant pipeline cost (typically minutes per PDB);
    # caching the result lets you iterate on extraction-side code without
    # re-running protassign + minimization every time. The cache is keyed
    # on the prepped-CIF path, which already encodes whether bio-unit
    # expansion was applied (the upstream call passes the expanded CIF as
    # input).
    if [[ -s "$PREPPED_CIF" && "$FORCE_PREP" == "0" ]]; then
        echo "=== [3/4] PrepWizard: reusing cached ${PREPPED_CIF#${PROJECT_DIR}/} (set FORCE_PREP=1 to re-prep) ==="
    else
        echo "=== [3/4] Preprocessing ${PDB_ID_LC} ==="
        python -m protein.clean.preprocess \
            "${PREPROCESS_FLAGS[@]}" \
            "$CIF_FOR_PREP" \
            "$PREPPED_CIF"
    fi

    # --- 4. Extract ligand-centered pockets --------------------------------
    # --include-resname <X>     : force-include resnames found in step 2.
    # --allow-openbabel-fallback: Schrodinger's structconvert -PDBx drops
    #                             the entire _struct_conn loop, so
    #                             intra-ligand bonds are unknown. Fall
    #                             back to OpenBabel geometric bond
    #                             perception when that happens.
    #
    # Note: no --biological-assembly flag here. When expansion was
    # requested, it happened in step 1c on the raw CIF before PrepWizard,
    # so the prepped CIF already contains the expanded atoms. Schrodinger
    # strips the assembly metadata tables from its output, so applying
    # expansion post-prep would fail anyway.
    if [[ "$USE_BIO_ASSEMBLY" -eq 1 ]]; then
        echo "=== [4/4] Extracting pockets for ${PDB_ID_LC} (biological assembly ${ASSEMBLY_ID}) ==="
    else
        echo "=== [4/4] Extracting pockets for ${PDB_ID_LC} ==="
    fi
    python -m protein.extract.find_pocket_processed \
        "$PREPPED_CIF" \
        --output-dir "$POCKETS_DIR" \
        --radius 10.0 \
        "${INCLUDE_FLAGS[@]}" \
        --allow-openbabel-fallback

    echo
    echo "=== Done. Outputs in ${POCKETS_DIR}/: ==="
    ls "$POCKETS_DIR/"
}

# --- Dispatcher ------------------------------------------------------------
# Single mode: errors propagate (set -e is in effect).
if [[ -n "$PDB_ID" ]]; then
    run_one "$PDB_ID"
    exit 0
fi

# Batch mode: per-PDB failures are isolated by running run_one in a
# subshell. set -e in the subshell aborts that one PDB on failure; the
# parent loop captures the exit code and continues to the next entry.
if [[ ! -f "$PDB_LIST" ]]; then
    echo "PDB list file not found: $PDB_LIST" >&2
    exit 2
fi

passed=()
failed=()

# `|| [[ -n "$raw_line" ]]` catches a trailing line with no newline.
while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    pdb_id="${raw_line%%#*}"            # strip inline comments
    pdb_id="${pdb_id//[$'\t\r\n ']/}"   # strip whitespace
    [[ -z "$pdb_id" ]] && continue

    echo
    echo "########## $pdb_id ##########"
    rc=0
    (run_one "$pdb_id") || rc=$?
    if [[ "$rc" -eq 0 ]]; then
        passed+=("$pdb_id")
    else
        failed+=("$pdb_id")
        echo "!!! $pdb_id FAILED (rc=$rc) — continuing"
    fi
done < "$PDB_LIST"

echo
echo "=== Summary ==="
echo "Passed (${#passed[@]}): ${passed[*]:-none}"
echo "Failed (${#failed[@]}): ${failed[*]:-none}"

(( ${#failed[@]} == 0 )) || exit 1
