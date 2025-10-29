#!/bin/bash
#SBATCH -J cari_vB
#SBATCH -A PAS2177
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 02:00:00
#SBATCH -o cari_vB.%A_%a.out
#SBATCH -e cari_vB.%A_%a.err
set -euo pipefail
export LANG=C
export LC_ALL=C

# ---- paths ----
CARI_HOME="/users/PAS2177/liu9756/RadiantNeF4HAPS/data/raw/CARI_7A_DVD"
WORK_LOC_BASE="/users/PAS2177/liu9756/RadiantNeF4HAPS/work/loc_vB"
OUT_BASE="/users/PAS2177/liu9756/RadiantNeF4HAPS/work/ans_vB"
mkdir -p "$OUT_BASE"

# ---- which date bucket? (export DATE_KEY=2002_01_00 etc.) ----
DATE_KEY=${DATE_KEY:? "DATE_KEY not set (e.g., 2002_01_00)"}
WORK_LOC="$WORK_LOC_BASE/$DATE_KEY"
OUT_ANS="$OUT_BASE/$DATE_KEY"
mkdir -p "$OUT_ANS"

# ---- short names per task ----
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SHORT_BASE=$(printf "VB_%s_%04d" "$DATE_KEY" "$TASK_ID")
SHORT_LOC="${SHORT_BASE}.LOC"


# -------- 1) CARI.INI 固定为无菜单/文件输出/Linux --------
cat > "$CARI_HOME/CARI.INI" <<'EOF'
DIAGNOSE  = NO!
VIEWER    = WRITE
OUTPUT    = FILE
OS        = LINUX
DISPLAY   = DOS
MENUS     = NO!
SUPERPOS  = NO!
GCR_MODEL = 3
NVCR      = 0
VEHICLE   = NO!
CARI7RPT  = NO!
END       = END
EOF



# ---- pick this part ----
RAW_PART=$(sed -n "$((TASK_ID+1))p" "$WORK_LOC/parts.txt")
[ -n "$RAW_PART" ] || { echo "parts.txt has no line $TASK_ID"; exit 3; }
RAW_SRC="$WORK_LOC/$RAW_PART"


# ---- sanitize to 12 fields & wrap START/STOP already in file; just copy ----
cp -f "$RAW_SRC" "$CARI_HOME/$SHORT_LOC"

# -------- 3) DEFAULT.INP：指向我们的短LOC（5行，2-4行前置空格）--------
cat > "$CARI_HOME/DEFAULT.INP" <<EOF
2002/01/00
 0
 0
 2
$SHORT_LOC
EOF
cp -f "$CARI_HOME/DEFAULT.INP" "$CARI_HOME/default.inp"

# ---- run from CARI_HOME; use temp copy of binary to avoid noexec ----
RUN_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
RUN_TMP="$(mktemp -d "${RUN_TMP_ROOT%/}/cari.${SLURM_JOB_ID:-$$}.XXXXXX")"
trap 'rm -rf "$RUN_TMP"' EXIT

CAND1="$CARI_HOME/cari7a_4.2.0(intel_linux)"
CAND2="$CARI_HOME/cari7a_4.2.0(intel_linux)/cari7a"
if [ -f "$CAND1" ] && [ -x "$CAND1" ]; then BIN_SRC="$CAND1"
elif [ -f "$CAND2" ] && [ -x "$CAND2" ]; then BIN_SRC="$CAND2"
elif [ -f "$CAND1" ]; then chmod u+rx "$CAND1"; BIN_SRC="$CAND1"
elif [ -f "$CAND2" ]; then chmod u+rx "$CAND2"; BIN_SRC="$CAND2"
else echo "ERROR: CARI binary not found"; exit 2; fi
cp -f "$BIN_SRC" "$RUN_TMP/cari7a"; chmod u+rx "$RUN_TMP/cari7a"

cd "$CARI_HOME"
set +e
"$RUN_TMP/cari7a"
rc=$?
set -e

# ---- collect outputs ----
mv -f "${SHORT_BASE%.LOC}.ANS" "$OUT_ANS/${SHORT_BASE%.LOC}.ANS" 2>/dev/null || true
cp -f "$CARI_HOME/$SHORT_LOC" "$OUT_ANS/$SHORT_LOC"

exit "$rc"
