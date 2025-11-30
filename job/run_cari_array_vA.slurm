#!/bin/bash
#SBATCH -J cari_vA
#SBATCH -A PAS2177
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 02:00:00
#SBATCH -o cari_vA.%j.out
#SBATCH -e cari_vA.%j.err

set -euo pipefail
ALT_KM=${ALT_KM:-12.0}

# ---- 0) 路径与任务号 ----
CARI_HOME="/users/PAS2177/liu9756/RadiantNeF4HAPS/data/raw/CARI_7A_DVD"
WORK_LOC="/users/PAS2177/liu9756/RadiantNeF4HAPS/work/loc_vA"
OUT_ANS_ROOT="/users/PAS2177/liu9756/RadiantNeF4HAPS/work/ans_vA"

ALT_STR=$(printf "%.1f" "$ALT_KM")
ALT_TAG="h${ALT_STR//./p}km"
OUT_ANS="${OUT_ANS_ROOT}/${ALT_TAG}"
mkdir -p "$OUT_ANS"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SHORT_BASE=$(printf "L%04d" "$TASK_ID")
SHORT_LOC="${SHORT_BASE}.LOC"

echo "[INFO] jobid=$SLURM_JOB_ID ALT_KM=$ALT_KM ALT_TAG=$ALT_TAG TASK_ID=$TASK_ID" >&2

# ---- 1) CARI.INI ----
cat > "$CARI_HOME/CARI.INI" <<'EOF'
DIAGNOSE  = NO!
VIEWER    = WRITE
OUTPUT    = FILE
OS        = LINUX
DISPLAY   = DOS
MENUS     = NO!
SUPERPOS  = NO!
GCR_MODEL = 4
NVCR      = 0
VEHICLE   = NO!
CARI7RPT  = NO!
END       = END
EOF

# ---- 2) 选本分块 LOC，做“紧凑+封装” ----
RAW_PART=$(sed -n "$((TASK_ID+1))p" "$WORK_LOC/parts.txt")
[ -n "$RAW_PART" ] || { echo "[ERR] parts.txt 没有第 $TASK_ID 项" >&2; exit 3; }
RAW_SRC="$WORK_LOC/$RAW_PART"
SAN_BODY="$(mktemp)"
SAN_LOG="$CARI_HOME/${SHORT_BASE}.sanitize_log.txt"
echo "[INFO] RAW_PART=$RAW_PART RAW_SRC=$RAW_SRC" >&2

awk -v FS="," -v OFS="," -v ALT_KM="$ALT_KM" '
function trim(s){gsub(/^[ \t]+|[ \t]+$/,"",s); return s}
BEGIN{ n_fix=0; n_keep=0; n_long=0 }
{
  raw=$0
  if (raw ~ /^[ \t]*$/) next
  if (raw ~ /^[ \t]*[C!]/) next
  if (raw ~ /^[ \t]*START-/) next
  if (raw ~ /^[ \t]*STOP--/)  next

  n = split(raw, a, ",")
  if (n < 12) next

  ns = trim(a[1]); lat = trim(a[2]); ew = trim(a[3]); lon = trim(a[4])
  un = trim(a[5]); alt = trim(a[6]); date=trim(a[7]); hr = trim(a[8])
  d  = trim(a[9]); p  = trim(a[10]); g  = trim(a[11]); s = trim(a[12])

  if (ew ~ /^[Ww]$/ && (lon+0) == 180) { lon="179.99999"; n_fix++ }
  # 覆盖高度为 ALT_KM（英尺）
  alt = sprintf("%.0f", ALT_KM * 3280.84)

  line = ns OFS lat OFS ew OFS lon OFS un OFS alt OFS date OFS hr OFS d OFS p OFS g OFS s
  if (length(line) > 66) n_long++
  print line; n_keep++
}
END{ printf("[INFO] awk kept=%d fix_lon180=%d too_long=%d\n", n_keep, n_fix, n_long) > "/dev/stderr" }
' "$RAW_SRC" > "$SAN_BODY" 2> "$SAN_LOG"

{
  echo "START--------------------------------------------------------"
  cat "$SAN_BODY"
  echo "STOP---------------------------------------------------------"
} > "$CARI_HOME/$SHORT_LOC"

# ---- 3) DEFAULT.INP ----
cat > "$CARI_HOME/DEFAULT.INP" <<EOF
2002/01/00
 0
 0
 2
$SHORT_LOC
EOF
cp -f "$CARI_HOME/DEFAULT.INP" "$CARI_HOME/default.inp"

# ---- 4) 拷贝可执行 ----
RUN_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
RUN_TMP="$(mktemp -d "${RUN_TMP_ROOT%/}/cari.${SLURM_JOB_ID:-$$}.XXXXXX")"
trap 'rm -rf "$RUN_TMP"' EXIT

CAND1="$CARI_HOME/cari7a_4.2.0(intel_linux)"
CAND2="$CARI_HOME/cari7a_4.2.0(intel_linux)/cari7a"
if   [ -f "$CAND1" ] && [ -x "$CAND1" ]; then BIN_SRC="$CAND1"
elif [ -f "$CAND2" ] && [ -x "$CAND2" ]; then BIN_SRC="$CAND2"
elif [ -f "$CAND1" ]; then chmod u+rx "$CAND1"; BIN_SRC="$CAND1"
elif [ -f "$CAND2" ]; then chmod u+rx "$CAND2"; BIN_SRC="$CAND2"
else echo "[ERR] CARI binary not found" >&2; exit 2; fi
cp -f "$BIN_SRC" "$RUN_TMP/cari7a"; chmod u+rx "$RUN_TMP/cari7a"

# ---- 5) 运行 ----
cd "$CARI_HOME"
echo "[INFO] running $RUN_TMP/cari7a in $PWD" >&2
set +e
"$RUN_TMP/cari7a"
rc=$?
set -e
echo "[INFO] cari7a exit code=$rc" >&2

# ---- 6) 收集 ANS ----
mkdir -p "$OUT_ANS"
mv -f "${SHORT_BASE}.ANS" "$OUT_ANS/${SHORT_BASE}.ANS" 2>/dev/null || true
cp -f "$CARI_HOME/$SHORT_LOC" "$OUT_ANS/${SHORT_BASE}.LOC"
cp -f "$SAN_LOG"             "$OUT_ANS/${SHORT_BASE}.sanitize_log.txt"

printf '{"alt_km": %.1f, "task_id": %d, "alt_tag": "%s", "src_part": "%s"}\n' \
  "$ALT_KM" "$TASK_ID" "$ALT_TAG" "$RAW_PART" > "$OUT_ANS/${SHORT_BASE}.meta.json"

exit "$rc"
