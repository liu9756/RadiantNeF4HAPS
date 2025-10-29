#!/bin/bash
#SBATCH -J cari_vA
#SBATCH -A PAS2177
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 02:00:00
#SBATCH -o cari_vA.%A_%a.out
#SBATCH -e cari_vA.%A_%a.err
set -euo pipefail

# -------- 0) 路径与任务号 --------
CARI_HOME="/users/PAS2177/liu9756/RadiantNeF4HAPS/data/raw/CARI_7A_DVD"
WORK_LOC="/users/PAS2177/liu9756/RadiantNeF4HAPS/work/loc_vA"
OUT_ANS="/users/PAS2177/liu9756/RadiantNeF4HAPS/work/ans_vA"
mkdir -p "$OUT_ANS"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SHORT_BASE=$(printf "L%04d" "$TASK_ID")
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
GCR_MODEL = 4
NVCR      = 0
VEHICLE   = NO!
CARI7RPT  = NO!
END       = END
EOF

# -------- 2) 选本分块 LOC，做“紧凑+封装”并写回短文件名 --------
RAW_PART=$(sed -n "$((TASK_ID+1))p" "$WORK_LOC/parts.txt")
[ -n "$RAW_PART" ] || { echo "parts.txt 没有第 $TASK_ID 项"; exit 3; }

RAW_SRC="$WORK_LOC/$RAW_PART"
SAN_BODY="$(mktemp)"
SAN_LOG="$CARI_HOME/${SHORT_BASE}.sanitize_log.txt"

# 2.1 生成紧凑数据体（去空格、修正W,180→179.99999、只保留12字段）
awk -v FS="," -v OFS="," '
function trim(s){gsub(/^[ \t]+|[ \t]+$/,"",s); return s}
BEGIN{ n_fix=0; n_keep=0; n_long=0 }
{
  raw=$0
  # 跳过空行/注释/原始START/STOP（我们统一注入）
  if (raw ~ /^[ \t]*$/) next
  if (raw ~ /^[ \t]*[C!]/) next
  if (raw ~ /^[ \t]*START-/) next
  if (raw ~ /^[ \t]*STOP--/)  next

  n = split(raw, a, ",")
  if (n < 12) next

  ns = trim(a[1]); lat = trim(a[2]); ew = trim(a[3]); lon = trim(a[4])
  un = trim(a[5]); alt = trim(a[6]); date=trim(a[7]); hr = trim(a[8])
  d  = trim(a[9]); p  = trim(a[10]); g  = trim(a[11]); s = trim(a[12])

  # 修正经度=180的西经
  if (ew ~ /^[Ww]$/ && (lon+0) == 180) { lon="179.99999"; n_fix++ }

  line = ns OFS lat OFS ew OFS lon OFS un OFS alt OFS date OFS hr OFS d OFS p OFS g OFS s
  if (length(line) > 66) n_long++
  print line; n_keep++
}
END{
  printf("kept=%d, fix_lon180=%d, too_long=%d\n", n_keep, n_fix, n_long) > "/dev/stderr"
}
' "$RAW_SRC" > "$SAN_BODY" 2> "$SAN_LOG"

# 2.2 注入 START/STOP 包裹，写回程序目录的短名文件（LF 行尾即可）
{
  echo "START--------------------------------------------------------"
  cat "$SAN_BODY"
  echo "STOP---------------------------------------------------------"
} > "$CARI_HOME/$SHORT_LOC"

# -------- 3) DEFAULT.INP：指向我们的短LOC（5行，2-4行前置空格）--------
cat > "$CARI_HOME/DEFAULT.INP" <<EOF
2002/01/00
 0
 0
 2
$SHORT_LOC
EOF
cp -f "$CARI_HOME/DEFAULT.INP" "$CARI_HOME/default.inp"

# -------- 4) 拷贝可执行到可执行临时目录，避免noexec --------
RUN_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
RUN_TMP="$(mktemp -d "${RUN_TMP_ROOT%/}/cari.${SLURM_JOB_ID:-$$}.XXXXXX")"
trap 'rm -rf "$RUN_TMP"' EXIT

CAND1="$CARI_HOME/cari7a_4.2.0(intel_linux)"
CAND2="$CARI_HOME/cari7a_4.2.0(intel_linux)/cari7a"
if   [ -f "$CAND1" ] && [ -x "$CAND1" ]; then BIN_SRC="$CAND1"
elif [ -f "$CAND2" ] && [ -x "$CAND2" ]; then BIN_SRC="$CAND2"
elif [ -f "$CAND1" ]; then chmod u+rx "$CAND1"; BIN_SRC="$CAND1"
elif [ -f "$CAND2" ]; then chmod u+rx "$CAND2"; BIN_SRC="$CAND2"
else echo "ERROR: CARI binary not found"; exit 2; fi
cp -f "$BIN_SRC" "$RUN_TMP/cari7a"; chmod u+rx "$RUN_TMP/cari7a"

# -------- 5) 运行（必须在程序目录）--------
cd "$CARI_HOME"
set +e
"$RUN_TMP/cari7a"
rc=$?
set -e

# -------- 6) 收集 ANS --------
mkdir -p "$OUT_ANS"
mv -f "${SHORT_BASE}.ANS" "$OUT_ANS/${SHORT_BASE}.ANS" 2>/dev/null || true

# 顺便把本次 LOC 及日志也存一份，便于追溯
cp -f "$CARI_HOME/$SHORT_LOC" "$OUT_ANS/${SHORT_BASE}.LOC"
cp -f "$SAN_LOG" "$OUT_ANS/${SHORT_BASE}.sanitize_log.txt"

exit "$rc"
