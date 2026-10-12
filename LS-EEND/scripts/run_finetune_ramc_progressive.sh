#!/bin/bash
set -e
set -o pipefail

# Progressive 4-stage finetune of LS-EEND on RAMC: 200 -> 400 -> 800 -> 1600 s.
#
# WHY FOUR STAGES. The paper (Sec. IV-C) sets the adaptation audio length by
# the corpus's average recording duration: "Audio length is increased from
# 200 s until it approximates the average recording duration of each
# dataset", giving 200 s for CALLHOME, 200->400 for DIHARD II/III, and
# 200->400->800->1600 for AMI. RAMC's average recording is 31.1 min
# (measured), against AMI's 35.6 min -- so RAMC is AMI-shaped and takes AMI's
# schedule. Fig. 3 shows the shortcut 400s->1600s loses real DER.
#
# Each stage is their AMI budget: 10 epochs, lr 1e-5, Adam, PIT loss.
# 40 epochs total, ~12x cheaper than the single 500-epoch run this replaces.
#
# Stage 1 starts from the released simulated checkpoint. Each later stage
# starts from the AVERAGED last-5 of the stage before it -- averaged rather
# than last, because that is how the authors hand off between their own
# stages (the released init, simu_1-8spk, is itself an average: their file is
# named ..._1-8spk_16_25_avg_model.ckpt, epochs 16-25 of a 25-epoch step).
#
#   ./scripts/run_finetune_ramc_progressive.sh
#   GPUS=1 ./scripts/run_finetune_ramc_progressive.sh
#   STAGES="800 1600" ./scripts/run_finetune_ramc_progressive.sh   # resume later stages
#
# Resumable: a stage whose averaged checkpoint already exists is skipped, so
# re-running after a crash picks up where it stopped.

LSEEND_ENV="${LSEEND_ENV:-/data/ocr/namvt17/FS-EEND/.venv}"
DATA_ROOT="${DATA_ROOT:-/data/ocr/namvt17/dataset/diarization}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT}"
GPUS="${GPUS:-0}"
LOG_DIR="${LOG_DIR:-logs/runs}"
BASE_CONFIG="${BASE_CONFIG:-conf/finetune_ramc_8k.yaml}"
CKPT="${CKPT:-./ckpt/simu/simu_1-8spk.ckpt}"
# The canonical schedule. STAGES may name a subset to resume, e.g.
# STAGES="800 1600"; stage NUMBERS are always looked up in ALL_STAGES so a
# partial run keeps the same dir names and still finds earlier stages.
ALL_STAGES="${ALL_STAGES:-200 400 800 1600}"
STAGES="${STAGES:-$ALL_STAGES}"
AVG_N="${AVG_N:-5}"
EPOCHS="${EPOCHS:-10}"

# Per-stage batch size. The authors use 8 throughout on AMI, but they also
# report hitting GPU memory limits at the longest lengths ("the audio length
# when fine-tuning on AMI is increased only up to 800 s, due to the limit of
# GPU memory"). Halving at 800 s and again at 1600 s keeps the schedule
# runnable; override with BATCHES="8 8 8 8" on a large card and say which was
# used, since batch size is a recipe deviation.
BATCHES="${BATCHES:-8 8 4 2}"

[ -d "$LSEEND_ENV" ] || { echo "missing conda env: $LSEEND_ENV" >&2; exit 1; }
[ -s "$CKPT" ] || { echo "missing init checkpoint: $CKPT -- see ./scripts/download_simu_ckpt.sh" >&2; exit 1; }
for d in "${OUT_ROOT}/ramc/kaldi_8k/train" "${OUT_ROOT}/ramc/kaldi_8k/dev"; do
  [ -d "$d" ] || { echo "missing: $d -- run ./scripts/prepare_8k_data.sh first" >&2; exit 1; }
done
mkdir -p "$LOG_DIR"

# AUTO-PRUNE. save_top_k: -1 is their setting and keeps every epoch (~150 MB),
# which is ~75 GB over 500 epochs. No config setting means "keep the last N"
# (with a monitor set, save_top_k keeps the BEST N), so a watcher is started
# alongside training and stopped with it. PRUNE_KEEP=0 disables.
PRUNE_KEEP="${PRUNE_KEEP:-6}"
PRUNE_EVERY="${PRUNE_EVERY:-600}"
PRUNER_PID=""
stop_pruner () {
  if [ -n "$PRUNER_PID" ]; then kill "$PRUNER_PID" 2>/dev/null || true; wait "$PRUNER_PID" 2>/dev/null || true; PRUNER_PID=""; fi
}
start_pruner () {  # <model log dir> <log file>
  [ "$PRUNE_KEEP" -gt 0 ] || return 0
  KEEP="$PRUNE_KEEP" WATCH="$PRUNE_EVERY" ./scripts/prune_ckpts.sh "$1" --yes >> "$2" 2>&1 &
  PRUNER_PID=$!
  echo "  auto-prune: newest ${PRUNE_KEEP} ckpts every ${PRUNE_EVERY}s (pid ${PRUNER_PID}, log $2)"
}
trap stop_pruner EXIT INT TERM

PY="${LSEEND_ENV}/bin/python"
read -r -a stage_arr <<< "$STAGES"
read -r -a all_arr <<< "$ALL_STAGES"
read -r -a batch_arr <<< "$BATCHES"

stage_index () {  # position of $1 in ALL_STAGES, 1-based
  local i=1 x
  for x in "${all_arr[@]}"; do
    [ "$x" = "$1" ] && { echo "$i"; return 0; }
    i=$(( i + 1 ))
  done
  echo "stage ${1}s is not in ALL_STAGES=\"${ALL_STAGES}\"" >&2
  return 1
}

init="$CKPT"
idx=0
for secs in "${stage_arr[@]}"; do
  n="$(stage_index "$secs")" || exit 1
  chunk=$(( secs * 10 ))            # 0.1 s units: 200 s -> 2000
  batch="${batch_arr[$(( n - 1 ))]:-${batch_arr[-1]}}"
  name="finetune_ramc_8k_s${n}_${secs}s"
  avg_out="logs/${name}/avg_last${AVG_N}.ckpt"

  if [ "$idx" -eq 0 ] && [ "$n" -gt 1 ]; then
    prev_secs="${all_arr[$(( n - 2 ))]}"
    prev="logs/finetune_ramc_8k_s$(( n - 1 ))_${prev_secs}s/avg_last${AVG_N}.ckpt"
    [ -s "$prev" ] || { echo "resuming at stage ${n} but stage $(( n - 1 )) output is missing: $prev" >&2; exit 1; }
    init="$prev"
    echo "[$(date '+%F %T')] resuming at stage ${n}, initialising from ${prev}"
  fi

  if [ -s "$avg_out" ]; then
    echo "[$(date '+%F %T')] stage ${n} (${secs}s) already done -> ${avg_out}, skipping"
    init="$avg_out"; idx=$(( idx + 1 )); continue
  fi

  run_config="${LOG_DIR}/${name}.run.yaml"
  sed -e "s|^  model_name:.*|  model_name: ${name}|" \
      -e "s|^  init_ckpt:.*|  init_ckpt: ${init}|" \
      -e "s|^  batch_size:.*|  batch_size: ${batch}|" \
      -e "s|^  max_epochs:.*|  max_epochs: ${EPOCHS}|" \
      -e "s|^  chunk_size:.*|  chunk_size: ${chunk}|" \
      -e "s|^  chunk_step:.*|  chunk_step: ${chunk}|" \
      -e "s|^  train_data_dir:.*|  train_data_dir: ${OUT_ROOT}/ramc/kaldi_8k/train|" \
      -e "s|^  val_data_dir:.*|  val_data_dir: ${OUT_ROOT}/ramc/kaldi_8k/dev|" \
      "$BASE_CONFIG" > "$run_config"

  echo "=================================================================="
  echo "[$(date '+%F %T')] RAMC stage ${n}/${#stage_arr[@]}: ${secs}s"
  echo "  chunk_size: ${chunk} (= ${secs} s)   batch: ${batch}   epochs: ${EPOCHS}"
  echo "  init:   ${init}"
  echo "  config: ${run_config}"
  echo "=================================================================="

  start_pruner "logs/${name}" "${LOG_DIR}/prune_ramc.log"

  CUDA_VISIBLE_DEVICES="${GPUS}" conda run -p "${LSEEND_ENV}" --no-capture-output \
      "$PY" train_dia_fintun_real.py --configs "${run_config}" --gpus 0, \
      2>&1 | tee "${LOG_DIR}/${name}.log"

  # Stop before averaging so the pruner cannot delete a file mid-read.
  stop_pruner

  # Newest version dir Lightning just wrote for this stage.
  vdir="$(find "logs/${name}" -maxdepth 1 -type d -name 'version_*' -printf '%T@ %p\n' \
          | sort -nr | head -1 | cut -d' ' -f2-)"
  [ -n "$vdir" ] || { echo "no version dir under logs/${name}" >&2; exit 1; }

  "$PY" scripts/average_ckpts.py "$vdir" --last-n "$AVG_N" -o "$avg_out"
  echo "[$(date '+%F %T')] stage ${n} done -> ${avg_out}"
  init="$avg_out"
  idx=$(( idx + 1 ))
done

echo
echo "[$(date '+%F %T')] all stages complete. Final model: ${init}"
echo "Score it with:"
echo "  ./scripts/run_infer_score.sh ramc ${init}"
