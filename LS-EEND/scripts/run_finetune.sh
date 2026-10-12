#!/bin/bash
set -e
# pipefail matters here: the training/inference command is piped into
# tee, and without it a crash inside conda run is masked and the script
# prints "finished" and exits 0.
set -o pipefail

# Finetunes LS-EEND on MSDWild or RAMC from the authors' released phase-1
# simulated checkpoint. One training stage only -- their phase 1 is already
# done for us, unlike EEND-EDA which has no released weights at all.
#
#   ./scripts/run_finetune.sh msdwild
#   ./scripts/run_finetune.sh ramc
#   GPUS=0 MAX_EPOCHS=50 ./scripts/run_finetune.sh msdwild
#
# PREREQUISITES (both idempotent, run once):
#   ./scripts/download_simu_ckpt.sh
#   ./scripts/prepare_8k_data.sh
#
# BUDGET: the authors' own, settled 2026-10-12 -- max_epochs 500, their
# early_stop_epoch (200 MSDWild / 500 RAMC), lr 1e-5, batch_size 4. Our own
# finetune protocol is also 500 epochs, so this is matched on both sides, not
# a deviation. MAX_EPOCHS= below overrides it into a run-local config copy;
# that is for smoke tests, not for a run you intend to report.
#
# BEFORE YOU RUN read the COST block in conf/finetune_<corpus>_8k.yaml. At
# batch 4 this is ~2,975 steps/epoch (MSDWild) / ~3,366 (RAMC), and on a 6 GB
# laptop that was 67 / 152 days for the full 500. The server is faster by an
# unmeasured factor: this script logs s/step, so read it a few minutes in and
# multiply before committing the machine.
#
# Lightning writes to logs/<model_name>/version_N/. Re-running starts a new
# version rather than resuming; pass --checkpoint_resume to continue instead.

CORPUS="${1:?usage: run_finetune.sh <msdwild>}"
case "$CORPUS" in
  msdwild) ;;
  ramc)
    echo "RAMC is not a single-stage finetune." >&2
    echo "Its average recording is 31.1 min, so by the paper's rule it takes the" >&2
    echo "AMI schedule: 200 -> 400 -> 800 -> 1600 s. conf/finetune_ramc_8k.yaml is" >&2
    echo "only the stage-1 template; running it alone trains 10 epochs at 200 s." >&2
    echo "Use:  ./scripts/run_finetune_ramc_progressive.sh" >&2
    exit 1 ;;
  *) echo "corpus must be msdwild (RAMC uses run_finetune_ramc_progressive.sh)" >&2; exit 1 ;;
esac

LSEEND_ENV="${LSEEND_ENV:-/data/ocr/namvt17/FS-EEND/.venv}"   # path-based, like custom-diaper/.venv
GPUS="${GPUS:-0}"
LOG_DIR="${LOG_DIR:-logs/runs}"
CONFIG="conf/finetune_${CORPUS}_8k.yaml"
# OUT_ROOT is where prepare_8k_data.sh wrote the 8 kHz Kaldi dirs; it defaults
# to the server's corpus tree (see that script's header). The config's own
# train_data_dir/val_data_dir are only defaults -- they are rewritten here so
# there is one source of truth for paths.
DATA_ROOT="${DATA_ROOT:-/data/ocr/namvt17/dataset/diarization}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT}"
CKPT="${CKPT:-./ckpt/simu/simu_1-8spk.ckpt}"
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

TRAIN_DIR="${OUT_ROOT}/${CORPUS}/kaldi_8k/train"
VAL_DIR="${OUT_ROOT}/${CORPUS}/kaldi_8k/dev"
for d in "$TRAIN_DIR" "$VAL_DIR"; do
  [ -d "$d" ] || { echo "missing: $d -- run ./scripts/prepare_8k_data.sh first" >&2; exit 1; }
done
[ -d "$LSEEND_ENV" ] || { echo "missing conda env: $LSEEND_ENV" >&2; echo "create it or pass LSEEND_ENV=<path>; see scripts/README.md" >&2; exit 1; }
[ -s "$CKPT" ] || { echo "missing init checkpoint: $CKPT -- see ./scripts/download_simu_ckpt.sh" >&2; exit 1; }

# Always write a run-local copy, so the config that actually ran is recorded
# next to the log and the checked-in one is never mutated.
RUN_CONFIG="${LOG_DIR}/finetune_${CORPUS}_8k.run.yaml"
sed -e "s|^  train_data_dir:.*|  train_data_dir: ${TRAIN_DIR}|"     -e "s|^  val_data_dir:.*|  val_data_dir: ${VAL_DIR}|"     -e "s|^  init_ckpt:.*|  init_ckpt: ${CKPT}|"     "$CONFIG" > "$RUN_CONFIG"
if [ -n "${MAX_EPOCHS:-}" ]; then
  sed -i "s|^  max_epochs:.*|  max_epochs: ${MAX_EPOCHS}|" "$RUN_CONFIG"
  echo "max_epochs overridden to ${MAX_EPOCHS}"
fi

echo "=================================================================="
echo "[$(date '+%F %T')] LS-EEND finetune: ${CORPUS}"
echo "  config: ${RUN_CONFIG}  (copy of ${CONFIG})"
echo "  train: ${TRAIN_DIR}"
echo "  valid: ${VAL_DIR}"
echo "  init:  ${CKPT}"
echo "  max_epochs: $(grep -m1 '^  max_epochs:' "$RUN_CONFIG" | tr -s ' ' | cut -d' ' -f3)"
echo "  batch_size: $(grep -m1 '^  batch_size:' "$RUN_CONFIG" | tr -s ' ' | cut -d' ' -f3)"
echo "  CUDA_VISIBLE_DEVICES=${GPUS}"
echo "=================================================================="

start_pruner "logs/finetune_${CORPUS}_8k" "${LOG_DIR}/prune_${CORPUS}.log"

CUDA_VISIBLE_DEVICES="${GPUS}" conda run -p "${LSEEND_ENV}" --no-capture-output \
    python train_dia_fintun_real.py --configs "${RUN_CONFIG}" --gpus 0, \
    2>&1 | tee "${LOG_DIR}/finetune_${CORPUS}.log"

stop_pruner

echo "[$(date '+%F %T')] finished. Next: ./scripts/run_infer_score.sh ${CORPUS} <ckpt>"
