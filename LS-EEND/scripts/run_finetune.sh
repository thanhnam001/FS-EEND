#!/bin/bash
set -e

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
#   DIAPER_ROOT=... ./scripts/prepare_8k_data.sh
#
# BEFORE YOU RUN: read the COST block in conf/finetune_<corpus>_8k.yaml and
# choose max_epochs on purpose. Their 500 is calibrated to CALLHOME's small
# epoch; on a 66 h / 150 h corpus it is very expensive. MAX_EPOCHS here
# rewrites that one field into a run-local copy of the config so the original
# stays untouched and the value actually used is recorded in the log.
#
# batch_size is 16 in the config and is the main throughput lever: measured
# ~15.5 GB at 16, so it needs a >=24 GB card. Drop to 8 or 4 on a smaller one
# (4 fits 6 GB) -- but note 8 thrashed on a 6 GB laptop, so size the batch to
# the card rather than splitting the difference.
#
# Lightning writes to logs/<model_name>/version_N/. Re-running starts a new
# version rather than resuming; pass --checkpoint_resume to continue instead.

CORPUS="${1:?usage: run_finetune.sh <msdwild|ramc>}"
case "$CORPUS" in
  msdwild|ramc) ;;
  *) echo "corpus must be msdwild or ramc" >&2; exit 1 ;;
esac

LSEEND_ENV="${LSEEND_ENV:-lseend}"
GPUS="${GPUS:-0}"
LOG_DIR="${LOG_DIR:-logs/runs}"
CONFIG="conf/finetune_${CORPUS}_8k.yaml"
mkdir -p "$LOG_DIR"

RUN_CONFIG="$CONFIG"
if [ -n "${MAX_EPOCHS:-}" ]; then
  RUN_CONFIG="${LOG_DIR}/finetune_${CORPUS}_8k.max${MAX_EPOCHS}.yaml"
  sed "s|^  max_epochs:.*|  max_epochs: ${MAX_EPOCHS}|" "$CONFIG" > "$RUN_CONFIG"
  echo "max_epochs overridden to ${MAX_EPOCHS} in ${RUN_CONFIG}"
fi

echo "=================================================================="
echo "[$(date '+%F %T')] LS-EEND finetune: ${CORPUS}"
echo "  config: ${RUN_CONFIG}"
echo "  max_epochs: $(grep -m1 '^  max_epochs:' "$RUN_CONFIG" | tr -s ' ' | cut -d' ' -f3)"
echo "  batch_size: $(grep -m1 '^  batch_size:' "$RUN_CONFIG" | tr -s ' ' | cut -d' ' -f3)"
echo "  CUDA_VISIBLE_DEVICES=${GPUS}"
echo "=================================================================="

CUDA_VISIBLE_DEVICES="${GPUS}" conda run -n "${LSEEND_ENV}" --no-capture-output \
    python train_dia_fintun_real.py --configs "${RUN_CONFIG}" --gpus 0, \
    2>&1 | tee "${LOG_DIR}/finetune_${CORPUS}.log"

echo "[$(date '+%F %T')] finished. Next: ./scripts/run_infer_score.sh ${CORPUS} <ckpt>"
