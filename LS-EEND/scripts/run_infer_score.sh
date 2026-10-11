#!/bin/bash
set -e

# Inference + dscore scoring for LS-EEND on MSDWild or RAMC, under OUR
# scoring protocol so the number is directly comparable to our own systems.
#
#   ./scripts/run_infer_score.sh msdwild ./ckpt/simu/simu_1-8spk.ckpt
#   ./scripts/run_infer_score.sh ramc logs/finetune_ramc_8k/version_0/epoch=49-step=42050.ckpt
#
# Passing the simulated checkpoint gives the ZERO-SHOT number. That is not a
# method comparison -- the model has never seen the corpus. Locally it scored
# MSDWild 43.74 at collar 0.25, failing by merging turns (4,106 segments
# averaging 10.07 s against a reference of 14,209 at 2.53 s). Use a finetuned
# checkpoint for anything you intend to report.
#
# Per-corpus conventions, applied automatically:
#   MSDWild -- median 11, dscore --collar 0.25, reference rttms/few.val.rttm
#   RAMC    -- median 1,  dscore collar 0,     reference kaldi/test/rttm with
#              G00000000 REMOVED. G is a non-speech/unintelligible bin, not a
#              speaker; the DiaPer authors scored RAMC without it (reproducing
#              their 21.11 exactly), so that is the convention we publish. The
#              with-G number is also printed since the official RAMC baseline
#              keeps it -- always say which convention a RAMC number uses.
#
# Checkpoint averaging: our protocol averages the last N checkpoints, and so
# does theirs (see the test block in train_dia_fintun_real.py). This script
# scores ONE checkpoint; average first if you want the protocol-matched number.

CORPUS="${1:?usage: run_infer_score.sh <msdwild|ramc> <ckpt>}"
CKPT="${2:?usage: run_infer_score.sh <msdwild|ramc> <ckpt>}"

DIAPER_ROOT="${DIAPER_ROOT:-/data/ocr/namvt17/custom-diaper}"
DSCORE_SRC="${DSCORE_SRC:-${DIAPER_ROOT}/dscore}"
DSCORE_ENV="${DSCORE_ENV:-${DIAPER_ROOT}/dscore/.dscore}"
LSEEND_ENV="${LSEEND_ENV:-lseend}"
GPUS="${GPUS:-0}"
DB="${DIAPER_ROOT}/database"

case "$CORPUS" in
  msdwild) MEDIAN=11; KALDI="${DB}/msdwild/kaldi_8k/test"; REF="${DB}/msdwild/rttms/few.val.rttm"; COLLAR="--collar 0.25" ;;
  ramc)    MEDIAN=1;  KALDI="${DB}/ramc/kaldi_8k/test";    REF="${DB}/ramc/kaldi/test/rttm";       COLLAR="" ;;
  *) echo "corpus must be msdwild or ramc" >&2; exit 1 ;;
esac

TAG="$(basename "$CKPT" .ckpt)"
OUT="results/${CORPUS}_8k/${TAG}/median${MEDIAN}"
mkdir -p "$OUT"

echo "[$(date '+%F %T')] inference: ${CORPUS}  ckpt=${CKPT}  median=${MEDIAN}"
CUDA_VISIBLE_DEVICES="${GPUS}" conda run -n "${LSEEND_ENV}" --no-capture-output \
    python batch_infer_kaldi.py \
      --kaldi_dir "$KALDI" --configs "conf/infer_${CORPUS}_8k.yaml" \
      --ckpt "$CKPT" --out_dir "${OUT}/rttms" --gpus 0 --median "$MEDIAN"

SYS="${OUT}/sys.rttm"
cat "${OUT}"/rttms/*.rttm > "$SYS"
n_sys=$(ls "${OUT}"/rttms/*.rttm | wc -l)
n_ref=$(awk '{print $2}' "$REF" | sort -u | wc -l)
echo "recordings: ${n_sys} predicted, ${n_ref} in reference"
[ "$n_sys" -eq "$n_ref" ] || echo "WARNING: count mismatch -- a partial run scores as 100% miss on the missing files"

score () {  # <label> <reference>
  echo "--- $1"
  conda run -p "${DSCORE_ENV}" --no-capture-output python -u "${DSCORE_SRC}/score.py" \
      $COLLAR -r "$2" -s "$SYS" 2>/dev/null | awk 'NR==1 || /OVERALL/'
}

if [ "$CORPUS" = "ramc" ]; then
  NOG="${OUT}/ref_noG.rttm"
  awk '$8!="G00000000"' "$REF" > "$NOG"
  score "RAMC collar 0, WITHOUT G00000000  (the convention we publish)" "$NOG"
  score "RAMC collar 0, with G00000000     (official RAMC baseline convention)" "$REF"
else
  score "MSDWild collar 0.25" "$REF"
fi

echo "[$(date '+%F %T')] done -> ${OUT}"
