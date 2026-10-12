#!/bin/bash
set -e
# pipefail matters here: the training/inference command is piped into
# tee, and without it a crash inside conda run is masked and the script
# prints "finished" and exits 0.
set -o pipefail

# Inference + dscore scoring for LS-EEND on MSDWild or RAMC, under OUR
# scoring protocol so the number is directly comparable to our own systems.
#
#   ./scripts/run_infer_score.sh msdwild ./ckpt/simu/simu_1-8spk.ckpt
#   ./scripts/run_infer_score.sh ramc logs/finetune_ramc_8k/version_0/epoch=49-step=42050.ckpt
#
# Passing the simulated checkpoint gives the ZERO-SHOT number. That is not a
# method comparison -- the model has never seen the corpus. Locally it scored
# MSDWild 43.98 at collar 0.25, failing by merging turns (4,106 segments
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

# DIAPER_ROOT is CODE only (dscore + its env). The corpora live in their own
# tree -- DATA_ROOT -- and the 8 kHz copies prepare_8k_data.sh made are under
# OUT_ROOT. Defaults are the server's real paths, taken from our own
# custom-diaper/scripts/precompute_{msdwild,ramc}.sh.
DIAPER_ROOT="${DIAPER_ROOT:-/data/ocr/namvt17/custom-diaper}"
DATA_ROOT="${DATA_ROOT:-/data/ocr/namvt17/dataset/diarization}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT}"
DSCORE_SRC="${DSCORE_SRC:-${DIAPER_ROOT}/dscore}"
DSCORE_ENV="${DSCORE_ENV:-${DIAPER_ROOT}/dscore/.dscore}"
LSEEND_ENV="${LSEEND_ENV:-/data/ocr/namvt17/FS-EEND/.venv}"   # path-based, like custom-diaper/.venv
GPUS="${GPUS:-0}"

# References come from the ORIGINAL 16 kHz tree: resampling does not move any
# timestamp, so the reference is identical and we score against the one our
# own systems are scored against.
case "$CORPUS" in
  msdwild) MEDIAN=11; KALDI="${OUT_ROOT}/msdwild/kaldi_8k/test"; REF="${DATA_ROOT}/msdwild/rttms/few.val.rttm"; COLLAR="--collar 0.25" ;;
  ramc)    MEDIAN=1;  KALDI="${OUT_ROOT}/ramc/kaldi_8k/test";    REF="${DATA_ROOT}/ramc/kaldi/test/rttm";       COLLAR="" ;;
  *) echo "corpus must be msdwild or ramc" >&2; exit 1 ;;
esac

for d in "$KALDI" "$DSCORE_SRC"; do
  [ -e "$d" ] || { echo "missing: $d" >&2; exit 1; }
done
[ -d "$LSEEND_ENV" ] || { echo "missing conda env: $LSEEND_ENV" >&2; echo "create it or pass LSEEND_ENV=<path>; see scripts/README.md" >&2; exit 1; }
[ -f "$REF" ] || { echo "missing reference rttm: $REF" >&2; exit 1; }

TAG="$(basename "$CKPT" .ckpt)"
OUT="results/${CORPUS}_8k/${TAG}/median${MEDIAN}"
mkdir -p "$OUT"

# score.txt is the artifact worth keeping: the DER used to go only to the
# terminal, so a finished run left the RTTMs on disk but not the number.
SCORE="${OUT}/score.txt"
{
  echo "corpus:  ${CORPUS}"
  echo "ckpt:    ${CKPT}"
  echo "config:  conf/infer_${CORPUS}_8k.yaml"
  echo "median:  ${MEDIAN}"
  echo "collar:  ${COLLAR:-0}"
  echo "ref:     ${REF}"
  echo "kaldi:   ${KALDI}"
  echo "date:    $(date '+%F %T')"
  echo
} > "$SCORE"

echo "[$(date '+%F %T')] inference: ${CORPUS}  ckpt=${CKPT}  median=${MEDIAN}"
CUDA_VISIBLE_DEVICES="${GPUS}" conda run -p "${LSEEND_ENV}" --no-capture-output \
    python batch_infer_kaldi.py \
      --kaldi_dir "$KALDI" --configs "conf/infer_${CORPUS}_8k.yaml" \
      --ckpt "$CKPT" --out_dir "${OUT}/rttms" --gpus 0 --median "$MEDIAN"

SYS="${OUT}/sys.rttm"
cat "${OUT}"/rttms/*.rttm > "$SYS"
n_sys=$(ls "${OUT}"/rttms/*.rttm | wc -l)
n_ref=$(awk '{print $2}' "$REF" | sort -u | wc -l)
echo "recordings: ${n_sys} predicted, ${n_ref} in reference" | tee -a "$SCORE"
if [ "$n_sys" -ne "$n_ref" ]; then
  echo "WARNING: count mismatch -- a partial run scores as 100% miss on the missing files" | tee -a "$SCORE"
fi

score () {  # <label> <reference>
  echo "--- $1" | tee -a "$SCORE"
  conda run -p "${DSCORE_ENV}" --no-capture-output python -u "${DSCORE_SRC}/score.py" \
      $COLLAR -r "$2" -s "$SYS" 2>/dev/null | awk 'NR==1 || /OVERALL/' | tee -a "$SCORE"
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
echo "score saved to ${SCORE}"
