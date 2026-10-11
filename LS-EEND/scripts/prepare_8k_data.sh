#!/bin/bash
set -e

# Builds the 8 kHz Kaldi dirs LS-EEND needs, from our 16 kHz ones.
#
# WHY 8 kHz: the released checkpoints' input_projection is (256, 345) =
# 23 mels x (2*7+1) context, and datasets/feature.py hard-wires
# logmel23_cummn to sr=8000 and never resamples. Our own pipeline is 40 mels
# at 16 kHz (= 600), so the released weights cannot take our features. Using
# their weights therefore requires 8 kHz input. State this in any comparison:
# it is near-lossless for RAMC (telephony) but a real handicap on MSDWild
# (wideband, in-the-wild video).
#
# Resampling preserves wall-clock time, so segments/rttm are copied verbatim
# and the scoring reference is unchanged (verified locally: durations match to
# <1 ms and the copied rttm is byte-identical to the 16 kHz dir's).
#
# Run from the LS-EEND dir on the server:
#   DIAPER_ROOT=/data/ocr/namvt17/custom-diaper ./scripts/prepare_8k_data.sh
# Idempotent: already-converted wavs are skipped, so re-running is cheap.

DIAPER_ROOT="${DIAPER_ROOT:-/data/ocr/namvt17/custom-diaper}"
PY="${PY:-python}"
DB="${DIAPER_ROOT}/database"

for corpus in msdwild ramc; do
  for split in train dev test; do
    src="${DB}/${corpus}/kaldi/${split}"
    out="${DB}/${corpus}/kaldi_8k/${split}"
    wavdir="${DB}/${corpus}_wavs_8k"
    if [ ! -d "$src" ]; then
      echo "[skip] no such dir: $src"
      continue
    fi
    echo "[$(date '+%F %T')] ${corpus}/${split}: ${src} -> ${out}"
    "$PY" scripts/make_8k_kaldi.py "$src" "$out" "$wavdir"
  done
done

echo "[$(date '+%F %T')] done. Disk: expect ~4.6 GB (MSDWild) + ~10.4 GB (RAMC)."
