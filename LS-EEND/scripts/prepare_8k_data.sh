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
# PATHS. On the server the corpora do NOT live under the repo -- they are in
# their own tree, which is what DATA_ROOT points at. Taken from our own
# scripts/precompute_{msdwild,ramc}.sh in custom-diaper, which read
#   $DATA_ROOT/msdwild/kaldi/{train,dev,test}   $DATA_ROOT/msdwild_wavs
#   $DATA_ROOT/ramc/kaldi/{train,dev,test}      $DATA_ROOT/ramc/MDT2021S003/WAV
# and write their precompute caches into the same tree, so it is writable.
# Locally that same structure sits under custom-diaper/database, so a local
# dev run just overrides DATA_ROOT.
#
#   # server (defaults are already the server's paths):
#   ./scripts/prepare_8k_data.sh
#   # local dev:
#   DATA_ROOT=D:/Python/custom-diaper/database ./scripts/prepare_8k_data.sh
#   # 8 kHz output somewhere else (e.g. a bigger disk):
#   OUT_ROOT=/data/ocr/namvt17/dataset/diarization_8k ./scripts/prepare_8k_data.sh
#
# Idempotent: already-converted wavs are skipped, so re-running is cheap.

DATA_ROOT="${DATA_ROOT:-/data/ocr/namvt17/dataset/diarization}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT}"
# Needs numpy, soundfile and scipy. Default to the project env rather than
# whatever `python` happens to resolve to on the server.
LSEEND_ENV="${LSEEND_ENV:-/data/ocr/namvt17/FS-EEND/.venv}"
PY="${PY:-${LSEEND_ENV}/bin/python}"

# Accept either an absolute interpreter path or a bare name on PATH.
command -v "$PY" >/dev/null 2>&1 || { echo "no python at: $PY" >&2; echo "pass PY=/path/to/python or LSEEND_ENV=<env>" >&2; exit 1; }

if [ ! -d "$DATA_ROOT" ]; then
  echo "DATA_ROOT does not exist: $DATA_ROOT" >&2
  echo "Point it at the tree holding msdwild/ and ramc/ (see header)." >&2
  exit 1
fi

echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_ROOT=$OUT_ROOT"

found=0
for corpus in msdwild ramc; do
  for split in train dev test; do
    src="${DATA_ROOT}/${corpus}/kaldi/${split}"
    out="${OUT_ROOT}/${corpus}/kaldi_8k/${split}"
    wavdir="${OUT_ROOT}/${corpus}_wavs_8k"
    if [ ! -d "$src" ]; then
      echo "[skip] no such dir: $src"
      continue
    fi
    found=$((found + 1))
    echo "[$(date '+%F %T')] ${corpus}/${split}: ${src} -> ${out}"
    "$PY" scripts/make_8k_kaldi.py "$src" "$out" "$wavdir"
  done
done

if [ "$found" -eq 0 ]; then
  echo "NOTHING WAS CONVERTED: no <corpus>/kaldi/<split> dirs found under $DATA_ROOT" >&2
  exit 1
fi

echo "[$(date '+%F %T')] done, ${found}/6 splits converted."
echo "Disk: expect ~4.6 GB (MSDWild) + ~10.4 GB (RAMC) under ${OUT_ROOT}."
