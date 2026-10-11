#!/bin/bash
set -e

# Fetches the authors' phase-1 simulated checkpoint (1-8 speakers), which is
# what both finetune configs initialise from.
#
# Google Drive serves a virus-scan interstitial for files this size, so the
# plain uc?export=download URL returns HTML rather than the model. Request
# once, scrape the confirm form's uuid, then re-request with confirm=t.
#
#   ./scripts/download_simu_ckpt.sh
# Writes ckpt/simu/simu_1-8spk.ckpt (~48 MB).

ID="1uWY8JvjHJJ-SvGiNS-6s3q10g4CY2ePt"
OUT="ckpt/simu/simu_1-8spk.ckpt"
mkdir -p "$(dirname "$OUT")"

if [ -s "$OUT" ]; then
  echo "already present: $OUT"
  exit 0
fi

tmp="$(mktemp)"
curl -sL "https://drive.usercontent.google.com/download?id=${ID}&export=download" -o "$tmp"
UUID="$(grep -oE 'name="uuid" value="[^"]+"' "$tmp" | sed 's/.*value="//;s/"//')"
if [ -z "$UUID" ]; then
  echo "could not find the confirm uuid; Drive may have changed its flow." >&2
  echo "Download manually from the LS-EEND README table into ${OUT}." >&2
  rm -f "$tmp"; exit 1
fi
rm -f "$tmp"

curl -L "https://drive.usercontent.google.com/download?id=${ID}&export=download&confirm=t&uuid=${UUID}" -o "$OUT"
ls -la "$OUT"
echo "expect ~49977074 bytes; a few-KB file means an HTML error page was saved."
