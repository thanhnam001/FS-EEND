#!/bin/bash
set -e

# Puts the authors' phase-1 simulated checkpoint (1-8 speakers) at
# ckpt/simu/simu_1-8spk.ckpt. Both finetune configs initialise from it, and
# the zero-shot sanity run uses it directly.
#
#   ./scripts/download_simu_ckpt.sh
#
# OFFLINE SERVERS: do not run this. Copy the file in instead -- it is already
# on the dev laptop at LS-EEND/ckpt/simu/simu_1-8spk.ckpt:
#
#   scp LS-EEND/ckpt/simu/simu_1-8spk.ckpt \
#       <server>:/path/to/FS-EEND/LS-EEND/ckpt/simu/
#
# then re-run this script to verify the checksum (it only downloads when the
# file is absent). Source of truth if you need to re-fetch it on any machine
# with internet:
#   https://drive.google.com/file/d/1uWY8JvjHJJ-SvGiNS-6s3q10g4CY2ePt/view
# Expected: 49977074 bytes,
#   sha256 9cbaceeae89c0fcca6bf7e38d47f33a0bb5fd6690081bacb72a24345594c9448

ID="1uWY8JvjHJJ-SvGiNS-6s3q10g4CY2ePt"
OUT="ckpt/simu/simu_1-8spk.ckpt"
WANT_SHA="9cbaceeae89c0fcca6bf7e38d47f33a0bb5fd6690081bacb72a24345594c9448"
WANT_BYTES=49977074
mkdir -p "$(dirname "$OUT")"

verify() {
  local bytes have
  bytes="$(stat -c '%s' "$OUT" 2>/dev/null || stat -f '%z' "$OUT")"
  if [ "$bytes" != "$WANT_BYTES" ]; then
    echo "SIZE MISMATCH: $OUT is $bytes bytes, expected $WANT_BYTES." >&2
    echo "A few-KB file means an HTML error page was saved; a short file means" >&2
    echo "the transfer was truncated. Re-copy it." >&2
    return 1
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    have="$(sha256sum "$OUT" | cut -d' ' -f1)"
    if [ "$have" != "$WANT_SHA" ]; then
      echo "SHA256 MISMATCH: got $have, expected $WANT_SHA" >&2
      return 1
    fi
    echo "verified: $OUT ($bytes bytes, sha256 ok)"
  else
    echo "verified size only: $OUT ($bytes bytes); sha256sum not available"
  fi
}

if [ -s "$OUT" ]; then
  verify
  exit $?
fi

# Google Drive serves a virus-scan interstitial for files this size, so the
# plain uc?export=download URL returns HTML rather than the model. Request
# once, scrape the confirm form's uuid, then re-request with confirm=t.
tmp="$(mktemp)"
if ! curl -sSL --connect-timeout 10 \
     "https://drive.usercontent.google.com/download?id=${ID}&export=download" -o "$tmp"; then
  rm -f "$tmp"
  echo "" >&2
  echo "No network access to Google Drive." >&2
  echo "This is expected on an offline server -- copy the checkpoint in by hand:" >&2
  echo "  scp LS-EEND/ckpt/simu/simu_1-8spk.ckpt <server>:$(pwd)/ckpt/simu/" >&2
  echo "then re-run this script to verify it." >&2
  exit 1
fi
UUID="$(grep -oE 'name="uuid" value="[^"]+"' "$tmp" | sed 's/.*value="//;s/"//')"
rm -f "$tmp"
if [ -z "$UUID" ]; then
  echo "could not find the confirm uuid; Drive may have changed its flow." >&2
  echo "Download manually from the link in this script's header into ${OUT}." >&2
  exit 1
fi

curl -L "https://drive.usercontent.google.com/download?id=${ID}&export=download&confirm=t&uuid=${UUID}" -o "$OUT"
verify
