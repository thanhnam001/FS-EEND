#!/bin/bash
set -e
set -o pipefail

# Keep only the newest N epoch checkpoints in a Lightning version dir.
#
# WHY THIS EXISTS: their config sets save_top_k: -1, so every epoch is kept
# (~150 MB each, ~75 GB over 500 epochs). There is no config setting that
# means "keep the last N": with a monitor set, save_top_k keeps the BEST N,
# and with monitor=None Lightning only allows -1/0/1. Since the averaging
# window we need is the LAST few epochs, pruning externally is the only way
# to cap disk without changing what gets averaged.
#
# Safe to run against a LIVE training job: it only removes epoch=*.ckpt files
# older than the newest N, never last.ckpt, never *avg*.ckpt, and Lightning
# does not read back old checkpoints during training.
#
#   ./scripts/prune_ckpts.sh logs/finetune_msdwild_8k/version_0            # dry run
#   ./scripts/prune_ckpts.sh logs/finetune_msdwild_8k/version_0 --yes
#   KEEP=5 WATCH=600 ./scripts/prune_ckpts.sh logs/.../version_0 --yes     # loop
#
# KEEP defaults to 6: the 5 the averaging window needs, plus one margin so a
# checkpoint being written right now is never the one deleted.

DIR="${1:?usage: prune_ckpts.sh <version_dir> [--yes]}"
YES="${2:-}"
KEEP="${KEEP:-6}"
WATCH="${WATCH:-0}"

[ -d "$DIR" ] || { echo "no such dir: $DIR" >&2; exit 1; }
case "$KEEP" in ''|*[!0-9]*) echo "KEEP must be a non-negative integer" >&2; exit 1 ;; esac
[ "$KEEP" -ge 1 ] || { echo "refusing KEEP=0; that would delete every checkpoint" >&2; exit 1; }

prune_once () {
  # Sort by the epoch number in the filename, not by mtime: a resumed run can
  # write an old epoch late.
  mapfile -t all < <(
    find "$DIR" -maxdepth 1 -name 'epoch=*.ckpt' -printf '%f\n' 2>/dev/null \
      | sed -E 's/^epoch=([0-9]+).*/\1 &/' | sort -n -k1,1 | cut -d' ' -f2-
  )
  local n=${#all[@]}
  if [ "$n" -le "$KEEP" ]; then
    echo "[$(date '+%F %T')] $n checkpoint(s), keep=$KEEP -- nothing to prune"
    return 0
  fi
  local drop=$(( n - KEEP ))
  local freed=0
  for f in "${all[@]:0:$drop}"; do
    sz=$(stat -c '%s' "$DIR/$f" 2>/dev/null || echo 0)
    if [ "$YES" = "--yes" ]; then
      rm -f -- "$DIR/$f"
    else
      echo "  would remove $f"
    fi
    freed=$(( freed + sz ))
  done
  echo "[$(date '+%F %T')] $n present, kept $KEEP, $( [ "$YES" = "--yes" ] && echo removed || echo 'would remove') $drop ($(( freed / 1024 / 1024 )) MB)"
  [ "$YES" = "--yes" ] || echo "  (dry run -- re-run with --yes to actually delete)"
}

if [ "$WATCH" -gt 0 ]; then
  [ "$YES" = "--yes" ] || { echo "WATCH mode requires --yes" >&2; exit 1; }
  echo "pruning $DIR every ${WATCH}s, keeping newest $KEEP. Ctrl-C to stop."
  while true; do prune_once; sleep "$WATCH"; done
else
  prune_once
fi
