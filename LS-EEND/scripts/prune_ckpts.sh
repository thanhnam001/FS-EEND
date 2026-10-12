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

# DIR may be a version dir, or a model dir holding version_* subdirs. In
# WATCH mode it may not exist yet -- the runner starts this before Lightning
# has created the log dir -- so only a one-shot run requires it up front.
if [ ! -d "$DIR" ] && [ "${WATCH:-0}" -le 0 ]; then
  echo "no such dir: $DIR" >&2; exit 1
fi
case "$KEEP" in ''|*[!0-9]*) echo "KEEP must be a non-negative integer" >&2; exit 1 ;; esac
[ "$KEEP" -ge 1 ] || { echo "refusing KEEP=0; that would delete every checkpoint" >&2; exit 1; }

# Every dir that could hold checkpoints: DIR itself plus any version_* under it.
target_dirs () {
  [ -d "$DIR" ] || return 0
  find "$DIR" -maxdepth 1 -type d -name 'version_*' -print 2>/dev/null
  printf '%s
' "$DIR"
}

prune_dir () {
  local DIR="$1"
  # Sort by the epoch number in the filename, not by mtime: a resumed run can
  # write an old epoch late.
  mapfile -t all < <(
    find "$DIR" -maxdepth 1 -name 'epoch=*.ckpt' -printf '%f\n' 2>/dev/null \
      | sed -E 's/^epoch=([0-9]+).*/\1 &/' | sort -n -k1,1 | cut -d' ' -f2-
  )
  local n=${#all[@]}
  [ "$n" -gt 0 ] || return 0
  if [ "$n" -le "$KEEP" ]; then
    echo "[$(date '+%F %T')] ${DIR}: $n checkpoint(s), keep=$KEEP -- nothing to prune"
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
  echo "[$(date '+%F %T')] ${DIR}: $n present, kept $KEEP, $( [ "$YES" = "--yes" ] && echo removed || echo 'would remove') $drop ($(( freed / 1024 / 1024 )) MB)"
  [ "$YES" = "--yes" ] || echo "  (dry run -- re-run with --yes to actually delete)"
}

prune_once () {
  local d
  while IFS= read -r d; do
    [ -n "$d" ] && prune_dir "$d"
  done < <(target_dirs)
}

if [ "$WATCH" -gt 0 ]; then
  [ "$YES" = "--yes" ] || { echo "WATCH mode requires --yes" >&2; exit 1; }
  echo "pruning $DIR every ${WATCH}s, keeping newest $KEEP. Ctrl-C to stop."
  while true; do prune_once; sleep "$WATCH"; done
else
  prune_once
fi
