#!/usr/bin/env python3
"""Average a window of training checkpoints into one weights-only file.

The authors average a window of epochs rather than scoring a single one --
their released model is named ..._1-8spk_16_25_avg_model.ckpt (epochs 16-25),
and conf/*.yaml carries start_epoch/end_epoch for exactly this. Our own
protocol averages the last N checkpoints too, so a protocol-matched number
needs this step between training and scoring.

train_dia_fintun_real.py also averages after fit(), but over EVERY .ckpt in
the folder (its epoch filters are commented out), which with save_top_k: -1
means all 500. This script makes the window explicit and can be re-run
without retraining.

Output format matches the released checkpoints: a flat {name: tensor} dict
with the lightning "model." prefix kept, which batch_infer_kaldi.py and
streaming_infer_dia.py both accept.

  python scripts/average_ckpts.py logs/finetune_msdwild_8k/version_0 \
      --first 491 --last 500 -o avg_491_500.ckpt
  python scripts/average_ckpts.py <dir> --last-n 10 -o avg_last10.ckpt
"""
import argparse
import os
import re
import sys

import torch

EPOCH_RE = re.compile(r"epoch=(\d+)")


def find_ckpts(d):
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".ckpt"):
            continue
        m = EPOCH_RE.search(f)
        if m is None:          # skips last.ckpt and any *_avg_model.ckpt
            continue
        out.append((int(m.group(1)), os.path.join(d, f)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--first", type=int, help="first epoch to include")
    ap.add_argument("--last", type=int, help="last epoch to include")
    ap.add_argument("--last-n", type=int, help="instead: the N highest epochs")
    a = ap.parse_args()

    found = find_ckpts(a.ckpt_dir)
    if not found:
        sys.exit("no epoch=*.ckpt files in %s" % a.ckpt_dir)

    if a.last_n:
        sel = found[-a.last_n:]
    else:
        lo = a.first if a.first is not None else found[0][0]
        hi = a.last if a.last is not None else found[-1][0]
        sel = [(e, p) for e, p in found if lo <= e <= hi]
    if not sel:
        sys.exit("no checkpoints matched the requested window; available "
                 "epochs %d..%d" % (found[0][0], found[-1][0]))

    print("averaging %d checkpoints (epochs %d..%d) of %d present"
          % (len(sel), sel[0][0], sel[-1][0], len(found)))

    acc = None
    for _, path in sel:
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if acc is None:
            acc = {k: v.float().clone() for k, v in sd.items()
                   if torch.is_tensor(v)}
            skipped = [k for k, v in sd.items() if not torch.is_tensor(v)]
            if skipped:
                print("  ignoring %d non-tensor entries" % len(skipped))
        else:
            if set(acc) != {k for k, v in sd.items() if torch.is_tensor(v)}:
                sys.exit("key mismatch in %s -- refusing to average "
                         "checkpoints from different models" % path)
            for k in acc:
                acc[k] += sd[k].float()
    for k in acc:
        acc[k] /= len(sel)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(acc, a.out)
    print("wrote %s (%d tensors, %.1f MB)"
          % (a.out, len(acc), os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main()
