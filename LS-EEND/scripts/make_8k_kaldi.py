#!/usr/bin/env python3
"""Make an 8 kHz copy of a Kaldi data dir, for baselines whose released
weights are 8 kHz (LS-EEND: logmel23_cummn is hard-wired to sr=8000 in
datasets/feature.py and nothing in that repo resamples).

Our corpora are 16 kHz mono PCM_16. This writes resampled wavs plus a new
Kaldi dir whose wav.scp points at them; every other Kaldi file is copied
verbatim, so segments/rttm timestamps stay valid (resampling preserves
wall-clock time).

Usage:
  python make_8k_kaldi.py <src_kaldi_dir> <out_kaldi_dir> <out_wav_dir>
"""
import os
import shutil
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 8000
# Copied as-is; only wav.scp is rewritten.
PASSTHROUGH = ("segments", "utt2spk", "spk2utt", "rttm", "reco2dur",
               "reco2num_spk", "utt2dur", "text")


def main(src, out, wavdir):
    os.makedirs(out, exist_ok=True)
    os.makedirs(wavdir, exist_ok=True)

    scp = os.path.join(src, "wav.scp")
    if not os.path.isfile(scp):
        sys.exit("no wav.scp in %s" % src)

    entries = []
    with open(scp) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            # rxfilenames in these dirs are plain relative paths, resolved
            # against the Kaldi dir itself.
            entries.append((parts[0], " ".join(parts[1:])))

    done = skipped = 0
    with open(os.path.join(out, "wav.scp"), "w", newline="\n") as w:
        for recid, rx in entries:
            src_wav = rx if os.path.isabs(rx) else os.path.normpath(
                os.path.join(src, rx))
            dst_wav = os.path.abspath(os.path.join(wavdir, recid + ".wav"))

            if os.path.exists(dst_wav):
                skipped += 1
            else:
                data, sr = sf.read(src_wav, dtype="float64", always_2d=True)
                data = data.mean(axis=1)          # force mono
                if sr != TARGET_SR:
                    g = np.gcd(int(sr), TARGET_SR)
                    data = resample_poly(data, TARGET_SR // g, int(sr) // g)
                # back to PCM_16, matching the source subtype
                data = np.clip(data, -1.0, 1.0)
                sf.write(dst_wav, (data * 32767).astype(np.int16),
                         TARGET_SR, subtype="PCM_16")
                done += 1
            w.write("%s %s\n" % (recid, dst_wav))
            if (done + skipped) % 200 == 0:
                print("  %d/%d" % (done + skipped, len(entries)), flush=True)

    for name in PASSTHROUGH:
        p = os.path.join(src, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(out, name))

    print("%s -> %s : %d converted, %d already present, %d total"
          % (src, out, done, skipped, len(entries)))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
