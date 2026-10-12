# Server setup for LS-EEND on MSDWild / RAMC

Mirrors how our DiaPer runs are organised: server-absolute paths, `conda run`,
one script per stage, dscore for scoring.

## What needs the internet (and what does not)

| Step | Internet? |
|---|---|
| `download_simu_ckpt.sh` | only if the ckpt is missing -- **copy it in instead**, see below |
| conda env creation | **yes** -- the one real blocker; see "Offline env" |
| `prepare_8k_data.sh` | no (but needs `DATA_ROOT`, see Paths) |
| `run_finetune.sh` / `run_infer_score.sh` | no |

### Checkpoint, offline
The file is already on the dev laptop. Copy it rather than downloading:
```bash
scp LS-EEND/ckpt/simu/simu_1-8spk.ckpt <server>:<repo>/LS-EEND/ckpt/simu/
./scripts/download_simu_ckpt.sh          # with the file present this only VERIFIES
```
Expected: **49,977,074 bytes**, sha256
`9cbaceeae89c0fcca6bf7e38d47f33a0bb5fd6690081bacb72a24345594c9448`.
Upstream source (any machine with internet):
<https://drive.google.com/file/d/1uWY8JvjHJJ-SvGiNS-6s3q10g4CY2ePt/view>

This is the only checkpoint our scripts use. The authors also publish
per-corpus finetuned ones (CALLHOME, DIHARD2/3, AMI) in the table in
`LS-EEND/README.md`; none covers MSDWild or RAMC, so they are not useful here.

### Offline env
`conda create` and `pip install` both reach out to the network. Options, best
first:
1. **Internal mirror** -- if the server has a PyPI/conda proxy, use it as-is;
   nothing below is needed.
2. **Wheelhouse.** On a *Linux* box with internet (wheels are platform-specific,
   so this cannot be built from the Windows laptop):
   ```bash
   pip download -d wheelhouse \
       torch==1.13.0+cu117 torchaudio==0.13.0+cu117 \
       --index-url https://download.pytorch.org/whl/cu117
   pip download -d wheelhouse "pip<24.1" "numpy<2" "pytorch-lightning==1.8.2" \
       "torchmetrics<1.0" pyyaml hyperpyyaml librosa soundfile scipy h5py \
       pyannote.metrics pyannote.core
   ```
   (~2.5 GB, almost all torch+cu117.) Then on the server:
   `pip install --no-index --find-links wheelhouse <same package list>`.
3. **conda-pack** is *not* an option from this laptop: the local envs are
   Windows, the server is Linux.

If the server already has a CUDA-enabled torch env, check it first --
LS-EEND needs only torch, pytorch-lightning 1.8.x, hyperpyyaml, librosa,
soundfile, scipy. It does not need a specific CUDA version beyond whatever
torch was built against.

## Once
```bash
conda create -n lseend python=3.9 -y
python -m pip install "pip<24.1"                 # PL 1.8.2 has a malformed
                                                 # dep spec that pip>=24.1 rejects
python -m pip install torch==1.13.0+cu117 torchaudio==0.13.0+cu117 \
    --index-url https://download.pytorch.org/whl/cu117
python -m pip install "numpy<2" "pytorch-lightning==1.8.2" "torchmetrics<1.0" \
    pyyaml hyperpyyaml librosa soundfile scipy h5py pyannote.metrics pyannote.core
```
Pin `numpy<2` and `torchmetrics<1.0`, and do **not** install
`pytorch-lightning==1.8.6`: it pulls torch 2.8 and destroys the cu117 build.
Call the env's interpreter directly -- `conda run -n <env> python -m pip ...`
has been seen to exit 0 while installing nothing.

```bash
./scripts/download_simu_ckpt.sh     # verify/fetch the ckpt
./scripts/prepare_8k_data.sh        # defaults are already the server's paths
```

## Paths

Three roots, deliberately separate, because on the server the corpora do **not**
live under the repo. The defaults below are the server's real paths, taken from
our own `custom-diaper/scripts/precompute_{msdwild,ramc}.sh`.

| Var | Default | Holds |
|---|---|---|
| `DIAPER_ROOT` | `/data/ocr/namvt17/custom-diaper` | code only: `dscore/` and `dscore/.dscore` |
| `DATA_ROOT` | `/data/ocr/namvt17/dataset/diarization` | the 16 kHz corpora + reference RTTMs (read) |
| `OUT_ROOT` | `$DATA_ROOT` | where the 8 kHz copies get written (~15 GB) |

Under `DATA_ROOT` the scripts expect exactly the layout our precompute scripts
already use:
```
msdwild/kaldi/{train,dev,test}      msdwild_wavs/
msdwild/rttms/few.val.rttm
ramc/kaldi/{train,dev,test}         ramc/MDT2021S003/WAV/
```
and they write `<corpus>/kaldi_8k/<split>/` + `<corpus>_wavs_8k/` under
`OUT_ROOT`. That tree is writable on the server -- our precompute caches are
already written there -- so `OUT_ROOT` only needs overriding to put 15 GB on a
different disk.

Locally the same structure sits under `custom-diaper/database`, so a dev run is
just:
```bash
DATA_ROOT=D:/Python/custom-diaper/database ./scripts/prepare_8k_data.sh
```

Every script now **fails fast** on a missing data dir, reference RTTM or
checkpoint rather than starting and dying later. `run_finetune.sh` rewrites
`train_data_dir`/`val_data_dir`/`init_ckpt` into a run-local copy of the
config, so `OUT_ROOT` is the single source of truth and the config that
actually ran is saved next to the log.

## `prepare_8k_data.sh` -- what it is

A one-off **resampling** pass: it writes an 8 kHz copy of our MSDWild and RAMC
audio and a parallel Kaldi dir pointing at it. Nothing is downloaded and no
features are computed; it is pure format conversion.

**Why it exists.** The released checkpoint's `input_projection` is (256, 345)
= 23 mels x (2*7+1) context, and `datasets/feature.py` hard-wires
`logmel23_cummn` to `sr=8000` and never resamples. Our own pipeline is 40 mels
at 16 kHz (= 600), so the released weights physically cannot take our
features. Using their weights therefore requires 8 kHz input. This must be
stated in any comparison: near-lossless for RAMC (telephony, 8 kHz content
anyway) but a real handicap on MSDWild (wideband, in-the-wild video).

**What it produces**, for each of `{msdwild, ramc}` x `{train, dev, test}`:
- `$OUT_ROOT/<corpus>_wavs_8k/<recid>.wav` -- mono PCM_16 at 8 kHz
  (`scipy.signal.resample_poly`)
- `$OUT_ROOT/<corpus>/kaldi_8k/<split>/` -- `wav.scp` rewritten to absolute paths
  into the above; `segments`, `utt2spk`, `spk2utt`, `rttm`, `reco2dur`,
  `reco2num_spk`, `utt2dur`, `text` copied **verbatim**

Timestamps are untouched because resampling preserves wall-clock time, so the
scoring reference is unchanged (verified: durations match to <1 ms, the copied
rttm is byte-identical to the 16 kHz dir's).

**Requirements**
- No network.
- `$DATA_ROOT/{msdwild,ramc}/kaldi/{train,dev,test}` must exist with the
  16 kHz audio they reference. A missing split is skipped with `[skip]`, and
  the tail line reports `N/6 splits converted` -- check it rather than
  assuming all six ran. A `DATA_ROOT` that does not exist, or that yields zero
  splits, is a hard error.
- Python with `numpy`, `soundfile`, `scipy` (the `lseend` env has all three;
  pass another with `PY=/path/to/python`).
- **Disk: ~15 GB** (~4.6 GB MSDWild + ~10.4 GB RAMC), on top of the 29 GB of
  16 kHz sources. 3,143 + 351 = 3,494 recordings, ~216 h.
- Time: well under an hour, single-threaded and I/O-bound (the resample itself
  runs ~2,500x realtime; writing ~15 GB dominates).
- **Idempotent** -- an already-converted wav is skipped, so an interrupted run
  is resumed by re-running it.

Inference alone only needs the `test` splits; `train`/`dev` are needed only if
you finetune.

## Train, then score

```bash
# 1. finetune (long -- see Cost)
GPUS=0 ./scripts/run_finetune.sh msdwild

# 2. average the last 10 epochs, as their recipe and ours both do
python scripts/average_ckpts.py logs/finetune_msdwild_8k/version_0     --last-n 10 -o logs/finetune_msdwild_8k/version_0/avg_last10.ckpt

# 3. score that averaged checkpoint
./scripts/run_infer_score.sh msdwild     logs/finetune_msdwild_8k/version_0/avg_last10.ckpt
```
Same with `ramc`. Step 2 is not optional for a reported number: the authors
average a window of epochs (their released model is `..._16_25_avg_model.ckpt`,
epochs 16-25) and so does our protocol. Scoring one epoch is a different,
noisier quantity.

**Disk: `save_top_k: -1` keeps every epoch.** The model is 12.47 M params, so
a lightning checkpoint with Adam state is ~150 MB and 500 epochs is **~75 GB
per corpus**. Check free space before launching -- filling the disk at epoch
400 loses the run. That setting is the authors', so it is left alone; if space
is short, lower `save_top_k` in the config and say so.

**Two upstream bugs are already fixed on this branch**, both of which only
surface at the very end of a long run:
- `train_dia_fintun_real.py` extracted `["state_dict"]` and then immediately
  overwrote it with the whole lightning checkpoint, so the post-`fit()`
  averaging died with `TypeError: unsupported operand type(s) for /: 'str'
  and 'int'` -- after all 500 epochs had run.
- the same file could not initialise from the released checkpoint at all
  (legacy `dec.attractor_decoder.layers.*` key names); it now applies the
  same remap the inference path already had. Zero-shot sanity check (no training, no `train`/`dev` data):
`./scripts/run_infer_score.sh msdwild ./ckpt/simu/simu_1-8spk.ckpt`
-> MSDWild 43.98 at collar 0.25, which is the expected *bad* number: the
simulated checkpoint has never seen MSDWild. Run this first -- it exercises the
env, the 8 kHz data and dscore in one go, in a few minutes.

## Three things that are easy to get wrong
1. **8 kHz is mandatory.** See `prepare_8k_data.sh` above.
2. **`chunk_size` in the infer config sets `max_seqlen`** and must cover the
   longest test recording, not the training chunk length. Their generic config
   (100 s) is 10x too small for MSDWild's 1320 s file and cost 0.24 DER.
3. **The training budget is settled: theirs.** DECIDED 2026-10-12 --
   `max_epochs: 500`, `early_stop_epoch` 200 (MSDWild, from their callhome)
   / 500 (RAMC, from their dihard3), `lr: 1e-5`, `batch_size: 4`, all exactly
   as the authors' configs. Our own finetune protocol is also 500 epochs, so
   the budget is matched on both sides; it is NOT a deviation and should not
   be caveated as one. `MAX_EPOCHS=` still exists on `run_finetune.sh` as an
   override for short smoke tests -- do not use it for a reported run.

   Know the wall-clock before starting (see "Cost" below).

## Cost

Their recipe at batch 4, measured per corpus. Steps/epoch is exact (audio
hours / chunk length / batch); the per-step time is the only estimate, and it
is from an **RTX 3060 Laptop 6 GB**, not the server.

| | MSDWild | RAMC |
|---|---|---|
| train audio | 66.1 h | 149.6 h |
| chunk | 20 s | 40 s |
| audio per step (batch 4) | 80 s | 160 s |
| steps / epoch | ~2,975 | ~3,366 |
| measured s/step (6 GB laptop) | 3.9 | ~7.8 |
| h / epoch (same card) | 3.2 | 7.3 |
| **500 epochs on that card** | **~67 days** | **~152 days** |
| earliest early stop | epoch ~201 (patience 200) | never -- patience = max_epochs |

A server GPU is faster, but **by how much has not been measured** -- do not
assume a factor. `run_finetune.sh` logs s/step; read it a few minutes in and
multiply by the steps/epoch x 500 above before committing the machine.

If that number is too large, the honest levers are, in order:
1. **`batch_size`** -- the only one that does not touch the epoch budget.
   Batch 4 leaves a 24 GB card mostly idle; batch 16 needs ~15.5 GB and cuts
   steps/epoch 4x. It IS a recipe deviation (same lr at a larger batch changes
   the optimisation), so it has to be stated if used.
2. **MSDWild first, RAMC later.** MSDWild is 2.3x cheaper and is the corpus
   where the 8 kHz handicap matters most, i.e. the more informative run.
3. **Cut `max_epochs`** -- last resort, and then it genuinely is a deviation
   from both their recipe and ours.
