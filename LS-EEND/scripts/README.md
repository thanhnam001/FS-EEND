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
   `"$ENV/bin/python" -m pip install --no-index --find-links wheelhouse
   <same package list>`.
3. **conda-pack** is *not* an option from this laptop: the local envs are
   Windows, the server is Linux.

If the server already has a CUDA-enabled torch env, check it first --
LS-EEND needs only torch, pytorch-lightning 1.8.x, hyperpyyaml, librosa,
soundfile, scipy. It does not need a specific CUDA version beyond whatever
torch was built against.

## Once

The env is **path-based**, at `/data/ocr/namvt17/FS-EEND/.venv`, mirroring
`custom-diaper/.venv`. Every script takes it as `LSEEND_ENV` and invokes it
with `conda run -p`, so a different location only needs
`LSEEND_ENV=<path>` in front of the command.

```bash
ENV=/data/ocr/namvt17/FS-EEND/.venv
conda create -p "$ENV" python=3.9 -y
# PL 1.8.2 ships a malformed dep spec that pip >= 24.1 rejects
"$ENV/bin/python" -m pip install "pip<24.1"
"$ENV/bin/python" -m pip install torch==1.13.0+cu117 torchaudio==0.13.0+cu117 \
    --index-url https://download.pytorch.org/whl/cu117
"$ENV/bin/python" -m pip install "numpy<2" "pytorch-lightning==1.8.2" \
    "torchmetrics<1.0" pyyaml hyperpyyaml librosa soundfile scipy h5py \
    pyannote.metrics pyannote.core
```
Pin `numpy<2` and `torchmetrics<1.0`, and do **not** install
`pytorch-lightning==1.8.6`: it pulls torch 2.8 and destroys the cu117 build.
Call the env's interpreter directly, as above -- `conda run -p <env> python
-m pip ...` has been seen to exit 0 while installing nothing.

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
| `LSEEND_ENV` | `/data/ocr/namvt17/FS-EEND/.venv` | this project's python env (`conda run -p`) |
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

Every script now **fails fast** on a missing env, data dir, reference RTTM or
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
- Python with `numpy`, `soundfile`, `scipy`. Defaults to
  `$LSEEND_ENV/bin/python`; override with `PY=/path/to/python` (a bare name on
  `PATH` works too).
- **Disk: ~15 GB** (~4.6 GB MSDWild + ~10.4 GB RAMC), on top of the 29 GB of
  16 kHz sources. 3,143 + 351 = 3,494 recordings, ~216 h.
- Time: well under an hour, single-threaded and I/O-bound (the resample itself
  runs ~2,500x realtime; writing ~15 GB dominates).
- **Idempotent** -- an already-converted wav is skipped, so an interrupted run
  is resumed by re-running it.

Inference alone only needs the `test` splits; `train`/`dev` are needed only if
you finetune.

## Train, then score

**MSDWild** -- single stage, their CALLHOME recipe:
```bash
GPUS=0 ./scripts/run_finetune.sh msdwild
/data/ocr/namvt17/FS-EEND/.venv/bin/python scripts/average_ckpts.py     logs/finetune_msdwild_8k/version_0 --last-n 5     -o logs/finetune_msdwild_8k/version_0/avg_last5.ckpt
./scripts/run_infer_score.sh msdwild logs/finetune_msdwild_8k/version_0/avg_last5.ckpt
```

**RAMC** -- four progressive stages, their AMI recipe. The runner chains the
stages, averages last-5 after each, and feeds that into the next:
```bash
GPUS=0 ./scripts/run_finetune_ramc_progressive.sh
./scripts/run_infer_score.sh ramc logs/finetune_ramc_8k_s4_1600s/avg_last5.ckpt
```
Resumable: a completed stage is skipped, and `STAGES="800 1600"` restarts from
part-way (stage numbers stay canonical, and it refuses to start if the previous
stage's output is missing). Batch sizes default to `8 8 4 2`; the authors use 8
throughout on AMI but report hitting GPU memory at the longest lengths, so
override with `BATCHES="8 8 8 8"` on a large card and record which you used.

Zero-shot sanity check (no training, only the `test` split):
`./scripts/run_infer_score.sh msdwild ./ckpt/simu/simu_1-8spk.ckpt`
-> MSDWild 43.98 at collar 0.25, the expected *bad* number. Run it first; it
exercises env, 8 kHz data and dscore in a few minutes.

## Three things that are easy to get wrong

1. **8 kHz is mandatory**, not a choice -- see `prepare_8k_data.sh` above. Say
   so in any comparison: near-lossless for RAMC, a real handicap on MSDWild.
2. **`chunk_size` is in 0.1 s units**, at training *and* inference. At
   inference it sets `max_seqlen` and must cover the longest test recording,
   not the training chunk length. Their generic infer config (10000 = 1000 s)
   is too small for MSDWild's 1320 s file and cost 0.24 DER.
3. **Budget and schedule are corpus-dependent.** See "Recipe mapping" below:
   MSDWild takes their CALLHOME recipe, RAMC their AMI one. Do not copy one
   corpus's `max_epochs` onto the other.

## Recipe mapping (read this before changing a config)

The paper sets the adaptation audio length by the corpus's **average recording
duration**: "Audio length is increased from 200 s until it approximates the
average recording duration of each dataset" (Sec. IV-C), giving 200 s for
CALLHOME, 200->400 for DIHARD II/III, 200->400->800->1600 for AMI. Applying
that rule to our measured averages:

| corpus | avg recording | matches their | schedule | epochs |
|---|---|---|---|---|
| MSDWild | 96 s | CALLHOME (126 s) | 200 s, single step | 500, patience 200 |
| RAMC | 1864 s (31.1 min) | AMI (35.6 min) | 200 -> 400 -> 800 -> 1600 s | 10 per stage, 40 total |

`chunk_size` is in **subsampled frames**: hop 80 @ 8 kHz = 10 ms, subsampling
10, so one unit is **0.1 s** and `chunk_size: 2000` = 200 s. That is what makes
their callhome 2000 / dihard3 4000 reproduce the paper's 200 s / 400 s. The
same units apply at inference, where `chunk_size` sets `max_seqlen` and must
cover the longest test recording: MSDWild 1320.5 s -> 13500, RAMC 1956.1 s ->
20000.

The paper gives epoch counts only for pre-training (Sim2spk / Sim{1-4} /
Sim{1-6} / Sim{1-8} for 100 / 50 / 50 / 25 epochs). Adaptation epoch counts and
the checkpoint-averaging windows exist only in the configs.

**Averaging window: last 5.** Their `*_infer.yaml` files record which epochs
were averaged for each published result -- callhome 416-420, dihard2 140-144,
dihard3 113-117, ami 1-9 -- i.e. the last 5 for real-data finetunes (last 10
for simulated steps; the released `simu_1-8spk` is `..._16_25_avg`, epochs
16-25 of a 25-epoch step).

## Cost

Steps per epoch is exact: audio / chunk length / batch.

| | MSDWild | RAMC |
|---|---|---|
| train audio | 66.1 h | 149.6 h |
| chunk | 200 s | 200 / 400 / 800 / 1600 s |
| batch | 4 | 8 / 8 / 4 / 2 |
| steps per epoch | ~297 | ~337 / 168 / 168 / 168 |
| epochs | 500 | 10 per stage, 40 total |

**Wall-clock has to be measured on the server.** An earlier estimate in this
repo (67 / 152 days) was derived assuming `chunk_size` was in raw 10 ms frames,
i.e. 20 s chunks rather than 200 s, so its realtime factor is not usable.
`run_finetune.sh` and the progressive runner both log s/step: read it a few
minutes in and multiply by the steps/epoch above.

**Disk.** `save_top_k: -1` is their setting and keeps every epoch (~150 MB).
RAMC's 40 epochs are fine (~6 GB). MSDWild's 500 are not (~75 GB) -- run
`scripts/prune_ckpts.sh` alongside it, which is safe against a live job:

```bash
KEEP=6 WATCH=600 ./scripts/prune_ckpts.sh logs/finetune_msdwild_8k/version_0 --yes
```
