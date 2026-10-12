# Server setup for LS-EEND on MSDWild / RAMC

Mirrors how our DiaPer runs are organised: server-absolute paths, `conda run`,
one script per stage, dscore for scoring.

## What needs the internet (and what does not)

| Step | Internet? |
|---|---|
| `download_simu_ckpt.sh` | only if the ckpt is missing -- **copy it in instead**, see below |
| conda env creation | **yes** -- the one real blocker; see "Offline env" |
| `prepare_8k_data.sh` | no |
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
./scripts/download_simu_ckpt.sh                       # verify/fetch the ckpt
DIAPER_ROOT=/data/ocr/namvt17/custom-diaper ./scripts/prepare_8k_data.sh
```

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
- `$DB/<corpus>_wavs_8k/<recid>.wav` -- mono PCM_16 at 8 kHz
  (`scipy.signal.resample_poly`)
- `$DB/<corpus>/kaldi_8k/<split>/` -- `wav.scp` rewritten to absolute paths
  into the above; `segments`, `utt2spk`, `spk2utt`, `rttm`, `reco2dur`,
  `reco2num_spk`, `utt2dur`, `text` copied **verbatim**

Timestamps are untouched because resampling preserves wall-clock time, so the
scoring reference is unchanged (verified: durations match to <1 ms, the copied
rttm is byte-identical to the 16 kHz dir's).

**Requirements**
- No network.
- `$DIAPER_ROOT/database/{msdwild,ramc}/kaldi/{train,dev,test}` must exist with
  the 16 kHz audio they reference. A missing split is skipped with `[skip]`,
  not an error -- check the log rather than assuming all six ran.
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
GPUS=0 MAX_EPOCHS=50 ./scripts/run_finetune.sh msdwild
./scripts/run_infer_score.sh msdwild logs/finetune_msdwild_8k/version_0/<ckpt>
```
Same with `ramc`. Zero-shot sanity check (no training, no `train`/`dev` data):
`./scripts/run_infer_score.sh msdwild ./ckpt/simu/simu_1-8spk.ckpt`
-> MSDWild 43.98 at collar 0.25, which is the expected *bad* number: the
simulated checkpoint has never seen MSDWild. Run this first -- it exercises the
env, the 8 kHz data and dscore in one go, in a few minutes.

## Three things that are easy to get wrong
1. **8 kHz is mandatory.** See `prepare_8k_data.sh` above.
2. **`chunk_size` in the infer config sets `max_seqlen`** and must cover the
   longest test recording, not the training chunk length. Their generic config
   (100 s) is 10x too small for MSDWild's 1320 s file and cost 0.24 DER.
3. **Choose `max_epochs` deliberately.** Their 500 suits CALLHOME's small
   epoch, not a 66 h / 150 h corpus. Measured 3.9 s/step at batch 4 on a 6 GB
   card (20.5x realtime); batch 16 needs ~15.5 GB. See the COST block in each
   finetune config, and report the budget used as a deviation from our own
   500-epoch protocol.
