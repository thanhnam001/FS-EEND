# Server setup for LS-EEND on MSDWild / RAMC

Mirrors how our DiaPer runs are organised: server-absolute paths, `conda run`,
one script per stage, dscore for scoring.

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
./scripts/download_simu_ckpt.sh                       # -> ckpt/simu/simu_1-8spk.ckpt
DIAPER_ROOT=/data/ocr/namvt17/custom-diaper ./scripts/prepare_8k_data.sh
```

## Train, then score
```bash
GPUS=0 MAX_EPOCHS=50 ./scripts/run_finetune.sh msdwild
./scripts/run_infer_score.sh msdwild logs/finetune_msdwild_8k/version_0/<ckpt>
```
Same with `ramc`. Zero-shot sanity check (no training):
`./scripts/run_infer_score.sh msdwild ./ckpt/simu/simu_1-8spk.ckpt`
-> MSDWild 43.98 at collar 0.25, which is the expected *bad* number: the
simulated checkpoint has never seen MSDWild.

## Three things that are easy to get wrong
1. **8 kHz is mandatory.** The released checkpoint's `input_projection` is
   (256, 345) = 23 mels x 15 context, and `datasets/feature.py` hard-wires
   `logmel23_cummn` to `sr=8000` and never resamples. Our 40-mel 16 kHz
   features (= 600) cannot be fed to these weights. Say so in any comparison:
   near-lossless for RAMC, a real handicap on MSDWild.
2. **`chunk_size` in the infer config sets `max_seqlen`** and must cover the
   longest test recording, not the training chunk length. Their generic config
   (100 s) is 10x too small for MSDWild's 1320 s file and cost 0.24 DER.
3. **Choose `max_epochs` deliberately.** Their 500 suits CALLHOME's small
   epoch, not a 66 h / 150 h corpus. Measured 3.9 s/step at batch 4 on a 6 GB
   card (20.5x realtime); batch 16 needs ~15.5 GB. See the COST block in each
   finetune config, and report the budget used as a deviation from our own
   500-epoch protocol.
