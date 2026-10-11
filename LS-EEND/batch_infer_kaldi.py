import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--OMP_NUM_THREADS', type=int, default=1)
_t, _ = parser.parse_known_args()
os.environ["OMP_NUM_THREADS"] = str(_t.OMP_NUM_THREADS)

import time
import warnings

import hyperpyyaml
import torch

from nnet.model.onl_conformer_retention_enc_1dcnn_tfm_retention_enc_linear_non_autoreg_pos_enc_l2norm_emb_loss_mask import (
    OnlineConformerRetentionDADiarization,
)
from datasets.feature import extract_fbank
from train.utils.make_rttm import make_rttm

warnings.filterwarnings("ignore")


def build_model(configs, ckpt, device):
    """Same construction and key remapping as streaming_infer_dia.predict,
    hoisted out of the per-file loop so the model is built once."""
    in_size = (2 * configs["data"]["context_recp"] + 1) * configs["data"]["feat"]["n_mels"]
    model = OnlineConformerRetentionDADiarization(
        n_speakers=configs["data"]["num_speakers"],
        in_size=in_size,
        **configs["model"]["params"],
    ).to(device)

    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k.replace("dec.attractor_decoder.layers.", "dec.layers."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[WARN] missing keys (%d): %s" % (len(missing), missing[:5]))
    if unexpected:
        print("[WARN] unexpected keys (%d): %s" % (len(unexpected), unexpected[:5]))
    model.eval()
    return model


def read_wav_scp(kaldi_dir):
    out = []
    with open(os.path.join(kaldi_dir, "wav.scp")) as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            rx = " ".join(p[1:])
            if not os.path.isabs(rx):
                rx = os.path.normpath(os.path.join(kaldi_dir, rx))
            out.append((p[0], rx))
    return out


def main():
    ap = ArgumentParser()
    ap.add_argument("--OMP_NUM_THREADS", type=int, default=1)
    ap.add_argument("--kaldi_dir", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--median", type=int, default=11,
                    help="median filter width; authors use 11 with a collar and 1 without")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N recordings")
    args = ap.parse_args()

    with open(args.configs) as f:
        configs = hyperpyyaml.load_hyperpyyaml(f)

    device = torch.device("cuda:%d" % args.gpus if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    model = build_model(configs, args.ckpt, device)
    max_nspks = configs["data"]["max_speakers"] + 2

    os.makedirs(args.out_dir, exist_ok=True)
    recs = read_wav_scp(args.kaldi_dir)
    if args.limit:
        recs = recs[:args.limit]
    print("[INFO] %d recordings" % len(recs))

    t0 = time.time()
    for i, (rec, wav) in enumerate(recs, 1):
        out = os.path.join(args.out_dir, rec + ".rttm")
        if os.path.exists(out):
            continue
        feat = extract_fbank(
            wav,
            context_size=configs["data"]["context_recp"],
            input_transform=configs["data"]["feat_type"],
            frame_size=configs["data"]["feat"]["win_length"],
            frame_shift=configs["data"]["feat"]["hop_length"],
            subsampling=configs["data"]["subsampling"],
        ).to(device)
        with torch.no_grad():
            # offline path; the authors verify it matches their streaming
            # path to atol/rtol 1e-3 in streaming_infer_dia.py
            batch_out, _, _ = model.test([feat], [len(feat)], max_nspks=max_nspks)
            pred = torch.sigmoid(batch_out[0][:, 1:]).detach().cpu().float()
        rttm = make_rttm(
            rec=rec,
            pred=pred,
            frame_shift=configs["data"]["feat"]["hop_length"],
            threshold=args.threshold,
            median=args.median,
            subsampling=configs["data"]["subsampling"],
            sampling_rate=configs["data"]["feat"]["sample_rate"],
        )
        with open(out, "w", newline="\n") as f:
            for spk in rttm:
                for line in rttm[spk]:
                    f.write(line + "\n")
        if i % 25 == 0 or i == len(recs):
            el = time.time() - t0
            print("  %d/%d  %.1fs elapsed, %.2fs/rec" % (i, len(recs), el, el / i), flush=True)

    print("[INFO] done -> %s" % args.out_dir)


if __name__ == "__main__":
    main()
